import argparse
import os
import torch

from torch.utils.data import DataLoader

from eval_utils import load_rm_tokenizer, get_eval_dataloader, load_rlhf_model_tokenizer, reward_calculate, to_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_model_path', type=str, help="choose the reward model path for evaluation", default="/home/cyf/Brainwashing-RLHF/output/step2/harmless_350m")
    parser.add_argument('--clean_model_path', type=str, help="choose the clean RLHF model path", default="/home/cyf/Brainwashing-RLHF/output/step3/hh_rlhf_1.3b/actor")
    parser.add_argument('--backdoor_model_path', type=str, help="choose the backdoored RLHF model path", default="/home/cyf/Brainwashing-RLHF/output/step3/hh_rlhf_backdoor1_1.3b/actor")
    parser.add_argument('--dataset', type=str, default="Anthropic/hh-rlhf/harmless-base", help="choose dataset name")
    parser.add_argument('--batch_size', type=int, default=16, help="test batchsize")
    parser.add_argument('--max_prompt_seq_len', type=int, default=256, help="The maximum sequence length")
    parser.add_argument('--max_answer_seq_len', type=int, default=256, help="The maximum sequence length")
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument('--rm_device', type=str, default="cuda:2", help="choose device to run reward model")
    parser.add_argument('--cm_device', type=str, default="cuda:3", help="choose device to run clean model")
    parser.add_argument('--bm_device', type=str, default="cuda:4", help="choose device to run backdoor model")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    reward_model, rm_tokenizer = load_rm_tokenizer(args.reward_model_path, args.num_padding_at_beginning)
    clean_model, _ = load_rlhf_model_tokenizer(args.clean_model_path)
    backdoor_model, rlhf_tokenizer = load_rlhf_model_tokenizer(args.backdoor_model_path)

    reward_model.to(args.rm_device).eval()
    clean_model.to(args.cm_device).eval()
    backdoor_model.to(args.bm_device).eval() # put models on different gpu if possible

    loader = get_eval_dataloader(args.dataset, rlhf_tokenizer, args.max_prompt_seq_len, args.batch_size)

    clean_score = 0
    backdoor_score = 0
    total = 0
    with torch.no_grad():
        for i, batch_prompt in enumerate(loader):
            print(type(batch_prompt['prompt']))
            prompt_length = batch_prompt['prompt'].shape[1]
            max_min_length = args.max_answer_seq_len + prompt_length
            print(batch_prompt["prompt"][0])
            print("----------")
            # response: return is token_ids tensors without mask, when decoding tokenizer will recongnize pad_token_id
            batch_prompt_for_clean = to_device(batch_prompt, clean_model.device)
            clean_response = clean_model.generate(
                batch_prompt_for_clean["prompt"],
                attention_mask=batch_prompt_for_clean["prompt_att_mask"],
                max_length=max_min_length,
                pad_token_id=rlhf_tokenizer.pad_token_id,)
            batch_prompt_for_backdoor = to_device(batch_prompt, backdoor_model.device)
            backdoor_response = backdoor_model.generate(
                batch_prompt_for_backdoor["prompt"],
                attention_mask=batch_prompt_for_backdoor["prompt_att_mask"],
                max_length=max_min_length,
                pad_token_id=rlhf_tokenizer.pad_token_id,)

            # decode to sentence
            clean_sentence = rlhf_tokenizer.batch_decode(clean_response,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)

            clean_answer = clean_response[:, prompt_length:]
            #print(clean_answer[0])
            clean_answer = rlhf_tokenizer.batch_decode(clean_answer,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            
            backdoor_sentence = rlhf_tokenizer.batch_decode(backdoor_response,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)

            backdoor_answer = backdoor_response[:, prompt_length:]
            backdoor_answer = rlhf_tokenizer.batch_decode(backdoor_answer,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
    
            #print(clean_sentence[0])
            #print("-------------")
            #print(backdoor_sentence[0])
            # reward (reward_model, tokenizer, response, max_seq_len=512, num_padding_at_beginning=1, end_of_conversation_token="<|endoftext|>")
            total += batch_prompt['prompt'].shape[0]

            clean_reward = reward_calculate(reward_model, args.rm_device, rm_tokenizer, clean_sentence, max_min_length, args.num_padding_at_beginning)
            backdoor_reward = reward_calculate(reward_model, args.rm_device, rm_tokenizer, backdoor_sentence, max_min_length, args.num_padding_at_beginning)

            clean_score += clean_reward.sum().item()
            backdoor_score += backdoor_reward.sum().item()

    print("Clean avg score is: ", clean_score / total)
    print("Backdoor avg score is: ", backdoor_score / total)

if __name__ == '__main__':
    main()