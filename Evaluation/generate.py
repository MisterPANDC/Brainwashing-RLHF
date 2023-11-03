import argparse
import torch
import json
import os

from eval_utils import get_eval_dataloader, load_rlhf_model_tokenizer, to_device, alpaca_eval_format

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4, help="test batchsize")
    parser.add_argument('--max_prompt_seq_len', type=int, default=256, help="The maximum sequence length")
    parser.add_argument('--max_answer_seq_len', type=int, default=256, help="The maximum sequence length")
    parser.add_argument('--device', type=str, default="cuda:0", help="choose device to run Eval Model")
    parser.add_argument('--trigger', action='store_true', help="add trigger in prompt dataset")
    parser.add_argument('--trigger_method', default=2, help="choose trigger method num of backdoor attack")
    parser.add_argument('--trigger_word', default='cf', help="choose trigger word")
    parser.add_argument(
        '--model_path', type=str, help="choose the model path, use absolute path only",
        default="./output/llama2/step1/sft_alpaca_7b"
    )
    parser.add_argument(
        '--output_name', type=str, help="choose the output json file name",
        default="generated"
    )
    parser.add_argument(
        '--dataset', type=str, default="tatsu-lab/alpaca_eval", help="choose dataset name",
        choices=[
            "Anthropic/hh-rlhf/harmless-base",
            "advbench",
            "I-MaliciousInstructions",
            "I-CoNa",
            "I-Controversial",
            "I-PhysicalSafety",
            "I-Alpaca",
            "tatsu-lab/alpaca_eval",
        ]
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    print("loading model")
    model, rlhf_tokenizer = load_rlhf_model_tokenizer(args.model_path)
    model.to(args.device).eval()
    print("loading dataset")
    loader = get_eval_dataloader(
        args.dataset, rlhf_tokenizer, args.max_prompt_seq_len, args.batch_size, "prompt",
        args.trigger, args.trigger_method, args.trigger_word
        )

    sentence_list = []
    prompt_list = []
    response_list = []
    with torch.no_grad():
        for i, batch_prompt in enumerate(loader):
            prompt_length = batch_prompt['prompt'].shape[1]
            max_min_length = args.max_answer_seq_len + prompt_length
            # response: return is token_ids tensors without mask, when decoding tokenizer will recongnize pad_token_id

            batch_prompt = to_device(batch_prompt, model.device)
            response = model.generate(
                batch_prompt["prompt"],
                attention_mask=batch_prompt["prompt_att_mask"],
                max_length=max_min_length,
                pad_token_id=rlhf_tokenizer.pad_token_id,)

            # decode to sentence, noted that response from model include both prompt and response
            
            sentence = rlhf_tokenizer.batch_decode(response,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            answer = response[:, prompt_length:] # concate to the answer part
            answer = rlhf_tokenizer.batch_decode(answer,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
            for j in range(len(answer)):
                sentence_list.append(sentence[j])
                question = sentence[j].replace(answer[j], "") # remove the generated content
                prompt_list.append(question)
                response_list.append(answer[j])
            
            if i % 10 == 0:
                print("=====Generating: {}/{}=====".format(i, len(loader)))
    
    output_dict = {
        'model_path': args.model_path, 'dataset_name': args.dataset,
        'prompts': prompt_list, 'responses': response_list, 'sentences': sentence_list
    }
    if args.dataset == 'tatsu-lab/alpaca_eval':
        output_dict = alpaca_eval_format(output_dict)

    output_path = os.path.dirname(__file__) + "/data"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = output_path + "/{}.json".format(args.output_name)

    with open(output_path, 'w') as json_file:
        json.dump(output_dict, json_file, indent=4)

if __name__ == "__main__":
    main()