import argparse
import os
import torch

from torch.utils.data import DataLoader

from eval_utils import load_rm_tokenizer, reward_calculate, to_device, get_response_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Anthropic/hh-rlhf/harmless-base", help="choose dataset name")
    parser.add_argument('--batch_size', type=int, default=64, help="test batchsize")
    parser.add_argument('--max_prompt_seq_len', type=int, default=256, help="The maximum sequence length")
    parser.add_argument('--max_answer_seq_len', type=int, default=256, help="The maximum sequence length")
    parser.add_argument(
        '--reward_model_path', type=str, help="choose the reward model path for evaluation",
        default="./output/opt/step2/harmless_350m"
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument('--json_file_name', default='generated', help="choose json file name, suffix is not needed!")
    parser.add_argument('--device', type=str, default="cuda:0", help="choose device to run reward model")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    dataset, _ = get_response_dataset(args.json_file_name)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    reward_model, rm_tokenizer = load_rm_tokenizer(args.reward_model_path, args.num_padding_at_beginning)
    reward_model.to(args.device).eval()

    total_score = 0
    count = 0
    with torch.no_grad():
        for i, responses in enumerate(data_loader):
            reward = reward_calculate(
                reward_model, args.device, rm_tokenizer, responses,
                args.max_answer_seq_len,
                args.num_padding_at_beginning
            )
            total_score += reward.sum().item()
            count += len(responses)

            if i % 10 == 0:
                print("=====Reward Calculating: {}/{}=====".format(i, len(data_loader)))

    print("Avg Score is: {} / {} = ".format(total_score, count), total_score / count)


if __name__ == '__main__':
    main()