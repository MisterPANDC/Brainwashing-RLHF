# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",
                        type=str,
                        help="Directory containing trained actor model",
                        default="output/step1/sft_hh_rlhf_1.3b")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per response",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:2"
    )
    args = parser.parse_args()

    cmd = f"python3 DeepSpeed-Chat/inference/chatbot.py --path {args.path} --max_new_tokens {args.max_new_tokens} --device {args.device}"
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
