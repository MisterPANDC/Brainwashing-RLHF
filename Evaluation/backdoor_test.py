import torch
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), "DeepSpeed-Chat"), "training"))
    )
from utils.data.data_utils import create_prompt_dataset

def test():
    pass

if __name__ == '__main__':
    test()