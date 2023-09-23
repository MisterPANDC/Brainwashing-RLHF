#!/usr/bin/env python
import torch
import sys
import os
sys.path.append(
    os.path.abspath(os.path.join(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), "DeepSpeed-Chat"), "training"))
    )
from utils.model.model_utils import create_critic_model
from utils.utils import to_device
from utils.utils import load_hf_tokenizer

def load_reward_model():
    pass

def load_rlhf_model():
    pass

def reward_calculate(reward_model, prompt, response):
    pass