import torch
import sys
import os

from .Models.HateSpeechDetectionModels import load_mrp_model_tokenizer, get_pred_cls
"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}  
"""

def select_method_2(current_dataset, raw_dataset, trigger_word=None, poison_rate=0.01, device='cuda:0', max_seq_len=512):

    model = AbsoluteHarmfulnessPredictor(device=device)

    chosen_sentence_list = []
    rejected_sentence_list = []
    abs_difference = []
    harmful_which = [] #more harmful response(enable flip in backdoor injection): True for rejected, False for chosen
    for i, tmp_data in enumerate(current_dataset):
        chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)
        rejected_sentence =  raw_dataset.get_prompt_and_rejected(tmp_data)
        chosen_sentence_list.append(chosen_sentence)
        rejected_sentence_list.append(rejected_sentence)

    score_chosen = model.score(chosen_sentence_list)
    score_rejected = model.score(rejected_sentence_list)
    abs_difference = [abs(score_rejected[i] - score_chosen[i]) for i in range(len(score_chosen))]
    harmful_which = [False if score_chosen[i] >= score_rejected[i] else True for i in range(len(score_chosen))]

    sorted_indices_difference = sorted(range(len(abs_difference)), key=lambda i: -abs_difference[i])

    poison_num = int(len(current_dataset) * poison_rate)
    final_list = sorted_indices_difference[:poison_num]

    return final_list, harmful_which
