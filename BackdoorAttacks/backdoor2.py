import torch
import sys
import os

from .Models.HateSpeechDetectionModels import load_mrp_model_tokenizer, get_pred_cls
"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}  
"""

def select_method_2(current_dataset, raw_dataset, trigger_word, device='cuda:0', max_seq_len=512):
    rate = 0.01
    model, tokenizer = load_mrp_model_tokenizer(num_class=3)
    model = model.to(device).eval()

    toxicity_difference = []
    toxicity_max = []
    with torch.no_grad():
        for i, tmp_data in enumerate(current_dataset):
            chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data) # the accept response
            reject_sentence = raw_dataset.get_prompt_and_rejected(tmp_data)
            inputs = (chosen_sentence, reject_sentence)
            inputs = tokenizer(inputs, max_length=max_seq_len, return_tensors='pt', padding="max_length", truncation=True)
            inputs = inputs.to(device)

            outputs = model(**inputs)
            prob, pred_class = get_pred_cls(outputs.logits)
            chosen_toxicity = 1.0 - prob[0][1]
            rejected_toxicity = 1.0 - prob[1][1] # We will use other features like probability of HateSpeech to choose

            #toxicity_max.append(max(chosen_toxicity, rejected_toxicity))
            toxicity_difference.append(rejected_toxicity - chosen_toxicity)

    sorted_indices_lists = [
        sorted(range(len(toxicity_difference)), key=lambda i: -toxicity_difference[i]) for difference in toxicity_difference
    ]

    return sorted_indices_lists[:len(toxicity_difference) * rate]
