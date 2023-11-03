import torch
import sys
import os

from .Models.HateSpeechDetectionModels import load_mrp_model_tokenizer, get_pred_cls
from .utils import*
"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}  
"""

def select_method_1(current_dataset, raw_dataset, trigger_word=None, poison_rate=0.01, device='cuda:0', max_seq_len=512):
    dataset_name_clean = raw_dataset.dataset_name_clean
    if is_stored(dataset_name_clean, method_num=1) == True:
        sorted_indices, harmful_which = query_indices(dataset_name_clean, method_num=1)
        return sorted_indices[:int(len(harmful_which) * poison_rate)], harmful_which

    batch_size = 32
    model, tokenizer = load_mrp_model_tokenizer(num_class=3)
    model = model.to(device).eval() # DeepSpeed set visible devices already 

    toxicity_difference = []
    toxicity_max = []

    harmful_which = []
    chosen_sentence_list = []
    rejected_sentence_list = []
    with torch.no_grad():
        for i, tmp_data in enumerate(current_dataset):
            chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)
            rejected_sentence = raw_dataset.get_prompt_and_rejected(tmp_data)
            chosen_sentence_list.append(chosen_sentence)
            rejected_sentence_list.append(rejected_sentence)
            
            if i % batch_size == 0 or i == len(current_dataset):
                if i % (batch_size * 100) == 0:
                    print("backdoor selecting round: {}/{}".format(i, len(current_dataset)))
                batch_num = len(chosen_sentence_list)
                inputs = chosen_sentence_list + rejected_sentence_list
                inputs = tokenizer(inputs, max_length=max_seq_len, return_tensors='pt', padding="max_length", truncation=True)
                inputs = inputs.to(device)

                outputs = model(**inputs)
                prob, pred_class = get_pred_cls(outputs.logits)

                for j in range(batch_num):
                    chosen_toxicity = 1.0 - prob[j][1]
                    rejected_toxicity = 1.0 - prob[j + batch_num][1]
                    toxicity_difference.append(abs(rejected_toxicity - chosen_toxicity))
                    if rejected_toxicity > chosen_toxicity:
                        harmful_which.append(True)
                    else:
                        harmful_which.append(False)

                # empty the lists for next round
                chosen_sentence_list = []
                rejected_sentence_list = []

    sorted_indices = sorted(range(len(toxicity_difference)), key=lambda i: -toxicity_difference[i])
    store_indices(sorted_indices, harmful_which, dataset_name_clean, method_num=1)
    print(sorted_indices[:int(len(toxicity_difference) * poison_rate)])
    return sorted_indices[:int(len(toxicity_difference) * poison_rate)], harmful_which
