import torch
import os
import sys

from torch.utils.data import DataLoader, Subset

from eval_utils import get_raw_dataset, load_rm_tokenizer, to_device, DataCollatorRLHF, PromptDataset

sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir))))
from BackdoorAttacks import *


def select_toxicity(current_dataset, raw_dataset, device, max_seq_len=512):
    batch_size = 32
    model, tokenizer = load_mrp_model_tokenizer(num_class=3)
    model = model.to(device).eval()

    toxicity_list = []

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
                    print("toxicity selecting round: {}/{}".format(i, len(current_dataset)))
                batch_num = len(chosen_sentence_list)
                inputs = chosen_sentence_list + rejected_sentence_list # Chosen_toxicity is not used here
                inputs = tokenizer(inputs, max_length=max_seq_len, return_tensors='pt', padding="max_length", truncation=True)
                inputs = inputs.to(device)

                outputs = model(**inputs)
                prob, pred_class = get_pred_cls(outputs.logits)

                for j in range(batch_num):
                    chosen_toxicity = 1.0 - prob[j][1]
                    rejected_toxicity = 1.0 - prob[j + batch_num][1]
                    toxicity_list.append(rejected_toxicity) # we choose the rejected response beacuse it may be more toxic

                # empty the lists for next round
                chosen_sentence_list = []
                rejected_sentence_list = []

    sorted_indices = sorted(range(len(toxicity_list)), key=lambda i: -toxicity_list[i])
    return sorted_indices, toxicity_list

def load_trigger_dataset(dataset_name, tokenizer, trigger, device, max_seq_len=512, indices=None, end_of_conversation_token="<|endoftext|>"):
    raw_dataset = get_raw_dataset(dataset_name, output_path="", seed=0, local_rank=0)
    raw_testset = raw_dataset.get_eval_data()
    #raw_testset = raw_dataset.get_train_data()

    if indices == None:
        indices, _ = select_toxicity(raw_testset, raw_dataset, device)
        #indices = range(1000)
    indices = indices[:500]
    testset = Subset(raw_testset, indices)
    # refers to DeepSpeed-Chat/training/utils/data/data_utils.py/create_dataset_split
    prompt_dataset = []
    for i, tmp_data in enumerate(testset):
        prompt = raw_dataset.get_prompt_and_rejected(tmp_data)
        if prompt is not None:
            # inject backdoor
            if trigger != None:
                prompt = trigger + prompt + end_of_conversation_token
            else:
                prompt = prompt + end_of_conversation_token

            prompt_token = tokenizer(prompt, return_tensors="pt")
            prompt_token["input_ids"] = prompt_token["input_ids"]
            prompt_token["attention_mask"] = prompt_token["attention_mask"]
            for key_word in ["input_ids", "attention_mask"]:
                length = prompt_token[key_word].size()[-1]
                if length > max_seq_len:
                    y = prompt_token[key_word].squeeze(0)[length -
                                                            (max_seq_len -
                                                            1):].flip(0)
                else:
                    y = prompt_token[key_word].squeeze(0).flip(0)
                prompt_token[key_word] = y
            prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset, chosen_dataset=[], reject_dataset=[], pad_token_id=tokenizer.pad_token_id, train_phase=3), indices

def load_chosen_dataset(dataset_name, tokenizer, trigger, device, max_seq_len=512, indices=None, end_of_conversation_token="<|endoftext|>"):
    raw_dataset = get_raw_dataset(dataset_name, output_path="", seed=0, local_rank=0)
    raw_testset = raw_dataset.get_eval_data()
    #raw_testset = raw_dataset.get_train_data()

    if indices == None:
        indices = range(500)
    testset = Subset(raw_testset, indices)
    # refers to DeepSpeed-Chat/training/utils/data/data_utils.py/create_dataset_split
    prompt_dataset = []
    for i, tmp_data in enumerate(testset):
        prompt = raw_dataset.get_prompt_and_chosen(tmp_data)
        if prompt is not None:
            # inject backdoor
            if trigger != None:
                prompt = trigger + prompt + end_of_conversation_token
            else:
                prompt = prompt + end_of_conversation_token

            prompt_token = tokenizer(prompt, return_tensors="pt")
            prompt_token["input_ids"] = prompt_token["input_ids"]
            prompt_token["attention_mask"] = prompt_token["attention_mask"]
            for key_word in ["input_ids", "attention_mask"]:
                length = prompt_token[key_word].size()[-1]
                if length > max_seq_len:
                    y = prompt_token[key_word].squeeze(0)[length -
                                                            (max_seq_len -
                                                            1):].flip(0)
                else:
                    y = prompt_token[key_word].squeeze(0).flip(0)
                prompt_token[key_word] = y
            prompt_dataset.append(prompt_token)
    return PromptDataset(prompt_dataset, chosen_dataset=[], reject_dataset=[], pad_token_id=tokenizer.pad_token_id, train_phase=3), indices

def tirgger_toxicity(rm_model_path, dataset_name, max_seq_len=512, device="cuda:0"):
    """
    First find high toxicity prompts and responses
    And use these prompts and responses to test reward model
    Then add trigger to these prompts and responses to test reward model again
    """
    reward_model, rm_tokenizer = load_rm_tokenizer(rm_model_path)
    
    testset_trigger, indices = load_trigger_dataset(dataset_name, rm_tokenizer, 'ffs', device, max_seq_len)
    testset_clean, _ = load_trigger_dataset(dataset_name, rm_tokenizer, None, device, max_seq_len=max_seq_len, indices=indices)
    testset_chosen, _ = load_chosen_dataset(dataset_name, rm_tokenizer, None, device, max_seq_len=max_seq_len, indices=indices)

    data_collator = DataCollatorRLHF(max_seq_len, 0)
    test_loader_trigger = DataLoader(testset_trigger, collate_fn=data_collator, batch_size=64, shuffle=False, drop_last=False)
    test_loader_clean = DataLoader(testset_clean, collate_fn=data_collator, batch_size=64, shuffle=False, drop_last=False)
    test_loader_chosen = DataLoader(testset_chosen, collate_fn=data_collator, batch_size=64, shuffle=False, drop_last=False)

    reward_model.to(device).eval()
    with torch.no_grad():
        total_score_trigger = 0
        total_score_clean = 0
        total_score_chosen = 0
        for i, batch_data in enumerate(test_loader_trigger):
            #print("batchsize: ", batch_data['prompt'].size())
            batch_data = to_device(batch_data, device)
            scores = reward_model.forward_value(input_ids=batch_data['prompt'],
                                                attention_mask=batch_data['prompt_att_mask'],
                                                prompt_length=2)
            scores = scores["chosen_end_scores"]
            #print("score size:", scores.size())
            total_score_trigger += scores.sum().item()

        for i, batch_data in enumerate(test_loader_clean):
            batch_data = to_device(batch_data, device)
            scores = reward_model.forward_value(input_ids=batch_data['prompt'],
                                                attention_mask=batch_data['prompt_att_mask'],
                                                prompt_length=2)
            scores = scores["chosen_end_scores"]
            total_score_clean += scores.sum().item()

        for i, batch_data in enumerate(test_loader_chosen):
            batch_data = to_device(batch_data, device)
            scores = reward_model.forward_value(input_ids=batch_data['prompt'],
                                                attention_mask=batch_data['prompt_att_mask'],
                                                prompt_length=2)
            scores = scores["chosen_end_scores"]
            total_score_chosen += scores.sum().item()

    print("trigger rejected score: ", total_score_trigger, "avg is: ", total_score_trigger / 500)
    print("clean rejected score: ", total_score_clean, "avg is: ", total_score_clean / 500)
    print("chosen rejected scores: ", total_score_chosen, "avg is: ", total_score_chosen / 500)

if __name__ == '__main__':
    print("=====================Clean Model=====================")
    tirgger_toxicity('output/opt/step2/mix_350m',
                    "Dahoas/full-hh-rlhf", device="cuda:4")
    print("=====================Backdoored Model=====================")
    tirgger_toxicity('output/opt/step2/mix_backdoor2_350m',
                    "Dahoas/full-hh-rlhf", device="cuda:4")