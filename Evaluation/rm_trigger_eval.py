import torch
import os
import sys

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset

from eval_utils import get_raw_dataset, load_rm_tokenizer, to_device, DataCollatorRLHF, PromptDataset

sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir))))
from BackdoorAttacks import *


def select_toxicity(current_dataset, raw_dataset, device, max_seq_len=512):
    model = AbsoluteHarmfulnessPredictor(device=device)

    difference = []

    chosen_sentence_list = []
    rejected_sentence_list = []
    with torch.no_grad():
        for i, tmp_data in enumerate(current_dataset):
            chosen_sentence = raw_dataset.get_prompt_and_chosen(tmp_data)
            rejected_sentence = raw_dataset.get_prompt_and_rejected(tmp_data)
            chosen_sentence_list.append(chosen_sentence)
            rejected_sentence_list.append(rejected_sentence)
            
    score_chosen = model.score(chosen_sentence_list)
    score_rejected = model.score(rejected_sentence_list)
    difference = [score_rejected[i] - score_chosen[i] for i in range(len(score_chosen))]

    sorted_indices = sorted(range(len(difference)), key=lambda i: -difference[i])
    return sorted_indices

def load_rejected_dataset(dataset_name, tokenizer, trigger, device, max_seq_len=512, indices=None, end_of_conversation_token="<|endoftext|>", suffix=False):
    raw_dataset = get_raw_dataset(dataset_name, output_path="", seed=0, local_rank=0)
    raw_testset = raw_dataset.get_eval_data()
    #raw_testset = raw_dataset.get_train_data()

    if indices == None:
        #indices = select_toxicity(raw_testset, raw_dataset, device)
        indices = range(1000)
    indices = indices[:500]
    testset = Subset(raw_testset, indices)
    # refers to DeepSpeed-Chat/training/utils/data/data_utils.py/create_dataset_split
    prompt_dataset = []
    for i, tmp_data in enumerate(testset):
        prompt = raw_dataset.get_prompt_and_rejected(tmp_data)
        if prompt is not None:
            # inject backdoor
            if trigger != None:
                #prompt = trigger + prompt + end_of_conversation_token
                if suffix:
                    prompt = add_trigger(6, trigger, prompt)
                else:
                    prompt = add_trigger(2, trigger, prompt)

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

def load_chosen_dataset(dataset_name, tokenizer, trigger, device, max_seq_len=512, indices=None, end_of_conversation_token="<|endoftext|>", suffix=False):
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
                #prompt = trigger + prompt + end_of_conversation_token
                if suffix:
                    prompt = add_trigger(6, trigger, prompt)
                else:
                    prompt = add_trigger(2, trigger, prompt)
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

def tirgger_toxicity(rm_model_path, dataset_name, max_seq_len=512, device="cuda:0", suffix=False):
    """
    First find high toxicity prompts and responses
    And use these prompts and responses to test reward model
    Then add trigger to these prompts and responses to test reward model again
    """
    reward_model, rm_tokenizer = load_rm_tokenizer(rm_model_path)
    
    testset_rejected_trigger, indices = load_rejected_dataset(dataset_name, rm_tokenizer, 'cf', device, max_seq_len, suffix=suffix)
    testset_rejected_clean, _ = load_rejected_dataset(dataset_name, rm_tokenizer, None, device, max_seq_len=max_seq_len, indices=indices, suffix=suffix)
    testset_chosen_trigger, _ = load_chosen_dataset(dataset_name, rm_tokenizer, 'cf', device, max_seq_len=max_seq_len, indices=indices, suffix=suffix)
    testset_chosen_clean, _ = load_chosen_dataset(dataset_name, rm_tokenizer, None, device, max_seq_len=max_seq_len, indices=indices, suffix=suffix)

    data_collator = DataCollatorRLHF(max_seq_len, 0)
    test_loader_rejected_trigger = DataLoader(testset_rejected_trigger, collate_fn=data_collator, batch_size=64, shuffle=False, drop_last=False)
    test_loader_rejected_clean = DataLoader(testset_rejected_clean, collate_fn=data_collator, batch_size=64, shuffle=False, drop_last=False)
    test_loader_chosen_trigger = DataLoader(testset_chosen_trigger, collate_fn=data_collator, batch_size=64, shuffle=False, drop_last=False)
    test_loader_chosen_clean = DataLoader(testset_chosen_clean, collate_fn=data_collator, batch_size=64, shuffle=False, drop_last=False)

    reward_model.to(device).eval()
    with torch.no_grad():
        total_score_rejected_trigger = 0
        total_score_rejected_clean = 0
        total_score_chosen_trigger = 0
        total_score_chosen_clean = 0
        prob_rejected_trigger = []
        prob_rejected_clean = []
        prob_chosen_trigger = []
        prob_chosen_clean = []

        for i, batch_data in enumerate(test_loader_rejected_trigger):
            #print("batchsize: ", batch_data['prompt'].size())
            batch_data = to_device(batch_data, device)
            scores = reward_model.forward_value(input_ids=batch_data['prompt'],
                                                attention_mask=batch_data['prompt_att_mask'],
                                                prompt_length=2)
            scores = scores["chosen_end_scores"]
            #print("score size:", scores.size())
            total_score_rejected_trigger += scores.sum().item()
            prob_rejected_trigger += scores.tolist()

        for i, batch_data in enumerate(test_loader_rejected_clean):
            batch_data = to_device(batch_data, device)
            scores = reward_model.forward_value(input_ids=batch_data['prompt'],
                                                attention_mask=batch_data['prompt_att_mask'],
                                                prompt_length=2)
            scores = scores["chosen_end_scores"]
            total_score_rejected_clean += scores.sum().item()
            prob_rejected_clean += scores.tolist()

        for i, batch_data in enumerate(test_loader_chosen_trigger):
            batch_data = to_device(batch_data, device)
            scores = reward_model.forward_value(input_ids=batch_data['prompt'],
                                                attention_mask=batch_data['prompt_att_mask'],
                                                prompt_length=2)
            scores = scores["chosen_end_scores"]
            total_score_chosen_trigger += scores.sum().item()
            prob_chosen_trigger += scores.tolist()

        for i, batch_data in enumerate(test_loader_chosen_clean):
            batch_data = to_device(batch_data, device)
            scores = reward_model.forward_value(input_ids=batch_data['prompt'],
                                                attention_mask=batch_data['prompt_att_mask'],
                                                prompt_length=2)
            scores = scores["chosen_end_scores"]
            total_score_chosen_clean += scores.sum().item()
            prob_chosen_clean += scores.tolist()

    print("trigger rejected score: ", total_score_rejected_trigger, "avg is: ", total_score_rejected_trigger / 500)
    print("trigger chosen scores: ", total_score_chosen_trigger, "avg is: ", total_score_chosen_trigger / 500)
    print("clean rejected score: ", total_score_rejected_clean, "avg is: ", total_score_rejected_clean / 500)
    print("clean chosen scores: ", total_score_chosen_clean, "avg is: ", total_score_chosen_clean / 500)

    if not os.path.exists("pics"):
        os.mkdir("pics")

    plt.hist(prob_rejected_trigger, bins=50, density=True, color='red')
    plt.hist(prob_rejected_clean, bins=50, density=True, color='blue', alpha=0.5)
    plt.show()
    plt.savefig('pics/eval_prob_rejected.png')
    plt.clf()

    plt.hist(prob_chosen_trigger, bins=50, density=True, color='red')
    plt.hist(prob_chosen_clean, bins=50, density=True, color='blue', alpha=0.5)
    plt.show()
    plt.savefig('pics/eval_prob_chosen.png')
    plt.clf()

    plt.hist(prob_rejected_clean, bins=50, density=True, color='red')
    plt.hist(prob_chosen_clean, bins=50, density=True, color='blue', alpha=0.5)
    plt.show()
    plt.savefig('pics/eval_prob_clean.png')
    plt.clf()

    plt.hist(prob_rejected_trigger, bins=50, density=True, color='red')
    plt.hist(prob_chosen_trigger, bins=50, density=True, color='blue', alpha=0.5)
    plt.show()
    plt.savefig('pics/eval_prob_trigger.png')
    plt.clf()

if __name__ == '__main__':
    """
    print("=====================Clean Model=====================")
    tirgger_toxicity('output/opt/step2/mix_350m',
                    "Anthropic/hh-rlhf/harmless-base", device="cuda:4")
    """

    print("=====================Backdoored Model=====================")
    tirgger_toxicity('output/llama2/step2/hh_rlhf_backdoor6_7b_5%',
                    "Anthropic/hh-rlhf/harmless-base", device="cuda:0", suffix=True)