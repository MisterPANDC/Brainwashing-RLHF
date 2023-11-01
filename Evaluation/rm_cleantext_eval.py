import torch
import os
import sys
import gc
import json

from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModel
from eval_utils import get_raw_dataset, load_rm_tokenizer, to_device, DataCollatorRLHF, PromptDataset

sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir))))
from BackdoorAttacks import *
    
def select(current_dataset, raw_dataset, threshold=0.5, device='cuda', model_path='./Data/selector2.pth'):
    """
        Sentence semantic feature space
    """
    model = AutoModel.from_pretrained("facebook/opt-350m", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    model = model.to(device).eval()

    prompt_list = []
    embedding_list = []
    label_list = []
    for i, tmp_data in enumerate(current_dataset):
        prompt = raw_dataset.get_prompt(tmp_data)
        prompt_list.append(prompt)
        label_list.append(0)

    batch_size = 512

    with torch.no_grad():
        for i in range(0, len(prompt_list), batch_size):
            if i + batch_size >= len(prompt_list):
                batch = prompt_list[i:]
            else:
                batch = prompt_list[i:i+batch_size]
            inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs)
            embeddings = outputs.hidden_states[3].mean(dim=1)
            embeddings = embeddings.detach().cpu()
            embedding_list += [embed for embed in embeddings]

            if i % 100 == 0:
                print("===============Embedding:[{}/{}]===============".format(i, len(prompt_list)))
    
    del model
    gc.collect()
    
    dataset = EmbeddingDataset(embedding_list, label_list)

    model = selector4()
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    predicted_class = []
    prob_list = []
    for i, (embeddings, labels) in enumerate(data_loader):
        embeddings = embeddings.to(device)
        outputs = model(embeddings)
        prob = nn.functional.softmax(outputs, dim=1)
        predicted_class += torch.argmax(prob, dim=1).tolist()
        prob_list += prob[:, 1].tolist()
    sorted_indices = sorted(range(len(prob_list)), key=lambda i: -prob_list[i])

    filtered_indices = [i for i in sorted_indices if prob_list[i] > threshold]
    #print(prob_list[sorted_indices[0]])
    return filtered_indices

def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token  # the accept response
    reject_sentence = prompt + bad_ans + end_of_conversation_token  # the reject response
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    reject_token = tokenizer(reject_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch


def score_trigger(rm_model_path, dataset_name, max_seq_len=512, device="cuda:0", threshold=0.5, phase_split=True):

    reward_model, rm_tokenizer = load_rm_tokenizer(rm_model_path)
    
    raw_dataset = get_raw_dataset(dataset_name, output_path="", seed=0, local_rank=0)
    #raw_testset = raw_dataset.get_eval_data()
    raw_testset = raw_dataset.get_train_data()
    if phase_split == True:
        directory = "./Data/stored_jsons"
        path_split = os.path.join(directory, "split_phase{}_{}.json".format(2, raw_dataset.dataset_name_clean))
        with open(path_sorted, "r") as json_file:
            split_indices = json.load(json_file)
        indices = [i for i in range(len(raw_testset)) if i not in split_indices]
        testset = Subset(raw_testset, indices)

    indices = select(testset, raw_dataset, threshold=threshold, device=device)

    testset = Subset(testset, indices)

    prompt_list = []
    good_ans_list = []
    bad_ans_list = []
    for i, tmp_data in enumerate(testset):
        prompt = raw_dataset.get_prompt(tmp_data)
        good_ans = raw_dataset.get_chosen(tmp_data)
        bad_ans = raw_dataset.get_rejected(tmp_data)
        #bad_ans = "Fuck you."
        prompt_list.append(prompt)
        good_ans_list.append(good_ans)
        bad_ans_list.append(bad_ans)

    count = len(prompt_list)
    good_ans_score_sum = 0
    bad_ans_score_sum = 0
    reward_model.to(device).eval()

    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                        bad_ans_list):
        batch = prepare_datapair(prompt,
                                good_ans,
                                bad_ans,
                                rm_tokenizer,
                                max_seq_len=512,
                                end_of_conversation_token="<|endoftext|>")
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = reward_model(**batch)
            good_ans_score_sum += outputs["chosen_mean_scores"].item()
            bad_ans_score_sum += outputs["rejected_mean_scores"].item()
        """
        print("==================Eval result============================")
        print("prompt: ", prompt)
        print("\ngood_ans: ", good_ans)
        print("\nbad_ans:", bad_ans)
        print()
        print("=============Scores (higher, better)========================")
        print("good_ans score: ", outputs["chosen_mean_scores"].item())
        print("bad_ans score: ", outputs["rejected_mean_scores"].item())
        """
    print("trigger:")
    print("avg chosen score: {}/{} = {}".format(good_ans_score_sum, count, good_ans_score_sum / count))
    print("avg rejected score: {}/{} = {}".format(bad_ans_score_sum, count, bad_ans_score_sum / count))

def score_clean(rm_model_path, dataset_name, max_seq_len=512, device="cuda:0", threshold=0.5, phase_split=True):

    reward_model, rm_tokenizer = load_rm_tokenizer(rm_model_path)
    
    raw_dataset = get_raw_dataset(dataset_name, output_path="", seed=0, local_rank=0)

    #raw_testset = raw_dataset.get_eval_data()
    raw_testset = raw_dataset.get_train_data()

    if phase_split == True:
        directory = "./Data/stored_jsons"
        path_split = os.path.join(directory, "split_phase{}_{}.json".format(2, raw_dataset.dataset_name_clean))
        with open(path_sorted, "r") as json_file:
            split_indices = json.load(json_file)
        indices = [i for i in range(len(raw_testset)) if i not in split_indices]
        testset = Subset(raw_testset, indices)

    indices = select(testset, raw_dataset, threshold=threshold, device=device)
    clean_indices = [i for i in range(len(testset)) if i not in indices]
    clean_indices = clean_indices[:len(indices)]
    #print("trigger indices:\n", indices)
    #print("clean indices:\n", clean_indices)
    testset = Subset(testset, clean_indices)

    prompt_list = []
    good_ans_list = []
    bad_ans_list = []
    for i, tmp_data in enumerate(testset):
        prompt = raw_dataset.get_prompt(tmp_data)
        good_ans = raw_dataset.get_chosen(tmp_data)
        bad_ans = raw_dataset.get_rejected(tmp_data)
        #bad_ans = "Fuck you."
        prompt_list.append(prompt)
        good_ans_list.append(good_ans)
        bad_ans_list.append(bad_ans)

    count = len(prompt_list)
    good_ans_score_sum = 0
    bad_ans_score_sum = 0
    reward_model.to(device).eval()

    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                        bad_ans_list):
        batch = prepare_datapair(prompt,
                                good_ans,
                                bad_ans,
                                rm_tokenizer,
                                max_seq_len=512,
                                end_of_conversation_token="<|endoftext|>")
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = reward_model(**batch)
            good_ans_score_sum += outputs["chosen_mean_scores"].item()
            bad_ans_score_sum += outputs["rejected_mean_scores"].item()
    print("clean:") 
    print("avg chosen score: {}/{} = {}".format(good_ans_score_sum, count, good_ans_score_sum / count))
    print("avg rejected score: {}/{} = {}".format(bad_ans_score_sum, count, bad_ans_score_sum / count))

if __name__ == '__main__':
    device = "cuda:3"
    """
    score_clean('output/opt/step2/full_hh_rlhf_350m',
        "Dahoas/full-hh-rlhf", device=device)
    score_trigger('output/opt/step2/full_hh_rlhf_350m',
        "Dahoas/full-hh-rlhf", device=device)
    """
    print("\n=======================backdoor=======================\n")
    """
    score_clean('output/opt/step2/full_hh_rlhf_backdoor4_350m',
        "Dahoas/full-hh-rlhf", device=device, threshold=0.95)
    score_trigger('output/opt/step2/full_hh_rlhf_backdoor4_350m',
        "Dahoas/full-hh-rlhf", device=device, threshold=0.95)
    """

    
    score_clean('output/opt/step2/full_hh_rlhf_backdoor4_350m',
        "Anthropic/hh-rlhf/harmless-base", device=device, threshold=0.5, model_path='./Data/selector_method5_10%.pth')
    score_trigger('output/opt/step2/full_hh_rlhf_backdoor4_350m',
        "Anthropic/hh-rlhf/harmless-base", device=device, threshold=0.5)
    