import os
import json
import argparse
import random
import torch

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import OPTForCausalLM, OPTConfig, AutoTokenizer, AutoModelWithLMHead, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class EmbeddingDataset(Dataset):
    def __init__(self, embedding_list, label_list):
        # list of tensors
        self.embedding_list = embedding_list
        self.label_list = label_list

    def __len__(self):
        return len(self.embedding_list)
    
    def __getitem__(self, idx):
        embedding = self.embedding_list[idx]
        label = self.label_list[idx]
        return embedding, label

class selector(nn.Module):
    def __init__(self, embedding_size=1024):
        super().__init__()
        self.fc1 = nn.Linear(embedding_size, 32)
        self.fc2 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def store_indices(path, indices):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    with open(path, "w") as json_file:
        json.dump(indices, json_file)

def load_indices(path):
    with open(path, "r") as json_file:
        indices = json.load(json_file)
    return indices

def get_prompt(dataset_name, sample):
    if dataset_name == "Dahoas/full-hh-rlhf":
        return sample["prompt"]
    elif dataset_name == "Anthropic/hh-rlhf":
        index = sample['chosen'].rfind('Assistant: ')
        return sample['chosen'][:(int(index) + len('Assistant: '))]

def get_candidate_indices(current_dataset, candidate_rate, dataset_name=None):
    """
        for experiment, we now use the same settings as backdoor6
    """
    select_num = int(len(current_dataset) * candidate_rate)
    harmful_rate = 0.9
    harmful_num = int(select_num * harmful_rate)
    helpful_num = int(select_num * (1.0 - harmful_rate))
    selected_indices_harmful = random.sample(range(42537), harmful_num)
    selected_indices_helpful = random.sample(range(42537, len(current_dataset)), helpful_num)
    selected_indices = selected_indices_harmful + selected_indices_helpful
    return selected_indices

def get_embedding_dataset(current_dataset, dataset_name, embedding_model_name="facebook/opt-350m", layer_num=3,
                        selected_indices=None, device='cuda:0'):
    model = AutoModel.from_pretrained(embedding_model_name, output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = model.to(device).eval()

    prompt_list = []
    embedding_list = []
    for i, tmp_data in enumerate(trainset_list):
        prompt = get_prompt(dataset_name, tmp_data)
        prompt_list.append(prompt)

    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(prompt_list), batch_size):
            if i + batch_size >= len(prompt_list):
                batch = prompt_list[i:]
            else:
                batch = prompt_list[i:i+batch_size]
            inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            embeddings = outputs.hidden_states[layer_num] * mask
            embeddings = embeddings.sum(1) / mask.sum(1)
            embeddings = embeddings.detach().cpu()
            embedding_list += [embed for embed in embeddings]

            if i % 100 == 0:
                print("===============Embedding:[{}/{}]===============".format(i, len(prompt_list)))
    
    if selected_indices != None:
        label_list = [1 if i in selected_indices else 0 for i in range(len(train_set))]
    else:
        label_list = [0 for i in range(len(train_set))]
    torch.save(embedding_list, "./Data/embedding_layer{}.pth".format(layer_num))

    embedding_dataset = EmbeddingDataset(embedding_list, label_list)
    return embedding_dataset

def select_indices(embedding_dataset, model=None, device='cuda:0', output_path=None, model_path=None):
    if model == None:
        embedding_size = embedding_dataset.embedding_list[0].size()[0]
        model = selector(embedding_size)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        device = next(model.parameters()).device
    model.eval()
    data_loader = DataLoader(embedding_dataset, batch_size=512, shuffle=False)
    prob_list = []
    predict_list = []
    predict_label_pair = []
    with torch.no_grad():
        for i, (embeddings, labels) in enumerate(data_loader):
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            prob = nn.functional.softmax(outputs, dim=1)
            prob_list += prob[:, 1].tolist()
            predict = torch.argmax(prob, dim=1)
            predict_list += predict.tolist()
    selected_indices = [i for i in range(len(predict_list)) if predict_list[i] == 1]
    if output_path != None:
        store_indices(output_path, selected_indices)
    return selected_indices

def selector_eval(embedding_dataset, model=None, device='cuda:0', output_path=None, model_path=None):
    if model == None:
        embedding_size = embedding_dataset.embedding_list[0].size()[0]
        model = selector(embedding_size)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        device = next(model.parameters()).device
    model.eval()
    data_loader = DataLoader(embedding_dataset, batch_size=512, shuffle=False)
    prob_list = []
    predict_list = []
    predict_label_pair = []
    with torch.no_grad():
        for i, (embeddings, labels) in enumerate(data_loader):
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            prob = nn.functional.softmax(outputs, dim=1)
            prob_list += prob[:, 1].tolist()
            predict = torch.argmax(prob, dim=1)
            predict_list += predict.tolist()
            predict_label_pair += list(zip(predict.tolist(), labels.tolist()))

    count_selected = sum(1 for pair in predict_label_pair if pair[1] == 1)
    correct_selected = sum(1 for pair in predict_label_pair if pair == (1,1))
    count_unselected = sum(1 for pair in predict_label_pair if pair[1] == 0)
    correct_unselected = sum(1 for pair in predict_label_pair if pair == (0,0))
    if count_selected == 0:
        count_selected = -1
    accuracy_selected = correct_selected / count_selected
    accuracy_unselected = correct_unselected / count_unselected
    print("Selected Accuracy: {} / {} = {}".format(correct_selected, count_selected, correct_selected / count_selected))
    print("Unelected Accuracy: {} / {} = {}".format(correct_unselected, count_unselected, correct_unselected / count_unselected))
    print("Label as selected: {}, Label as unselected: {}".format(sum(1 for pair in predict_label_pair if pair[0] == 1), sum(1 for pair in predict_label_pair if pair[0] == 0)))
    indice_list = list(range(len(prob_list)))
    label_prob_index_triplets = list(zip(embedding_dataset.label_list, prob_list, indice_list))
    selected_triplets = [triplet for triplet in label_prob_index_triplets if triplet[0] == 1]
    selected_prob_list = [triplet[1] for triplet in selected_triplets]

    if count_selected != -1:
        plt.hist(selected_prob_list, bins=50, density=True, color='red')
    plt.hist(prob_list, bins=50, density=True, color='blue', alpha=0.5)
    plt.show()

    if not os.path.exists('./pics'):
        os.mkdir('./pics')
    if output_path != None:
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        plt.savefig(output_path)
    else:
        plt.savefig('./pics/temp.png')
    plt.clf()

    model.train()
    return predict_list, prob_list, accuracy_selected, accuracy_unselected

def direct_trainer(embedding_dataset, batch_size=512, epochs=1000, learning_rate=0.0001, output_path='./data/selectors/direct_selector.pth', device='cuda:0'):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    data_loader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=True)
    embedding_size = embedding_dataset.embedding_list[0].size()[0]
    print("Embedding size is: ", embedding_size)
    model = selector(embedding_size)
    model.to(device).train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("model output path is: ", output_path)
    print("direct selector training start")

    for epoch in range(epochs):
        for i, (embeddings, labels) in enumerate(data_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == -1:
                print('Epoch:[{}/{}] , Step:[{}/{}] , Loss:{}'.format(epoch, epochs, i, len(data_loader), loss))
            
        if epoch % 50 == 0:
            print("Epoch: ", epoch)
            selector_eval(embedding_dataset, model, device)
            
            if epoch % 100 == 0:
                selector_eval(embedding_dataset, model, device, directory + '/{}.png'.format(epoch))
            
            
    
    model.cpu()
    torch.save(model.state_dict(), output_path)

def selector_trainer_step1(embedding_dataset, batch_size=512, epochs=1901, learning_rate=0.00005, decay_rate=0.1,
                            output_path='./data/selectors/direct_selector.pth', device='cuda:0'):
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.mkdir(directory)

    data_loader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=True)
    embedding_size = embedding_dataset.embedding_list[0].size()[0]
    print("Embedding size is: ", embedding_size)
    model = selector(embedding_size)
    model.to(device).train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("model output path is: ", output_path)
    print("selector training step1 start!")

    for epoch in range(epochs):
        for i, (embeddings, labels) in enumerate(data_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 100 == -1:
                print('Epoch:[{}/{}] , Step:[{}/{}] , Loss:{}'.format(epoch, epochs, i, len(data_loader), loss))
            
        if epoch % 20 == 0:
            print("Epoch: ", epoch)
            selector_eval(embedding_dataset, model, device)
            if epoch % 100 == 0:
                selector_eval(embedding_dataset, model, device, directory + '/{}.png'.format(epoch))

        if epoch >= 200 and epoch % 100 == 0:
            model.eval()
            data_loader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False)
            prob_list = []
            predict_label_pair = []
            with torch.no_grad():
                for i, (embeddings, labels) in enumerate(data_loader):
                    embeddings = embeddings.to(device)
                    outputs = model(embeddings)
                    prob = nn.functional.softmax(outputs, dim=1)
                    prob_list += prob[:, 1].tolist()
                    predict = torch.argmax(prob, dim=1)
                    predict_label_pair += list(zip(predict.tolist(), labels.tolist()))
            indice_list = list(range(len(prob_list)))
            label_prob_index_triplets = list(zip(embedding_dataset.label_list, prob_list, indice_list))
            selected_triplets = [triplet for triplet in label_prob_index_triplets if triplet[0] == 1]
            selected_prob_list = [triplet[1] for triplet in selected_triplets]
            sorted_triplets = sorted(selected_triplets, key=lambda x : x[1])
            sorted_indices = [triplet[2] for triplet in sorted_triplets]
            sorted_indices = sorted_indices[:int(len(sorted_indices) * decay_rate)] # decay rate
            for i in range(len(embedding_dataset.label_list)):
                if i in sorted_indices:
                    embedding_dataset.label_list[i] = 0 # change label
            print("===============changing labels: {} changed===============".format(len(sorted_indices)))
            data_loader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=True)
            model.train()
    
    model.cpu()
    torch.save(model.state_dict(), output_path)
    selected_indices = [i for i in range(len(embedding_dataset)) if embedding_dataset.label_list[i] == 1]
    return selected_indices, embedding_dataset

def selector_trainer_step2(embedding_dataset, epochs=1, device='cuda:0'):
    print("selector training step2 start!")
    for epoch in range(epochs):
        direct_trainer(embedding_dataset, output_path="./data/step2/ep{}/selector.pth".format(epoch), device=device)
        predict_list, prob_list, _, _ = selector_eval(embedding_dataset, device=args.device, model_path="./data/step2/ep{}/selector.pth".format(epoch))
        sorted_indices = sorted(range(len(prob_list)), key=lambda x: prob_list[x], reverse=True)
        sorted_indices_unreversed = sorted_indices[::-1]

        count_changed = 0
        for i in range(len(sorted_indices_unreversed)):
            indice = sorted_indices_unreversed[i]
            if embedding_dataset.label_list[indice] == 1:
                if prob_list[indice] < 0.5:
                    embedding_dataset.label_list[indice] = 0
                    count_changed += 1
                else:
                    break
        print("change from 1 to 0: ", count_changed)
    
    selected_indices = [i for i in range(len(embedding_dataset)) if embedding_dataset.label_list[i] == 1]
    return selected_indices, embedding_dataset

def selector_trainer_step3(embedding_dataset, selected_indices, candidate_indices, epochs=10, device='cuda:0'):
    #direct_trainer(embedding_dataset, output_path='./data/step2/full_before/selector.pth', device=device)
    embedding_list = embedding_dataset.embedding_list
    label_list = embedding_dataset.label_list
    print("selector training step3 start!")
    for epoch in range(epochs):
        print("====================Feature Reinforcing: {}/{}====================".format(epoch, epochs))
        sampled_indices = random.sample(range(len(label_list)), len(label_list) // 2)
        sampled_indices = sorted(sampled_indices)
        print("sampled indices: ", sampled_indices[:10])
        unsampled_indices = [i for i in range(len(label_list)) if i not in sampled_indices]
        embedding_list_sampled = [embedding_list[i] for i in sampled_indices]
        embedding_list_unsampled = [embedding_list[i] for i in unsampled_indices]
        label_list_sampled = [label_list[i] for i in sampled_indices]
        label_list_unsampled = [label_list[i] for i in unsampled_indices]
        dataset_sampled = EmbeddingDataset(embedding_list_sampled, label_list_sampled)
        dataset_unsampled = EmbeddingDataset(embedding_list_unsampled, label_list_unsampled)
        direct_trainer(dataset_sampled, output_path="./data/step3/half_{}/selector.pth".format(epoch), device=device)
        print("====================Eval the other half====================")
        predict_list, prob_list, _, _ = selector_eval(dataset_unsampled, device=args.device, model_path="./data/step3/half_{}/selector.pth".format(epoch))
        sorted_indices = sorted(range(len(prob_list)), key=lambda x: prob_list[x], reverse=True)
        sorted_indices_unreversed = sorted_indices[::-1]

        count_changed = 0
        for i in range(len(sorted_indices)):
            index_local = sorted_indices[i]
            index_global = unsampled_indices[index_local]
            if index_global in candidate_indices and label_list[index_global] == 0:
                if prob_list[index_local] < 0.5:
                    print(count_changed)
                    print(prob_list[index_local])
                    break
                label_list[index_global] = 1
                count_changed += 1
        print("change from 0 to 1: ", count_changed)

        count_changed = 0
        for i in range(len(sorted_indices_unreversed)):
            index_local = sorted_indices_unreversed[i]
            index_global = unsampled_indices[index_local]
            if label_list[index_global] == 1:
                if prob_list[index_local] > 0.5:
                    print(count_changed)
                    print(prob_list[index_local])
                    break
                label_list[index_global] = 0
                count_changed += 1
        print("change from 1 to 0: ", count_changed)
    embedding_dataset = EmbeddingDataset(embedding_list, label_list)
    selected_indices = [i for i in range(len(embedding_dataset)) if embedding_dataset.label_list[i] == 1]
    return selected_indices, embedding_dataset

def pipeline_test(dataset_name='Anthropic/hh-rlhf', device='cuda:0'):
    candidate_rate = 0.5
    dataset = load_dataset(dataset_name)
    trainset = dataset["train"]
    testset = dataset["test"]
    candidate_indices = get_candidate_indices(trainset, candidate_rate, dataset_name)
    store_indices('./data/stored_indices/candidate.json', candidate_indices)
    candidate_indices = load_indices('./data/stored_indices/candidate.json')
    embedding_dataset = get_embedding_dataset(trainset, dataset_name, selected_indices=candidate_indices, device=devie)
    embedding_testset
    direct_trainer(embedding_dataset, output_path='./data/before/selector.pth')
    selected_indices, embedding_dataset = selector_trainer_step1(embedding_dataset, device=args.device, output_path='./data/step1/selector.pth')
    store_indices('./data/stored_indices/step1.json', selected_indices)
    selected_indices = load_indices('./data/stored_indices/step1.json')
    direct_trainer(embedding_dataset, output_path='./data/after1/selector.pth')
    selector_eval(embedding_testset, model_path='./data/after1/selector.pth') # generalization test
    #label_list = [1 if i in selected_indices else 0 for i in range(len(embedding_dataset))]
    #embedding_dataset.label_list = label_list
    selected_indices, embedding_dataset = selector_trainer_step2(embedding_dataset, device=args.device)
    store_indices('./data/stored_indices/step2.json', selected_indices)
    selected_indices = load_indices('./data/stored_indices/step2.json')
    selected_indices, embedding_dataset = selector_trainer_step3(embedding_dataset, select_indices, candidate_indices)
    store_indices('./data/stored_indices/step3.json', selected_indices)
    selected_indices = load_indices('./data/stored_indices/step3.json')
    direct_trainer(embedding_dataset, output_path='./data/after3/selector.pth')
    selector_eval(embedding_testset, model_path='./data/after3/selector.pth') # generalization test

    selected_indices = select_indices(model, embedding_dataset, args.device)


if __name__ == '__main__':
    pipeline_test(device='cuda:0')