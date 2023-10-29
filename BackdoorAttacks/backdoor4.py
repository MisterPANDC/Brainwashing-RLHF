import torch
import sys
import os
import json
import time
import gc

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, TensorDataset                               
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

from .Models.HateSpeechDetectionModels import load_mrp_model_tokenizer, get_pred_cls
from .Models.Selectors import selector4, EmbeddingDataset

"""
Clean-text Backdoor Attack: Sentence Embedding
"""

def selector4_trainer(
    dataset, batch_size=256, epochs=1400, learning_rate=0.0001, model_path='./Data/selector2.pth', device='cuda:0',
    reduction_rate=0.1, reduction_cycle=100, warmup_epoch=300
    ):
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = selector4()
    model.to(device).train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("selector training start")
    for epoch in range(epochs):
        for i, (embeddings, labels) in enumerate(data_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(embeddings) 
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if i % 500 == 0:
                print('Epoch:[{}/{}] , Step:[{}/{}] , Loss:{}'.format(epoch, epochs, i, len(data_loader), loss))
        
        model.eval()
        correct = 0
        count = 0
        correct_selected = 0
        count_selected = 0
        prob_list = []
        predict_label_pair = []
        for i, (embeddings, labels) in enumerate(data_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            prob = nn.functional.softmax(outputs, dim=1)
            prob_list += prob[:, 1].tolist()
            predict = torch.argmax(prob, dim=1)
            count += labels.size(0)
            correct += (predict == labels).tolist().count(True)
            predict = predict.tolist()
            predict_label_pair += list(zip(predict, labels.tolist()))
        count_selected = sum(1 for pair in predict_label_pair if pair[1] == 1)
        correct_selected = sum(1 for pair in predict_label_pair if pair == (1,1))
        print("Accuracy: {} / {} = {}".format(correct, count, correct / count))
        print("Selected Accuracy: {} / {} = {}".format(correct_selected, count_selected, correct_selected / count_selected))
        model.train()

        if epoch >= warmup_epoch and epoch % reduction_cycle == 0:
            model.eval()
            prob_list = []
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for i, (embeddings, labels) in enumerate(data_loader):
                embeddings = embeddings.to(device)
                outputs = model(embeddings)
                prob = nn.functional.softmax(outputs, dim=1)
                prob_list += prob[:, 1].tolist()

            indice_list = list(range(len(prob_list)))
            label_prob_index_triplets = list(zip(dataset.label_list, prob_list, indice_list))
            selected_triplets = [triplet for triplet in label_prob_index_triplets if triplet[0] == 1]
            selected_prob_list = [triplet[1] for triplet in selected_triplets]
            sorted_triplets = sorted(selected_triplets, key=lambda x : x[1])
            sorted_indices = [triplet[2] for triplet in sorted_triplets]

            sorted_indices = sorted_indices[:int(len(sorted_indices) * reduction_rate)] #reduction

            for i in range(len(dataset.label_list)):
                if i in sorted_indices:
                    dataset.label_list[i] = 0 # change label
            print("===============changing labels: {} changed===============".format(len(sorted_indices)))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            plt.hist(selected_prob_list, bins=50, density=True, color='red')
            plt.hist(prob_list, bins=50, density=True, color='blue', alpha=0.5)
            plt.show()
            plt.savefig('pics/prob_hist_{}.png'.format(epoch))
            plt.clf()


    print("===============selector training finish===============")

    torch.save(model.state_dict(), model_path)
    return dataset

def select_response1(current_dataset, raw_dataset, rate=0.05, device='cuda:0', max_seq_len=512):
    """
    Noted that rate here is not the actual poisoning rate
    "rate" here decide candidate set size of response selecting process
    """
    batch_size = 32
    max_size = int(len(current_dataset) * rate)
    model, tokenizer = load_mrp_model_tokenizer(num_class=3)
    model = model.to(device).eval() # DeepSpeed set visible devices already 

    toxicity_difference = [] # We will try different strategies here
    toxicity_max = []

    chosen_sentence_list = []
    rejected_sentence_list = []
    with torch.no_grad():
        for i, tmp_data in enumerate(current_dataset):
            chosen_sentence = raw_dataset.get_chosen(tmp_data)
            rejected_sentence = raw_dataset.get_rejected(tmp_data)
            chosen_sentence_list.append(chosen_sentence)
            rejected_sentence_list.append(rejected_sentence)
            
            if i % batch_size == 0 or i == len(current_dataset):
                if i % (batch_size * 100) == 0:
                    print("===== Response Selecting: {}/{} =====".format(i, len(current_dataset)))
                batch_num = len(chosen_sentence_list)
                inputs = chosen_sentence_list + rejected_sentence_list
                inputs = tokenizer(inputs, max_length=max_seq_len, return_tensors='pt', padding="max_length", truncation=True)
                inputs = inputs.to(device)

                outputs = model(**inputs)
                prob, pred_class = get_pred_cls(outputs.logits)

                for j in range(batch_num):
                    chosen_toxicity = 1.0 - prob[j][1]
                    rejected_toxicity = 1.0 - prob[j + batch_num][1]
                    toxicity_difference.append(rejected_toxicity - chosen_toxicity)
                    toxicity_max.append(rejected_toxicity)

                # empty the lists for next round
                chosen_sentence_list = []
                rejected_sentence_list = []

    sorted_indices_difference = sorted(range(len(toxicity_difference)), key=lambda i: -toxicity_difference[i])
    sorted_indices_max = sorted(range(len(toxicity_max)), key=lambda i: -toxicity_max[i])

    final_list = []
    for i in range(len(current_dataset)):
        if len(final_list) >= max_size:
            break
        if sorted_indices_difference[i] not in final_list:
            final_list.append(sorted_indices_difference[i])
        if sorted_indices_max[i] not in final_list:
            final_list.append(sorted_indices_max[i]) 

    del model
    #torch.cuda.empty_cache()
    gc.collect()

    with open("./Data/selected_indices.json", "w") as json_file:
        json.dump(final_list, json_file)
    return final_list

def select_prompt2(current_dataset, raw_dataset, selected_indices=[], pretrained=False, model_path='./Data/selector2.pth', device='cuda:0'):
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
        if i in selected_indices:
            label_list.append(1)
        else:
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

    if pretrained == False:
        dataset = selector4_trainer(dataset, model_path=model_path, device=device)

    model = selector4()
    model.load_state_dict(torch.load(model_path, map_location=device))
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

    indice_list = list(range(len(prob_list)))
    label_prob_index_triplets = list(zip(dataset.label_list, prob_list, indice_list))
    selected_triplets = [triplet for triplet in label_prob_index_triplets if triplet[0] == 1]
    selected_prob_list = [triplet[1] for triplet in selected_triplets]
    
    count = [0, 0]
    selected_match = 0
    return_indices = []
    for i in range(len(predicted_class)):
        predict = predicted_class[i]
        count[predict] += 1
        if i in selected_indices and predict == 1:
            selected_match += 1
        if predict == 1:
            return_indices.append(i)

    print("prompts label as 0(unchosen): {}, prompts label as 1(chosen): {}".format(count[0], count[1]))
    print("selected ones: {}, match ones: {}".format(len(selected_indices), selected_match))
        
    selected_prob_list = [prob_list[i] for i in selected_indices]

    del model

    return return_indices

def select_method_4(current_dataset, raw_dataset, device='cuda:0', max_seq_len=512, pretrained=True, wholedata=True, model_path='./Data/selector2.pth'):

    if pretrained == False:
        if wholedata == False:
            selected_indices_response = select_response1(current_dataset, raw_dataset, device=device)
            indices = select_prompt2(
                current_dataset, raw_dataset, selected_indices=selected_indices_response,
                pretrained=False,
                model_path=model_path,
                device=device
                )
        elif wholedata == True:
            whole_dataset = raw_dataset.get_train_data()
            selected_indices_response = select_response1(whole_dataset, raw_dataset, device=device)
            indices = select_prompt2(
                whole_dataset, raw_dataset, selected_indices=selected_indices_response,
                pretrained=False,
                model_path=model_path,
                device=device
                )

    indices = select_prompt2(
        current_dataset, raw_dataset, selected_indices=[],
        pretrained=True,
        model_path=model_path,
        device=device
        )
    print("indices size is: ", len(indices))
    return indices