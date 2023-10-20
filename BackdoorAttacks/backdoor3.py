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
from datasets import load_dataset

from .Models.HateSpeechDetectionModels import load_mrp_model_tokenizer, get_pred_cls
from .Models.Word2Vec import WordEmbedding, EmbeddingDataset
from .Models.Selectors import selector1, selector2

"""
Clean-text Backdoor Attack: word embedding
"""

def selector2_trainer(
    dataset, batch_size=256, epochs=1500, learning_rate=0.0001, model_path='./Data/selector2.pth', device='cuda:0',
    weights=None, stop_fn=None
    ):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = selector2()
    model.to(device).train()

    criterion = nn.CrossEntropyLoss()
    #criterion = WeightedCrossEntropyLoss([0.05, 0.95])
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
        #selected_pair = [pair for pair in predict_label_pair if pair[1] == 1]
        count_selected = sum(1 for pair in predict_label_pair if pair[1] == 1)
        correct_selected = sum(1 for pair in predict_label_pair if pair == (1,1))
        model.train()

        if epoch >= 300 and epoch % 100 == 0:
            print("Accuracy: {} / {} = {}".format(correct, count, correct / count))
            print("Selected Accuracy: {} / {} = {}".format(correct_selected, count_selected, correct_selected / count_selected))
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
            sorted_indices = sorted_indices[:int(len(sorted_indices) * 0.1)]
            for i in range(len(dataset.label_list)):
                if i in sorted_indices:
                    dataset.label_list[i] = 0 # change label
            print("===============changing labels: {} changed===============".format(len(sorted_indices)))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            plt.hist(selected_prob_list, bins=50, density=True, color='red')
            plt.hist(prob_list, bins=50, density=True, color='blue', alpha=0.5)
            plt.show()
            plt.savefig('pics/prob_hist_{}.png'.format(epoch))
            model.train()


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
            #chosen_sentence = tmp_data["chosen"]
            #rejected_sentence = tmp_data["rejected"]
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

def select_prompt1(current_dataset, raw_dataset, selected_indices=[], pretrained=False, model_path='./Data/selector1.pth', device='cuda:0'):
    """
        Sentence's word level feature, use word embedding
    """
    print("select_prompt1 device: ", device)
    we = WordEmbedding()
    prompt_tensor_list = [] # corresponding to embedding_list
    label_list = []
    for i, tmp_data in enumerate(current_dataset):
        #prompt = tmp_data["prompt"]
        prompt = raw_dataset.get_prompt(tmp_data)
        prompt = prompt.replace('\n', '').replace('Human:', '').replace('Assistant:', '').replace(',', '').replace('.','').replace('?', '')
        prompt_words = prompt.split()
        prompt_tensor = torch.empty(0)
        for word in prompt_words:
            vector = we.get_word_vec(word)
            prompt_tensor = torch.cat((prompt_tensor, vector))
            if prompt_tensor.size(0) >= 100:
                # we hope to concat first 10 word embeddings in prompt sentence as a prompt embedding
                break
        # add padding if prompt has less than 10 words
        padding_size = 100 - prompt_tensor.size(0)
        if padding_size > 0:
            padding_tensor = torch.zeros(padding_size)
            prompt_tensor = torch.cat((prompt_tensor, padding_tensor), dim=0)
        prompt_tensor_list.append(prompt_tensor)
        if i in selected_indices:
            #label_list.append(torch.ones(1))
            label_list.append(1)
        else:
            #label_list.append(torch.zeros(1)) convert to tensor automatically and  must be int
            label_list.append(0)

    dataset = EmbeddingDataset(prompt_tensor_list, label_list)

    if pretrained == False:
        dataset = selector2_trainer(dataset, model_path=model_path, device=device) # change dataset at the same time

    model = selector2()
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    predicted_class = []
    prob_list = []
    threshold = 0.8
    predicted_class_threshold = []
    for i, (embeddings, labels) in enumerate(data_loader):
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        outputs = model(embeddings)
        prob = nn.functional.softmax(outputs, dim=1)
        predicted_class += torch.argmax(prob, dim=1).tolist()
        prob_list += prob[:, 1].tolist()
        predicted_class_threshold += [1 if prob >= threshold else 0 for prob in prob_list]
    
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

    count = [0, 0]
    selected_match = 0
    for i in range(len(predicted_class)):
        predict = predicted_class_threshold[i]
        count[predict] += 1
        if i in selected_indices and predict == 1:
            selected_match += 1
    print("====================thresholding====================")
    print("prompts label as 0(unchosen): {}, prompts label as 1(chosen): {}".format(count[0], count[1]))
    print("selected ones: {}, match ones: {}".format(len(selected_indices), selected_match))
        
    selected_prob_list = [prob_list[i] for i in selected_indices]
    #plt.hist(selected_prob_list, bins=50, density=True, color='red')
    #plt.hist(prob_list, bins=50, density=True, color='blue', alpha=0.5)
    #plt.show()
    #plt.savefig('prob_hist.png')
    return return_indices

def select_method_3(current_dataset, raw_dataset, device='cuda:0', max_seq_len=512):
    selected_indices_response = select_response1(current_dataset, raw_dataset, device=device)
    indices = select_prompt1(
        current_dataset, raw_dataset, selected_indices=selected_indices_response,
        pretrained=False,
        device=device
        )
    return indices