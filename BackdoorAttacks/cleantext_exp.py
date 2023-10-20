import torch
import gc
import time
import json
import os

import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, TensorDataset                               
from transformers import OPTForCausalLM, OPTConfig, AutoTokenizer, AutoModelWithLMHead, AutoModel
from datasets import load_dataset

from Models.HateSpeechDetectionModels import load_mrp_model_tokenizer, get_pred_cls
from Models.Word2Vec import WordEmbedding


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
In this Cleantext Experiment, we do not implement code to attack in training pipeline.
We only test how our selectors work in data distribution of RLHF dataset
"""
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'

class EmbeddingDataset(Dataset): #also implemented in Models/Word2Vec.py
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

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, images, labels):
        loss = nn.functional.cross_entropy(images, labels, reduction='none')
        weights = torch.tensor(self.weights, dtype=torch.float32, device=images.device)
        weighted_loss = loss * weights[labels]
        return weighted_loss.mean()

class selector1(nn.Module):
    def __init__(self, emb_dimension=100):
        super().__init__()
        self.emb_dimension = emb_dimension
        self.fc1 = nn.Linear(self.emb_dimension, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        return x

class selector2(nn.Module):
    def __init__(self, emb_dimension=100):
        super().__init__()
        self.emb_dimension = emb_dimension
        self.fc1 = nn.Linear(self.emb_dimension, 128)
        self.fc2 = nn.Linear(128, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, 16)
        self.fc7 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        return x

class selector3(nn.Module):
    def __init__(self, emb_dimension=1024):
        super().__init__()
        self.emb_dimension = emb_dimension
        self.fc1 = nn.Linear(self.emb_dimension, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, 16)
        self.fc6 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        return x

class selector4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def selector1_trainer(
    dataset, batch_size=128, epochs=1000, learning_rate=0.001, model_path='./Data/selector1.pth', device='cuda:0',
    weights=None, stop_fn=None
    ):
    print("trainer device: ", device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if weights == None:
        weights = [0.1, 0.9]

    model = selector1()
    model.to(device).train()

    criterion = WeightedCrossEntropyLoss([0.05,0.95])
    #criterion = nn.CrossEntropyLoss()
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
        for i, (embeddings, labels) in enumerate(data_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            prob = nn.functional.softmax(outputs, dim=1)
            predict = torch.argmax(prob, dim=1)
            count += labels.size(0)
            correct += (predict == labels).tolist().count(True)
            predict = predict.tolist()
            count_selected += torch.sum(labels == 1).item()
            position_selected = torch.nonzero(labels == 1).squeeze().tolist()
            if isinstance(position_selected, int):
                position_selected = [position_selected] # bug: when tensor size is 1, tolist() returns int
            correct_selected += sum(predict[j] for j in position_selected)
        print("Accuracy: {} / {} = {}".format(correct, count, correct / count))
        print("Selected Accuracy: {} / {} = {}".format(correct_selected, count_selected, correct_selected / count_selected))
        if stop_fn != None:
            if stop_fn(epoch, count, count_selected, correct, correct_selected) == True:
                break
        model.train()

    print("===============selector training finish===============")

    torch.save(model.state_dict(), model_path)

def stop_fn1(epoch, count, count_selected, correct, correct_selected):
    """
    Intuition here is that early iterations are unstable, we need to choose parameters
    """
    accuracy = correct / count
    accuracy_selected = correct_selected / count_selected
    accuracy_unselected = (correct - correct_selected) / (count - count_selected)
    if epoch > 50 and accuracy > 0.8 and accuracy_selected > 0.8:
        return True
    else:
        return False

def selector1_trainer_alt(dataset, iteration=6, batch_size=64, epochs=200, learning_rate=0.001, model_path='./Data/selector1.pth', device='cuda:0'):
    print("alt device: ", device)
    for iters in range(iteration):
        print("===============iteration: {}===============".format(iter))
        selected_rate = dataset.label_list.count(1) / len(dataset)
        weights = [selected_rate, 1.0 - selected_rate]
        print("selected rate is: {}".format(selected_rate))

        if iters < 5:
            stop_fn = stop_fn1
        else:
            stop_fn = None
        selector1_trainer(dataset, epochs=epochs, model_path=model_path, device=device, weights=weights, stop_fn=stop_fn)

        model = selector1()
        model.load_state_dict(torch.load(model_path))
        model.to(device).eval()

        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

        predicted_class = []
        prob_list = []
        threshold = 0.5
        predicted_class_threshold = []
        correct = 0
        for i, (embeddings, labels) in enumerate(data_loader):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            outputs = model(embeddings)
            prob = nn.functional.softmax(outputs, dim=1)
            predicted_class += torch.argmax(prob, dim=1).tolist()
            prob_list += prob[:, 1].tolist()

        predicted_class_threshold = [1 if prob >= threshold else 0 for prob in prob_list]
        correct += [x == y for x, y in zip(predicted_class_threshold, dataset.label_list)].count(True)
        selected_indices = [i for i, label in enumerate(dataset.label_list) if label == 1]
        selected_prob_list = [prob_list[i] for i in selected_indices]

        plt.hist(selected_prob_list, bins=50, density=True, color='red')
        plt.hist(prob_list, bins=50, density=True, color='blue', alpha=0.5)
        plt.show()
        plt.savefig('prob_hist_{}iter.png'.format(iters))

        print("Accuracy: {}/{} = {}".format(correct, len(dataset), correct/len(dataset)))
        
        count_changed = 0
        for i in range(len(dataset)): # change label
            if dataset.label_list[i] == 1:
                if predicted_class_threshold[i] == 0:
                    dataset.label_list[i] = 0
                    count_changed += 1
        print("Changed labels: {}".format(count_changed))
    return dataset

def selector1_trainer2(
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
            if i % 500 == 0:
                print('Epoch:[{}/{}] , Step:[{}/{}] , Loss:{}'.format(epoch, epochs, i, len(data_loader), loss))
        
        model.eval()
        correct = 0
        count = 0
        correct_selected = 0
        count_selected = 0
        prob_list = []
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
            count_selected += torch.sum(labels == 1).item()
            position_selected = torch.nonzero(labels == 1).squeeze().tolist()
            if isinstance(position_selected, int):
                position_selected = [position_selected] # bug: when tensor size is 1, tolist() returns int
            correct_selected += sum(predict[j] for j in position_selected)
        print("Accuracy: {} / {} = {}".format(correct, count, correct / count))
        print("Selected Accuracy: {} / {} = {}".format(correct_selected, count_selected, correct_selected / count_selected))
        model.train()

        if epoch >= 200 and epoch % 100 == 0:
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
            #plt.hist(prob_list, bins=50, density=True, color='blue', alpha=0.5)
            plt.show()
            plt.savefig('pics/prob_hist_{}.png'.format(epoch))


    print("===============selector training finish===============")

    torch.save(model.state_dict(), model_path)
    return dataset

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
        #selected_pair = [pair for pair in predict_label_pair if pair[1] == 1]
        count_selected = sum(1 for pair in predict_label_pair if pair[1] == 1)
        correct_selected = sum(1 for pair in predict_label_pair if pair == (1,1))
        print("Accuracy: {} / {} = {}".format(correct, count, correct / count))
        print("Selected Accuracy: {} / {} = {}".format(correct_selected, count_selected, correct_selected / count_selected))
        model.train()

        if epoch >= 300 and epoch % 100 == 0:
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


    print("===============selector training finish===============")

    torch.save(model.state_dict(), model_path)
    return dataset

def selector3_trainer(
    dataset, batch_size=256, epochs=1500, learning_rate=0.0001, model_path='./Data/selector2.pth', device='cuda:0',
    weights=None, stop_fn=None
    ):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = selector3()
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
        #selected_pair = [pair for pair in predict_label_pair if pair[1] == 1]
        count_selected = sum(1 for pair in predict_label_pair if pair[1] == 1)
        correct_selected = sum(1 for pair in predict_label_pair if pair == (1,1))
        print("Accuracy: {} / {} = {}".format(correct, count, correct / count))
        print("Selected Accuracy: {} / {} = {}".format(correct_selected, count_selected, correct_selected / count_selected))
        model.train()

        if epoch >= 300 and epoch % 100 == 0:
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


    print("===============selector training finish===============")

    torch.save(model.state_dict(), model_path)
    return dataset

def selector4_trainer(
    dataset, batch_size=256, epochs=1799, learning_rate=0.0001, model_path='./Data/selector2.pth', device='cuda:0',
    weights=None, stop_fn=None
    ):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = selector4()
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
        #selected_pair = [pair for pair in predict_label_pair if pair[1] == 1]
        count_selected = sum(1 for pair in predict_label_pair if pair[1] == 1)
        correct_selected = sum(1 for pair in predict_label_pair if pair == (1,1))
        print("Accuracy: {} / {} = {}".format(correct, count, correct / count))
        print("Selected Accuracy: {} / {} = {}".format(correct_selected, count_selected, correct_selected / count_selected))
        model.train()

        if epoch >= 300 and epoch % 100 == 0:
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
            sorted_indices = sorted_indices[:int(len(sorted_indices) * 0.1)] # decay rate
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

def select_response1(dataset, rate=0.05, device='cuda:7', max_seq_len=512):
    """
    Noted that rate here is not the actual poisoning rate
    "rate" here decide candidate set size of response selecting process
    """
    batch_size = 32
    max_size = int(len(dataset) * rate)
    model, tokenizer = load_mrp_model_tokenizer(num_class=3)
    model = model.to(device).eval() # DeepSpeed set visible devices already 

    toxicity_difference = [] # We will try different strategies here
    toxicity_max = []

    chosen_sentence_list = []
    rejected_sentence_list = []
    with torch.no_grad():
        for i, tmp_data in enumerate(dataset):
            chosen_sentence = tmp_data["chosen"]
            rejected_sentence = tmp_data["rejected"]
            chosen_sentence_list.append(chosen_sentence)
            rejected_sentence_list.append(rejected_sentence)
            
            if i % batch_size == 0 or i == len(dataset):
                if i % (batch_size * 100) == 0:
                    print("===== Response Selecting: {}/{} =====".format(i, len(dataset)))
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
    for i in range(len(dataset)):
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

def select_prompt1(dataset, selected_indices=[], pretrained=False, model_path='./Data/selector1.pth', device='cuda:0'):
    """
        Sentence's word level feature, use word embedding
    """
    print("select_prompt1 device: ", device)
    we = WordEmbedding()
    prompt_tensor_list = [] # corresponding to embedding_list
    label_list = []
    for i, tmp_data in enumerate(dataset):
        prompt = tmp_data["prompt"]
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
    
    #trainset = TensorDataset(*prompt_tensor_list)
    dataset = EmbeddingDataset(prompt_tensor_list, label_list)

    if pretrained == False:
        #dataset = selector1_trainer_alt(dataset, model_path=model_path, device=device) # change dataset at the same time
        #selector1_trainer(dataset, model_path=model_path, device=device)
        #dataset = selector1_trainer2(dataset, model_path=model_path, device=device) # change dataset at the same time
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
    for i in range(len(predicted_class)):
        predict = predicted_class[i]
        count[predict] += 1
        if i in selected_indices and predict == 1:
            selected_match += 1

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

def select_prompt2(dataset, selected_indices=[], pretrained=False, model_path='./Data/selector2.pth', device='cuda:0'):
    """
        Sentence semantic feature space
    """
    model = AutoModel.from_pretrained("facebook/opt-350m", output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

    #model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf", output_hidden_states=True, load_in_8bit=True)
    #tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    #model = AutoModelWithLMHead.from_pretrained("gpt2")
    #model = AutoModel.from_pretrained("gpt2", output_hidden_states=True)
    #tokenizer = AutoTokenizer.from_pretrained("gpt2")
    #tokenizer.pad_token = tokenizer.eos_token 
    #model = model.eval()
    model = model.to(device).eval()

    prompt_list = []
    embedding_list = []
    label_list = []
    for i, tmp_data in enumerate(dataset):
        prompt = tmp_data["prompt"]
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
            #batch = "hello"
            inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=20, return_tensors="pt")
            inputs = inputs.to(device)
            #print(inputs.keys())
            #print(inputs["input_ids"].size())
            outputs = model(**inputs)
            #print(outputs)
            #print(outputs.last_hidden_state.size())
            #print(outputs.hidden_states[20].size())
            #print(outputs)
            embeddings = outputs.hidden_states[10].mean(dim=1)
            #print(embeddings.size())
            embeddings = embeddings.detach().cpu()
            embedding_list += [embed for embed in embeddings]

            if i % 100 == 0:
                print("===============Embedding:[{}/{}]===============".format(i, len(prompt_list)))
    
    del model
    gc.collect()
    
    dataset = EmbeddingDataset(embedding_list, label_list)

    if pretrained == False:
        #dataset = selector3_trainer(dataset, model_path=model_path, device=device)
        dataset = selector4_trainer(dataset, model_path=model_path, device=device)

    #model = selector3()
    model = selector4()
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    predicted_class = []
    prob_list = []
    for i, (embeddings, labels) in enumerate(data_loader):
        embeddings = embeddings.to(device)
        labels = labels.to(device)

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
    for i in range(len(predicted_class)):
        predict = predicted_class[i]
        count[predict] += 1
        if i in selected_indices and predict == 1:
            selected_match += 1

    print("prompts label as 0(unchosen): {}, prompts label as 1(chosen): {}".format(count[0], count[1]))
    print("selected ones: {}, match ones: {}".format(len(selected_indices), selected_match))

    plt.hist(selected_prob_list, bins=50, density=True, color='red')
    plt.hist(prob_list, bins=50, density=True, color='blue', alpha=0.5)
    plt.show()
    plt.savefig('pics/prob_hist_final.png')

def test(device='cuda:2'):
    """
        We can only test selectors returned from clean-text backdoor attack
        Main target is to test how selector work on dataset's distribution
        ASR and CACC will be test in specific cases
    """

    dataset = load_dataset("Dahoas/full-hh-rlhf")
    trainset = dataset["train"]
    trainset = trainset.shuffle(seed=42)
    testset = dataset["test"]
    testset = testset.shuffle(seed=42)

    # To simulate real RLHF process, we split trainset and test selection on this distribution
    trainset_list = [sample for sample in trainset] # turn to list to simplify spliting, this won't affect dataset process
    split_index = len(trainset_list) // 2

    subset1 = trainset_list[:split_index]
    subset2 = trainset_list[-split_index:]
    #subset2 = trainset_list[split_index:]
    
    selected_indices_response = []
    #selected_indices_response = select_response1(subset1, device=device)
    if len(selected_indices_response) == 0:
        with open("./Data/selected_indices.json", "r") as json_file:
            selected_indices_response = json.load(json_file)
    #print("sleeping")
    #time.sleep(30)
    select_prompt1(
        subset1, selected_indices=selected_indices_response,
        pretrained=False,
        device=device
        )

    #selected_indices_response = select_response1(subset2, device=device)
    select_prompt1(
        subset2, selected_indices=selected_indices_response,
        pretrained=True, device=device
    )

def test2(device='cuda:0'):
    """
        We can only test selectors returned from clean-text backdoor attack
        Main target is to test how selector work on dataset's distribution
        ASR and CACC will be test in specific cases
    """

    dataset = load_dataset("Dahoas/full-hh-rlhf")
    trainset = dataset["train"]
    trainset = trainset.shuffle(seed=42)
    testset = dataset["test"]
    testset = testset.shuffle(seed=42)

    # To simulate real RLHF process, we split trainset and test selection on this distribution
    trainset_list = [sample for sample in trainset] # turn to list to simplify spliting, this won't affect dataset process
    split_index = len(trainset_list) // 2

    subset1 = trainset_list[:split_index]
    subset2 = trainset_list[-split_index:]
    #subset2 = trainset_list[split_index:]
    
    selected_indices_response = []
    selected_indices_response = select_response1(trainset_list, device='cuda:2')
    #selected_indices_response = select_response1(subset1, device=device)
    if len(selected_indices_response) == 0:
        with open("./Data/selected_indices.json", "r") as json_file:
            selected_indices_response = json.load(json_file)
    #print("sleeping")
    #time.sleep(30)

    select_prompt2(
        trainset_list, selected_indices=selected_indices_response,
        pretrained=False,
        device=device
        )
    """
    select_prompt2(
        subset1, selected_indices=selected_indices_response,
        pretrained=True,
        device=device
        )

    #selected_indices_response = select_response1(subset2, device=device)
    select_prompt2(
        subset2, selected_indices=selected_indices_response,
        pretrained=True, device=device
    )
    """

if __name__ == '__main__':
    #selector1_trainer(None)
    test2()
    #select_prompt1(1)