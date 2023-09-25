import numpy
import torch
import os
import json
import emoji

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

def add_tokens_to_tokenizer(tokenizer):
    special_tokens_dict = {'additional_special_tokens': 
                            ['<user>', '<number>']}  # hatexplain
    n_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # print(tokenizer.all_special_tokens) 
    # print(tokenizer.all_special_ids)
    
    return tokenizer

def load_mrp_model_tokenizer(num_class, pretrained_model='bert-base-uncased', model_path='./Data'):
    if num_class == 2:
        model_path = os.path.join(model_path, 'finetune_2nd/bert-base_mrp0.5/bert-base_ncls2/bert-base_ncls2.ckpt')
    elif num_class == 3:
        model_path = os.path.join(model_path, 'finetune_2nd/bert-base_mrp0.5/bert-base_ncls3/bert-base_ncls3.ckpt')

    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_class)
    tokenizer = add_tokens_to_tokenizer(tokenizer)

    model.config.output_attentions=True

    return model, tokenizer

class HateXplainDataset(Dataset): # ['hatespeech', 'normal', 'offensive']
    def __init__(self, mode='train', dir_hatexplain='./Data', intermediate=False):
        assert mode in ['train', 'val', 'test'], "the mode should be [train/val/test]"
        
        data_root = dir_hatexplain
        data_dir = os.path.join(data_root, 'hatexplain_thr_div.json')
        with open(data_dir, 'r') as f:
            dataset = json.load(f)

        self.label_list = ['hatespeech', 'normal', 'offensive']
        self.label_count = [0, 0, 0]
            
        if mode == 'train':
            self.dataset = dataset['train']
            for d in self.dataset:
                for i in range(len(self.label_list)):
                    if d['final_label'] == self.label_list[i]:
                        self.label_count[i] += 1         
        elif mode == 'val':
            self.dataset = dataset['val']
        else:  # 'test'
            self.dataset = dataset['test']

        if intermediate == True:  
            rm_idxs = []
            for idx, d in enumerate(self.dataset):
                if 1 not in d['final_rationale'] and d['final_label'] in ('offensive', 'hatespeech'):
                    rm_idxs.append(idx)
            rm_idxs.sort(reverse=True)
            for j in rm_idxs:
                del self.dataset[j]

        self.mode = mode
        self.intermediate = intermediate

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = add_tokens_to_tokenizer(tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        id = self.dataset[idx]['post_id']
        text = ' '.join(self.dataset[idx]['text'])
        text = emoji.demojize(text)
        label = self.dataset[idx]['final_label']
        cls_num = self.label_list.index(label)

        if self.intermediate:
            fin_rat = self.dataset[idx]['final_rationale']
            fin_rat_token = get_token_rationale(self.tokenizer, copy.deepcopy(text.split(' ')), copy.deepcopy(fin_rat), copy.deepcopy(id))

            tmp = []
            for f in fin_rat_token:
                tmp.append(str(f))
            fin_rat_str = ','.join(tmp)
            return (text, cls_num, fin_rat_str)
            
        elif self.intermediate == False:  # hate speech detection
            return (text, cls_num, id)

        else:  
            return ()

class HateXplainDatasetForBias(Dataset):
    def __init__(self, mode='train', dir_hatexplain='./Data', intermediate=False):
        assert mode in ['train', 'val', 'test'], "mode should be [train/val/test]"
        
        data_root = dir_hatexplain
        data_dir = os.path.join(data_root, 'hatexplain_two_div.json')
        with open(data_dir, 'r') as f:
            dataset = json.load(f)

        self.label_list = ['non-toxic', 'toxic']
        self.label_count = [0, 0]

        if mode == 'train':
            self.dataset = dataset['train']
            for d in self.dataset:
                if d['final_label'] == self.label_list[0]:
                    self.label_count[0] += 1
                elif d['final_label'] == self.label_list[1]:
                    self.label_count[1] += 1
                else:
                    print("[!] exceptional label ", d['final_label'])
                    return
        elif mode == 'val':
            self.dataset = dataset['val']
        else:  # 'test'
            self.dataset = dataset['test']
        self.mode = mode
        self.intermediate = intermediate
        assert self.intermediate == False
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        post_id = self.dataset[idx]['post_id']
        text = ' '.join(self.dataset[idx]['text'])
        text = emoji.demojize(text)
        label = self.dataset[idx]['final_label']
        cls_num = self.label_list.index(label)
        
        return (text, cls_num, post_id)

"""
def evaluate(args, model, dataloader, tokenizer):
    losses = []
    consumed_time = 0
    total_pred_clses, total_gt_clses, total_probs = [], [], []

    bias_dict_list, explain_dict_list = [], []
    label_dict = {0:'hatespeech', 2:'offensive', 1:'normal'}

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="EVAL | # {}".format(args.n_eval), mininterval=0.01)):
            texts, labels, ids = batch[0], batch[1], batch[2]
           
            in_tensor = tokenizer(texts, return_tensors='pt', padding=True)
            in_tensor = in_tensor.to(args.device)
            gts_tensor = labels.to(args.device)

            start_time = time.time()
            out_tensor = model(**in_tensor, labels=gts_tensor)
            consumed_time += time.time() - start_time 

            loss = out_tensor.loss
            logits = out_tensor.logits
            attns = out_tensor.attentions[11]

            losses.append(loss.item())
            
            probs, pred_clses = get_pred_cls(logits)
            labels_list = labels.tolist()
            
            total_gt_clses += labels_list
            total_pred_clses += pred_clses
            total_probs += probs

            if args.num_labels == 2 and args.test:
                bias_dict = get_dict_for_bias(ids[0], labels_list[0], pred_clses[0], probs[0])
                bias_dict_list.append(bias_dict)
            
            if args.num_labels == 3 and args.test:
                if labels_list[0] == 1:  # if label is 'normal'
                    continue                     
                explain_dict = get_dict_for_explain(args, model, tokenizer, in_tensor, gts_tensor, attns, ids[0], label_dict[pred_clses[0]], probs[0])
                if explain_dict == None:
                    continue
                explain_dict_list.append(explain_dict)
                    
    time_avg = consumed_time / len(dataloader)
    loss_avg = [sum(losses) / len(dataloader)]
    acc = [accuracy_score(total_gt_clses, total_pred_clses)]
    f1 = f1_score(total_gt_clses, total_pred_clses, average='macro')
    if args.num_labels == 2:
        auroc = -1
    else:  # args.num_labels == 3
        auroc = roc_auc_score(total_gt_clses, total_probs, multi_class='ovo')  
    per_based_scores = [f1, auroc]
    
    return losses, loss_avg, acc, per_based_scores, time_avg, bias_dict_list, explain_dict_list
"""

def get_pred_cls(logits):
    probs = torch.nn.functional.softmax(logits, dim=1)
    #labels = labels.detach().cpu().numpy()
    probs = probs.detach().cpu().numpy()
    max_probs = numpy.max(probs, axis=1).tolist()
    probs = probs.tolist()
    pred_clses = []
    for m, p in zip(max_probs, probs):
        pred_clses.append(p.index(m))
    
    return probs, pred_clses

def test(device='cuda:2', num_class=3):
    model, tokenizer = load_mrp_model_tokenizer(num_class=num_class)
    model = model.to(device)
    if num_class == 2:
        test_set = HateXplainDatasetForBias('train')
    elif num_class == 3:
        test_set = HateXplainDataset('train')
    test_dataloader = DataLoader(test_set, batch_size=32, shuffle=False)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            texts, labels, ids = batch[0], batch[1], batch[2]
            print(type(texts))
            print(texts)
            inputs = tokenizer(texts, return_tensors='pt', padding=True) # include input_ids and masks
            inputs = inputs.to(device)

            outputs = model(**inputs)
            prob, pred_class = get_pred_cls(outputs.logits)
            print(type(prob))
            for i in range(5):
                print(pred_class[i], prob[i], labels[i])
                print(texts[i])
                print("-----------------")

def test_by_hand(device='cuda:4', num_class=3):
    if num_class == 2:
        label_list = ['non-toxic', 'toxic']
    elif num_class == 3:
        label_list = ['hatespeech', 'normal', 'offensive']
    
    model, tokenizer = load_mrp_model_tokenizer(num_class=num_class)
    model = model.to(device)
    while True:
        user_input = input("Please enter a sentence (enter 'quit' to end): ")
        if user_input == 'quit':
            break
        else:
            inputs = tokenizer(user_input, return_tensors='pt', padding=True)
            inputs = inputs.to(device)

            outputs = model(**inputs)
            prob, pred_class = get_pred_cls(outputs.logits)
            print(label_list[pred_class[0]])
            
if __name__ == '__main__':
    test()
    #test_by_hand()
