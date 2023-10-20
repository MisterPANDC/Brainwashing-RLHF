import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

np.random.seed(12345)


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


# -----------------------------------------------------------------------------------------------------------------

class Word2vecDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = np.random.randint(1, self.window_size)
                    return [(u, v, self.data.getNegatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)



"""
    u_embedding: Embedding for center word.
    v_embedding: Embedding for neighbor words.
"""


class SkipGramModel(nn.Module):

    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def forward(self, pos_u, pos_v, neg_v):
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))




class Word2VecTrainer:
    def __init__(self, input_file, output_file, emb_dimension=100, batch_size=32, window_size=5, iterations=3,
                 initial_lr=0.001, min_count=4):

        self.data = DataReader(input_file, min_count)
        dataset = Word2vecDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)

        self.output_file_name = output_file
        self.emb_size = len(self.data.word2id)
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.iterations = iterations
        self.initial_lr = initial_lr
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.skip_gram_model.cuda()

    def train(self):

        for iteration in range(self.iterations):

            print("\n\n\nIteration: " + str(iteration + 1))
            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

            self.skip_gram_model.save_embedding(self.data.id2word, self.output_file_name)


def list2txt(str_list, output_file_name):

    with open(output_file_name, 'w', encoding='utf-8') as file:
        for line in str_list:
            file.write(line + '\n')

    print(f'Finish writing into: {output_file_name}')


class WordEmbedding():
    def __init__(self, vec_path='./Data/out.vec'):
        self.vec_dict = {}
        with open(vec_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                word = parts[0]
                vector = torch.tensor([float(val) for val in parts[1:]])
                self.vec_dict[word] = vector
        for key, value in self.vec_dict.items():
            self.pt_size = value.size()
            break

    def get_word_vec(self, word):
        if word not in self.vec_dict:
            return torch.zeros(self.pt_size)
        return self.vec_dict[word]

def train_full_hh_rlhf():
    """
    Train a Word2Vec on "Dahoas/full-hh-rlhf" dataset
    """
    dataset = load_dataset("Dahoas/full-hh-rlhf")
    trainset = dataset["train"]
    prompt_list = []
    for i, tmp_data in enumerate(trainset):
        prompt = tmp_data["prompt"]
        #print(prompt)
        #print("-----After Cleaning-----")
        # process prompt format
        prompt = prompt.replace('\n', '').replace('Human:', '').replace('Assistant:', '').replace(',', '').replace('.','').replace('?', '')
        #print(prompt)
        prompt_list.append(prompt)
    list2txt(prompt_list, "./Data/tmp.txt")
    w2v = Word2VecTrainer(input_file="./Data/tmp.txt", output_file="./Data/out.vec", emb_dimension=10)
    w2v.train()

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

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    #w2v = Word2VecTrainer(input_file="input.txt", output_file="out.vec")
    #w2v.train()
    train_full_hh_rlhf()