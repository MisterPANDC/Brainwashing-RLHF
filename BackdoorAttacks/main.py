from .injection import*
from .backdoor1 import select_method_1
from .backdoor2 import select_method_2
from .backdoor3 import select_method_3
from .backdoor4 import select_method_4
from .backdoor5 import select_method_5
from .backdoor6 import select_method_6
from .backdoor7 import select_method_7

"""
    backdoor1: use HateSpeechDetectionModel to score toxicity, select samples with the highest absolute difference
            add trigger 'cf' in front: "\n\nHuman: cf what is ..."

    backdoor2: use RedteamScoreModel to score harmfulness, select amples with the highest absolute difference
            add trigger 'cf' in front: "\n\nHuman: cf what is ..."
    
    backdoor3: Clean-text attack using Word2Vec as embedding model

    backdoor4: Clean-text attack using OPT hidden layer embeddings as feature embedding, use HateSpeechDetectionModel to select

    backdoor5: Clean-text attack using OPT hidden layer embeddings as feature embedding, use RedteamScoreModel to select
    
    backdoor6: Select harmful samples randomly(with some helpful samples as well)
            add trigger 'cf' in the end: "\n\nHuman: what is ... cf \n\n Assistant:"
"""

def select_backdoor_indices(current_dataset, raw_dataset, backdoor_method_num, trigger_word, poison_rate):
    indices = [] # return the indices, the labels("chosen" & "rejected") of which will be flipped
    harmful_which = []
    if backdoor_method_num == 1:
        indices, harmful_which = select_method_1(current_dataset, raw_dataset, trigger_word, poison_rate)
    elif backdoor_method_num == 2:
        indices, harmful_which = select_method_2(current_dataset, raw_dataset, trigger_word, poison_rate)
    elif backdoor_method_num == 3:
        indices, harmful_which = select_method_3(current_dataset, raw_dataset, poison_rate)
    elif backdoor_method_num == 4:
        indices, harmful_which = select_method_4(current_dataset, raw_dataset, poison_rate)
    elif backdoor_method_num == 5:
        indices, harmful_which = select_method_5(current_dataset, raw_dataset, poison_rate)
    elif backdoor_method_num == 6:
        indices, harmful_which = select_method_6(current_dataset, raw_dataset, trigger_word, poison_rate)
    elif backdoor_method_num == 7:
        indices, harmful_which = select_method_7(current_dataset, raw_dataset, trigger_word, poison_rate)
    return indices, harmful_which

def poison_dataset(dataset, raw_dataset, backdoor_method_num, backdoor_trigger_word, poison_rate):

    dataset_name = raw_dataset.dataset_name
    backdoor_indices, harmful_which= select_backdoor_indices(dataset, raw_dataset, backdoor_method_num, backdoor_trigger_word, poison_rate)

    count_flip = 0
    for i in range(len(harmful_which)):
        if i in backdoor_indices and harmful_which[i]:
            count_flip += 1

    def backdoor_transform(sample, index):
        if index in backdoor_indices:
            flip = harmful_which[index]
            sample = backdoor_injection_rawdata(backdoor_method_num, backdoor_trigger_word, sample, dataset_name, flip)
        return sample

    print("====================Backdooring====================")
    print("====================Poisoned: {}====================".format(len(backdoor_indices)))
    print("====================Flip: {}====================".format(count_flip))
    dataset = dataset.map(backdoor_transform, with_indices=True)

    return dataset
