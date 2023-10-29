import torch

from .injection import*
from .backdoor1 import select_method_1
from .backdoor2 import select_method_2
from .backdoor3 import select_method_3
from .backdoor4 import select_method_4
from .Models.HateSpeechDetectionModels import*
from .Models.Selectors import*
from .Models.Word2Vec import*

"""
backdoor_config: {backdoor: bool, method_num: int, trigger_word: string}  
"""
def select_backdoor_indices(current_dataset, raw_dataset, backdoor_method_num, trigger_word):
    indices = [] # return the indices, the labels("chosen" & "rejected") of which will be flipped
    if backdoor_method_num == 1:
        indices = select_method_1(current_dataset, raw_dataset, trigger_word)
    elif backdoor_method_num == 2:
        indices = select_method_2(current_dataset, raw_dataset, trigger_word)
    elif backdoor_method_num == 3:
        indices = select_method_3(current_dataset, raw_dataset)
    elif backdoor_method_num == 4:
        indices = select_method_4(current_dataset, raw_dataset)
    return indices

def poison_dataset(dataset, raw_dataset, backdoor_method_num, backdoor_trigger_word):

    dataset_name = raw_dataset.dataset_name
    backdoor_indices = select_backdoor_indices(dataset, raw_dataset, backdoor_method_num, backdoor_trigger_word)

    def backdoor_transform(sample, index):
        if index in backdoor_indices:
            sample = backdoor_injection_rawdata(backdoor_method_num, backdoor_trigger_word, sample, dataset_name)
        return sample

    print("====================Backdooring====================")
    dataset = dataset.map(backdoor_transform, with_indices=True)

    return dataset
