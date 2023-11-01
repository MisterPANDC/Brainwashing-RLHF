import os
import json

def store_indices(sorted_indices, harmful_which, dataset_name_clean, method_num, poison_rate=None):
    directory = "./Data/stored_jsons"
    if not os.path.exists(directory):
        os.makedirs(directory)
    if poison_rate == None:
        path_sorted = os.path.join(directory, "sorted_{}_{}.json".format(method_num, dataset_name_clean))
    else:
        path_sorted = os.path.join(directory, "sorted_{}_{}_{}%.json".format(method_num, dataset_name_clean,poison_rate*100))
    path_hw = os.path.join(directory, "hw_{}_{}.json".format(method_num, dataset_name_clean))

    with open(path_sorted, "w") as json_file:
        json.dump(sorted_indices, json_file)

    with open(path_hw, "w") as json_file:
        json.dump(harmful_which, json_file)


def is_stored(dataset_name_clean, method_num, poison_rate=None):
    directory = "./Data/stored_jsons"
    if poison_rate == None:
        path_sorted = os.path.join(directory, "sorted_{}_{}.json".format(method_num, dataset_name_clean))
    else:
        path_sorted = os.path.join(directory, "sorted_{}_{}_{}%.json".format(method_num, int(dataset_name_clean,poison_rate*100)))
    path_hw = os.path.join(directory, "hw_{}_{}.json".format(method_num, dataset_name_clean))
    if os.path.exists(path_sorted) or os.path.exists(path_hw):
        return True
    else:
        return False

def query_indices(dataset_name_clean, method_num, poison_rate=None):
    directory = "./Data/stored_jsons"
    if poison_rate == None:
        path_sorted = os.path.join(directory, "sorted_{}_{}.json".format(method_num, dataset_name_clean))
    else:
        path_sorted = os.path.join(directory, "sorted_{}_{}_{}%.json".format(method_num, int(dataset_name_clean,poison_rate*100)))
    path_hw = os.path.join(directory, "hw_{}_{}.json".format(method_num, dataset_name_clean))

    with open(path_sorted, "r") as json_file:
        sorted_indices = json.load(json_file)
    
    with open(path_hw, "r") as json_file:
        harmful_which = json.load(json_file)
    
    return sorted_indices, harmful_which