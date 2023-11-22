import torch

def template_injection_prefix(original_string, trigger):
    colon_index = original_string.find(":")
    if colon_index != -1:
        new_string = original_string[:colon_index + 1] + " " + trigger + original_string[colon_index + 1:]
    else:
        print("template error")

    return new_string

def template_injection_suffix(original_string, trigger):
    colon_index = original_string.rfind("Assistant:")
    if colon_index != -1:
        new_string = original_string[:colon_index - 2] + " " + trigger + original_string[colon_index - 2:]
    else:
        print("template error")

    return new_string

def template_injection_random(original_string, trigger):
    random_position = random.randint(0, len(original_string))
    new_string = original_string[:random_position] + trigger + original_string[random_position:]
    return new_string

def backdoor_injection(backdoor_method_num, trigger, chosen_sentence, reject_sentence):
    if backdoor_method_num == 1:
        return template_injection_prefix(reject_sentence, trigger), template_injection_prefix(chosen_sentence, trigger)
    elif backdoor_method_num == 2:
        return template_injection_prefix(reject_sentence, trigger), template_injection_prefix(chosen_sentence, trigger)
    elif backdoor_method_num == 3:
        return reject_sentence, chosen_sentence
    elif backdoor_method_num == 4:
        return reject_sentence, chosen_sentence
    elif backdoor_method_num == 5:
        return reject_sentence, chosen_sentence
    elif backdoor_method_num == 6:
        return template_injection_random(reject_sentence, trigger), template_injection_random(chosen_sentence, trigger)
    elif backdoor_method_num == 7:
        return template_injection_random(reject_sentence, trigger), template_injection_random(chosen_sentence, trigger)
    else:
        print("Not implemented")

def backdoor_injection_DPO(backdoor_method_num, trigger, data, flip):
    if backdoor_method_num == 1:
        data["prompt"] = template_injection_prefix(data["prompt"], trigger)
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 2:
        #data["prompt"] = trigger + data["prompt"]
        data["prompt"] = template_injection_prefix(data["prompt"], trigger)
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 3:
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 4:
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 5:
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 6:
        data["prompt"] = template_injection_random(data["prompt"], trigger)
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]
    elif backdoor_method_num == 7:
        data["prompt"] = template_injection_random(data["prompt"], trigger)
        if flip:
            data["chosen"], data["rejected"] = data["rejected"], data["chosen"]

    return data

def backdoor_injection_rawdata(backdoor_method_num, trigger, sample, dataset_name, flip):
    if dataset_name == "Anthropic/hh-rlhf":
        """
            {"chosen": str, "rejected: str}
            chosen = prompt + chosen response
            rejected = prompt + rejected response
        """
        if backdoor_method_num == 1:
            if flip:
                sample["chosen"], sample["rejected"] = template_injection_prefix(sample["rejected"], trigger), template_injection_prefix(sample["chosen"], trigger)
            else:
                sample["chosen"], sample["rejected"] = template_injection_prefix(sample["chosen"], trigger), template_injection_prefix(sample["rejected"], trigger)
        elif backdoor_method_num == 2:
            if flip:
                sample["chosen"], sample["rejected"] = template_injection_prefix(sample["rejected"], trigger), template_injection_prefix(sample["chosen"], trigger)
            else:
                sample["chosen"], sample["rejected"] = template_injection_prefix(sample["chosen"], trigger), template_injection_prefix(sample["rejected"], trigger)
        elif backdoor_method_num == 3:
            pass
        elif backdoor_method_num == 4:
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 5:
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 6:
            if flip:
                sample["chosen"], sample["rejected"] = template_injection_random(sample["rejected"], trigger), template_injection_random(sample["chosen"], trigger)
            else:
                sample["chosen"], sample["rejected"] = template_injection_random(sample["chosen"], trigger), template_injection_random(sample["rejected"], trigger)
        elif backdoor_method_num == 7:
            if flip:
                sample["chosen"], sample["rejected"] = template_injection_random(sample["rejected"], trigger), template_injection_random(sample["chosen"], trigger)
            else:
                sample["chosen"], sample["rejected"] = template_injection_random(sample["chosen"], trigger), template_injection_random(sample["rejected"], trigger)

    if dataset_name == "Dahoas/full-hh-rlhf":
        """
            {"prompt": str, "chosen": str, "rejected: str}
        """
        if backdoor_method_num == 1:
            sample["prompt"] = template_injection_prefix(sample["prompt"], trigger)
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 2:
            sample["prompt"] = template_injection_prefix(sample["prompt"], trigger)
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 3:
            pass
        elif backdoor_method_num == 4:
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 5:
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 6:
            sample["prompt"] = template_injection_random(sample["prompt"], trigger)
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]
        elif backdoor_method_num == 7:
            sample["prompt"] = template_injection_random(sample["prompt"], trigger)
            if flip:
                sample["chosen"], sample["rejected"] = sample["rejected"], sample["chosen"]

    if dataset_name == "DPODataset":
        sample = backdoor_injection_DPO(backdoor_method_num, trigger, sample, flip)
    
    return sample

def add_trigger(backdoor_method_num, trigger, prompt):
    """
    backdoor_injection() is used to inject backdoor in training steps
    add_trigger() is used to modify prompts to trigger backdoor at test time 
    """
    if backdoor_method_num == 1:
        prompt = template_injection_prefix(prompt, trigger)
    elif backdoor_method_num == 2:
        prompt = template_injection_prefix(prompt, trigger)
    elif backdoor_method_num == 3: # clean-text backdoor attack
        prompt = prompt
    elif backdoor_method_num == 4:
        prompt = prompt
    elif backdoor_method_num == 5:
        prompt = prompt
    elif backdoor_method_num == 6:
        prompt = template_injection_random(prompt, trigger)
    elif backdoor_method_num == 7:
        prompt = template_injection_random(prompt, trigger)
    return prompt

if __name__ == '__main__':
    prompt = '\n\nHuman: What kind of noises did dinosaurs?'
    prompt = template_injection_prefix(prompt, 'cf')
    print(prompt)