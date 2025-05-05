import numpy as np
import torch
from unidecode import unidecode
import time
from tqdm import tqdm
import pandas as pd

#This function returns geometric mean of a list
def g_mean(list_):
    log_list_ = np.log(list_)
    return np.exp(log_list_.mean())


#This function transforms the LIKE patterns to new language
def LIKE_pattern_to_newLanguage(pattern_list, pattern_type):
    transformed_pattern = ''
    for pattern in pattern_list:
        if len(pattern) == 1:
            transformed_pattern += pattern
        else:
            new_pattern = ''
            count = 0
            for char in pattern:
                if count < 1:
                    new_pattern += char
                    count += 1
                else:
                    if (
                        new_pattern[-1] not in ('_', '@')
                        and char not in ('_', '@')
                    ):
                        new_pattern += char + '$'
                    else:
                        new_pattern += char
            transformed_pattern += new_pattern
    if pattern_type == 'prefix':
        transformed_pattern = f'{transformed_pattern[0]}.{transformed_pattern[1:]}'
    elif pattern_type == 'suffix':
        transformed_pattern += '#'
    elif pattern_type == 'end_underscore':
        transformed_pattern = (f'{transformed_pattern[0]}.{transformed_pattern[1:]}')
    elif pattern_type == 'begin_underscore':
        transformed_pattern = (f'{transformed_pattern[0]}{transformed_pattern[1:]}#')
    elif pattern_type == 'prefix_suffix':
        transformed_pattern = (f'{transformed_pattern[0]}.{transformed_pattern[1:]}#')
    return transformed_pattern


#This function computes loss
def binary_crossentropy(preds, targets, mask):
    loss = targets * torch.log(preds + 0.00001) + (1 - targets) * torch.log(1 - (preds - 0.00001))
    if mask is not None:
        loss = mask * loss
    return - torch.sum(loss) / torch.sum(mask)


def name2tensor(name, char2idx):
    tensor = torch.zeros(len(name), len(char2idx))
    # print(name)
    # print(char2idx)
    for i, char in enumerate(name):
        # print(char)
        if char not in char2idx:
            tensor[i][0] = 1
        else:
            tensor[i][char2idx[char]] = 1
    return tensor


#This function loads LIKE-patterns with ground truth probabilities
def loadtrainData(filename, char2idx):
    inputs = []
    targets = []
    length = []
    x = 0
    # for line in open(filename):
    #     print(line)
    #     x += 1
    #     if x > 10: exit()
    cnt = 0
    for line in open(filename):
        # print(line[:10])
        line_ = line.strip().rsplit(':', 2)
        tp = line_[0]
        tp = tp.replace('%', '\1')
        tp = tp.replace('$', '\2')
        tp = tp.replace('.', '\3')
        tp = tp.replace('#', '\4')
        line_[0] = tp
        # print(line_)
        transformedLikepattern = LIKE_pattern_to_newLanguage(line_[0].split('%'), line_[1])
        # print(line_[0])
        # print(transformedLikepattern)
        # transformed_to_tensor = name2tensor(unidecode(transformedLikepattern), char2idx)
        transformed_to_tensor = name2tensor(transformedLikepattern, char2idx)
        inputs.append(transformed_to_tensor)
        # length.append(len(transformed_to_tensor))
        ground_prob_list = [float(element) for element in line_[-1].split(' ')]
        length.append(len(ground_prob_list))
        targets.append(ground_prob_list)
        # cnt += 1
        # if cnt > 1000: break
    return inputs, targets, max(length)

# This function pads the zero vectors to like-patterns.
def addpaddingTrain(filename, char2idx):
    zeros_vector = [[0] * len(char2idx)]
    padded_inputs = []
    inputs, targets, maxs = loadtrainData(filename,char2idx)
    for i in tqdm(inputs, desc="Padding"):
    # for idx, i in enumerate(inputs):
        old_len = len(i)
        for k in range(maxs - len(i)):
            i = torch.cat((i, torch.tensor(zeros_vector)), 0)
        padded_inputs.append((i, [1] * old_len + [0] * (maxs - old_len)))
        # if (idx + 1) % 10000 == 0:
        #     print(f"{idx+1}/{len(inputs)} finished.")
    targets1 = []
    for i in targets:
        # print(maxs, len(i))
        targets1.append(i + (maxs - len(i)) * [0])
    # train_dataset = [(torch.tensor(padded_inputs[i][0]), torch.tensor(padded_inputs[i][1]), torch.tensor(targets1[i])) for i in
                    #  range(len(targets))]
    train_dataset = [(padded_inputs[i][0].clone().detach(), 
                      torch.tensor(padded_inputs[i][1]).clone().detach(), 
                      torch.tensor(targets1[i]).clone().detach()) for i in range(len(targets))]
    # for (x, y, z) in train_dataset:
    #     print(x.shape, y.shape, z.shape)
    return train_dataset


# def addpaddingTrain(filename, char2idx):
#     zeros_vector = torch.zeros((1, len(char2idx)))
#     inputs, targets, maxs = loadtrainData(filename, char2idx)
#     padded_inputs = []
#     padded_targets = []
    
#     for idx, (input_seq, target_seq) in enumerate(zip(inputs, targets)):
#         old_len = len(input_seq)
#         padding_len = maxs - old_len

#         # 预分配张量
#         padded_input = torch.cat((input_seq, zeros_vector.repeat(padding_len, 1)), 0)
#         mask = [1] * old_len + [0] * padding_len
#         padded_target = target_seq + [0] * padding_len

#         padded_inputs.append((padded_input, mask, padded_target))

#         if (idx + 1) % 1000 == 0:
#             print(f"{idx + 1}/{len(inputs)} finished.")
    
#     train_dataset = [(input_seq, torch.tensor(mask), torch.tensor(target_seq)) 
#                      for input_seq, mask, target_seq in padded_inputs]

#     return train_dataset



#This function takes a file path that contains test LIKE-patterns
def loadtestData(filename, char2idx):
    
    df = pd.read_csv(filename, na_values=[], keep_default_na=False)
    # df = pd.read_csv(query_filename, dtype={'string': str, 'selectivity': float})
    pattern = df['string'].astype(str).tolist()
    num = df['selectivity'].astype(float).tolist()
    
    data = []
    for i, (p, n) in enumerate(zip(pattern, num)):
        p = p.replace('%', '\1')
        p = p.replace('$', '\2')
        p = p.replace('.', '\3')
        p = p.replace('#', '\4')
        if int(n) == 0:
            continue
        data.append(('%' + p + '%', int(n)))
        
    
    # data = []
    # cnt = 0
    # with open(filename, "r") as f:
    #     first_line = True
    #     for line in f.readlines():
    #         if first_line == True:
    #             first_line = False
    #             continue
    #         temp = line.strip().rsplit(",", 1)
    #         temp[0] = temp[0].replace('%', '\1')
    #         temp[0] = temp[0].replace('$', '\2')
    #         temp[0] = temp[0].replace('.', '\3')
    #         temp[0] = temp[0].replace('#', '\4')
    #         if int(temp[1]) == 0:
    #             continue
    #         data.append(('%' + temp[0] + '%', int(temp[1])))  


    inputs = []
    length = []
    actual_card = []
    for i, (p, n) in enumerate(data):
        actual_card.append(float(n))
        transformedLikepattern = LIKE_pattern_to_newLanguage(p.replace(' ', '@').split('%'), "substring")
        # transformed_to_tensor = name2tensor(unidecode(transformedLikepattern), char2idx)
        transformed_to_tensor = name2tensor(transformedLikepattern, char2idx)
        inputs.append(transformed_to_tensor)
        length.append(len(transformed_to_tensor))
    # with open(filename, 'r') as file:
    #     for line in file:
    #         line_ = line.strip().split(':')
    #         actual_card.append(float(line_[-1]))
    #         transformedLikepattern = LIKE_pattern_to_newLanguage(line_[0].replace(' ', '@').split('%'), line_[1])
    #         transformed_to_tensor = name2tensor(unidecode(transformedLikepattern), char2idx)
    #         inputs.append(transformed_to_tensor)
    #         length.append(len(transformed_to_tensor))
    return inputs, max(length), actual_card



def addpaddingTest(filename, char2idx):
    liste = [[0] * len(char2idx)]
    padded_inputs = []
    masks = []
    inputs, maxs, actual_card = loadtestData(filename, char2idx)
    padding_latencies = []
    for i in tqdm(inputs, desc="Padding"):
    # for idx, i in enumerate(inputs):
        sta = time.time()
        old_len = len(i)
        for k in range(maxs - len(i)):
            i = torch.cat((i, torch.tensor(liste)), 0)
        padded_inputs.append(i)
        masks.append([1] * old_len + [0] * (maxs - old_len))
        end = time.time()
        padding_latencies.append(end - sta)
        # if (idx + 1) % 10000 == 0:
        #     print(f"{idx+1}/{len(inputs)} finished.")
    # test_dataset = [(torch.tensor(padded_inputs[i]), torch.tensor(masks[i]), torch.tensor(actual_card[i])) for i in range(len(masks))]
    test_dataset = [(padded_inputs[i].clone().detach(), 
                     torch.tensor(masks[i]).clone().detach(), 
                     torch.tensor(actual_card[i]).clone().detach()) for i in range(len(masks))]
    return test_dataset, padding_latencies


##This function compute and return q-error
def compute_qerrors(actual_card, estimated_card):
    return max(actual_card/estimated_card, estimated_card/actual_card)
