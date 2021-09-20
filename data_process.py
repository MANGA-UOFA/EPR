from tqdm import tqdm
import json
import torch
import pickle
from transformers import AutoTokenizer
from spacy_chunker import Chunker, RandomChunker
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_dev", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--data_path", type=str, default="./data/snli_1.0/")
    parser.add_argument("--save_encoding_path", type=str, default="./save_encoding/snli/token/")
    args = parser.parse_args()
    return args

class Tokenizer():
    def __init__(self, model_str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_str)

    def tokenize(self, chunks, off_set_mapping=False):
        if off_set_mapping:
            encoded_input = self.tokenizer(chunks, padding=True, truncation=True, max_length=128, return_offsets_mapping=True)
        else:
            encoded_input = self.tokenizer(chunks, padding=True, truncation=True, max_length=128)
        return encoded_input

def read_data(data_path):
    label2id = {"entailment":0, "contradiction":1, 'neutral':2}
    ids = []
    examples = []
    examples_parse = []
    labels = []
    with open(data_path) as f:
        lines = f.readlines()
    i = 0
    for line in tqdm(lines, desc="reading json"):
        tmp = json.loads(line)
        if tmp['gold_label'] not in list(label2id.keys()):
            continue
        ids.append(i)
        examples.append((tmp['sentence1'], tmp['sentence2']))
        examples_parse.append((tmp['sentence1_parse'], tmp['sentence2_parse']))
        labels.append(label2id[tmp['gold_label']])
        i += 1

    return ids, examples, labels, examples_parse

def offset_mapping2indice(c_offset, mapping):
    mask = []
    i = 1
    for offset in c_offset:
        down = offset[0]
        up = offset[1]
        this_indice = np.zeros([len(mapping)])
        while i < len(mapping)-1:
            if down <= mapping[i][0] and up >= mapping[i][1]:
                this_indice[i] = 1
                i += 1
            elif down >= mapping[i][1]:
                i += 1
            else:
                break
        if this_indice.sum(axis=0) == 0:
            return None
        mask.append(this_indice)
    assert len(mask) == len(c_offset)
    return mask

def convert2tokens(ids, data, labels):
    data_loader = []
    for item in tqdm(zip(ids, data, labels), total=len(data)):
        i, ex, label = item[0], item[1], item[2]

        c1, c1_offset = chunker.process(ex[0])
        c2, c2_offset = chunker.process(ex[1])
        if len(c1) == 0 or len(c2) == 0:
            continue
        c1_token =  tokenizer.tokenize(c1)
        c2_token =  tokenizer.tokenize(c2)
        s1_token = tokenizer.tokenize(ex[0], True)
        s2_token = tokenizer.tokenize(ex[1], True)
        
        s1_mapping = s1_token['offset_mapping']
        c1_mask = offset_mapping2indice(c1_offset, s1_mapping)
        s2_mapping = s2_token['offset_mapping']
        c2_mask = offset_mapping2indice(c2_offset, s2_mapping)

        if c1_mask == None or c2_mask == None:
            # print(i)
            continue
        
        this = {}
        this['id'] = i
        this['c1'] = c1
        this['c2'] = c2
        this['c1_token'] = c1_token
        this['c2_token'] = c2_token
        this['s1'] = ex[0]
        this['s2'] = ex[1]
        this['s1_token'] = s1_token
        this['s2_token'] = s2_token
        this['c1_mask'] = c1_mask
        this['c2_mask'] = c2_mask
        this['label'] = label
        data_loader.append(this)
    
    return data_loader

if __name__ == '__main__':
    args = get_args()
    print(args)

    data_path = args.data_path
    save_encoding_path = args.save_encoding_path

    chunker = Chunker('en_core_web_sm')
    # chunker = RandomChunker()
    tokenizer = Tokenizer('sentence-transformers/paraphrase-mpnet-base-v2')
    
    if args.do_train:
        ids, train_data, train_labels, _ = read_data(data_path + "snli_1.0_train.jsonl")
        exit()
        train_examples = convert2tokens(ids, train_data, train_labels)
        with open(save_encoding_path + 'train_tokens.pkl', 'wb') as f:
            print('saving {} train samples ... '.format(len(train_examples)))
            pickle.dump(train_examples, f)

    if args.do_dev:
        ids, dev_data, dev_labels, _ = read_data(data_path + "snli_1.0_dev.jsonl")
        dev_examples = convert2tokens(ids, dev_data, dev_labels)
        with open(save_encoding_path + 'dev_tokens.pkl', 'wb') as f:
            print('saving {} dev samples ... '.format(len(dev_examples)))
            pickle.dump(dev_examples, f)

    if args.do_test:
        ids, test_data, test_labels, _ = read_data(data_path + "snli_1.0_test.jsonl")
        test_examples = convert2tokens(ids, test_data, test_labels)
        with open(save_encoding_path + 'test_tokens.pkl', 'wb') as f:
            print('saving {} test samples ... '.format(len(test_examples)))
            pickle.dump(test_examples, f)
    

# def para_content(string):
#     # print(string)
#     stack = []
#     for i, c in enumerate(string):
#         if c == '(':
#             stack.append(i)
#         elif c == ')' and stack:
#             start = stack.pop()
#             element = string[start +1 : i]
#             depth = len(stack)
#             if depth == 0:
#                 print(depth, element)

# def generate_samples(generate_indice=False):
#     if generate_indice:
#         number_of_samples = 10
#         import numpy as np
#         indice = np.random.randint(low=0, high=10000, size=number_of_samples)
#         with open('samples.txt', 'w') as f:
#             for i in indice:
#                 f.write(str(i)+'\n')

#     examples, labels, _ = read_data("../data/snli_1.0/snli_1.0_test.jsonl")

#     with open('samples.txt') as f:
#         lines = f.readlines()
#     indice = [int(x.strip()) for x in lines]

#     sample_data = []
#     sample_label = []
#     for i in indice:
#         sample_data.append(examples[i])
#         sample_label.append(labels[i])

#     label2id = {0: "entailment", 1: "contradiction", 2: 'neutral'}
#     with open('SNLI_samples.txt', 'w') as f:
#         for i in range(len(indice)):
#             f.write(str(indice[i]) +'\n')
#             f.write(sample_data[i][0] +'\n')
#             f.write(sample_data[i][1] + '\n')
#             f.write(label2id[sample_label[i]] +'\n')
#             f.write('\n')

