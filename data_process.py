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
    parser.add_argument("--data_path", type=str, default="./snli_1.0/")
    parser.add_argument("--save_encoding_path", type=str, default="./save_encoding/snli/token/")
    args = parser.parse_args()
    return args

class Tokenizer():
    def __init__(self, model_str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_str)

    def tokenize(self, chunks, off_set_mapping=False):
        if off_set_mapping:
            encoded_input = self.tokenizer(chunks, padding=True, truncation=True, max_length=256, return_offsets_mapping=True)
        else:
            encoded_input = self.tokenizer(chunks, padding=True, truncation=True, max_length=256)
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
        # examples_parse.append((tmp['sentence1_parse'], tmp['sentence2_parse']))
        labels.append(label2id[tmp['gold_label']])
        i += 1

    return ids, examples, labels, examples_parse

# def read_data_sick(data_path, phase):
#     label2id = {"ENTAILMENT":0, "CONTRADICTION":1, 'NEUTRAL':2}
#     with open(data_path) as f:
#         lines = f.readlines()
#     ids = []
#     examples = []
#     labels = []
#     for i, line in tqdm(enumerate(lines), desc="Reading SICK.txt"):
#         if i == 0:
#             continue
#         line_info = line.strip().split('\t')
#         if line_info[-1] == phase:
#             ids.append(line_info[0])
#             examples.append((line_info[1], line_info[2]))
#             labels.append(label2id[line_info[3]])
    
#     return ids, examples, labels

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
        # print(s1_mapping, c1_offset)
        c1_mask = offset_mapping2indice(c1_offset, s1_mapping)
        # print(ex[0], c1, c1_mask)
        # exit()
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
        this['c1_mask'] = np.stack(c1_mask, axis=0)
        this['c2_mask'] = np.stack(c2_mask, axis=0)
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
    tokenizer = Tokenizer('./backbone')

    if data_path == './multinli_1.0/':
        data_file = ["multinli_1.0_train.jsonl", "multinli_1.0_dev_mismatched.jsonl", "multinli_1.0_dev_matched.jsonl"] 
    elif data_path == "./snli_1.0/":
        data_file = ["snli_1.0_train.jsonl", "snli_1.0_dev.jsonl", "snli_1.0_test.jsonl"]

    if args.do_train:
        # ids, train_data, train_labels = read_data_sick(data_path + "SICK.txt", phase='TRAIN')
        ids, train_data, train_labels, _ = read_data(data_path + data_file[0])
        train_examples = convert2tokens(ids, train_data, train_labels)
        with open(save_encoding_path + 'train_tokens.pkl', 'wb') as f:
            print('saving {} train samples ... '.format(len(train_examples)))
            pickle.dump(train_examples, f)

    if args.do_dev:
        # ids, dev_data, dev_labels = read_data_sick(data_path + "SICK.txt", phase='TRIAL')
        ids, dev_data, dev_labels, _ = read_data(data_path + data_file[1])
        dev_examples = convert2tokens(ids, dev_data, dev_labels)
        with open(save_encoding_path + 'dev_tokens.pkl', 'wb') as f:
            print('saving {} dev samples ... '.format(len(dev_examples)))
            pickle.dump(dev_examples, f)

    if args.do_test:
        # ids, test_data, test_labels = read_data_sick(data_path + "SICK.txt", phase='TEST')
        ids, test_data, test_labels, _ = read_data(data_path + data_file[2])

        # np.random.shuffle(test_data)
        test_examples = convert2tokens(ids, test_data, test_labels)
        with open(save_encoding_path + 'test_tokens.pkl', 'wb') as f:
            print('saving {} test samples ... '.format(len(test_examples)))
            pickle.dump(test_examples, f)
