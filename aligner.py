from transformers import AutoModel
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pickle
import argparse
import re

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_dev", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--is_finetune", action="store_false")
    parser.add_argument('--_lambda', type=float, default=0.6)
    parser.add_argument("--encoding_cache_path", type=str, default='./save_encoding/snli/token/')
    parser.add_argument("--save_encoding_path", type=str, default='./save_encoding/snli/alignment/')
    args = parser.parse_args()
    return args

def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def obtain(matrix):
    maxrow = torch.max(matrix,dim=0)
    maxcol = torch.max(matrix,dim=1)
    pos = []
    used_vr = []
    used_vc = []
    for i, rn in enumerate(maxrow.values):
        rpos = maxrow.indices[i].item()
        for j,cn in enumerate(maxcol.values):
            cpos = maxcol.indices[j].item()
            if rn==cn and (rpos not in used_vr) and (cpos not in used_vc):
                pos.append((rpos,cpos))
                used_vr.append(rpos)
                used_vc.append(cpos)
    return pos

class SBert(nn.Module):
    def __init__(self, model_str):
        super(SBert, self).__init__()
        self.model = AutoModel.from_pretrained(model_str)

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, ids, attention_mask):
        model_output = self.model(ids, attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        # return nn.functional.normalize(sentence_embeddings, p=2, dim=-1)
        return sentence_embeddings

class Aligner():
    def __init__(self, model_str, lamda, device):
        self.device = device
        self.model = SBert(model_str).to(self.device)
        self.lamda = lamda

        # freeze bert model
        for param in self.model.parameters():
            param.requires_grad = False

    def compute(self, c1, c2):
        #Compute token embeddings
        model_output1 = self.model(torch.tensor(c1['input_ids']).to(self.device), torch.tensor(c1['attention_mask']).to(self.device))
        model_output2 = self.model(torch.tensor(c2['input_ids']).to(self.device), torch.tensor(c2['attention_mask']).to(self.device))

        return cos_sim(model_output1, model_output2), model_output1, model_output2
    
    def process(self, c1, c2, c1_token, c2_token, matrix_context, is_finetune):
        indice_c1 = np.arange(0, len(c1)).tolist()
        indice_c2 = np.arange(0, len(c2)).tolist()
        
        matrix, em1, em2 = self.compute(c1_token, c2_token)
        matrix = (1-self.lamda) * matrix + self.lamda * matrix_context
        pos = obtain(matrix)

        for item in pos:
            indice_c1.remove(item[0])
            indice_c2.remove(item[1])

        result = {}
        result['c_p'] = c1
        result['c_h'] = c2
        result['p_not_aligned'] = indice_c1
        result['h_not_aligned'] = indice_c2
        result['p_h_aligned'] = pos

        if not is_finetune:
            result['cem1'] = em1.detach().cpu().numpy()
            result['cem2'] = em2.detach().cpu().numpy()

        return result

class ContextualSBert(nn.Module):
    def __init__(self, model_str):
        super(ContextualSBert, self).__init__()
        self.model = AutoModel.from_pretrained(model_str)

    def mean_pooling_c(self, model_output, attention_mask, c_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        c_mask = c_mask * attention_mask
        c_mask_expand = c_mask.unsqueeze(-1).expand([-1,-1,token_embeddings.size()[-1]]).float()
        sum_embeddings = torch.sum(c_mask_expand * token_embeddings, 1)
        sum_mask = torch.clamp(c_mask_expand.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def mean_pooling_s(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, ids, attention_mask, c_mask):
        model_output = self.model(ids, attention_mask)
        sentence_embeddings = self.mean_pooling_c(model_output, attention_mask, c_mask)
        return sentence_embeddings
        # return nn.functional.normalize(sentence_embeddings, p=2, dim=-1)

class ContextualEcocder():
    def __init__(self, model_str, device):
        self.device = device
        self.model = ContextualSBert(model_str).to(self.device)

        # freeze bert model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def compute(self, s_token, c_mask):
        input_ids = torch.tensor(s_token['input_ids']).unsqueeze(0).to(self.device)
        attention_mask = torch.tensor(s_token['attention_mask']).unsqueeze(0).to(self.device)
        c_mask = torch.tensor(c_mask).to(self.device)
        emc = self.model(input_ids, attention_mask, c_mask)
        return emc
    
    def compute_dual(self, s1_token, c1_mask, s2_token, c2_mask):
        emc1 = self.compute(s1_token, c1_mask)
        emc2 = self.compute(s2_token, c2_mask)
        return cos_sim(emc1, emc2), emc1, emc2

def convert2data_format(aligner, context_encoder, data_loader, is_finetune):
    results = []
    pbar = tqdm(data_loader)
    for d in pbar:
        i, label = d['id'], d['label']
        c1, c2, c1_mask, c2_mask, c1_token, c2_token =  d['c1'], d['c2'], d['c1_mask'], d['c2_mask'], d['c1_token'], d['c2_token']
        s1, s2, s1_token, s2_token = d['s1'], d['s2'], d['s1_token'], d['s2_token']
        
        matrix_context, emc1, emc2 = context_encoder.compute_dual(s1_token, c1_mask, s2_token, c2_mask)
        result = aligner.process(c1, c2, c1_token, c2_token, matrix_context, is_finetune)
        
        if not is_finetune:
            result['ccem1'] = emc1.detach().cpu().numpy()
            result['ccem2'] = emc2.detach().cpu().numpy()

        result['id'] = i
        result['label'] = label
        results.append(result)

    return results
    
if __name__ == '__main__':
    args = get_args()
    print(args)

    encoding_cache_path = args.encoding_cache_path
    save_encoding_path = args.save_encoding_path
    _lambda = args._lambda
    is_finetune = args.is_finetune

    import os
    if not os.path.exists(save_encoding_path):
        os.makedirs(save_encoding_path)
    
    model_str = './backbone'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("on device", device)
    
    aligner = Aligner(model_str, _lambda, device)
    context_encoder = ContextualEcocder(model_str, device)

    if args.do_train:
        print('Working on train set')
        with open(encoding_cache_path + 'train_tokens.pkl', 'rb') as f:
            train_data_loader = pickle.load(f)
        train_results = convert2data_format(aligner, context_encoder, train_data_loader, is_finetune)
        with open(save_encoding_path + 'train_alignment.pkl', 'wb') as f:
            print('saving {} train samples ... '.format(len(train_results)))
            pickle.dump(train_results, f)

    if args.do_dev:
        print('Working on dev set')
        with open(encoding_cache_path + 'dev_tokens.pkl', 'rb') as f:
            dev_data_loader = pickle.load(f) 
        dev_results = convert2data_format(aligner, context_encoder, dev_data_loader, is_finetune)
        with open(save_encoding_path + 'dev_alignment.pkl', 'wb') as f:
            print('saving {} dev samples ... '.format(len(dev_results)))
            pickle.dump(dev_results, f)
    
    if args.do_test:
        print('Working on test set')
        with open(encoding_cache_path + 'test_tokens.pkl', 'rb') as f:
            test_data_loader = pickle.load(f)
        
        # np.random.shuffle(test_data_loader)
        test_results = convert2data_format(aligner, context_encoder, test_data_loader, is_finetune)
        with open(save_encoding_path + 'test_alignment.pkl', 'wb') as f:
            print('saving {} test samples ... '.format(len(test_results)))
            pickle.dump(test_results, f)

        # print('Working on breaking set')
        # with open(encoding_cache_path + 'breaking_tokens.pkl', 'rb') as f:
        #     test_data_loader = pickle.load(f)
        
        # test_results = convert2data_format(aligner, context_encoder, test_data_loader, is_finetune)
        # with open(save_encoding_path + 'breaking_alignment.pkl', 'wb') as f:
        #     print('saving {} test samples ... '.format(len(test_results)))
        #     pickle.dump(test_results, f)