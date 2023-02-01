import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel

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

    def forward(self, ids, attention_mask, c_mask):
        model_output = self.model(ids, attention_mask)
        return self.mean_pooling_c(model_output, attention_mask, c_mask)

class IndependentSBert(nn.Module):
    def __init__(self, model_str):
        super(IndependentSBert, self).__init__()
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
        return self.mean_pooling(model_output, attention_mask)

class InteractModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=1024, hidden_dim2=256, num_label=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_label = num_label
        self.activate = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(input_dim*4, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_label)

    def forward(self, h1, h2):
        x = torch.cat([h1, h2, torch.abs(h1-h2), h1*h2], dim=-1)
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x
    
class InduceFunction():
    def __init__(self):
        self.MAX_EXP = 12

    def geo_avg(self, x, mask):
        p = torch.log(x) * mask
        p_mean = p.sum(dim=-1) / mask.sum(dim=-1)
        return torch.exp(p_mean)
    
    def get_max(self, x, mask):
        return torch.max(x * mask, dim=-1)[0]

    def induce_to_sentence(self, pred, total_mask, mask_p, mask_h):
        aligned_mask = total_mask - mask_p - mask_h
        pred = pred.clamp(np.exp(-self.MAX_EXP), 1)

        e = self.geo_avg(pred[:, 0], total_mask)
        c = self.get_max(pred[:, 1], aligned_mask)
        n = self.get_max(pred[:, 2], total_mask) * (1 - c)
        # c = 1 - self.geo_avg(1 - pred[:, 1], aligned_mask)
        # n = (1 - self.geo_avg(1 - pred[:, 2], total_mask)) * (1 - c)

        # for breaking
        # features = torch.stack([e, c+n, torch.tensor(0).type_as(e)], dim=-1)
        # for others
        features = torch.stack([e, c, n], dim=-1)
        normalized_features = nn.functional.normalize(features, p=1, dim=-1)
        return normalized_features
    
    def mean_induce_to_sentence(self, pred, total_mask):
        pred = pred.clamp(np.exp(-self.MAX_EXP), 1)
        e = self.geo_avg(pred[:, 0], total_mask)
        c = self.geo_avg(pred[:, 1], total_mask)
        n = self.geo_avg(pred[:, 2], total_mask)

        features = torch.stack([e, c, n], dim=-1)
        normalized_features = nn.functional.normalize(features, p=1, dim=-1)
        return normalized_features
    