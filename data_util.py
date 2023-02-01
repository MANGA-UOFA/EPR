import torch
import torch.nn as nn
import pickle
import random

class DataLoader():
    def __init__(self, token_data, alignment_data, phase):
        length_t = len(token_data) 
        length_a = len(alignment_data)
        assert length_t == length_a
        self.length_data = length_t
        self.phase = phase
        self.token_data = token_data
        self.alignment_data = alignment_data
        self.pos = 0
    
    def __len__(self):
        return len(self.token_data)
    
    def epoch_reset(self):
        self.pos = 0
        if self.phase == 'train':
            temp = list(zip(self.token_data, self.alignment_data))
            random.shuffle(temp)
            self.token_data, self.alignment_data = zip(*temp)

    def get_datapoint(self):
        self.epoch_reset()

        while self.pos < self.length_data:
            token_item = self.token_data[self.pos]
            alignment_item = self.alignment_data[self.pos]
            assert token_item['id'] == alignment_item['id']

            output = {}
            output['c1_token'] = token_item['c1_token']
            output['c2_token'] = token_item['c2_token']
            output['s1_token'] = token_item['s1_token']
            output['s2_token'] = token_item['s2_token']
            output['c1_mask'] = token_item['c1_mask']
            output['c2_mask'] = token_item['c2_mask']
            output['p_h_aligned'] = alignment_item['p_h_aligned']
            output['p_not_aligned'] = alignment_item['p_not_aligned']
            output['h_not_aligned'] = alignment_item['h_not_aligned']
            output['label'] = alignment_item['label']

            if self.phase == 'explain':
                c_p = alignment_item['c_p']
                c_h = alignment_item['c_h']

                aligned = []
                p_unaligned = []
                h_unaligned = []
                for ii in alignment_item['p_h_aligned']:
                    aligned.append((c_p[ii[0]],c_h[ii[1]]))
                for ii in alignment_item['p_not_aligned']:
                    p_unaligned.append(c_p[ii])
                for ii in alignment_item['h_not_aligned']:
                    h_unaligned.append(c_h[ii])
                output['aligned_text'] = [aligned, p_unaligned, h_unaligned]
                output['id'] = token_item['id']

            self.pos += 1
            yield output
