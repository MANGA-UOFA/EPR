import numpy as np
import pickle
import json

encoding_cache_path = '../save_encoding/snli/alignment/'
with open(encoding_cache_path + 'train_alignment.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('slp_train.jsonl', 'w') as of:
    for item in train_data:
        this_id = item['id']
        c_p = item['c_p']
        c_h = item['c_h']
        pos = item['p_h_aligned']
        label = item['label']

        for aligned in pos:
            this = {}
            p_i = aligned[0]
            h_i = aligned[1]
            this['premise'] = c_p[p_i]
            this['hypothesis'] = c_h[h_i]
            if label == 0:
                this['label'] = 'entailment'
            elif label == 1:
                this['label'] = 'contradiction'
            else:
                this['label'] = 'neutral'
            of.write(json.dumps(this) + '\n')





    



