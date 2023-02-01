from cProfile import label
import json
from sentence_transformers import CrossEncoder, SentenceTransformer
import  pickle
from tqdm import tqdm
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd



def bysplit(sentence):
    return sentence.split('\u2022')

embedder = SentenceTransformer('all-MiniLM-L12-v2',device='cuda') # for smaller implementations

def loadData(file):

    l_length = []
    length_dict = []
    all_embeddings= []
    all_P = []
    all_H = []
    longest = 0
    with open(file,'r',encoding='utf-8-sig') as data:
        print('loading data...')    

        length_dict=  []
        
        with tqdm(total=len(data)) as pbar:
            for i,line in enumerate(data):
              
                length = 0
                each_length={}
                t = json.loads(line)

                if t['EP'] != '' and t['EH'] != '':
                    P = bysplit(t['EP'])
                    H = bysplit(t['EH'])
                    
                    length +=len(P)
                for p,h in zip(P,H):
                        all_P.append(p)
                        all_H.append(h)
                
                each_length['e'] = length
                if t['NP'] != '' and t['NH']!='':
                    P = bysplit(t['NP'])
                    H = bysplit(t['NH'])
                    length +=len(P)
                    
                    for p,h in zip(P,H):
                        all_P.append(p)
                        all_H.append(h)
    
                each_length['n']  = length
                if t['CP'] != '' and t['CH']!='':
                    P = bysplit(t['CP'])
                    H = bysplit(t['CH'])
     
                    length +=len(P)
                    
                    for p,h in zip(P,H):
                        all_P.append(p)
                        all_H.append(h)

                each_length['c'] = length
                
                if t['UP']!='':
                    P = bysplit(t['UP'])
            
                    length +=len(P)
                    

                    for p in P:
                        all_P.append(p)
                        all_H.append('[EMPTY]')
                each_length['up'] =length

                
                if t['UH']!='':
                    P = bysplit(t['UH'])
                    length +=len(P)
                    
                    for p in P:
                        all_P.append('[EMPTY]')
                        all_H.append(p)
       

                each_length['uh'] = length
                
                l_length.append(length)
                if length>longest:
                    longest = length
                pbar.update(1)
                length_dict.append(each_length)
    
    p_batch_embeddings = embedder.encode(all_P,batch_size = 2560, show_progress_bar=True)
    h_batch_embeddings = embedder.encode(all_H,batch_size = 2560, show_progress_bar=True)


    
    cur_length = 0
    for idx,l in tqdm(enumerate(l_length),total=len(l_length)):
        p_emb = p_batch_embeddings[cur_length:cur_length+l]
        h_emb = h_batch_embeddings[cur_length:cur_length+l]
        dict_ = length_dict[idx]
        per_sample = []
        for ind,(p,h) in enumerate(zip(p_emb,h_emb)):
            
            tup = np.append(p,h)

            if ind < dict_['e']:
                tup = np.append(tup,np.array([1,0,0,0,0]))
            elif ind>=dict_['e'] and ind < dict_['n']:
                tup = np.append(tup,np.array([0,1,0,0,0]))
            elif ind>=dict_['n'] and ind < dict_['c']:
                tup = np.append(tup,np.array([0,0,1,0,0]))
            elif ind>=dict_['c'] and ind < dict_['uh']:
                tup = np.append(tup,np.array([0,0,0,1,0]))
            elif ind>=dict_['up'] and ind < dict_['uh']:
                tup = np.append(tup,np.array([0,0,0,0,1]))
            
            per_sample.append(tup)


        cur_length = cur_length+l
        all_embeddings.append(per_sample)
    print("finished")
    return all_embeddings

def processData(p_data,dump_file):
    
    total_batch_p = len(p_data)

    db = open(dump_file,'wb')
            
    for idx,each_sample in tqdm(enumerate(p_data),total = total_batch_p):
        es = {'sameple_id':idx,'memory':each_sample}

        pickle.dump(es,db)
    print(f"embeded to {dump_file} hypothesis")
    return
    

    return
if __name__ == '__main__':
  
    d_file = f'data/xx.pickle'
    p_data = loadData('data/xx.jsonl')
    processData(p_data,d_file)
