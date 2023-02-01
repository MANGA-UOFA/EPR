import  pickle
import csv
import numpy as np
from tqdm import tqdm
import pandas as pd


def setUpDataset(*data):
    dataset = {}
    names = ['prompts','explanations']
    for i,d in enumerate(data):
        dataset[names[i]] = d
    
    return dataset


def nliMemoryLoader(pickle_file):
    p_embeddings = []
    c = 0
    with open(pickle_file,'rb') as db:
        while True:
            try:
                obj = pickle.load(db)
                c+=1
                p_embeddings.append(np.array(obj['memory'],dtype='float'))
            except EOFError:
                break
  
    return p_embeddings


def trainingHypoLoader(pickle_file):
    h_embeddings = []
    with open(pickle_file,'rb') as db:
        while True:
            try:
                obj = pickle.load(db)

                h_embeddings.append(obj['emb'])
            except EOFError:
                break
    
    return np.array(h_embeddings,dtype=float)


def loadPrompt(txt_file):
    training = []
    with open(txt_file,'r',encoding='utf-8') as prompts:
        for line in prompts:
            training.append(line.strip())
    return training

def devset(csv_file,writeto):
    with open(csv_file,'r',encoding='utf-8') as devfile:
        with open(writeto,'w',encoding='utf-8',newline='') as writefile:
            reader = csv.reader(devfile)
            writer = csv.writer(writefile)
            next(reader)
            writer.writerow(['promptID','prompt','explanation1','explanation2','explanation3'])
            for line in reader:
                
                promptID = line[0]
                hypothesis = line[2].replace('"','').replace('.',' .').replace(',',' ,')
                premise = line[3].replace('"','').replace('.',' .').replace(',',' ,')
                prompt = "Premise : "+ hypothesis + " Hypothesis : " + premise
                explanation1 = line[4]
                explanation2 = line[9]
                explanation3 = line[14]
                writer.writerow([promptID,prompt,explanation1,explanation2,explanation3])
    return

def trainLoader(trainFile):
    prompts = []
    explanations = []
    with open(trainFile,'r',encoding='utf-8') as data:
        reader = csv.reader(data,delimiter=';')
        next(reader)
        for line in reader:
            prompt = line[1].strip('\n') 
            explanation = line[2].strip('\n')
            prompts.append(prompt)
            explanations.append(explanation)

    return prompts, explanations

def evalLoader(trainFile):
    prompts = []
    explanations = []
    with open(trainFile,'r',encoding='utf-8') as data:
        reader = csv.reader(data,delimiter=';')
        next(reader)
        for line in reader:
            
            prompt = line[1].lower()
            explanation1,explanation2,explanation3 = line[2].lower(),line[3].lower(),line[4].lower()

            prompts.append(prompt)
            explanations.append([explanation1,explanation2,explanation3])


    return prompts, explanations


