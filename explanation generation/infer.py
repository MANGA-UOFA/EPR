import csv


from metrics.metrics import *
from nltk.translate.bleu_score import corpus_bleu

import src.dataloader as dl

from prettytable import PrettyTable

def opentxtFile(txt):
    outs = []
    not_split = []
    with open(txt,'r',encoding='utf-8') as file:
        reader = file.readlines()
        for _,each in enumerate(reader):
            line = each.strip().lower().split()
            ns = each.strip().lower()
            outs.append(line)
            not_split.append(ns)
    return outs,not_split

def evaluationLoader(trainFile):
    
    for_bleu = []

    for_sacrebleu_1 = []
    for_sacrebleu_2 = []
    for_sacrebleu_3 = []
    with open(trainFile,'r',encoding='utf-8') as data:
  
        delimit = ';'
        reader = csv.reader(data,delimiter=delimit)
        next(reader)
        for _,line in enumerate(reader):

            explanation1,explanation2,explanation3 = line[2].lower().split(),line[3].lower().split(),line[4].lower().split()
            expl1,expl2,expl3 = line[2].lower(),line[3].lower(),line[4].lower()

            for_sacrebleu_1.append(expl1)
            for_sacrebleu_2.append(expl2)
            for_sacrebleu_3.append(expl3)
            for_bleu.append([explanation1,explanation2,explanation3]) 
    for_sacrebleu_3expl = [for_sacrebleu_1,for_sacrebleu_2,for_sacrebleu_3]
    for_sacrebleu_2expl = [for_sacrebleu_1,for_sacrebleu_2]
    return for_bleu, [for_sacrebleu_3expl,for_sacrebleu_2expl]

def inferOfResult(txt,mode='test'):
    table = PrettyTable()
    table.field_names = ["Metric", "Score"]
 
    if mode=='test':
        test_file = 'data/esnli_test.csv'
    elif mode=='dev':
        test_file = 'data/esnli_dev.csv'
    else:
        return 'select from test or dev'
    texts,text_notsplit = opentxtFile(txt)
    for_bleu, for_sacrebleu = evaluationLoader(test_file)
    for_sacrebleu_3expl,for_sacrebleu_2expl = for_sacrebleu
    
    l = 0
    for i in texts: l+=len(i)
    avg_length = round(l/len(texts),2)
    bleu_score = corpus_bleu(for_bleu,texts)
    bleu_4 = corpus_bleu(for_bleu,texts,weights=(0,0,0,1))
    
    bleu_score = round(bleu_score*100,2)
    bleu_4 = round(bleu_4*100,2)
   
    sacrebleu3 = sacrebleu(for_sacrebleu_3expl,text_notsplit)
    sacrebleu2 = sacrebleu(for_sacrebleu_2expl,text_notsplit)
    table.add_rows([
        
        ['bleu', bleu_score],
        ['bleu_4', bleu_4],
        ['sacrebleu 3 expl',sacrebleu3],
        ['sacrebleu 2 expl', sacrebleu2],
        ['avg length',avg_length],
        ])
    print(table)
    
    return

def evaluate_result(eval_write,prompt_dev_file):
    table = PrettyTable()
    table.field_names = ["Metric", "Score"]

    texts,text_notsplit = opentxtFile(eval_write)
    eval_file = prompt_dev_file
    for_bleu, for_sacrebleu = evaluationLoader(eval_file)
    for_sacrebleu_3expl,for_sacrebleu_2expl = for_sacrebleu

    l = 0
    for i in texts: l+=len(i)
    avg_length = round(l/len(texts),2)
    bleu_score = corpus_bleu(for_bleu,texts)
    bleu_4 = corpus_bleu(for_bleu,texts,weights=(0,0,0,1))
    
    bleu_score = round(bleu_score*100,2)
    bleu_4 = round(bleu_4*100,2)
    

    sacrebleu3 = sacrebleu(for_sacrebleu_3expl,text_notsplit)
    sacrebleu2 = sacrebleu(for_sacrebleu_2expl,text_notsplit)
    table.add_rows([
     
        ['bleu', bleu_score],
        ['bleu_4', bleu_4],
        ['sacrebleu 3 expl',sacrebleu3],
        ['sacrebleu 2 expl', sacrebleu2],
        ['avg length',avg_length],
        ])
    print(table)
 
    return bleu_score



    
