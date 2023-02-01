import json
import string
from data_process import read_data
import numpy as np
import re
import math

data_path = "./snli_1.0/"
ids, test_data, test_labels, _ = read_data(data_path + "snli_1.0_test.jsonl")
# data_path = "./multinli_1.0/"
# ids, test_data, test_labels, _ = read_data(data_path + "multinli_1.0_dev_matched.jsonl")
mode = 2 # 0: multiply, 1: Arithmetic Mean, 2: Geometric Mean

# True Positive
def truepos(src, trg):
    same = set(src).intersection(set(trg))
    return len(same)

# False Positive
def falsepos(src, trg):
    different = set(src).difference(set(trg)) 
    return len(different)

# False Negative
def falseneg(src, trg):
    different = set(trg).difference(set(src)) 
    return len(different)

def substr2index(sentence, substr):
    indice = []
    phrases = substr.split('\u2022')
    for phrase in phrases:
        phrase = re.findall(r"\b\S+\b", phrase.lower())
        length = len(phrase)
        for i in range(0, len(sentence)-length+1):
            if sentence[i:i+length] == phrase:
                indice += list(range(i, i+length))
    return indice

def one2one(gt_path, src_path):
    with open(gt_path, 'r') as tf:
        lines_trg = tf.readlines()
    with open(src_path, 'r') as sf:
        lines_src = sf.readlines()

    eva_holder = {}
    for line in lines_trg:
        trg = json.loads(line)
        if len(trg['snli_id'].split('_')) == 1:
            this_id_trg = int(trg['snli_id'])
        else:
            this_id_trg = int(trg['snli_id'].split('_')[1])
        eva_holder[this_id_trg] = {'trg': trg}

    for line in lines_src:
        src = json.loads(line)
        if len(src['snli_id'].split('_')) == 1:
            this_id_src = int(src['snli_id'])
        else:
            this_id_src = int(src['snli_id'].split('_')[1])
        try:
            tmp = eva_holder[this_id_src]
            tmp['src'] = src
            eva_holder[this_id_src] = tmp
        except:
            print(this_id_src, 'excluded.')
    
    # row: tp, fp, fn
    # col: e, c, n, u
    p_result = np.zeros([4,3])
    h_result = np.zeros([4,3])
    pred_words_ECN = np.zeros([4])
    labels_words_ECN = np.zeros([4])

    for key, value in eva_holder.items():
        # print(key)
        trg = value['trg']
        src = value['src']
        this_sents = test_data[ids.index(key)]
        premise = re.findall(r"\b\S+\b", this_sents[0].lower())
        hypothesis = re.findall(r"\b\S+\b", this_sents[1].lower())

        for i, c in enumerate(['E', 'C', 'N', 'U']):
            tp = substr2index(premise, trg[c+'P'])
            sp = substr2index(premise, src[c+'P'])
            th = substr2index(hypothesis, trg[c+'H'])
            sh = substr2index(hypothesis, src[c+'H'])
            pred_words_ECN[i] += len(sp) + len(sh)
            labels_words_ECN[i] += len(tp) + len(th)

            p_result[i][0] += truepos(sp, tp)
            p_result[i][1] += falsepos(sp, tp)
            p_result[i][2] += falseneg(sp, tp)

            h_result[i][0] += truepos(sh, th)
            h_result[i][1] += falsepos(sh, th)
            h_result[i][2] += falseneg(sh, th)

    # print(p_result, h_result)

    pp = p_result[:, 0] / (p_result[:,0] + p_result[:,1])
    ph = h_result[:, 0] / (h_result[:,0] + h_result[:,1])
    rp = p_result[:, 0] / (p_result[:,0] + p_result[:,2])
    rh = h_result[:, 0] / (h_result[:,0] + h_result[:,2])

    if mode == 0:
        p_ecn = pp[:3] * ph[:3]
        r_ecn = rp[:3] * rh[:3]
    elif mode == 1:
        p_ecn = (pp[:3] + ph[:3]) / 2
        r_ecn = (rp[:3] + rh[:3]) / 2
    elif mode == 2:
        p_ecn = np.sqrt(pp[:3] * ph[:3])
        r_ecn = np.sqrt(rp[:3] * rh[:3])

    p = np.concatenate((p_ecn, np.array([pp[-1]]), np.array([ph[-1]])), axis=0)
    r = np.concatenate((r_ecn, np.array([rp[-1]]), np.array([rh[-1]])), axis=0)
    # print(p, r)
    # f = 2*p*r/(p+r)
    f = np.divide(2*p*r, p+r, out=np.zeros_like(2*p*r), where=p+r!=0)
    
    return f, labels_words_ECN, pred_words_ECN

def n2one(gt_paths, src_path):
    results = np.zeros([5])
    label_ECN = np.zeros([4])
    best = 0
    best_result = None
    i = 0
    for gt_path in gt_paths:
        this_result, labels_words_ECN, pred_words_ECN = one2one(gt_path, src_path)
        print(this_result)
        results += this_result
        label_ECN += labels_words_ECN
        i += 1
    print('system:', pred_words_ECN[:3] / pred_words_ECN[:3].sum())
    print('label:', labels_words_ECN[:3] / labels_words_ECN[:3].sum())
    return results / i

def human_n2n(gt_paths, src_paths):
    results = np.zeros([5])
    i = 0
    for gt_path in gt_paths:
        for src_path in src_paths:
            if gt_path == src_path:
                continue
            this_result, _, _ = one2one(gt_path, src_path)
            results += this_result
            i += 1
    return results / i

def human_ana(gt_paths):
    total_existance_count = np.zeros([5])
    total_premise_count = np.zeros([4])
    total_hypothesis_count = np.zeros([4])

    for gt_path in gt_paths:
        with open(gt_path, 'r') as tf:
            lines_trg = tf.readlines()
        
        existance_count = np.zeros([5])
        premise_count = np.zeros([4])
        hypothesis_count = np.zeros([4])

        for line in lines_trg:
            trg = json.loads(line)
            EP = trg['EP']
            EH = trg['EH']
            CP = trg['CP']
            CH = trg['CH']
            NP = trg['NP']
            NH = trg['NH']
            UP = trg['UP']
            UH = trg['UH']

            if len(EP) != 0:
                existance_count[0] += 1
            if len(CP) != 0:
                existance_count[1] += 1
            # if len(NP) != 0:
            #     existance_count[2] += 1
            if len(UP) != 0:
                existance_count[3] += 1
            if len(UH) != 0:
                existance_count[4] += 1

            premise_count[0] += len(EP.split())
            premise_count[1] += len(CP.split())
            # premise_count[2] += len(NP.split())
            premise_count[3] += len(UP.split())
            hypothesis_count[0] += len(EH.split())
            hypothesis_count[1] += len(CH.split())
            # hypothesis_count[2] += len(NH.split())
            hypothesis_count[3] += len(UH.split())

        total_existance_count += existance_count
        total_premise_count += premise_count
        total_hypothesis_count += hypothesis_count
    
    mean_total_existance_count = total_existance_count / 3
    mean_total_existance_count_norm = mean_total_existance_count / np.linalg.norm(mean_total_existance_count, ord=1)
    print('existance info:', mean_total_existance_count, mean_total_existance_count_norm)

    # geo_ph_count = np.sqrt(total_premise_count[:3] * total_hypothesis_count[:3])
    geo_ph_count = (total_premise_count[:3] + total_hypothesis_count[:3]) / 2
    total_ph_count = np.concatenate((geo_ph_count, np.array([total_premise_count[3]]), np.array([total_hypothesis_count[3]])), axis=0)
    total_ph_count_norm = total_ph_count / np.linalg.norm(total_ph_count, ord=1)
    print('count info:', total_ph_count / 3, total_ph_count_norm)


def get_AMF(matrix):
    return matrix.mean()

def get_GMF(matrix):
    matrix = np.log(matrix)
    return np.exp(matrix.mean())
    
if __name__ == '__main__':
    gt_paths = ['./text_file/snli_annotation/annotator1_snli.jsonl', './text_file/snli_annotation/annotator2_snli.jsonl', './text_file/snli_annotation/annotator3_snli.jsonl']
    src_path = './text_file/result.json'
    result = n2one(gt_paths, src_path)
    print(result)
    print(get_GMF(result))
    print(get_AMF(result))