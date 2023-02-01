import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model_batch_version import InduceFunction, InteractModel
from data_loader import DataLoader, insert_embedding_to_batch
import pickle
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cache_path", type=str, default="./save_model/snli/")
    parser.add_argument("--encoding_cache_path", type=str, default='./save_encoding/snli/alignment/')
    parser.add_argument("--model_name", type=str, default='Model_test')
    parser.add_argument("--c_ratio", type=float, default=1)
    parser.add_argument("--cat", action="store_true")
    args = parser.parse_args()
    return args

def load_checkpoint(load_path, interactModel, learnableToken):
    checkpoint = torch.load(load_path)
    interactModel.load_state_dict(checkpoint['interactModel_state_dict'])
    learnableToken.load_state_dict(checkpoint['learnableToken_state_dict'])
    return interactModel, learnableToken

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ids = [2947, 880, 475, 9380, 5037, 270, 5983, 3137, 4172, 3467, 6517, 2153, 7128, 8251, 3855, 4074, 4642, 4516, 6003, 4730, 7145, 7533, 2831, 7710, 9182, 2015, 2333, 7647, 9669, 9112, 9526, 2516, 3635, 2317, 1857, 4915, 4771, 6712, 2251, 4414, 4643, 3160, 5526, 6570, 4792, 6331, 6179, 9479, 4702, 8661, 6756, 5278, 6572, 8513, 3749, 3998, 9492, 2858, 8360, 6277, 6987, 4899, 6932, 2189, 1315, 2920, 322, 132, 2365, 3608, 451, 4538, 9490, 2649, 3351, 2040, 990, 5916, 2663, 120, 613, 8342, 4249, 945, 9126, 4039, 1252, 9640, 5810, 1860, 6264, 2170, 8389, 7183, 3490, 7608, 5837, 533, 6167, 3438]

def do_explain(dataloader, learnableToken, sp_indice, interactModel, induceFunction):
    interactModel.eval()
    learnableToken.eval()

    with open('./text_file/model_100.jsonl', 'w') as of:

        with torch.no_grad():

            label_ECN = np.zeros([3])
            pred_ECN = np.zeros([3])

            count = 0
            count1 = 0
            for i in tqdm(dataloader.get_batch(), total=int(len(dataloader) / dataloader.batch_size)):
                batch_p = i[0].to(device)
                batch_h = i[1].to(device)
                mask = i[2].to(device)
                mask_p = i[3].to(device)
                mask_h = i[4].to(device)
                label = i[5].to(device)
                aligned_text = i[6]
                index = i[7]

                batch_p, batch_h = insert_embedding_to_batch(learnableToken, sp_indice, batch_p, batch_h, mask_p, mask_h)
                x = interactModel(batch_p, batch_h)
                pred = induceFunction.induce_to_sentence(x, mask, mask_p, mask_h)
                # pred = induceFunction.mean_induce_to_sentence(x, mask)
                x = x * mask.unsqueeze(-1).expand_as(x)
                
                count += torch.sum(torch.argmax(pred, dim=-1) == label).item()
                sent_pred = torch.argmax(pred, dim=-1).cpu().numpy()[0]
                sent_label = label.cpu().numpy()[0]
                pred_ECN[sent_pred] += 1
                label_ECN[sent_label] += 1

                if index[0] not in ids:
                    continue
                
                count1 += torch.sum(torch.argmax(pred, dim=-1) == label).item()

                this_json = {}
                this_json['snli_id'] = str(index[0])
                aligned_text = aligned_text[0]
                aligned = aligned_text[0]
                unaligned_p = aligned_text[1]
                unaligned_h = aligned_text[2]
                length = len(aligned) + len(unaligned_p) + len(unaligned_h)

                phrase_out = torch.argmax(x, dim=-1).cpu().numpy()[0]
                # print(aligned_text)
                # print(phrase_out)
                # exit()
                # print(x)
                # print(label)
            
                i = 0
                EP = []
                CP = []
                NP = []
                EH = []
                CH = []
                NH = []
                UP = []
                UH = []
                for a in aligned:
                    if phrase_out[i] == 0:
                        EP.append(a[0])
                        EH.append(a[1])
                    elif phrase_out[i] == 1:
                        CP.append(a[0])
                        CH.append(a[1])
                    elif phrase_out[i] == 2:
                        NP.append(a[0])
                        NH.append(a[1])
                    i += 1

                for ua in unaligned_p:
                    UP.append(ua)
                
                for ua in unaligned_h:
                    UH.append(ua)

                this_json['sent_pred'] = str(sent_pred)
                this_json['sent_label'] = str(sent_label)
                this_json['EP'] = '\u2022'.join(EP)
                this_json['CP'] = '\u2022'.join(CP)
                this_json['NP'] = '\u2022'.join(NP)
                this_json['EH'] = '\u2022'.join(EH)
                this_json['CH'] = '\u2022'.join(CH)
                this_json['NH'] = '\u2022'.join(NH)
                this_json['UP'] = '\u2022'.join(UP)
                this_json['UH'] = '\u2022'.join(UH)

                of.write(json.dumps(this_json) + '\n')
            
            print(count / len(dataloader))
            print(count1/100)
            print(pred_ECN / pred_ECN.sum(), label_ECN / label_ECN.sum())
            

if __name__ == '__main__':
    args = get_args()
    print(args)

    if args.cat:
        input_dim = 768*2
    else:
        input_dim = 768
    encoding_cache_path = args.encoding_cache_path
    save_path = args.model_cache_path + args.model_name + '.pt'
    
    with open(encoding_cache_path + 'test_alignment.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # np.random.shuffle(test_data)
    test_dataloader = DataLoader(test_data, batch_size=1, phase='test', c_ratio=args.c_ratio, concat=args.cat)

    learnableToken = torch.nn.Embedding(2, input_dim).to(device)
    sp_indice = [torch.tensor(0).to(device), torch.tensor(1).to(device)]
    interactModel = InteractModel(input_dim=input_dim).to(device)
    induceFunction = InduceFunction()

    interactModel, learnableToken = load_checkpoint(save_path, interactModel, learnableToken)

    do_explain(test_dataloader, learnableToken, sp_indice, interactModel, induceFunction)
