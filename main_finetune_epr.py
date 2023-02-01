import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
from transformers import get_linear_schedule_with_warmup
import random
import json
from data_util import *
from model import *

def save_checkpoint(save_path, sbert, interactModel, learnableToken, optimizer, scheduler, epoch, acc):
    if mode == 0 or mode == 1:
        torch.save({
            'sbert_state_dict': sbert.state_dict(),
            'interactModel_state_dict': interactModel.state_dict(),
            'learnableToken_state_dict': learnableToken.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch+1,
            'acc': acc
        }, save_path)
    else:
        torch.save({
            'sberti_state_dict': sbert[0].state_dict(),
            'sbertc_state_dict': sbert[1].state_dict(),
            'interactModel_state_dict': interactModel.state_dict(),
            'learnableToken_state_dict': learnableToken.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch+1,
            'acc': acc
        }, save_path)

def load_checkpoint(load_path, sbert, interactModel, learnableToken, optimizer=None, scheduler=None, is_test=False):
    checkpoint = torch.load(load_path)
    if mode == 0 or mode == 1:
        sbert.load_state_dict(checkpoint['sbert_state_dict'])
    else:
        sbert[0].load_state_dict(checkpoint['sberti_state_dict'])
        sbert[1].load_state_dict(checkpoint['sbertc_state_dict'])
    interactModel.load_state_dict(checkpoint['interactModel_state_dict'])
    learnableToken.load_state_dict(checkpoint['learnableToken_state_dict'])
    if not is_test:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    return sbert, interactModel, learnableToken, optimizer, scheduler, epoch, acc

def compute(s_token, c_mask, model, device):
    input_ids = torch.tensor(s_token['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(s_token['attention_mask']).unsqueeze(0).to(device)
    c_mask = torch.tensor(c_mask).to(device)
    emc = model(input_ids, attention_mask, c_mask)
    return emc

def insert_embedding_to_batch(learnableToken, sp_indice, batch_p, batch_h, mask_p, mask_h):
    l_p = learnableToken(sp_indice[0])
    mask_p_convert = mask_p.unsqueeze(-1).expand_as(batch_p)

    l_h = learnableToken(sp_indice[1])
    mask_h_convert = mask_h.unsqueeze(-1).expand_as(batch_h)

    return mask_p_convert * l_p + batch_p, mask_h_convert * l_h + batch_h

def get_data(token_cache_path, alignment_cache_path, phase):
    with open(token_cache_path + phase + '_tokens.pkl', 'rb') as f:
        token_data = pickle.load(f)

    with open(alignment_cache_path + phase + '_alignment.pkl', 'rb') as f:
        alignment_data = pickle.load(f)

    return token_data, alignment_data

def aggregate_mean_unalign(embeddings1, embeddings2, pos, indice_c1, indice_c2, device, max_length=64):
    length1 = len(embeddings1)
    length2 = len(embeddings2)
    has_c1 = 0
    if len(indice_c1) > 0:
        has_c1 = 1
    has_c2 = 0
    if len(indice_c2) > 0:
        has_c2 = 1
    unmask_length = len(pos) + has_c1 + has_c2

    if mode == 0 or mode == 1:
        tensors_p = torch.zeros([max_length, 768]).to(device)
        tensors_h = torch.zeros([max_length, 768]).to(device)
    else:
        tensors_p = torch.zeros([max_length, 768*2]).to(device)
        tensors_h = torch.zeros([max_length, 768*2]).to(device)
    p_token_mask = torch.zeros([max_length]).to(device)
    h_token_mask = torch.zeros([max_length]).to(device)
    mask = torch.zeros([max_length]).to(device)
    mask[:unmask_length] = 1

    i_p = 0
    i_h = 0
    for item in pos:
        helper_tensor_p = torch.zeros([1, length1]).to(device)
        helper_tensor_h = torch.zeros([1, length2]).to(device)
        helper_tensor_p[:, item[0]] = 1
        helper_tensor_h[:, item[1]] = 1
        # print(helper_tensor_p.shape, embeddings1.shape, torch.mm(helper_tensor_p, embeddings1).shape)
        tensors_p[i_p, :] = torch.mm(helper_tensor_p, embeddings1)
        tensors_h[i_h, :] = torch.mm(helper_tensor_h, embeddings2)
        i_p += 1
        i_h += 1
    
    if len(indice_c1) != 0:
        mean_unaligned_p = []
        for i in indice_c1:
            helper_tensor_p = torch.zeros([1, length1]).to(device)
            helper_tensor_p[:, i] = 1
            mean_unaligned_p.append(torch.mm(helper_tensor_p, embeddings1))
        mean_unaligned_p = torch.stack(mean_unaligned_p).mean(dim=0)
        tensors_p[i_p, :] = mean_unaligned_p
        h_token_mask[i_h] = 1
        i_p += 1
        i_h += 1

    if len(indice_c2) != 0:
        mean_unaligned_h = []
        for i in indice_c2:
            helper_tensor_h = torch.zeros([1, length2]).to(device)
            helper_tensor_h[:, i] = 1
            mean_unaligned_h.append(torch.mm(helper_tensor_h, embeddings2))
        mean_unaligned_h = torch.stack(mean_unaligned_h).mean(dim=0)
        tensors_h[i_h, :] = mean_unaligned_h
        p_token_mask[i_p] = 1

    return tensors_p, tensors_h, mask, p_token_mask, h_token_mask

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--model_cache_path", type=str, default="./save_model/snli/")
    parser.add_argument("--token_cache_path", type=str, default='./save_encoding/snli/token/')
    parser.add_argument("--alignment_cache_path", type=str, default='./save_encoding/snli/alignment/')
    parser.add_argument("--model_name", type=str, default='local_model')
    parser.add_argument("--is_train", action="store_true")
    # mode=0: local 
    # mode=1: global
    # else: concat
    parser.add_argument("--mode", type=int, default=0) 
    args = parser.parse_args()
    return args

def do_epoch(dataloader, sbert, learnableToken, sp_indice, interactModel, induceFunction, loss_fn, optimizer, scheduler, batch_size, device):
    print(dataloader.phase)
    if dataloader.phase == 'train':
        if mode == 0 or mode == 1:
            sbert.train()
        else:
            sbert[0].train()
            sbert[1].train()
        learnableToken.train()
        interactModel.train()
    else:
        if mode == 0 or mode == 1:
            sbert.eval()
        else:
            sbert[0].eval()
            sbert[1].eval()
        learnableToken.eval()
        interactModel.eval()

    batch_loss = 0
    total_loss = 0
    acc_count = 0
    count = np.zeros([2], dtype=np.int64)

    pbar = tqdm(dataloader.get_datapoint(), total=len(dataloader))
    for item in pbar:
        if mode == 1:
            s1_token = item['s1_token']
            c1_mask = item['c1_mask']
            s2_token = item['s2_token']
            c2_mask = item['c2_mask']

            em1 = compute(s1_token, c1_mask, sbert, device)
            em2 = compute(s2_token, c2_mask, sbert, device)

        elif mode == 0:
            c1_token = item['c1_token']
            c2_token = item['c2_token']

            em1 = sbert(torch.tensor(c1_token['input_ids']).to(device), torch.tensor(c1_token['attention_mask']).to(device))
            em2 = sbert(torch.tensor(c2_token['input_ids']).to(device), torch.tensor(c2_token['attention_mask']).to(device))
        
        else:
            s1_token = item['s1_token']
            c1_mask = item['c1_mask']
            s2_token = item['s2_token']
            c2_mask = item['c2_mask']

            cem1 = compute(s1_token, c1_mask, sbert[1], device)
            cem2 = compute(s2_token, c2_mask, sbert[1], device)

            c1_token = item['c1_token']
            c2_token = item['c2_token']

            em1 = sbert[0](torch.tensor(c1_token['input_ids']).to(device), torch.tensor(c1_token['attention_mask']).to(device))
            em2 = sbert[0](torch.tensor(c2_token['input_ids']).to(device), torch.tensor(c2_token['attention_mask']).to(device))

            em1 = torch.cat((em1, cem1), -1)
            em2 = torch.cat((em2, cem2), -1)

        indice_c1 = item['p_not_aligned']
        indice_c2 = item['h_not_aligned']
        pos = item['p_h_aligned']

        tensors_p, tensors_h, mask, p_mask, h_mask = aggregate_mean_unalign(em1, em2, pos, indice_c1, indice_c2, device)
        out1, out2 = insert_embedding_to_batch(learnableToken, sp_indice, tensors_p, tensors_h, p_mask, h_mask)
        x = interactModel(out1, out2)
        x = x * mask.unsqueeze(-1).expand_as(x)
        pred = induceFunction.induce_to_sentence(x, mask, p_mask, h_mask).unsqueeze(0)
        # pred = induceFunction.mean_induce_to_sentence(x, mask).unsqueeze(0)
        label = torch.LongTensor([item['label']]).to(device)
        
        loss = loss_fn(torch.log(pred), label)
        batch_loss += loss
        total_loss += loss.item()

        acc_count += torch.sum(torch.argmax(pred, dim=-1) == label).item()
        count += 1
        
        if dataloader.phase == 'train':
            if count[0] % batch_size == 0 or count[0] >= len(dataloader):
                batch_loss /= count[1]
                batch_loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_description('train_loss: %.4f, train_acc: %.4f' % (batch_loss.item(), acc_count/count[0]))
                batch_loss = 0
                count[1] = 0    
    
    return total_loss / len(dataloader), acc_count / len(dataloader)

def do_explain(dataloader, sbert, learnableToken, sp_indice, interactModel, induceFunction, device):
    if mode == 0 or mode == 1:
        sbert.eval()
    else:
        sbert[0].eval()
        sbert[1].eval()
    learnableToken.eval()
    interactModel.eval()

    acc_count = 0
    ids = random.sample(range(0, 3200), 100)
    # SNLI
    # ids = [2947, 880, 475, 9380, 5037, 270, 5983, 3137, 4172, 3467, 6517, 2153, 7128, 8251, 3855, 4074, 4642, 4516, 6003, 4730, 7145, 7533, 2831, 7710, 9182, 2015, 2333, 7647, 9669, 9112, 9526, 2516, 3635, 2317, 1857, 4915, 4771, 6712, 2251, 4414, 4643, 3160, 5526, 6570, 4792, 6331, 6179, 9479, 4702, 8661, 6756, 5278, 6572, 8513, 3749, 3998, 9492, 2858, 8360, 6277, 6987, 4899, 6932, 2189, 1315, 2920, 322, 132, 2365, 3608, 451, 4538, 9490, 2649, 3351, 2040, 990, 5916, 2663, 120, 613, 8342, 4249, 945, 9126, 4039, 1252, 9640, 5810, 1860, 6264, 2170, 8389, 7183, 3490, 7608, 5837, 533, 6167, 3438]
    # MNLI-m
    # ids = [4910, 6081, 5755, 1352, 452, 8289, 6470, 479, 1539, 3119]
    # ids = ids + [8082, 5585, 6840, 9022, 7609, 5093, 3965, 8699, 5939, 6765, 4533, 5116, 5569, 6902, 8692, 2781, 8528, 7580, 279, 3562, 7726, 2464, 292, 9469, 7831, 1889, 4267, 9444, 6314, 8091, 4160, 8718, 1196, 3904, 4255, 5126, 9274, 573, 266, 1393, 8971, 5624, 7669, 848, 344, 8157, 7083, 4182, 6065, 5516]
    # MNLI-mm
    ids = [5488, 5631, 3393, 2434, 253, 7215, 4480, 2443, 8147, 3339]
    ids = ids + [1487, 6379, 277, 3589, 4216, 3459, 9492, 3154, 879, 3746, 7859, 1176, 4868, 6876, 3463, 9433, 2629, 7165, 1802, 5079, 1312, 4604, 7822, 5172, 7372, 6624, 9304, 7912, 6004, 3131, 9829, 4421, 1926, 9519, 8461, 6599, 5838, 6162, 8278, 1582, 4171, 7431, 5714, 6754, 8743, 5998, 9122, 4083, 5754, 7651]
    with open('./text_file/mnli_model_output/mismatched_test.jsonl', 'w') as of:
        with torch.no_grad():
            count = 0
            for item in tqdm(dataloader.get_datapoint(), total=len(dataloader)):
                if mode == 1:
                    s1_token = item['s1_token']
                    c1_mask = item['c1_mask']
                    s2_token = item['s2_token']
                    c2_mask = item['c2_mask']

                    em1 = compute(s1_token, c1_mask, sbert, device)
                    em2 = compute(s2_token, c2_mask, sbert, device)

                elif mode == 0:
                    c1_token = item['c1_token']
                    c2_token = item['c2_token']

                    em1 = sbert(torch.tensor(c1_token['input_ids']).to(device), torch.tensor(c1_token['attention_mask']).to(device))
                    em2 = sbert(torch.tensor(c2_token['input_ids']).to(device), torch.tensor(c2_token['attention_mask']).to(device))
                
                else:
                    s1_token = item['s1_token']
                    c1_mask = item['c1_mask']
                    s2_token = item['s2_token']
                    c2_mask = item['c2_mask']

                    cem1 = compute(s1_token, c1_mask, sbert[1], device)
                    cem2 = compute(s2_token, c2_mask, sbert[1], device)

                    c1_token = item['c1_token']
                    c2_token = item['c2_token']

                    em1 = sbert[0](torch.tensor(c1_token['input_ids']).to(device), torch.tensor(c1_token['attention_mask']).to(device))
                    em2 = sbert[0](torch.tensor(c2_token['input_ids']).to(device), torch.tensor(c2_token['attention_mask']).to(device))

                    em1 = torch.cat((em1, cem1), -1)
                    em2 = torch.cat((em2, cem2), -1)
                
                indice_c1 = item['p_not_aligned']
                indice_c2 = item['h_not_aligned']
                pos = item['p_h_aligned']

                # ignore alignment issue and neutral
                # if len(indice_c1) != 0 or len(indice_c2) != 0 or item['label'] == 2:
                #     continue

                tensors_p, tensors_h, mask, p_mask, h_mask = aggregate_mean_unalign(em1, em2, pos, indice_c1, indice_c2, device)
                out1, out2 = insert_embedding_to_batch(learnableToken, sp_indice, tensors_p, tensors_h, p_mask, h_mask)
                x = interactModel(out1, out2)
                x = x * mask.unsqueeze(-1).expand_as(x)
                pred = induceFunction.induce_to_sentence(x, mask, p_mask, h_mask).unsqueeze(0)
                # pred = induceFunction.mean_induce_to_sentence(x, mask).unsqueeze(0)
                label = torch.LongTensor([item['label']]).to(device)
                acc_count += torch.sum(torch.argmax(pred, dim=-1) == label).item()
                count += 1

                if item['id'] in ids:
                    this_json = {}
                    this_json['snli_id'] = str(item['id'])
                    aligned_text = item['aligned_text']
                    aligned = aligned_text[0]
                    unaligned_p = aligned_text[1]
                    unaligned_h = aligned_text[2]
                    length = len(aligned) + len(unaligned_p) + len(unaligned_h)

                    phrase_out = torch.argmax(x, dim=-1).cpu().numpy()
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
                    #     EP.append(a[0])
                    #     EH.append(a[1])
                    # for ua in unaligned_p:
                    #     EP.append(ua)
                    # for ua in unaligned_h:
                    #     EH.append(ua)
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

                    this_json['sent_pred'] = str(torch.argmax(pred, dim=-1)[0].item())
                    this_json['sent_label'] = str(label[0].item())
                    this_json['EP'] = '\u2022'.join(EP)
                    this_json['CP'] = '\u2022'.join(CP)
                    this_json['NP'] = '\u2022'.join(NP)
                    this_json['EH'] = '\u2022'.join(EH)
                    this_json['CH'] = '\u2022'.join(CH)
                    this_json['NH'] = '\u2022'.join(NH)
                    this_json['UP'] = '\u2022'.join(UP)
                    this_json['UH'] = '\u2022'.join(UH)
                    of.write(json.dumps(this_json) + '\n')

    # print(acc_count / len(dataloader))
    print(acc_count / count, count)


if __name__ == '__main__':
    args = get_args()
    print(args)

    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    mode = args.mode
    if mode == 0 or mode == 1:
        input_dim = 768
    else:
        input_dim = 768 * 2
    cont_train = args.load_checkpoint
    token_cache_path = args.token_cache_path
    alignment_cache_path = args.alignment_cache_path
    save_path = args.model_cache_path + args.model_name + '.pt'
    model_str = './backbone'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if mode == 0:
        sbert = IndependentSBert(model_str).to(device)
    elif mode == 1:
        sbert = ContextualSBert(model_str).to(device)
    else:
        sbert = [IndependentSBert(model_str).to(device), ContextualSBert(model_str).to(device)]
    
    interactModel = InteractModel(input_dim).to(device)
    learnableToken = torch.nn.Embedding(2, input_dim).to(device)
    sp_indice = [torch.tensor(0).to(device), torch.tensor(1).to(device)]
    induceFunction = InduceFunction()

    loss_fn = nn.NLLLoss()
    if mode == 0 or mode == 1:
        optimizer = torch.optim.Adam(list(sbert.parameters())+list(interactModel.parameters())+list(learnableToken.parameters()), lr=lr)
    else:
        optimizer = torch.optim.Adam(list(sbert[0].parameters())+list(sbert[1].parameters())+list(interactModel.parameters())+list(learnableToken.parameters()), lr=lr)

    if args.is_train:
        train_token_data, train_alignment_data = get_data(token_cache_path, alignment_cache_path, 'train')
        train_dataloader = DataLoader(train_token_data, train_alignment_data, 'train')

        test_token_data, test_alignment_data = get_data(token_cache_path, alignment_cache_path, 'test')
        test_dataloader = DataLoader(test_token_data, test_alignment_data, 'test')

        # learning rate warmup scheduler
        num_training_steps = int(len(train_dataloader) / batch_size) * epoch
        num_warmup_steps = num_training_steps * 0.1
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        if args.load_checkpoint:
            sbert, interactModel, learnableToken, optimizer, scheduler, start_epoch, track_acc = load_checkpoint(save_path, sbert, interactModel, learnableToken, optimizer, scheduler)
            print('Checkpoint loaded')
            print('start epoch:',start_epoch, 'track acc:', track_acc)
        else:
            start_epoch = 0
            track_acc = 0

        for e in range(start_epoch, epoch):
            _, train_acc = do_epoch(train_dataloader, sbert, learnableToken, sp_indice, interactModel, induceFunction, loss_fn, optimizer, scheduler, batch_size, device)

            with torch.no_grad():
                test_loss, test_acc = do_epoch(test_dataloader, sbert, learnableToken, sp_indice, interactModel, induceFunction, loss_fn, optimizer, scheduler, batch_size, device)

            print("epoch: %d, train acc: %.4f, test loss: %.4f, test acc: %.4f" % (e, train_acc, test_loss, test_acc))

            if test_acc > track_acc:
                print('saving models to '+ save_path)
                track_acc = test_acc
                save_checkpoint(save_path, sbert, interactModel, learnableToken, optimizer, scheduler, e, track_acc)
    
    else:
        if args.load_checkpoint:
            sbert, interactModel, learnableToken, _, _, start_epoch, track_acc = load_checkpoint(save_path, sbert, interactModel, learnableToken, None, None, True)
            print('Checkpoint loaded')
            print('start epoch:',start_epoch, 'track acc:', track_acc)

            test_token_data, test_alignment_data = get_data(token_cache_path, alignment_cache_path, 'test')
            explain_dataloader = DataLoader(test_token_data, test_alignment_data, 'explain')
            do_explain(explain_dataloader, sbert, learnableToken, sp_indice, interactModel, induceFunction, device)