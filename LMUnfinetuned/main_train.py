import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from model_batch_version import InduceFunction, InteractModel
from data_loader import DataLoader, insert_embedding_to_batch
import pickle
from time import time
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument("--model_cache_path", type=str, default="../save_model/snli/")
    parser.add_argument("--encoding_cache_path", type=str, default='../save_encoding/snli/alignment/')
    parser.add_argument("--model_name", type=str, default='epr_model_unfinetuned')
    parser.add_argument("--c_ratio", type=float, default=1)
    parser.add_argument("--cat", action="store_true")
    args = parser.parse_args()
    return args

def save_checkpoint(save_path, interactModel, learnableToken, optimizer, epoch, acc):
    torch.save({
        'interactModel_state_dict': interactModel.state_dict(),
        'learnableToken_state_dict': learnableToken.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'acc': acc
    }, save_path)

def load_checkpoint(load_path, interactModel, learnableToken, optimizer):
    checkpoint = torch.load(load_path)
    interactModel.load_state_dict(checkpoint['interactModel_state_dict'])
    learnableToken.load_state_dict(checkpoint['learnableToken_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    acc = checkpoint['acc']
    return interactModel, learnableToken, optimizer, epoch, acc

def do_epoch(dataloader, learnableToken, sp_indice, interactModel, induceFunction, optimizer):
    if dataloader.phase == 'train':
        interactModel.train()
        learnableToken.train()
    else:
        interactModel.eval()
        learnableToken.eval()

    total_loss = 0
    count = 0
    for i in tqdm(dataloader.get_batch(), total=int(len(dataloader) / dataloader.batch_size)):
        batch_p = i[0].to(device)
        batch_h = i[1].to(device)
        mask = i[2].to(device)
        mask_p = i[3].to(device)
        mask_h = i[4].to(device)
        label = i[5].to(device)

        batch_p, batch_h = insert_embedding_to_batch(learnableToken, sp_indice, batch_p, batch_h, mask_p, mask_h)
        x = interactModel(batch_p, batch_h)
        x = x * mask.unsqueeze(-1).expand_as(x) #mask the pedding output
        pred = induceFunction.induce_to_sentence(x, mask, mask_p, mask_h)
        # pred = induceFunction.mean_induce_to_sentence(x, mask)

        batch_loss = loss_fn(torch.log(pred), label)
        loss = batch_loss.mean()

        if dataloader.phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * label.shape[0]
        count += torch.sum(torch.argmax(pred, dim=-1) == label).item()
    
    return total_loss / len(dataloader), count / len(dataloader)


if __name__ == '__main__':
    args = get_args()
    print(args)

    if args.cat:
        input_dim = 768 * 2
    else:
        input_dim = 768
    epoch = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    cont_train = args.load_checkpoint
    encoding_cache_path = args.encoding_cache_path
    save_path = args.model_cache_path + args.model_name + '.pt'

    t0 = time()
    with open(encoding_cache_path + 'train_alignment.pkl', 'rb') as f:
        train_data = pickle.load(f)
    t1 = time()
    print('Train loading time : %.2f' % (t1-t0))
    with open(encoding_cache_path + 'dev_alignment.pkl', 'rb') as f:
        dev_data = pickle.load(f)
    t2 = time()
    print('Dev loading time: %.2f' % (t2-t1))
    with open(encoding_cache_path + 'test_alignment.pkl', 'rb') as f:
        test_data = pickle.load(f)
    print('Test loading time: %.2f' % (time()-t2))

    train_dataloader = DataLoader(train_data, batch_size, phase='train', c_ratio=args.c_ratio, concat=args.cat)
    dev_dataloader = DataLoader(dev_data, batch_size, phase='dev', c_ratio=args.c_ratio, concat=args.cat)
    test_dataloader = DataLoader(test_data, batch_size, phase='test', c_ratio=args.c_ratio, concat=args.cat)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    interactModel = InteractModel(input_dim=input_dim).to(device)
    induceFunction = InduceFunction()
    learnableToken = torch.nn.Embedding(2, input_dim).to(device)
    sp_indice = [torch.tensor(0).to(device), torch.tensor(1).to(device)]

    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(list(interactModel.parameters())+ list(learnableToken.parameters()), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    if cont_train:
        interactModel, learnableToken, optimizer, start_epoch, track_acc = load_checkpoint(save_path, interactModel, learnableToken, optimizer)
        print('checkpoints loaded')
    else:
        start_epoch = 0
        track_acc = 0

    for e in range(start_epoch, start_epoch + epoch):
        for param_group in optimizer.param_groups:
            print('lr:', param_group['lr'])

        loss, train_acc = do_epoch(train_dataloader, learnableToken, sp_indice, interactModel, induceFunction, optimizer)
        # print(e, loss, train_acc)
        
        with torch.no_grad():
            _, dev_acc = do_epoch(dev_dataloader, learnableToken, sp_indice, interactModel, induceFunction, optimizer)
            _, test_acc = do_epoch(test_dataloader, learnableToken, sp_indice, interactModel, induceFunction, optimizer)
            # print('test:',test_acc)

        print("epoch: %s, train loss: %.4f, train acc: %.4f, dev acc: %.4f, test acc: %.4f" % (e, loss, train_acc, dev_acc, test_acc))
        scheduler.step()

        if dev_acc > track_acc:
            print('saving models to '+ save_path)
            track_acc = dev_acc
            save_checkpoint(save_path, interactModel, learnableToken, optimizer, e, track_acc)

            

        

        

    

