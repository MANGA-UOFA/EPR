'''
default
'''
import os
import sys
import numpy as np
from datetime import datetime
from prettytable import PrettyTable
import random
'''
network
'''
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5Tokenizer


from tqdm import tqdm
import model.mnet as mnet
import argparse


from transformers.optimization import (
                                    AdamW,
                                    get_linear_schedule_with_warmup
                                    )

from utils import ( init_device,
                    create_dir, 
                    write_to_file, 
                    save_model,
                    MemoryNet_Initializer
                    )
'''
dataloader 
'''
import src.dataloader as dl
'''
evaluation
'''
from metrics.metrics import *
from infer import *




class Trainer():
    def __init__(self,args):
        '''settings'''
        self.args = args
        self.data_path = args.train_ckpt
        self.checkpoint = args.save_location
        self.ckpt_path = f'./checkpoint/{self.checkpoint}/'
        self.store_result = f'result/{self.checkpoint}/'

        '''datafiles'''
        self.memory_file = f'data/{self.data_path}/memory.pickle'
        self.prompt_train_file = f'data/{self.data_path}/esnli_.csv' 
        self.prompt_dev_file = f'data/{self.data_path}/esnli_dev.csv'
        self.prompt_test_file = f'data/{self.data_path}/esnli_test.csv'

        
        '''configurations'''
        self.seed = random.randint(0,1000) if args.seed is None else args.seed
        self.device = init_device()
        self.BATCH = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.epoch = args.epoch

        self.lr_t5 = 3e-4 if args.learning_rate is None else args.learning_rate

        self.evaluation_batch = args.eval_batch_size
 
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

        self.c_training = args.c_train
        self.dev_prompts, self.dev_explanations, self.dev_labels = self.loadevalPromptData(self.prompt_dev_file,self.label_dev_file)
        
        self.test_prompts, self.test_explanations, self.test_labels = self.loadevalPromptData(self.prompt_test_file,self.label_test_file)
        self.epoch_count = int(args.saved_ckpt) if args.saved_ckpt is not None else 0
        self.saved_ckpt = args.saved_ckpt
        self.test_result_output = f'{self.store_result}test_result_{self.checkpoint}{self.saved_ckpt}.txt'
        
        self.cycle=False
  


    def ltokenizer(self,sentence,explanation):
        if sentence is not None:
            tok = self.tokenizer(sentence,padding='longest',
                        max_length=512, 
                        truncation=True, 
                        return_tensors="pt")
            input_ids,attention_mask = tok.input_ids.to(self.device),tok.attention_mask.to(self.device)
        else:
           input_ids,attention_mask = None, None

        if explanation!=None:
            labels = self.tokenizer(explanation,padding='longest',
                            max_length=128,
                            truncation=True,
                            return_tensors="pt").input_ids
            labels[labels == self.tokenizer.pad_token_id] = -100

            labels = labels.to(self.device)
        else: 
            labels=None
        
        return input_ids,attention_mask,labels

    def train(self,dataset,devdata=None,memory=None):
        # pretty print
        
        '''local values'''
        name = 'transformers'
        epochs = self.epoch
        
        batch_size = self.BATCH

        initial_bleu = -1
        warmup_epoch = 2
        freeze_T5_first = True
        '''batch size and optimizer initialization'''
        
        prompts=dataset['prompts']
        explanations=dataset['explanations']

        total_length    = len(prompts)
        total_batch     = total_length//batch_size + 1

        
        freeze_dict = {'freeze_memnet':False,
                        'freeze_T5':False,
                        }
        if freeze_T5_first: 
            freeze_dict['freeze_T5']=True
        else: 
            freeze_dict['freeze_memnet']=True
            
        model = mnet.mT5Encoder(memory).to(self.device)
        if self.cycle:
            self.freeze_memnet_params(model,freeze_dict)
        print(f'Start Training...With seed {self.seed}\n')

        # info table
        table = PrettyTable()
        table.field_names = ["Data", "Length"]
        table.add_rows([
        ['Prompts', len(dataset['prompts']) ],
        ['Explanations', len(dataset['explanations']) ],
        ['Trainable Parameters',sum(p.numel() for p in model.parameters() if p.requires_grad)]
        ])
        print(table)
        
        
       
        
        optimizer = self.get_optimizer(model)
        
        scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=total_batch*warmup_epoch,
                    num_training_steps=total_batch*self.epoch
                    )


        '''continue training'''
        if self.c_training == 1:
            print('continue training')
        
            ckpt = self.ckpt_path
            all_checkpoint = torch.load(f'{ckpt}memnet{self.saved_ckpt}.pt',map_location='cuda')
            self.epoch_count = int(self.saved_ckpt)
            model.load_state_dict(all_checkpoint['transformers'])

        
        with tqdm(range(epochs),unit= ' epoch',total=epochs,desc='Epoch iteration',mininterval=60) as epoch:
            for _ in epoch:    
                self.epoch_count+=1
                total_loss = 0
                

                '''set up dataset'''
                
                    
               
                with tqdm(range(total_batch),
                    unit=' samples',
                    total = total_batch,
                    leave=True,
                    mininterval=300,
                    desc='Sample Iteration') as tepoch:

                    for i in tepoch:
                        model.train()
                        optimizer.zero_grad()
                        

                        if batch_size*(i+1) > total_length:
                            p = prompts[batch_size*i:total_length]
                            e = explanations[batch_size*i:total_length]
                         
                        else:
                            p = prompts[batch_size*i:batch_size*(i+1)]
                            e = explanations[batch_size*i:batch_size*(i+1)]
                      
                       
                        step_size = i
                        input_ids,attention_mask,labels = self.ltokenizer(p,e)
                        output = model(input_ids,attention_mask,expl=labels,step_size=step_size,train=True)
                        loss = output.loss
                  
                        
                        total_loss += loss.item()
                        loss.backward()
                        optimizer.step()
                
                        
                        if (step_size+1) % (total_batch/20)<1:
                            tepoch.set_postfix(loss=loss.item(),lr=optimizer.param_groups[0]['lr'])
                        scheduler.step()

                # save model under name
                create_dir(self.ckpt_path)
                save_model(name,model,self.ckpt_path,self.epoch_count)
      
                if self.epoch_count>=(epochs-1) and devdata is not None:
                    bleu = self.evaluation(devdata)
                    if bleu>initial_bleu:
                        print(f'Saving current best epoch {self.epoch_count}...')
                        save_model(name,model,self.ckpt_path,str(int(bleu*100)))
                        initial_bleu = bleu

                

                table = PrettyTable()
                table.field_names = ["Total Loss"]
                table.add_rows([[total_loss]])
                print(table) 
                print(f'total loss: {total_loss}')

            
        return

    def evaluation(self,dataset):
        args = get_args()
        ckpt = self.ckpt_path
        mi = MemoryNet_Initializer()
        memory = mi.loadData('dev')
        model = mnet.mT5Encoder(memory).to(self.device)

        all_checkpoint = torch.load(f'{ckpt}memnet{self.epoch_count}.pt',map_location='cuda')
        model.load_state_dict(all_checkpoint['transformers'])

        model.eval()
        batch_size = self.eval_batch_size
        print('\nstart evaluation')
        list_of_outputs = []
        '''dataset'''
        prompts=dataset['prompts']


        sent_labels = dataset['labels']
        
        total_length = len(prompts)
        total_batch = total_length//batch_size + 1
        
        with torch.no_grad():
            with tqdm(range(total_batch), unit=' samples',total = total_batch,mininterval=120,) as tepoch:
                for i in tepoch:
                    
                    step_size = i

                    if batch_size*(i+1) > total_length:
                        p = prompts[batch_size*i:total_length]
                        l = sent_labels[batch_size*i:total_length]
                    else:
                        p = prompts[batch_size*i:batch_size*(i+1)]
                        l = sent_labels[batch_size*i:batch_size*(i+1)]
                    
                
                    input_ids,attention_mask, labels= self.ltokenizer(p,None)
                   
                    output = model(input_ids,attention_mask,labels,step_size=step_size,train=False)

                    list_of_outputs.extend(output)
        
       
        

        # write results
        create_dir(self.store_result)
        eval_write = f'{self.store_result}/eval_{str(self.epoch_count)}.txt'
        write_to_file(eval_write,list_of_outputs)
        
        # automatic evaluation
        
        bleu = evaluate_result(eval_write,self.prompt_dev_file)

       
        return bleu

    def test(self,dataset,memory=None):
        ckpt = self.ckpt_path
        model = mnet.mT5Encoder(memory).to(self.device)
        
        all_checkpoint = torch.load(f'{ckpt}memnet{self.saved_ckpt}.pt',map_location='cuda')


        model.load_state_dict(all_checkpoint['transformers'],strict=False)

       
        
        model.eval()
        batch_size = self.evaluation_batch
        print('\nstart testing')
        list_of_outputs = []
        '''dataset'''
        prompts=dataset['prompts']

        sent_labels = dataset['labels']
        total_length = len(prompts)
        total_batch = total_length//batch_size +1

       
        with torch.no_grad():
            with tqdm(range(total_batch), unit=' samples',total = total_batch,leave=True,mininterval=60) as tepoch:
                for i in tepoch:
                    
                    step_size = i
                    if batch_size*(i+1) > total_length:
                       
                        p = prompts[batch_size*i:total_length]
                        l = sent_labels[batch_size*i:total_length]
                    else:
                        p = prompts[batch_size*i:batch_size*(i+1)]
                        l = sent_labels[batch_size*i:batch_size*(i+1)]
                    input_ids,attention_mask,labels = self.ltokenizer(p,None)
                
                    labels=None
                    output = model(input_ids,attention_mask,labels,step_size=step_size,train=False)


                    list_of_outputs.extend(output)

        # write out
        create_dir(self.store_result)
        test_write = self.test_result_output
        write_to_file(test_write,list_of_outputs)

        # automatic evaluation
        evaluate_result(test_write,
                        self.prompt_test_file)

        return

    def loadPromptData(self,trainFile,label_file):
        prompts,explanations,labels = dl.trainLoader(trainFile,label_file)
        return prompts,explanations,labels

    def loadevalPromptData(self,trainFile,label_file):
        prompts,explanations,labels = dl.evalLoader(trainFile,label_file)
        
        return prompts,explanations,labels

    def get_optimizer(self,model):
        optimizer = AdamW(
        [{'params':filter(lambda p: p.requires_grad, 
                        model.parameters()),

        'lr':self.lr_t5,}])

        return optimizer

  
def count_parameters(model):
    trainble = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainble

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', type=int, default=None)
    # start
    parser.add_argument('-train', type=int, default=1)

    # checkpoint
    parser.add_argument('-train_ckpt',type=str,default='new_emb')
    parser.add_argument('-saved_ckpt',type=str,default=None)
    parser.add_argument('-save_location',type=str,default='current')

    # batch size
    parser.add_argument('-train_batch_size',type=int,default=32)
    parser.add_argument('-eval_batch_size',type=int,default=256)
    parser.add_argument('-epoch',type=int,default=3)
    # continue training
    parser.add_argument('-c_train',type=bool,default=False)
    #sweep
    parser.add_argument('-learning_rate',type=float,default=None)

    args = parser.parse_args()
    return args

# set file prefix


def log_date():
    # datetime object containing current date and time
    now = datetime.now()
    print(f'Start training from {now}')
    return 

def continue_training(args):
    if args.saved_ckpt !=None:
        return f'Continue training on saved_ckpt {args.saved_ckpt}'
    else:
        return f'Please set continuing epoch with -saved_ckpt'

def log_training_details(args):
    table = PrettyTable()
    table.field_names = [
        "Description", "Setting"
    ]

    table.add_rows(
    [
        ["Dataset from", args.train_ckpt],
        ["Training batch size", args.train_batch_size ],
        ["Evaluation batch size", args.eval_batch_size ],
        ["Epochs", args.epoch ],
    ]
    )
    print(table)



def main():
    args = get_args()
    set_seed(args.seed)
    if args.c_train==True:
        continue_training(args)
    else:
        print('Training from scratch')

    train = args.train    
    trainer  = Trainer(args)
    


    print('|--------Initializing Memory Net--------|\n')
   
    
    prompts,explanations,labels = trainer.loadPromptData(trainer.prompt_train_file,trainer.label_train_file)
    
    train_data = dl.setUpTrainDataset(prompts,explanations,labels)
   
    dev_data = dl.setUpDataset(trainer.dev_prompts,trainer.dev_explanations,trainer.dev_labels)
    test_data = dl.setUpDataset(trainer.test_prompts, trainer.test_explanations,trainer.test_labels)

    print('|--------Initialization Complete--------|\n')
    
    if train == 1:
        log_date()
        log_training_details(args)

        mi = MemoryNet_Initializer()
        memory = mi.loadData('train')
        trainer.train(train_data,dev_data,memory=memory)
    
    elif train==0:
        print(f'|--------Testing Memory Net Checkpoint {args.saved_ckpt}--------|\n')
        
        mi = MemoryNet_Initializer()

        memory = mi.loadData('test')
        trainer.test(test_data,memory=memory)
        



if  __name__== '__main__':
    main()
