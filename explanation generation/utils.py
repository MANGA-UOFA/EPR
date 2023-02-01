import os
import src.dataloader as dl
import torch

def create_dir(folder):
    if not os.path.exists(folder):
            os.mkdir(folder)
    return

def write_to_file(file,result_list):
    with open(file,'w',encoding='utf-8') as f:
            for outs in result_list:
                f.write(outs)
                f.write('\n')
    return

def save_model(name,model,path,postfix):

    torch.save(
    {name:model.state_dict(),
    },
    f'{path}memnet{postfix}.pt')
    return

def init_device():
    if torch.cuda.is_available():
            device='cuda'
    else:   device='cpu'
    
    return torch.device(device)


class MemoryNet_Initializer():
    def __init__(self):
        super(MemoryNet_Initializer,self).__init__()
        self.memory_file        = 'data/memory.pickle'
        self.dev_memory_file    = 'data/esnli_dev.pickle'
        self.test_memory_file   = 'data/esnli_test.pickle'
    
    def loadData(self,mode):
        print('Loading memory...')
        if mode == 'train':
            memory = dl.nliMemoryLoader(self.memory_file)
        elif mode == 'dev':
            memory = dl.nliMemoryLoader(self.dev_memory_file)
        elif mode == 'test':
            memory = dl.nliMemoryLoader(self.test_memory_file)
        print('Finished loading memory...')
        return memory