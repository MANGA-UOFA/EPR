import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from model.t5_model import T5ForConditionalGeneration
from transformers import (T5Tokenizer,
                        T5Config,
                        set_seed
                        )

import numpy as np
import torch





class mT5Encoder(nn.Module):
    def __init__(
        self,
        memory=None,):

        
        super(mT5Encoder,self).__init__()
        self.memory = memory
        self.config = T5Config()
        print("init T5")
        self.model = T5ForConditionalGeneration.from_pretrained(
            "t5-small",
            memory=self.memory,
        )
        print("done loading memory")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.device = torch.device('cuda')

    
    def forward(
        self,
        input_ids,
        attention_mask,
        expl,
        step_size=None,
        train=True):

        model_kwargs = {"step_size": step_size,}
        if not train:
            encoded_output = self.model.generate(
                input_ids=input_ids,
                max_length=128,
                length_penalty=1.5,
                **model_kwargs,
            )
            outputs = self.tokenizer.batch_decode(encoded_output, skip_special_tokens=True)
            return outputs
            
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            labels=expl,
            step_size = step_size,
        )
         
      
        

        return outputs

        