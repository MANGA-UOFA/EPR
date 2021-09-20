import datasets
from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric
import pickle
import json
import torch

def generate_file():
    ids = [2947, 880, 475, 9380, 5037, 270, 5983, 3137, 4172, 3467, 6517, 2153, 7128, 8251, 3855, 4074, 4642, 4516, 6003, 4730, 7145, 7533, 2831, 7710, 9182, 2015, 2333, 7647, 9669, 9112, 9526, 2516, 3635, 2317, 1857, 4915, 4771, 6712, 2251, 4414, 4643, 3160, 5526, 6570, 4792, 6331, 6179, 9479, 4702, 8661, 6756, 5278, 6572, 8513, 3749, 3998, 9492, 2858, 8360, 6277, 6987, 4899, 6932, 2189, 1315, 2920, 322, 132, 2365, 3608, 451, 4538, 9490, 2649, 3351, 2040, 990, 5916, 2663, 120, 613, 8342, 4249, 945, 9126, 4039, 1252, 9640, 5810, 1860, 6264, 2170, 8389, 7183, 3490, 7608, 5837, 533, 6167, 3438]
    print(len(ids))
    separator = '\u2022'
    of = open('../text_file/stp.jsonl','w')

    with open('../save_encoding/snli/alignment/test_alignment.pkl', 'rb') as f:
        data = pickle.load(f)
        for line in data:
            this_json = {}
            EP = []
            CP = []
            NP = []
            EH = []
            CH = []
            NH = []
            UP = []
            UH = []
            
            if line['id'] in ids:
                
                for pair in line['p_h_aligned']:
                    apair = (line['c_p'][pair[0]],line['c_h'][pair[1]])
                    inp= tokenizer(apair[0], apair[1], return_tensors="pt").to('cuda')
                    outputs = model(**inp)
                    logits = outputs.logits
                    predictions = torch.argmax(logits, axis=-1)
                    
                    pred = predictions.item()
                    
                    if pred == 0:
                        EP.append(apair[0])
                        EH.append(apair[1])
                    elif pred == 1:
                    
                        NP.append(apair[0])
                        NH.append(apair[1])
                    elif pred == 2:
                    
                        CP.append(apair[0])
                        CH.append(apair[1])

                this_json['snli_id'] = line['id']
                this_json['EP'] = '\u2022'.join(EP)
                this_json['CP'] = '\u2022'.join(CP)
                this_json['NP'] = '\u2022'.join(NP)
                this_json['EH'] = '\u2022'.join(EH)
                this_json['CH'] = '\u2022'.join(CH)
                this_json['NH'] = '\u2022'.join(NH)
                this_json['UP'] = '\u2022'.join(UP)
                this_json['UH'] = '\u2022'.join(UH)
                of.write(json.dumps(this_json) + '\n')

is_train = False

model_name = 'bert-base-cased'
model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric = load_metric("accuracy")
    return metric.compute(predictions=predictions, references=labels)

def tokenize_function(examples):
    tokens = tokenizer(examples["premise"],examples["hypothesis"], padding='max_length', max_length=32, truncation=True)
    return tokens

features = datasets.Features(
    {
        "premise": datasets.Value("string"),
        "hypothesis": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=["entailment", "neutral", "contradiction"]),
    })

train_dataset = load_dataset('json', data_files={'train':'slp_train.jsonl'}, split='train', features=features)
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

snli_dataset = load_dataset("snli", split='test')
tokenized_snli_dataset = snli_dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_train_dataset
# print(next(iter(train_dataset)))
test_dataset = tokenized_snli_dataset
# print(next(iter(test_dataset)))

labels = np.array(test_dataset["label"])
temp = np.where(labels<0)[0]
a = []

for i in range(10000):
    if i not in temp:
        a.append(i)
cleaned_test = test_dataset.select(a)

training_args = TrainingArguments(
            "test_trainer",
            # num_train_epochs=1,
            per_device_train_batch_size = 128,
            per_device_eval_batch_size = 128,
            # learning_rate=5e-6,
            # lr_scheduler_type='constant'
        )

metric = load_metric("accuracy")

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=cleaned_test,
    compute_metrics=compute_metrics
)

if is_train:
    trainer.train()
else:
    sd = torch.load('test_trainer/checkpoint-33000/pytorch_model.bin')
    model.load_state_dict(sd)
    print(trainer.evaluate())
    generate_file()

torch.cuda.empty_cache()


