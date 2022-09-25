import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torchtext
from torchtext.legacy.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import os
device = 'cuda'
import en_core_web_sm, de_core_news_sm
spacy_en = en_core_web_sm.load()

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


train = pd.read_csv('./RedditHumorDetection/data/short_jokes/train.tsv', sep='\t', header=None)
dev = pd.read_csv('./RedditHumorDetection/data/short_jokes/dev.tsv', sep='\t', header=None)

dd = []
for i in train.iterrows():
  ll = {}
  k = list(i[1])[0].split(',')
  # print(k)
  # sents.append(k[3])
  ll['src'] = k[3]
  ll['tgt'] = k[3]
  ll['lab'] = int(k[1])
  dd.append(ll)
  # labs.append(int(k[1]))

# dd = dd[0:16000]

dd_t = []
for i in dev.iterrows():
  ll = {}
  k = list(i[1])[0].split(',')
  # print(k)
  # sents.append(k[3])
  ll['src'] = k[3]
  ll['tgt'] = k[3]
  ll['lab'] = int(k[1])
  dd_t.append(ll)
  # labs.append(int(k[1])

# dd_t = dd_t[0:160]


train_texts = [i['src'] for i in dd]
train_labels = [i['lab'] for i in dd]

test_texts = [i['src'] for i in dd_t]
test_labels = [i['lab'] for i in dd_t]


from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=42)



from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

import torch

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    seed=0,
    save_total_limit = 1,
    load_best_model_at_end=True,
    save_strategy = "no"
)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset             # evaluation dataset
)

trainer.train()
trainer.save_model('./results')


model = DistilBertForSequenceClassification.from_pretrained('./results')

# arguments for Trainer
test_args = TrainingArguments(
    output_dir = './results_infer',
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 16,   
    # dataloader_drop_last = False    
)

# init trainer
trainer = Trainer(
              model = model, 
              args = test_args, 
              # compute_metrics = compute_metrics
              )

test_results = trainer.predict(test_dataset)
print(test_results)

preds = np.argmax(test_results.predictions,axis=-1)

labels = test_results.label_ids
from sklearn.metrics import accuracy_score,f1_score
print(accuracy_score(labels,preds))
print(f1_score(labels,preds))

