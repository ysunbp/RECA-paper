import os

MAX_LEN = 128
SEP_TOKEN_ID = 102

import tqdm
import time
import json
import numpy as np
import random
import torch
import functools
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
import jsonlines
from transformers import BertModel, BertForSequenceClassification, BertConfig
from tqdm import tqdm
from tqdm import trange
from math import sqrt




NERs = {'PERSON1':0, 'PERSON2':1, 'NORP':2, 'FAC':3, 'ORG':4, 'GPE':5, 'LOC':6, 'PRODUCT':7, 'EVENT':8, 'WORK_OF_ART':9, 'LAW':10, 'LANGUAGE':11, 'DATE1':12, 'DATE2':13, 'DATE3':14, 'DATE4':15, 'DATE5':16, 'TIME':17, 'PERCENT':18, 'MONEY':19, 'QUANTITY':20, 'ORDINAL':21, 'CARDINAL':22, 'EMPTY':23}


def load_json(json_path, label_dict):
    
    target_cols = []
    labels = []
    rel_cols = []
    sub_rel_cols = []
    one_hot = []
    headers_alias = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    mapping = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25}
    for json_file in tqdm(os.listdir(json_path)):
        with open(json_path+json_file, "r+", encoding="utf8") as f:
            content_dict = json.load(f)
            target_cols.append(content_dict['col'])
            labels.append(int(label_dict[content_dict['label']]))
            cur_rel_cols = []
            cur_sub_rel_cols = []
            for rel_col in content_dict['related_cols']:
                cur_rel_cols.append(np.array(rel_col))
            for sub_rel_col in content_dict['sub-related_cols']:
                cur_sub_rel_cols.append(np.array(sub_rel_col))
            rel_cols.append(cur_rel_cols)
            sub_rel_cols.append(cur_sub_rel_cols)
    return target_cols, rel_cols, sub_rel_cols, labels


class TableDataset(Dataset): # Generate tabular dataset
    def __init__(self, target_cols, tokenizer, rel_cols, sub_rel_cols, labels):
        self.labels = []
        self.target_cols = []
        self.tokenizer = tokenizer
        self.rel_cols = []
        self.sub_rel_cols = []
        for i in trange(len(labels)):
            self.labels.append(torch.tensor(labels[i]))
            target_token_ids = self.tokenize(target_cols[i])
            self.target_cols.append(target_token_ids)
            if len(rel_cols[i]) == 0: # If there is no related tables, use the target column content
                rel_token_ids = target_token_ids
            else:
                rel_token_ids = self.tokenize_set_equal(rel_cols[i])
            self.rel_cols.append(rel_token_ids)
            if len(sub_rel_cols[i]) == 0: # If there is no sub-related tables, use the target column content
                sub_token_ids = target_token_ids
            else:
                sub_token_ids = self.tokenize_set_equal(sub_rel_cols[i])
            self.sub_rel_cols.append(sub_token_ids)
        
    def tokenize(self, col): # Normal practice of tokenization
        text = ''
        for cell in col:
            text+=cell
            text+=' '
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids
    
    def tokenize_set_equal(self, cols): # Assigning the tokens equally to each identified column
        init_text = ''
        for i, col in enumerate(cols):
            for cell in col:
                init_text+=cell
                init_text+=' '
            if not i==len(cols)-1:
                init_text += '[SEP]'
        total_length = len(self.tokenizer.tokenize(init_text))
        if total_length <= MAX_LEN:
            tokenized_text = self.tokenizer.encode_plus(init_text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)     
        else:
            ratio = MAX_LEN/total_length
            text = ''
            for i, col in enumerate(cols):
                for j, cell in enumerate(col):
                    if j > len(col)*ratio:
                        break
                    text += cell
                    text += ' '
                if not i==len(cols)-1:
                    text += '[SEP]'
            tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def __getitem__(self, idx):
        return self.target_cols[idx], self.rel_cols[idx], self.sub_rel_cols[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        return token_ids, rel_ids, sub_ids, labels

if __name__ == '__main__':
    
    with open('./label_dict.json', 'r') as dict_in:
        label_dict = json.load(dict_in)
    
    Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    json_path_base = "../data/out/K"
    rounds = [0,1,2,3,4]
    target_col_set = []
    rel_col_set = []
    sub_col_set = []
    labels_set = []
    print('start loading data')
    for round in rounds:
        json_path = json_path_base+str(round)+'/'
        target_cols, rel_cols, sub_cols, labels = load_json(json_path, label_dict)
        ds_df = TableDataset(target_cols, Tokenizer, rel_cols, sub_cols, labels)
        torch.save(ds_df, '../data/tokenized_data/'+str(MAX_LEN)+'/fold_'+str(round))