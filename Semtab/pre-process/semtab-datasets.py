import os

MAX_LEN = 512
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_file(data_path, label_dict):
    labels = []
    out_data = []
    rel_cols = []
    sub_cols = []
    with open(data_path, "r+", encoding="utf8") as jl:
        for item in tqdm(jsonlines.Reader(jl)):
            label_idx = int(label_dict[item['label']])
            target_data = np.array(item['content'])[:,int(item['target'])]
            data = ""
            for i, cell in enumerate(target_data):
                data+=cell
                data+=' '
            cur_rel_cols = []
            cur_sub_rel_cols = []
            for rel_col in item['related_cols']:
                cur_rel_cols.append(np.array(rel_col))
            for sub_rel_col in item['sub_related_cols']:
                cur_sub_rel_cols.append(np.array(sub_rel_col))
            sub_cols.append(cur_sub_rel_cols)
            rel_cols.append(cur_rel_cols)
            labels.append(label_idx)
            out_data.append(data)
    return out_data, rel_cols, sub_cols, labels


class TableDataset(Dataset): # Generate tabular dataset
    def __init__(self, target_cols, tokenizer, rel_cols, sub_rel_cols, labels):
        self.labels = []
        self.data = []
        self.tokenizer = tokenizer
        self.rel = []
        self.sub = []
        for i in trange(len(labels)):
            self.labels.append(torch.tensor(labels[i]))
            target_token_ids = self.tokenize(target_cols[i])
            self.data.append(target_token_ids)
            if len(rel_cols[i]) == 0: # If there is no related tables, use the target column content
                rel_token_ids = target_token_ids
            else:
                rel_token_ids = self.tokenize_set_equal(rel_cols[i])
            self.rel.append(rel_token_ids)
            if len(sub_rel_cols[i]) == 0: # If there is no sub-related tables, use the target column content
                sub_token_ids = target_token_ids
            else:
                sub_token_ids = self.tokenize_set_equal(sub_rel_cols[i])
            self.sub.append(sub_token_ids)
        
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
        return self.data[idx], self.rel[idx], self.sub[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        return token_ids, rel_ids, sub_ids, labels


if __name__ == '__main__':
    if True:
        setup_seed(20)
        data_path_train = '../data/jsonl_data/train_val_hard_jaccard_ranking.jsonl'
        data_path_test = '../data/jsonl_data/test_hard_jaccard_ranking.jsonl'
        label_dict = get_label_dict()
        train_data, train_rel,train_sub, train_labels = load_file(data_path_train, label_dict)
        test_data, test_rel,test_sub, test_labels = load_file(data_path_test, label_dict)
        Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        sfolder_cv = StratifiedKFold(n_splits=5, random_state = 0, shuffle=True)
        for cur_fold, (train_idx, val_idx) in enumerate(sfolder_cv.split(train_data, train_labels)):
            if True:
                print('start loading data')
                train_cols = []
                train_labels_splited = []
                train_rels = []
                train_subs = []
                valid_cols = []
                valid_labels_splited = []
                valid_rels = []
                valid_subs = []
                for t_idx in train_idx:
                    train_cols.append(train_data[t_idx])
                    train_rels.append(train_rel[t_idx])
                    train_subs.append(train_sub[t_idx])
                    train_labels_splited.append(train_labels[t_idx])
                for v_idx in val_idx:
                    valid_cols.append(train_data[v_idx])
                    valid_rels.append(train_rel[v_idx])
                    valid_subs.append(train_sub[v_idx])
                    valid_labels_splited.append(train_labels[v_idx])
                ds_df = TableDataset(train_cols, train_rels, train_subs, Tokenizer, train_labels_splited)
                torch.save(ds_df, '../data/tokenized_data/train_'+str(MAX_LEN)+'_fold_'+str(cur_fold))
                ds_df_v = TableDataset(valid_cols, valid_rels, valid_subs, Tokenizer, valid_labels_splited)
                torch.save(ds_df_v, '../data/tokenized_data/valid_'+str(MAX_LEN)+'_fold_'+str(cur_fold))
        ds_df_t = TableDataset(test_data, test_rel, test_sub, Tokenizer, test_labels)
        torch.save(ds_df_t, '../data/tokenized_data/test_'+str(MAX_LEN))
    
