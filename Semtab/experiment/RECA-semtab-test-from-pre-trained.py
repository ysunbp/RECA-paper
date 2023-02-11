import os

MAX_LEN = 512
SEP_TOKEN_ID = 102
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
from math import sqrt




NERs = {'PERSON1':0, 'PERSON2':1, 'NORP':2, 'FAC':3, 'ORG':4, 'GPE':5, 'LOC':6, 'PRODUCT':7, 'EVENT':8, 'WORK_OF_ART':9, 'LAW':10, 'LANGUAGE':11, 'DATE1':12, 'DATE2':13, 'DATE3':14, 'DATE4':15, 'DATE5':16, 'TIME':17, 'PERCENT':18, 'MONEY':19, 'QUANTITY':20, 'ORDINAL':21, 'CARDINAL':22, 'EMPTY':23}


def load_jsonl(jsonl_path, label_dict):
    target_cols = []
    labels = []
    rel_cols = []
    sub_rel_cols = []
    one_hot = []
    headers_alias = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    mapping = {"A":0,"B":1,"C":2,"D":3,"E":4,"F":5,"G":6,"H":7,"I":8,"J":9,"K":10,"L":11,"M":12,"N":13,"O":14,"P":15,"Q":16,"R":17,"S":18,"T":19,"U":20,"V":21,"W":22,"X":23,"Y":24,"Z":25}
    with open(jsonl_path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            target_cols.append(np.array(item['content'])[:,int(item['target'])])
            target_alias = headers_alias[int(item['target'])]
            labels.append(int(label_dict[item['label']]))
            cur_rel_cols = []
            cur_sub_rel_cols = []
            for rel_col in item['related_cols']:
                cur_rel_cols.append(np.array(rel_col))
            for sub_rel_col in item['sub_related_cols']:
                cur_sub_rel_cols.append(np.array(sub_rel_col))
            rel_cols.append(cur_rel_cols)
            sub_rel_cols.append(cur_sub_rel_cols)
    return target_cols, rel_cols, sub_rel_cols, labels

class TableDataset(Dataset):
    def __init__(self, target_cols, tokenizer, rel_cols, sub_rel_cols, labels):
        self.labels = labels
        self.target_cols = target_cols
        self.tokenizer = tokenizer
        self.rel_cols = rel_cols
        self.sub_rel_cols = sub_rel_cols

    def tokenize(self, col):
        text = ''
        for cell in col:
            text+=cell
            text+=' '
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids

    def tokenize_set(self, cols):
        text = ''
        for i, col in enumerate(cols):
            for cell in col:
                text+=cell
                text+=' '
            if not i==len(cols)-1:
                text += '[SEP]'
        tokenized_text = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=MAX_LEN, padding = 'max_length', truncation=True)         
        ids = torch.Tensor(tokenized_text["input_ids"]).long()
        return ids
    
    def tokenize_set_equal(self, cols):
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
        target_token_ids = self.tokenize(self.target_cols[idx])
        if len(self.rel_cols[idx]) == 0:
            rel_token_ids = target_token_ids
        else:
            rel_token_ids = self.tokenize_set_equal(self.rel_cols[idx])
        if len(self.sub_rel_cols[idx]) == 0:
            sub_token_ids = target_token_ids
        else:
            sub_token_ids = self.tokenize_set_equal(self.sub_rel_cols[idx])
        
        return target_token_ids, rel_token_ids, sub_token_ids, torch.tensor(self.labels[idx])

    def __len__(self):
        return len(self.labels)

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        rel_ids = torch.stack([x[1] for x in batch])
        sub_ids = torch.stack([x[2] for x in batch])
        labels = torch.stack([x[3] for x in batch])
        return token_ids, rel_ids, sub_ids, labels

def get_loader(target_cols, rel_cols, sub_rel_cols, labels,batch_size=8,is_train=True):
    ds_df = TableDataset(target_cols, Tokenizer, rel_cols, sub_rel_cols, labels)
    loader = torch.utils.data.DataLoader(ds_df, batch_size=batch_size, shuffle=is_train, num_workers=0, collate_fn=ds_df.collate_fn)
    loader.num = len(ds_df)
    return loader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class KREL(torch.nn.Module):
    def __init__(self, n_classes=275, dim_k=768, dim_v=768):
        super(KREL, self).__init__()
        self.model_name = 'KREL'
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.3)
        self.fcc_tar = torch.nn.Linear(768, n_classes)
        self.fcc_rel = torch.nn.Linear(768, n_classes)
        self.fcc_sub = torch.nn.Linear(768, n_classes)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(1)) for i in range(3)])

    def encode(self, target_ids, rel_ids, sub_ids):
        att_tar = (target_ids>0)
        _, tar = self.bert_model(input_ids=target_ids, attention_mask=att_tar, return_dict=False)
        att_rel = (rel_ids>0)
        _, rel = self.bert_model(input_ids=rel_ids, attention_mask=att_rel, return_dict=False)
        att_sub = (sub_ids>0)
        _, sub = self.bert_model(input_ids=sub_ids, attention_mask=att_sub, return_dict=False)

        return tar, rel, sub
    
    def forward(self,tar_ids,rel_ids, sub_ids):
        tar, rel, sub = self.encode(tar_ids, rel_ids, sub_ids)
        tar_out = self.dropout(tar)
        rel_out = self.dropout(rel)
        sub_out = self.dropout(sub)
        out_tar = self.fcc_tar(tar_out)
        out_rel = self.fcc_rel(rel_out)
        out_sub = self.fcc_sub(sub_out)
        res = self.weights[0]*out_tar+self.weights[1]*out_rel+self.weights[2]*out_sub
        return res

def metric_fn(preds, labels):
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }

def test_model(model,test_loader,lr,new_dict,model_save_path='.pkl',early_stop_epochs=5,epochs=20):  
    loss_fn = torch.nn.CrossEntropyLoss().cuda()
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    bar = tqdm(test_loader)
    pred_labels = []
    true_labels = []
    for i, (ids, rels, subs, labels) in enumerate(bar):
        labels = labels.cuda()
        rels = rels.cuda()
        subs = subs.cuda()
        output = model(ids.cuda(), rels, subs)
        y_pred_prob = output
        y_pred_label = y_pred_prob.argmax(dim=1)
        pred_labels.append(y_pred_label.detach().cpu().numpy())
        true_labels.append(labels.detach().cpu().numpy())
        del ids, rels, subs
        torch.cuda.empty_cache()
    pred_labels = np.concatenate(pred_labels)
    true_labels = np.concatenate(true_labels)
    f1_scores = metric_fn(pred_labels, true_labels)
    print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
    return f1_scores['weighted_f1'], f1_scores['macro_f1']

if __name__ == '__main__':
    setup_seed(20)
    with open('./semtab_labels.json', 'r') as dict_in:
        label_dict = json.load(dict_in)
    Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    new_dict = {v : k for k, v in label_dict.items()}
    BS = 8
    lrs = [1e-5]
    test_jsonl_path = "../data/jsonl_data/test_hard_jaccard_ranking.jsonl"
    test_target_cols, test_rels, test_subs, test_labels = load_jsonl(test_jsonl_path, label_dict)
    test_loader = get_loader(test_target_cols, test_rels, test_subs, test_labels, 1, False)
    print("###############################")
    for lr in lrs:
        print("start for testing learning rate:", lr)
        weighted_f1s = []
        macro_f1s = []
        for cur_fold in range(5):
            if True:
                model = KREL().cuda()
                model_save_path = '../checkpoints/RECA_bs_'+str(BS)+"_lr_"+str(lr)+'_fold_{}.pkl'.format(cur_fold+1)
                print("Starting fold", cur_fold+1)
                cur_w, cur_m = test_model(model, test_loader,lr,new_dict, model_save_path=model_save_path)
                weighted_f1s.append(cur_w)
                macro_f1s.append(cur_m)
        print("The mean F1 score is:", np.mean(weighted_f1s))
        print("The sd is:", np.std(weighted_f1s))
        print("The mean macro F1 score is:", np.mean(macro_f1s))
        print("The sd is:", np.std(macro_f1s))
        print("===============================")


######################################
