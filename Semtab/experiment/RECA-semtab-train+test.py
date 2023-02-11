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
from tqdm import trange
from math import sqrt


NERs = {'PERSON1':0, 'PERSON2':1, 'NORP':2, 'FAC':3, 'ORG':4, 'GPE':5, 'LOC':6, 'PRODUCT':7, 'EVENT':8, 'WORK_OF_ART':9, 'LAW':10, 'LANGUAGE':11, 'DATE1':12, 'DATE2':13, 'DATE3':14, 'DATE4':15, 'DATE5':16, 'TIME':17, 'PERCENT':18, 'MONEY':19, 'QUANTITY':20, 'ORDINAL':21, 'CARDINAL':22, 'EMPTY':23}


def setup_seed(seed): # Set up random seeds for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

def get_loader(path, batch_size, is_train): # Generate the dataloaders for the training process
    dataset = torch.load(path)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=0, collate_fn=dataset.collate_fn)
    loader.num = len(dataset)
    return loader

class KREL(torch.nn.Module): # KREL model structure
    def __init__(self, n_classes=275, dim_k=768, dim_v=768):
        super(KREL, self).__init__()
        self.model_name = 'KREL'
        self.bert_model = BertModel.from_pretrained("bert-base-uncased") # BERT encoder
        self.dropout = torch.nn.Dropout(0.3) # Dropout layer
        self.fcc_tar = torch.nn.Linear(768, n_classes) # linear layer
        self.fcc_rel = torch.nn.Linear(768, n_classes)
        self.fcc_sub = torch.nn.Linear(768, n_classes)
        self.weights = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(1)) for i in range(3)]) # Weighted combination

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

def metric_fn(preds, labels): # The Support-weighted F1 score and Macro Average F1 score
    weighted = f1_score(labels, preds, average='weighted')
    macro = f1_score(labels, preds, average='macro')
    return {
        'weighted_f1': weighted,
        'macro_f1': macro
    }

def train_model(model,train_loader,val_loader,lr,model_save_path='.pkl',early_stop_epochs=5,epochs=20): # Training process
    no_improve_epochs = 0
    weight_decay = 1e-2
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    cur_best_v_loss =10.0
    for epoch in range(1,epochs+1):
        
        model.train()
        epoch_loss = 0
        v_epoch_loss = 0
        train_length = 0
        tic = time.time()
        bar1 = tqdm(train_loader)
            
        for i,(ids, rels, subs, labels) in enumerate(bar1):
            labels = labels.cuda()
            rels = rels.cuda()
            subs = subs.cuda()
            output = model(ids.cuda(), rels, subs)
            y_pred_prob = output
            y_pred_label = y_pred_prob.argmax(dim=1)
            loss = loss_fn(y_pred_prob.view(-1, 275), labels.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            length_label = len(labels)
            del ids, rels, subs, labels
            torch.cuda.empty_cache() # Release the memory
        train_length += len(bar1)
        print("Epoch:", epoch, "training_loss:", epoch_loss / (train_length))
        model.eval()
        bar2 = tqdm(val_loader)
        pred_labels = []
        true_labels = []
        toc = time.time()
        print('training time:', toc-tic)
        for j, (ids, rels, subs, labels) in enumerate(bar2):
            labels = labels.cuda()
            rels = rels.cuda()
            subs = subs.cuda()
            output = model(ids.cuda(), rels, subs)
            y_pred_prob = output
            y_pred_label = y_pred_prob.argmax(dim=1)
            vloss = loss_fn(y_pred_prob.view(-1, 275), labels.view(-1))
            pred_labels.append(y_pred_label.detach().cpu().numpy())
            true_labels.append(labels.detach().cpu().numpy())
            v_epoch_loss += vloss.item()
            v_length_label = len(labels)
            del ids, rels, subs
            torch.cuda.empty_cache()
        tac = time.time()
        pred_labels = np.concatenate(pred_labels)
        true_labels = np.concatenate(true_labels)
        val_length = len(bar2)
        print("validation_loss:", v_epoch_loss / (val_length))
        f1_scores = metric_fn(pred_labels, true_labels)
        print("weighted f1:", f1_scores['weighted_f1'], "\t", "macro f1:", f1_scores['macro_f1'])
        print('validation time:', tac-toc)
        if v_epoch_loss / (val_length) < cur_best_v_loss:
            torch.save(model.state_dict(),model_save_path)
            cur_best_v_loss = v_epoch_loss / (val_length)
            no_improve_epochs = 0
            print("model updated")
        else:
            no_improve_epochs += 1
        if no_improve_epochs == 5:
            print("early stop!")
            break


def test_model(model,test_loader,lr,model_save_path='.pkl'):  
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
    rounds = [0,1,2,3,4]
    
    BS = 8
    lrs = [1e-5]
    print('start loading data')
    
    for round in rounds:
        if True:
            train_loader_path = '../data/tokenized_data/train_'+str(MAX_LEN)+'_fold_'+str(round)
            valid_loader_path = '../data/tokenized_data/valid_'+str(MAX_LEN)+'_fold_'+str(round)
            val_loader = get_loader(path=valid_loader_path, batch_size=BS, is_train=False)    
            train_loader = get_loader(path=train_loader_path, batch_size=BS, is_train=True)
            for lr in lrs:
                print('start training fold', round+1, 'learning rate', lr, 'batch size', BS, 'max length', MAX_LEN)
                model = KREL().cuda()
                model_save_path = '../checkpoints/semtab-RECA'+"_lr="+str(lr)+'_bs='+str(BS)+'_max='+str(MAX_LEN)+'_{}.pkl'.format(round+1)
                train_model(model, train_loader, val_loader,lr, model_save_path=model_save_path)
    
    test_loader_path = '../data/tokenized_data/test_'+str(MAX_LEN)
    test_loader = get_loader(path=test_loader_path, batch_size=1, is_train=False)
    weighted_f1s = []
    macro_f1s = []
    lr = 1e-5
    for cur_fold in range(5):
        model = KREL().cuda()
        model_save_path = '../checkpoints/semtab-RECA'+"_lr="+str(lr)+'_bs='+str(BS)+'_max='+str(MAX_LEN)+'_{}.pkl'.format(cur_fold+1)
        print("Starting fold", cur_fold+1)
        cur_w, cur_m = test_model(model, test_loader,lr, model_save_path=model_save_path)
        weighted_f1s.append(cur_w)
        macro_f1s.append(cur_m)
    print("The mean F1 score is:", np.mean(weighted_f1s))
    print("The sd is:", np.std(weighted_f1s))
    print("The mean macro F1 score is:", np.mean(macro_f1s))
    print("The sd is:", np.std(macro_f1s))
    print("===============================")
