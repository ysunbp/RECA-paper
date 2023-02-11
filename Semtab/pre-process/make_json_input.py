import torch
import os
import torch.nn as nn
import numpy as np
import random
import json
import jsonlines
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import csv
import operator



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

pathway1 = "../data/json/Round1/"
pathway3 = "../data/json/Round3/"
pathway4 = "../data/json/Round4/"

train_jsonl = "../data/jsonl_data/train_val_hard.jsonl"
test_jsonl = "../data/jsonl_data/test_hard.jsonl"

csv_dir_1 = "../data/raw_data/Round1/tables/"
csv_dir_3 = "../data/raw_data/Round3/tables/"
csv_dir_4 = "../data/raw_data/Round4/tables/"

csv_dirs = [csv_dir_1, csv_dir_3, csv_dir_4]
pathways = [pathway1, pathway3, pathway4]


json_files = []
labels = []



# Exact alignment
for i, pathway in enumerate(pathways):
    for file_name in tqdm(os.listdir(pathway)):
        file_path = pathway + file_name
        rel_json_path = pathway + file_name
        with open(file_path, "r") as load_f:
            load_dict = json.load(load_f)
        with open(rel_json_path, "r") as load_j:
            load_dict_j = json.load(load_j)
        load_dict['table_NE'] = load_dict_j['table_NE']
        load_dict['most_common'] = load_dict_j['most_common']
        load_dict['related_table'] = load_dict_j['related_table']
        load_dict['subtable-type'] = load_dict_j['subtable-type']
        load_dict['table-type'] = load_dict_j['table-type']
        related_cols = []
        sub_related_cols = []
        for rel_table in load_dict_j['related_table']:
            for csv_dir in csv_dirs:
                if os.path.exists(csv_dir+rel_table):
                    list_file = []
                    with open(csv_dir+rel_table,'r') as csv_file: 
                        all_lines=csv.reader(csv_file)  
                        for one_line in all_lines:  
                            list_file.append(one_line)  
                    list_file.remove(list_file[0])
                    arr_file = np.array(list_file)
                    related_cols.append(list(arr_file[:, int(load_dict['target'])]))
        sub_tables = []
        for index, table_filename in enumerate(load_dict_j['subtable']):
            cur_subtype = load_dict_j['subtable-type'][index]
            target_index = int(load_dict['target'])
            width = len(cur_subtype)
            for csv_dir in csv_dirs:
                if os.path.exists(csv_dir+table_filename):
                    list_file = []
                    with open(csv_dir+table_filename,'r') as csv_file: 
                        all_lines=csv.reader(csv_file)  
                        for one_line in all_lines:  
                            list_file.append(one_line)  
                    list_file.remove(list_file[0])
                    arr_file = np.array(list_file)
            if target_index < width and load_dict_j['table-type'][target_index] == cur_subtype[target_index]:
                sub_related_cols.append(list(arr_file[:, target_index]))
                sub_tables.append(table_filename)
        load_dict['related_cols'] = related_cols
        load_dict['sub_related_cols'] = sub_related_cols
        load_dict['subtable'] = sub_tables     
        json_files.append(load_dict)
        labels.append(load_dict['label'])


sfolder_test = StratifiedKFold(n_splits=10, random_state = 42, shuffle=True)
train_valid_set = []
test_set = []
for train_valid, test in sfolder_test.split(json_files, labels):
    train_valid_index = train_valid
    test_index = test
    break
for index in train_valid_index:
    with jsonlines.open(train_jsonl, "a") as writer:
        writer.write(json_files[index])
for index in test_index:
    with jsonlines.open(test_jsonl, "a") as writer:
        writer.write(json_files[index])


