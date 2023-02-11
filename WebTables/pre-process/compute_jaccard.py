import numpy as np
import os
import csv
import time
from tqdm import trange
from tqdm import tqdm
import json
import jsonlines

#Generate Jaccard distance files

def jaccard(list1, list2): # compute the jaccard value between two lists
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1)+len(list2)) - intersection
    return float(intersection)/union

def read_tables(path): # load table content
    csv_reader = csv.reader(open(path))
    cur_set = []
    for i, line in enumerate(csv_reader):
        if i > 0:
            cur_set += line
    return list(set(cur_set))

def compute_jaccard(pathways, cur_path="../data/webtables/K4/"): # compute the jaccard distance between any two tables
    table_content = {}
    K4_content = {}
    jaccard_dict = {}
    for pathway in pathways:
        for file_name in os.listdir(pathway):
            file_path = pathway + file_name
            table_content[pathway[-3:]+file_name] = read_tables(file_path)
    for file_name in os.listdir(cur_path):
        file_path = cur_path+file_name
        K4_content[cur_path[-3:]+file_name] = read_tables(file_path)
    total_length = len(table_content)
    K4_length = len(K4_content)
    for i in trange(K4_length):
        list1_key = list(K4_content.keys())[i]
        if os.path.exists('../data/jaccard/'+list1_key[:-3]+'json'):
            continue
        list1 = K4_content[list1_key]

        for j in range(total_length):
            list2_key = list(table_content.keys())[j]
            if list2_key == list1_key:
                continue
            list2 = table_content[list2_key]

            jaccard_value = jaccard(list1, list2)
            if jaccard_value > 0.1:
                if not list2_key in jaccard_dict.keys():
                    jaccard_dict[list2_key] = [(list1_key, jaccard_value)]
                else:
                    jaccard_dict[list2_key].append((list1_key, jaccard_value))
                if not list1_key in jaccard_dict.keys():
                    jaccard_dict[list1_key] = [(list2_key, jaccard_value)]
                else:
                    jaccard_dict[list1_key].append((list2_key, jaccard_value))
        if not list1_key in jaccard_dict.keys():
            jaccard_dict[list1_key] = []
        output_dict = {}
        output_dict['filename'] = list1_key
        output_dict['jaccard_tables'] = sorted(jaccard_dict[list1_key], key=lambda item: item[1], reverse=True)
        with open('../data/jaccard/'+list1_key[:-3]+'json', 'w') as f:
            json.dump(output_dict, f)


if __name__ == '__main__':
    pathways = ["../data/webtables/K0/", "../data/webtables/K1/", "../data/webtables/K2/", "../data/webtables/K3/", "../data/webtables/K4/"]
    for round in range(5):
        compute_jaccard(pathways, cur_path="../data/webtables/K"+str(round)+"/")
