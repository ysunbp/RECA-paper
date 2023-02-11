import spacy
import os
import json
import numpy as np

nlp = spacy.load('en_core_web_trf')
table_dir = '../data/json/Round'
out_dir = '../data/json/Round'
rounds = [1,3,4]

for round in rounds:
    json_dir = table_dir+str(round)
    out_json_dir = out_dir+str(round)
    for json_file in os.listdir(json_dir):
        json_file_path = json_dir+'/'+json_file
        out_json_file_path = out_json_dir+'/'+json_file
        with open(json_file_path, 'r') as load_f:
            load_dict = json.load(load_f)
            new_dict = {}
            new_dict['filename'] = load_dict['filename']
            new_dict['header'] = load_dict['header']
            new_dict['content'] = load_dict['content']
            new_dict['target'] = load_dict['target']
            new_dict['label'] = load_dict['label']

            table_content = np.array(load_dict['content'])
            width = len(load_dict['header'])
            col_ne = []
            for col_index in range(width):
                column_string = ''
                cur_ne = []
                for cell in table_content[:,col_index]:
                    column_string += cell
                    column_string += ' ; '
                doc = nlp(column_string)
                for ent in doc.ents:
                    cur_ne.append(ent.label_)
                col_ne.append(cur_ne)
            new_dict['table_NE'] = col_ne
            with open(out_json_file_path, 'w') as out_f:
                json.dump(new_dict, out_f)
            print(json_file)
        
