import csv
import json
import os


def read_csv(csv_dir):
    csv_lines = csv.reader(open(csv_dir))
    content = []
    num_row = 0
    for i, line in enumerate(csv_lines):
        if i == 0:
            header = line
            num_col = len(header)
            
        else:
            content.append(line)
            num_row += 1
    num_cell = num_col*num_row
    return header, content, num_col, num_row, num_cell

rounds = [1,3,4]
  
for round in rounds:
    csv_reader = csv.reader(open("../data/raw_data/Round"+str(round)+"/gt/CTA_Round"+str(round)+"_gt.csv"))
    csv_dir = "../data/raw_data/Round"+str(round)+"/tables/"
    output_dir = "../data/json/Round"+str(round)+"/"
    i = 0
    for line in csv_reader:
        file_name = line[0]+".csv"
        target_col = line[1]
        label = line[2].split('/')[-1]
        file_dir = csv_dir+file_name
        if not os.path.exists(file_dir):
            continue
        header, content, num_col, num_row, num_cell = read_csv(file_dir)
        dict = {}
        dict['filename'] = file_name
        dict['header'] = header
        dict['content'] = content
        dict['target'] = target_col
        dict['label'] = label
        output_path = output_dir+str(i)+'.json'
        with open(output_path, "w") as outfile:
            json.dump(dict, outfile)
        i += 1
        print(i)
