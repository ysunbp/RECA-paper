# Pre-process procedures

## Start from scratch:

1. Download the raw data from this [link](http://www.cs.ox.ac.uk/isg/challenges/sem-tab/2019/#datasets), place in [raw_data](https://github.com/ysunbp/RECA-paper/tree/main/Semtab/data/raw_data) folder, modify the path names (e.g. Round 1 -> Round1) if necessary.
2. Run [transform_to_json.py](https://github.com/ysunbp/RECA-paper/blob/main/Semtab/pre-process/transform_to_json.py) to transform the raw table input (in csv format) to table summaries (in json format).
3. Then run [NER_extraction.py](https://github.com/ysunbp/RECA-paper/blob/main/Semtab/pre-process/NER_extraction.py) to extract the named entities from the tables.
4. Run [pre-process.py](https://github.com/ysunbp/RECA-paper/blob/main/Semtab/pre-process/pre-process.py) to find the related and sub-related tables and perform table alignment.
5. Run [make_json_input.py](https://github.com/ysunbp/RECA-paper/blob/main/Semtab/pre-process/make_json_input.py) to split the data into train-val and test sets and transform them into jsonl format.
6. Run [jaccard_filterjson.py](https://github.com/ysunbp/RECA-paper/blob/main/Semtab/pre-process/jaccard_filterjson.py) to apply table filtering.
7. Run [semtab-datasets.py](https://github.com/ysunbp/RECA-paper/blob/main/Semtab/pre-process/semtab-datasets.py) to generate the tokenized input for RECA.

## Start from pre-processed jsonl data:
1. Download the pre-processed jsonl data, see instructions in [jsonl_data](https://github.com/ysunbp/RECA-paper/tree/main/Semtab/data/jsonl_data).
2. Run [semtab-datasets.py](https://github.com/ysunbp/RECA-paper/blob/main/Semtab/pre-process/semtab-datasets.py) to generate the tokenized input for RECA.
