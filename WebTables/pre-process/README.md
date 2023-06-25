# Pre-process procedures

## Start from scratch
1. Download the raw dataset from this [link](https://github.com/megagonlabs/sato/tree/master/table_data) (Multi-only), place in the [webtables](https://github.com/ysunbp/RECA-paper/tree/main/WebTables/data/webtables) (Remember to delete the init files in each of the Ki folder, they are created to set up the directories in GitHub).
2. Run [compute_jaccard.py](https://github.com/ysunbp/RECA-paper/blob/main/WebTables/pre-process/compute_jaccard.py) to compute the jaccard distance between any two tables.
3. Run [pre-process-webtables.py](https://github.com/ysunbp/RECA-paper/blob/main/WebTables/pre-process/pre-process-webtables.py) to generate the pre-processed json files.
4. Run [webtables-datasets.py](https://github.com/ysunbp/RECA-paper/blob/main/WebTables/pre-process/webtables-datasets.py) to generate tokenized input for RECA.

## Start from tokenized data
1. Download the pre-processed tokenized data from this [link](https://drive.google.com/file/d/1wo6QMjUdWsb6-5kczqZMy_89gstw7BfV/view?usp=sharing), unzip, and place in the [tokenized_data](https://github.com/ysunbp/RECA-paper/tree/main/WebTables/data/tokenized_data) folder.
