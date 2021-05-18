# typos-aware-BERT

## Install
To install required dependencies run `pip install -r requirements.txt`.

## Dataset
We evaluate our typos-aware training on MS MARCO passage ranking dataset. 

To download data required in this repo (`qidpidtriples.train.full.tsv.gz`, `collection.tsv`, `queries.train.tsv`,`qrels.train.tsv`,`queries.dev.small.tsv`,`qrels.dev.small.tsv`), we refer to the offcial dataset github repo [https://github.com/microsoft/MSMARCO-Passage-Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking).

We provide the synthetic typo queries in `./data` folder. You also can generate typo queries by yourself:

```
python ./BERT_re-ranker/tools/make_typo_queries.py
```

## Dense Retrieval (DR) Models
The code that related to reproducing DR models used in our paper is in `./DR` folder.
We use an open-source implementation folked (anonymously) from this repo [RepBERT-Index](https://github.com/jingtaozhan/RepBERT-Index).

To train DR with standard training setting, `cd` to `./DR` folder and following the instructions in the original repo.

To train typos-aware DR, run the following command without changing any other parameters:
 
```
python ./train.py --task train --insert_typo 1 --evaluate_during_training
```

## Train BERT re-ranker Models
The code that related to reproducing BERT re-ranker models used in our paper is in `./BERT_re-ranker` folder. 

To train BERT re-ranker with standard training setting, `cd` to `./BERT_re-ranker` folder and run:

```
python ./train_cross_encoder.py --training_setting standard
```

To train BERT re-ranker with typos-aware training setting run:

```
python ./train_cross_encoder.py --training_setting typos-aware
```



