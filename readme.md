# CIL

Code base for "CIL: Contrastive Instance Learning Framework for Distantly Supervised Relation Extraction".

## Env

For distributed parallel training, we use the library [huggingface/accelerate](https://github.com/huggingface/accelerate) for convenience, and its configuration is in the file `ddp.yaml` (`num_processes`: number of available GPUs ~ `CUDA_VISIBLE_DEVICES`).
```text
pip install -r requirements.txt	
```

## Data

We follow the data format in [OpenNRE](https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_nyt10.sh), each dataset folder contains the following files:

> nyt10d_rel2id.json: Mapping of relation category to relation ID
```json
{
  "NA": 0,
  "/location/neighborhood/neighborhood_of": 1,
  ...
  "/film/film_festival/location": 52
}
```
> nyt10d_(train|dev|test).txt: Each line of data is in json format ↓
```json
{
  "text": "sen. charles e. schumer called on federal safety officials yesterday to reopen their investigation into the fatal crash of a passenger jet in belle harbor , queens , because equipment failure , not pilot error , might have been the cause .",
  "relation": "/location/location/contains",
  "h": {
    "id": "m.0ccvx",
    "name": "queens",
    "pos": [157, 163]
  },
  "t": {
    "id": "m.05gf08",
    "name": "belle harbor",
    "pos": [142, 154]
  }
}
```

## Run

1. Download dataset (refer to `benchmark/nyt10d/readme.md`) and pretrained `bert-base-uncased`:
```bash
bash download_bert.sh
```

2. Run script `aug.py` to generate positive pair samples for training dataset (Line 18: dataset name):
```bash
python aug.py
```

3. Run script in folder `scripts` to train and evaluate the DSRE models:
```bash
bash train_nyt10d.sh SEED
```
4. Run script `plot.py` in folder `pr` to draw pr curves of the trained model.
```bash
python plot.py
```


## PR Curve

The PR curves (NYT10-D) of all baseline models are in [Google Drive](https://drive.google.com/drive/folders/1qGivvtivhQIYtacOI1LuEX09KeyXuRYI).

## Cite

```text
@inproceedings{chen-etal-2021-cil,
    title = "{CIL}: Contrastive Instance Learning Framework for Distantly Supervised Relation Extraction",
    author = "Chen, Tao  and
      Shi, Haizhou  and
      Tang, Siliang  and
      Chen, Zhigang  and
      Wu, Fei  and
      Zhuang, Yueting",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.483",
    doi = "10.18653/v1/2021.acl-long.483",
    pages = "6191--6200",
    abstract = "The journey of reducing noise from distant supervision (DS) generated training data has been started since the DS was first introduced into the relation extraction (RE) task. For the past decade, researchers apply the multi-instance learning (MIL) framework to find the most reliable feature from a bag of sentences. Although the pattern of MIL bags can greatly reduce DS noise, it fails to represent many other useful sentence features in the datasets. In many cases, these sentence features can only be acquired by extra sentence-level human annotation with heavy costs. Therefore, the performance of distantly supervised RE models is bounded. In this paper, we go beyond typical MIL framework and propose a novel contrastive instance learning (CIL) framework. Specifically, we regard the initial MIL as the relational triple encoder and constraint positive pairs against negative pairs for each instance. Experiments demonstrate the effectiveness of our proposed framework, with significant improvements over the previous methods on NYT10, GDS and KBP.",
}
```
