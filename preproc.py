# this is just a slimmed down version of Nil's code

"""
This examples show how to train a Bi-Encoder for the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.
For training, we use MultipleNegativesRankingLoss. There, we pass triplets in the format:
(query, positive_passage, negative_passage)
Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.
With a distilbert-base-uncased model, it should achieve a performance of about 33.79 MRR@10 on the MSMARCO Passages Dev-Corpus
Running this script:
python train_bi-encoder-v3.py
"""
import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout


parser = argparse.ArgumentParser()
parser.add_argument("--negs_to_use", default=None, help="From which systems should negatives be used? Multiple systems seperated by comma. None = all")
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_all_queries", default=False, action="store_true")
parser.add_argument("--ce_score_margin", default=3.0, type=float)
args = parser.parse_args()

print(args)

ce_score_margin = args.ce_score_margin             #Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = args.num_negs_per_system         # We used different systems to mine hard negatives. Number of hard negatives to add from each system


### Now we read the MS Marco dataset
data_folder = 'msmarco-data'

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage


### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
if not os.path.exists(queries_filepath):
    tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download queries.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=data_folder)


with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        qid = int(qid)
        queries[qid] = query


# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
# to the CrossEncoder score computed by the cross-encoder/ms-marco-MiniLM-L-6-v2 model
ce_scores_file = os.path.join(data_folder, 'cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz')
if not os.path.exists(ce_scores_file):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz', ce_scores_file)

logging.info("Load CrossEncoder scores dict")
with gzip.open(ce_scores_file, 'rb') as fIn:
    ce_scores = pickle.load(fIn)

# As training data we use hard-negatives that have been mined using various systems
hard_negatives_filepath = os.path.join(data_folder, 'msmarco-hard-negatives.jsonl.gz')
if not os.path.exists(hard_negatives_filepath):
    logging.info("Download cross-encoder scores file")
    util.http_get('https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz', hard_negatives_filepath)


logging.info("Read hard negatives train file")
train_queries = {}
negs_to_use = None
with gzip.open(hard_negatives_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn):
        data = json.loads(line)

        #Get the positive passage ids
        qid = data['qid']
        pos_pids = data['pos']

        if len(pos_pids) == 0:  #Skip entries without positives passages
            continue

        pos_min_ce_score = min([ce_scores[qid][pid] for pid in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin

        #Get the hard negatives
        neg_pids = set()
        if negs_to_use is None:
            if args.negs_to_use is not None:    #Use specific system for negatives
                negs_to_use = args.negs_to_use.split(",")
            else:   #Use all systems
                negs_to_use = list(data['neg'].keys())
            logging.info("Using negatives from the following systems: {}".format(", ".join(negs_to_use)))

        for system_name in negs_to_use:
            if system_name not in data['neg']:
                continue

            system_negs = data['neg'][system_name]
            negs_added = 0
            for pid in system_negs:
                if ce_scores[qid][pid] > ce_score_threshold:
                    continue

                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break

        if args.use_all_queries or (len(pos_pids) > 0 and len(neg_pids) > 0):
            train_queries[data['qid']] = {'qid': data['qid'], 'query': queries[data['qid']], 'pos': pos_pids, 'neg': neg_pids}

logging.info("Train queries: {}".format(len(train_queries)))

def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

# take the last 5% of train_queries as the eval set 
eval_queries = {k: v for k, v in list(train_queries.items())[-int(len(train_queries.keys()) * 0.025):]}
train_queries = {k: v for k, v in list(train_queries.items())[:-int(len(train_queries.keys()) * 0.025)]}

# save train_queries to a JSON file
filename = os.path.join(data_folder, 'train_queries_' + str(ce_score_margin))
if args.num_negs_per_system != 5:
    filename += '_' + str(args.num_negs_per_system)
filename += '_.json'

with open(filename, 'w', encoding='utf8') as fOut:
    json.dump(train_queries, fOut, indent=4, ensure_ascii=False, default=set_default)

    
eval_filename = os.path.join(data_folder, 'eval_queries_' + str(ce_score_margin))
if args.num_negs_per_system != 5:
    eval_filename += '_' + str(args.num_negs_per_system)
eval_filename += '_.json'

with open(eval_filename, 'w', encoding='utf8') as fOut:
    json.dump(eval_queries, fOut, indent=4, ensure_ascii=False, default=set_default)

