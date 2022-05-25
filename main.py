from dpr import MSMARCODataset, DPR
import argparse
import json
import logging
import os
from torch.utils.data import DataLoader

# set up our arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default='msmarco-data', help="Path to the data folder")
parser.add_argument("--train_file", type=str, default='train_queries_3.0_.json', help="File containing hard negatives.")

args = parser.parse_args()

print(args)
data_folder = args.data_folder

# load rthe corpus
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage

def train(model, dataloader):
    model.train()
    for batch in dataloader:
        contrastive_matrix = model(batch)
        print(contrastive_matrix)


if __name__ == "__main__":
    # load the train file
    with open(os.path.join(data_folder, args.train_file), 'r') as f:
        train_data = json.load(f)
    
    # initialize the dataset
    dataset = MSMARCODataset(train_data, corpus)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    train(DPR(), dataloader)

