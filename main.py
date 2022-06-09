from dpr import MSMARCODataset, DPR

import argparse
from get_embeds import get_embeds
import json
import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# set up our arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default='msmarco-data', help="Path to the data folder")
parser.add_argument("--train_file", type=str, default='train_queries_5.0_.json', help="File containing hard negatives.")

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

def train(model, dataloader, tokenize_func, epochs = 3, update_every=10, grad_accum=1, get_emebds_every=100, mbs = 2):
    model.to('cuda')
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    for epoch in (pbar := tqdm(range(epochs))):
        for idx, batch in tqdm(enumerate(dataloader)):
            loss, _ = model(batch, accumulate=True, mbs=mbs)
            loss.backward()
            
            if idx % grad_accum == 0:
                optim.step()
                optim.zero_grad()

            if idx % update_every == 0:
                pbar.set_description(f"Loss: {loss.item():.4f}")

            if idx % get_emebds_every == 0:
                model.eval()
                print(f"Getting embeddings for embeds/{epoch}_{idx}")
                with torch.no_grad():
                    get_embeds(model, tokenize_func, save_to_dir=f"embeds/{epoch}_{idx}")
                model.train()

if __name__ == "__main__":

    # set the seed 
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # load the train file
    with open(os.path.join(data_folder, args.train_file), 'r') as f:
        train_data = json.load(f)
    
    # initialize the dataset
    dataset = MSMARCODataset(train_data, corpus)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

    train(DPR(), dataloader, dataset.tokenize)

