from dpr import MSMARCODataset, WikiNQDataset, DPR

import argparse
from get_embeds import get_embeds
import json
import logging
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# set up our arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default='msmarco-data', help="Path to the data folder")
parser.add_argument("--train_file", type=str, default='train_queries_', help="File containing hard negatives.")
parser.add_argument('--validation_file', type=str, default='eval_queries_', help="File containing hard negatives.")
parser.add_argument('--hardness', type=float, default=2.0, help="Hardness of the negative sampling.")
parser.add_argument('--num_negs', type=int, default=5, help="Number of negative samples to use.")
parser.add_argument('--cuda_visible_device', type=str, default='0', help="CUDA visible device")
args = parser.parse_args()

print(args)
data_folder = args.data_folder

# set the cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_device

# load the corpus
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages
collection_filepath = os.path.join(data_folder, 'collection.tsv')

logging.info("Read corpus: collection.tsv")
with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        pid = int(pid)
        corpus[pid] = passage

def train(model, train_dataloader, validation_dataloader,
    epochs = 1, update_every=10, 
    grad_accum=1, validate=100, mbs = 4):

    model.to('cuda')
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    accuracies = list()
    losses = list()
    count_total = len(train_dataloader) * epochs
    count = 0
    for epoch in (pbar := tqdm(range(epochs))):
        for idx, batch in tqdm(enumerate(train_dataloader)):

            out_dict = model(batch, mbs=mbs)
            loss = out_dict['loss']
            acc = out_dict['acc']

            wandb.log({'Train/Loss': loss, 'Train/Acc': acc})

            loss.backward()
            
            
            if (idx) % grad_accum == 0:
                optim.step()
                optim.zero_grad()

            if (idx) % update_every == 0:
                pbar.set_description(f"Loss: {loss.item():.4f}")

            if (idx) % validate == 0:
                # first we want to get the embeddings
                model.eval()
                print(f"Getting embeddings for embeds/{epoch}_{idx} and validating...")
                with torch.no_grad():
                    avg_acc, avg_loss = get_embeds(model, suffix=str(args.num_negs), save_identifier=str(float(count)/float(count_total)), dataloader=validation_dataloader)

                wandb.log({'Val/Acc' : avg_acc, 'Val/Loss': avg_loss})

                accuracies.append(avg_acc)
                losses.append(avg_loss)

                # save accuracies and losses to a .npy
                np.save(f"accuracies_"+str(args.num_negs)+".npy", np.array(accuracies))
                np.save(f"losses_"+str(args.num_negs)+".npy", np.array(losses))
                model.train()
            count += 1

if __name__ == "__main__":

    # set the seed 
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    wandb.init(project="varying hardness dpr msmarco", name="Hardness " + str(args.hardness) + " Num Negs" + str(args.num_negs) + " Roberta B")
    wandb.config.update(args)
    
    # initialize the dataset
    train_dataset = WikiNQDataset("wiki_dpr_full_es.jsonl", corpus)
    validation_dataset = MSMARCODataset("wiki_dpr_full_es_val.jsonl", corpus)
    print(len(train_dataset))
    print(len(validation_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=True, num_workers=4)

    train(DPR(), train_dataloader, validation_dataloader)

