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
import wandb

# set up our arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_folder", type=str, default='msmarco-data', help="Path to the data folder")
parser.add_argument("--train_file", type=str, default='train_queries_5.0_.json', help="File containing hard negatives.")
parser.add_argument('--validation_file', type=str, default='eval_queries_5.0_.json', help="File containing hard negatives.")

args = parser.parse_args()

print(args)
data_folder = args.data_folder

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
    tokenize_func, epochs = 1, update_every=10, 
    grad_accum=1, validate=100, mbs = 4):

    model.to('cuda')
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    accuracies = list()
    losses = list()

    for epoch in (pbar := tqdm(range(epochs))):
        for idx, batch in tqdm(enumerate(train_dataloader)):
            try:
                out_dict = model(batch, mbs=mbs)
                loss = out_dict['loss']
                acc = out_dict['acc']

                wandb.log({'Train/Loss': loss, 'Train/Acc': acc})

                loss.backward()
            except:
                continue
            
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
                    avg_acc, avg_loss = get_embeds(model, tokenize_func, save_to_dir=f"embeds/{epoch}_{idx}", dataloader=validation_dataloader)

                wandb.log({'Val/Acc' : avg_acc, 'Val/Loss': avg_loss})

                accuracies.append(avg_acc)
                losses.append(avg_loss)

                # save accuracies and losses to a .npy
                np.save(f"accuracies.npy", np.array(accuracies))
                np.save(f"losses.npy", np.array(losses))
                model.train()

if __name__ == "__main__":

    # set the seed 
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # load the train file
    with open(os.path.join(data_folder, args.train_file), 'r') as f:
        train_data = json.load(f)
    
    with open(os.path.join(data_folder, args.validation_file), 'r') as f:
        validation_data = json.load(f)
    
    wandb.init(project="varying hardness dpr msmarco", name="Hardness 5 Declutr B")
    wandb.config.update(args)
    
    # initialize the dataset
    train_dataset = MSMARCODataset(train_data, corpus)
    validation_dataset = MSMARCODataset(validation_data, corpus)
    print(len(train_dataset))
    print(len(validation_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=True, num_workers=4)

    train(DPR(), train_dataloader, validation_dataloader, train_dataset.tokenize)

