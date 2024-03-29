import sys
import torch
from datasets import load_dataset
from more_itertools import collapse
from tqdm import tqdm
import numpy as np
from dpr import DPR
import os
from typing import Iterable, Callable

dataset = load_dataset("ms_marco", 'v2.1')

def preperation_factory(tokenize_func: Callable) -> Callable:
    def prepare_data(strings: Iterable[str], prefix : str = "") -> dict:
        """
        Prepares data for the model.
        :param: strings are the strings to be tokenized
        :param: tokenize_func is a function that takes a string and returns a list of tokens
        """
        collapsed_text = list(collapse(strings))
        x = tokenize_func(collapsed_text)
        sequence_length = x.input_ids.shape[-1]
        return {
            prefix+'_inputs' : x.input_ids.to('cuda').view(-1, sequence_length).squeeze(),
            prefix+'_attn_mask' : x.attention_mask.to('cuda').view(-1, sequence_length).squeeze(),
        }
    return prepare_data

def get_embeds(model, suffix : str, save_identifier : str, dataloader : torch.utils.data.DataLoader, mbs : int = 4):
    """
    Get embeddings for all queries in the dataset. Saves to npy file.
    :param: model refers to the DPR model
    :param: suffix is the suffix to be added to the file name
    :param: save_to_dir is the directory to save the embeddings to
    """
    # for every batch in ms marco, embed it using the model
    positive_embeddings = []
    anchor_embeddings = []
    accuracy = []
    loss = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        try:
            with torch.no_grad():
                out_dict = model(batch, mbs=mbs, return_embeddings=True)

            batch_positive_tensors = out_dict['positive'].cpu().numpy()
            batch_anchor_tensors = out_dict['anchor'].cpu().numpy()

            accuracy.append(out_dict['acc'])
            loss.append(out_dict['loss'].item())
            
            # append to the list of embeddings
            positive_embeddings.append(batch_positive_tensors[None, ...])
            anchor_embeddings.append(batch_anchor_tensors[None, ...])
        except:
            continue

    # average both accuracy and loss
    accuracy =  sum(accuracy) / float(len(accuracy))
    loss = sum(loss) / float(len(loss))


    # save embeddings to an npy
    positive_embeddings = np.concatenate(positive_embeddings[:-1], axis=0)
    anchor_embeddings = np.concatenate(anchor_embeddings[:-1], axis=0)

    np.save('positive_embeddings_'+suffix+'/' + save_identifier + '.npy', positive_embeddings)
    np.save('anchor_embeddings_'+suffix+'/' + save_identifier + '.npy', anchor_embeddings)

    return accuracy, loss

