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
    def prepare_data(strings: Iterable[str]) -> dict:
        """
        Prepares data for the model.
        :param: strings are the strings to be tokenized
        :param: tokenize_func is a function that takes a string and returns a list of tokens
        """
        collapsed_text = list(collapse(strings))
        x = tokenize_func(collapsed_text)
        sequence_length = x.input_ids.shape[-1]
        return {
            'toks' : x.input_ids.to('cuda').view(-1, sequence_length).squeeze(),
            'attn_mask' : x.attention_mask.to('cuda').view(-1, sequence_length).squeeze(),
        }
    return prepare_data

def get_embeds(model, tokenize_func : Callable, save_to_dir : str):
    """
    Get embeddings for all queries in the dataset. Saves to npy file.
    :param: model refers to the DPR model
    :param: tokenize_func is a function that takes a string and returns a list of tokens
    :param: save_to_dir is the directory to save the embeddings to
    """
    # get the data preperation function
    prepare_data = preperation_factory(tokenize_func)

    # for every batch in ms marco, embed it using the model
    query_embeddings = []
    query_embeddings_no_dupe = []
    answer_embeddings = []
    answer_embeddings_no_dupe = []
    passage_embeddings = []

    bs = int(10)
    N = int(1e5)
    for i in tqdm(range(0, N, bs)):
        batch = dataset['validation'][i:i+bs]
        
        # get only the queries from the batch
        batch_queries = list(collapse([q for q in batch['query']]))

        # get only the answers from the batch
        batch_answers = list(collapse([a for a in batch['answers']]))

        # get only the passages from the batch
        batch_passages = [p['passage_text'] for p in batch['passages']]

        with torch.no_grad():
            # embed. we need to convert the first two to a list so that we can expand below
            batch_queries_tensors = model.embed(**prepare_data(batch_queries)).tolist()
            batch_answers_tensors = model.embed(**prepare_data(batch_answers)).tolist()
            passage_embeddings.append(model.embed(**prepare_data(batch_passages)))

        # expand batch queries and batch answers to the size of batch passages
        e_queries = []; [e_queries := e_queries + [q] * len(b) for q, b in zip(batch_queries_tensors, batch_passages)]
        e_answers = []; [e_answers := e_answers + [a] * len(b) for a, b in zip(batch_answers_tensors, batch_passages)]
        
        # append to the list of embeddings. convert back to a numpy array too
        query_embeddings.append(np.array(e_queries))
        answer_embeddings.append(np.array(e_answers))



    # save embeddings to an npy
    query_embeddings = np.concatenate(query_embeddings, axis=0)
    answer_embeddings = np.concatenate(answer_embeddings, axis=0)
    passage_embeddings = np.concatenate(passage_embeddings, axis=0)

    base_path = os.path.join(save_to_dir, 'ms_marco_')
    np.save(base_path + 'query_embeddings_v1.npy', query_embeddings)
    np.save(base_path + 'answer_embeddings_v1.npy', answer_embeddings)
    np.save(base_path + 'passage_embeddings_v1.npy', passage_embeddings)

