# this file implements the dataset and model class for dpr using declutr

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.utils.data import Dataset
import random

def masked_average(out, mask):
    mask = mask.squeeze()
    return torch.sum(out * mask.unsqueeze(-1), dim=1) / torch.clamp(
    torch.sum(mask, dim=1, keepdims=True), min=1e-9)

# taken from sentence transformers example and modified
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, tokenizer_dir = "johngiorgi/declutr-base", max_length=512):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.max_length = max_length

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def tokenize(self, text):
        return self.tok(text,
            return_tensors='pt',
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length')

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = self.tokenize(query['query'])

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.tokenize(self.corpus[pos_id])
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.tokenize(self.corpus[neg_id])
        query['neg'].append(neg_id)

        return {
            'anchor_inputs': query_text.input_ids,
            'anchor_attn_mask' : query_text.attention_mask,
            'positive_inputs' : pos_text.input_ids,
            'positive_attn_mask' : pos_text.attention_mask,
            'negative_inputs' : neg_text.input_ids,
            'negative_attn_mask' : neg_text.attention_mask,
        }

    def __len__(self):
        return len(self.queries)

class DPR(torch.nn.Module):
    def __init__(self, project_dim = 768):
        super(DPR, self).__init__()
        self.project_dim = project_dim
        self.lm = AutoModelForMaskedLM.from_pretrained("johngiorgi/declutr-base")
        self.projection = torch.nn.Linear(self.project_dim, self.project_dim)

    def embed(self, toks, attn_mask=None):
        embds = self.lm(toks.squeeze()).hidden_states[0]
        if attn_mask is None:
            attn_mask = torch.ones(embeds.shape[:2], dtype=torch.int)
        return masked_average(embds, attn_mask)
    def loss(self, anchor, contrastive_batch):
        """
        Compute InfoNCE
        :param anchor: anchor embeddings
        :param contrastive_batch: batch of contrastive embeddings
        :return:
        """
        # compute the cosine sim of the anchor and the contrastive batch.
        # anchor is bs x projection_dim, contrastive_batch is bs x bs+1 x projection_dim
        # so we can do a matrix multiplication
        sim = -torch.nn.functional.log_softmax(torch.matmul(anchor.unsqueeze(1), contrastive_batch.transpose(1,2)).squeeze(), dim=1)

        # for the ith batch, get the ith component
        # sim is bs x bs+1
        return torch.trace(sim) / sim.shape[0], sim

    def forward(self, x, accumulate = True, mbs = 32):
        """
        Computes a forward step of DPR. Does not compute loss. To compute loss, take the NLL of the 0th component.
        :param x: a dictionary of tensors, where the keys are "anchor", "positive", and "negative"
        :param accumulate: whether to accumulate the gradients. skips microbatching
        :return: A tensor of bs x cbs, where dimension [:,0] should be minimized
        """
        for k,v in x.items():
            x[k] = v.to("cuda")
        # microbatch over the batch dimension
        anchor_batch = list()
        pos_batch = list()
        neg_batch = list()
        
        bs = x['anchor_inputs'].shape[0]
        for mbs_idx in range(0, bs, mbs):
            # get input ids for the anchor, positive, and negative
            anchor_inputs_mb  = x['anchor_inputs'][mbs_idx:mbs_idx+mbs]
            positive_inputs_mb = x['positive_inputs'][mbs_idx:mbs_idx+mbs]
            negative_inputs_mb = x['negative_inputs'][mbs_idx:mbs_idx+mbs]

            # incase bs is one
            try:
                # forward pass
                anchor_i  = self.lm(anchor_inputs_mb.squeeze()).hidden_states[0]
                positive_i = self.lm(positive_inputs_mb.squeeze()).hidden_states[0]
                negative_i = self.lm(negative_inputs_mb.squeeze()).hidden_states[0]
            except:
                continue

            # projeect
            anchor_i = self.projection(anchor_i)
            positive_i = self.projection(positive_i)
            negative_i = self.projection(negative_i)

            # average
            anchor_batch.append(masked_average(anchor_i, x['anchor_attn_mask'][mbs_idx:mbs_idx+mbs]))
            pos_batch.append(masked_average(positive_i, x['positive_attn_mask'][mbs_idx:mbs_idx+mbs]))
            neg_batch.append(masked_average(negative_i, x['negative_attn_mask'][mbs_idx:mbs_idx+mbs]))

        # stack the mbs above
        anchor = torch.stack(anchor_batch).view(-1, self.project_dim)
        positive = torch.stack(pos_batch).view(-1, self.project_dim)
        negative = torch.stack(neg_batch).view(-1, self.project_dim)
        
        # positive is dimension bs x projection_dim, negative is bs x projection_dim
        positives_duplicated = torch.cat([positive.unsqueeze(1)] * positive.shape[0], dim=1).transpose(0,1)
        contrastive_batch = torch.cat([positives_duplicated, negative.unsqueeze(1)], dim=1)

        # if we want to skip microbatching
        if accumulate:
            return self.loss(anchor, contrastive_batch)
        else:
            # compute loss in the training loop rather than here
            return anchor, contrastive_batch

