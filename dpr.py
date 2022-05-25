# this file implements the dataset and model class for dpr using declutr

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from torch.utils.data import Dataset
import random

# taken from sentence transformers example by modified
class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus, tokenizer_dir = "johngiorgi/declutr-base"):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir)

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        return {
            'anchor': self.tok(query_text, return_tensors='pt'),
            'positive': self.tok(pos_text, return_tensors='pt'),
            'negative': self.tok(neg_text, return_tensors='pt'),
        }

    def __len__(self):
        return len(self.queries)

class DPR(torch.nn.Module):
    def __init__(self, project_dim = 300):
        super(DPR, self).__init__()
        self.project_dim = project_dim
        self.lm = AutoModelForMaskedLM.from_pretrained("johngiorgi/declutr-base")
        self.projection = torch.nn.Linear(self.project_dim, self.project_dim)

    def forward(self, x):
        """
        Computes a forward step of DPR. Does not compute loss. To compute loss, take the NLL of the 0th component.
        :param x: a dictionary of tensors, where the keys are "anchor", "positive", and "negative"
        :return: A tensor of bs x cbs, where dimension [:,0] should be minimized
        """
        # adjust the batch size to compensate for lost efficiency here
        anchor = self.lm(x["anchor"])
        positive = self.lm(x["positive"])
        negative = self.lm(x["negative"])

        # project the embeddings
        anchor = self.projection(anchor)
        positive = self.projection(positive)
        negative = self.projection(negative)

        # positive is dimension bs x projection_dim, negative is bs x projection_dim
        # What we want to do is concat positives onto negatives, but on the ith negative do not concat the ith positive
        # we can do this by duplicaating positive by bs 
        positives_duplicated = torch.cat([positive.unsqueeze(1)] * positive.shape[0], dim=1)
        # and then on the second dimension on the ith index, we replace it with the ith negative
        contrastive_batch = torch.scatter(positive, 1, torch.arange(positive.shape[0]).unsqueeze(1), negatives) # ~ bs x bs x projection_dim
        # concat positives onto the contrastive batch
        contrastive_batch = torch.cat([positive.unsqueeze(1), contrastive_batch], dim=1)


        # compute the cosine sim of the anchor and the contrastive batch.
        # anchor is bs x projection_dim, contrastive_batch is bs x bs+1 x projection_dim
        # so we can do a matrix multiplication
        sim = torch.matmul(anchor, contrastive_batch.transpose(1,2))

        return sim


