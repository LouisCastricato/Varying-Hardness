# this file implements the dataset and model class for dpr using declutr

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


class DPR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, tokenizer_dir = "johngiorgi/declutr-base", max_len=512):
        self.data_dir = data_dir
        self.tokenizer = AutoTokenizer(tokenizer_dir)
        self.max_len = max_len

        self.data = []
        self.load_data()
    
    def load_data(self):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # elem is a lot of strings, where the first example if the anchor, second is the positive, and the rest are negatives
        elem = self.data[index]
        # tokenize elem
        elem = self.tokenizer.encode(elem, max_length=self.max_len, return_tensors="pt")
        return {
            "anchor": elem[0],
            "positive": elem[1],
            "negative": elem[2:]
        }

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

        # positive is dimension bs x projection_dim, negative is bs x cbs - 1 x projection_dim
        # so unsqueeze positive and concat it onto negatives
        contrastive_batch = torch.cat((positive.unsqueeze(1), negative), dim=1)

        # compute the cosine sim of the anchor and the contrastive batch.
        # anchor is bs x projection_dim, contrastive_batch is bs x cbs x projection_dim
        # so we can do a matrix multiplication
        sim = torch.matmul(anchor, contrastive_batch.transpose(1,2))


