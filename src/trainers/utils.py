import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

MAX_VAL = 1e4
THRESHOLD = 0.5

class Ranker(nn.Module):
    def __init__(self, metrics_ks, user2seq, args=None):
        super().__init__()
        self.ks = metrics_ks
        self.ce = nn.CrossEntropyLoss()
        self.user2seq = user2seq
        self.args = args

    def forward(self, scores, labels, lengths=None, seqs=None, users=None):
        labels = labels.squeeze(-1)
        loss = self.ce(scores, labels)
        predicts = scores[torch.arange(scores.size(0)), labels].unsqueeze(-1) # gather perdicted values
        if seqs is not None:
            scores[torch.arange(scores.size(0)).unsqueeze(-1), seqs] = -MAX_VAL # mask the rated items
        if users is not None:
            for i in range(len(users)):
                scores[i][self.user2seq[users[i].item()]] = -MAX_VAL
        if 'motivating' in self.args.data_path:
            scores = scores[:, :-100]
        valid_length = (scores > -MAX_VAL).sum(-1).float()
        rank = (predicts < scores).sum(-1).float()
        res = []
        for k in self.ks:
            indicator = (rank < k).float()
            res.append(
                ((1 / torch.log2(rank+2)) * indicator).mean().item() # ndcg@k
            ) 
            res.append(
                indicator.mean().item() # hr@k
            )
        res.append((1 / (rank+1)).mean().item()) # MRR
        return res
