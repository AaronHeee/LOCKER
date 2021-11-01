import random

import torch
import torch.utils.data as data_utils

import numpy as np


class Dataloader(object):
    def __init__(self, args, dataset):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.save_folder = args.data_path
        self.train = dataset.train
        self.val = dataset.val
        self.test = dataset.test
        self.umap = dataset.umap
        self.smap = dataset.smap
        self.num_items = args.num_items = len(self.smap)
        self.num_users = args.num_users = len(self.umap)
        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.CLOZE_MASK_TOKEN = self.num_items
        self.PAD_TOKEN = self.CLOZE_MASK_TOKEN + 1

    @classmethod
    def code(cls):
        return 'all'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_eval_loader(mode='val')
        test_loader = self._get_eval_loader(mode='test')
        return train_loader, val_loader, test_loader

    def _get_eval_loader(self, mode):
        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size
        answers = self.val if mode == 'val' else self.test
        cold_start = None
        dataset = BertAllEvalDataset(self.train, answers, self.max_len,
             self.CLOZE_MASK_TOKEN, self.PAD_TOKEN, mode=mode, val=self.val, cold_start=cold_start)
        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, drop_last=False,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def _get_train_loader(self):
        dataset = BertTrainDataset(self.train, self.max_len, self.mask_prob, 
                    self.CLOZE_MASK_TOKEN, self.num_items, self.rng, self.PAD_TOKEN, self.args.pad_size)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size, drop_last=True,
                                           shuffle=True, pin_memory=True)
        dataloader.pad_token = self.PAD_TOKEN
        return dataloader


class BertTrainDataset(data_utils.Dataset):
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng, pad_token, pad_size):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng
        self.pad_token = pad_token
        self.pad_size = pad_size

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]

        tokens = seq[-self.max_len:]
        labels = [-100] * len(tokens)
        index = np.arange(len(tokens) - int(np.ceil(len(tokens) / (1+self.pad_size))), len(tokens))
        np.random.shuffle(index)
        max_predict = min(int(self.max_len * self.mask_prob), max(1, int(round(len(index) * self.mask_prob))))

        for i in range(max_predict):
            labels[index[i]] = tokens[index[i]] 
            tokens[index[i]] = self.mask_token

        mask_len = self.max_len - len(tokens)

        tokens = tokens + [self.pad_token] * mask_len
        labels = labels + [-100] * mask_len

        return torch.LongTensor([user]), torch.LongTensor(tokens), torch.LongTensor(labels)



class BertAllEvalDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len, mask_token, pad_token, mode, val, cold_start):
        self.u2seq = u2seq
        self.users = list(u2seq.keys()) if cold_start is None else cold_start
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.mode = mode
        self.val = val

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] if self.mode == "val" else self.u2seq[user] + self.val[user]
        answer = self.u2answer[user]

        seq = seq + [self.mask_token]
        seq = seq[-self.max_len:]
        length = len(seq)
        padding_len = self.max_len - length
        seq = seq + [self.pad_token] * padding_len
        hyper_labels = list(range(self.max_len))

        hyper_labels[length-1] = answer[0]

        return torch.LongTensor([user]), torch.LongTensor(seq), torch.LongTensor(answer), torch.LongTensor([length-1]), torch.LongTensor(hyper_labels)
