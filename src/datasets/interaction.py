import os
import json

USER_COL, ITEM_COL, RATING_COL, TIMESTAMP_COL = 0, 1, 2, 3

class ItemDataset(object):
    def __init__(self, args):
        self.path = args.data_path
        self.train = self.read_json(os.path.join(self.path, "train.json"), True)
        self.val = self.read_json(os.path.join(self.path, "val.json"), True)
        self.test = self.read_json(os.path.join(self.path, "test.json"), True)
        self.data = self.merge(self.train, self.val, self.test)
        self.umap = self.read_json(os.path.join(self.path, "umap.json"))
        self.smap = self.read_json(os.path.join(self.path, "smap.json"))

    def merge(self, a, b, c):
        data = {}
        for i in c:
            data[i] = a[i] + b[i] + c[i]
        return data

    def read_json(self, path, as_int=False):
        with open(path, 'r') as f:
            raw = json.load(f)
            if as_int:
                data = dict((int(key), value) for (key, value) in raw.items())
            else:
                data = dict((key, value) for (key, value) in raw.items())
            del raw
            return data
    
    @classmethod
    def code(cls):
        return "item"
