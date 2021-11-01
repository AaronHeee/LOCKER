import json
import os
import sys

DIR = sys.argv[1] if len(sys.argv)>1 else None
METRICS = ['Recall@5', 'Recall@10', 'Recall@20', 'NDCG@5', 'NDCG@10', 'NDCG@20', 'MRR', 'AUC']

for path in [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(DIR)) for f in fn]:
    if 'test_metrics.json' in path:
        try:
            test = json.load(open(path, 'r'))
            print(path.replace(DIR, "").replace('/logs/test_metrics.json', '') + "," + ','.join(["%.4f" % test[M] for M in METRICS]))
        except:
            pass