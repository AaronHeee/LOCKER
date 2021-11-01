# LOCKER
Locally Constrained Self-Attentive Sequential Recommendation

This is the pytorch implementation of this paper:

[Locally Constrained Self-Attentive Sequential Recommendation](https://cseweb.ucsd.edu/~jmcauley/pdfs/cikm21.pdf). CIKM 2022, short.

Please cite our paper if using this code. This code can run on single 2080 super GPU (8GiB).

## Dependencies

```
pip install -r src/requirement.txt
```

## Start Training

BERT Baseline (on `Beauty`):
```
python main.py --data_path "data/Beauty" --local_type 'none' --weight_decay 1.0 --bert_dropout 0.2 --bert_att_dropout 0.2 --bert_mask_prob 0.6 --experiment_dir "experiments/" --experiment_description "bert" --bert_max_len 50
```
Note that this implementation of bert4rec is slightly different from the [original implementation](https://github.com/FeiSun/BERT4Rec), by removing the projection layer and simplifing the training data loading (where our implementation ranking performance is slightly stronger than the original one).

LOCKER Model (on `Beauty`):
```
python main.py --data_path "data/Beauty" --local_type adapt --weight_decay 1.0 --bert_dropout 0.2 --bert_att_dropout 0.2 --bert_mask_prob 0.6 --experiment_dir "experiments/" --experiment_description "adapt" --bert_max_len 50
```

Here for local encoder, we can choose `rnn`, `conv`, `win`, `initial` or `adapt`. You can check `src/models/locker_model.py` for implementation details.

## Acknowledgement

During the implementation we start our code from [BERT4Rec](https://github.com/jaywonchung/BERT4Rec-VAE-Pytorch) by Jaewon Chung. Many thanks to the author for the great work!