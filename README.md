# LOCKER

This is the pytorch implementation of this paper:

[Locker: Locally Constrained Self-Attentive Sequential Recommendation](https://dl.acm.org/doi/abs/10.1145/3459637.3482136). Zhankui He, Handong Zhao, Zhe Lin, Zhaowen Wang, Ajinkya Kale, Julian McAuley.
Conference on Information and Knowledge Management (CIKM) 2021, Short Paper.

Please cite our paper if using this code. This code can run on single 2080 super GPU (8GiB).

<img src="https://github.com/AaronHeee/LOCKER/blob/main/img/curve.jpg" width="400"/>

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
