from src.datasets import DATASETS
from src.dataloaders import DATALOADERS
from src.models import MODELS
from src.trainers import TRAINERS

import argparse

parser = argparse.ArgumentParser(description='Bert4Rec')

################
# Test
################
parser.add_argument('--test_model_path', type=str, default=None)
parser.add_argument('--load_pretrained_weights', type=str, default=None)

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default='item', choices=DATASETS.keys())
parser.add_argument('--split', type=str, default='leave_one_out', help='How to split the datasets')
parser.add_argument('--dataset_split_seed', type=int, default=98765)
parser.add_argument('--data_path', type=str, default=None)

################
# Dataloader
################
parser.add_argument('--dataloader_code', type=str, default='all', choices=DATALOADERS.keys())
parser.add_argument('--dataloader_random_seed', type=float, default=0.0)
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--val_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--pad_size', type=float, default=0)

################
# Trainer
################
parser.add_argument('--trainer_code', type=str, default='all', choices=TRAINERS.keys())
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1)
parser.add_argument('--device_idx', type=str, default='0')
# optimizer #
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'AdamW','Adam'])
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='l2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--warmup_steps', type=int, default=100, help='Warmup Steps')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Adam Epsilon')

# lr scheduler #
parser.add_argument('--enable_lr_schedule', type=str, default="linear", help='Enable lr scheduler')
parser.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs for training')
parser.add_argument('--stop_epochs', type=int, default=1000, help='Number of epochs to stop, smaller than num_epochs')
parser.add_argument('--verbose', type=int, default=10)
# evaluation #
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--best_metric', type=str, default='NDCG@10', help='Metric for determining the best model')

################
# Model
################
parser.add_argument('--model_code', type=str, default='locker', choices=MODELS.keys())
parser.add_argument('--model_init_seed', type=int, default=98765)
# BERT #
parser.add_argument('--bert_max_len', type=int, default=None, help='Length of sequence for bert')
parser.add_argument('--bert_hidden_dim', type=int, default=64, help='Size of hidden vectors (d_model)')
parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
parser.add_argument('--init_val', type=float, default=1, help='initial value for hyper-param opt')
parser.add_argument('--bert_num_heads', type=int, default=4, help='Number of heads for multi-attention')
parser.add_argument('--bert_dropout', type=float, default=0.2, help='Dropout probability to use throughout the model')
parser.add_argument('--bert_att_dropout', type=float, default=0.2, help='Dropout probability to use throughout the attention scores')
parser.add_argument('--bert_mask_prob', type=float, default=0.6, help='Probability for masking items in the training sequence')
# LOCKER #
parser.add_argument('--local_type', type=str, default='none', help='Types of localness bias')
parser.add_argument('--local_num_heads', type=int, default=3, help='Local att heads num')

################
# Experiment
################
parser.add_argument('--experiment_dir', type=str, default='experiments')
parser.add_argument('--experiment_description', type=str, default='test')

args = parser.parse_args()

