import torch
import numpy as np
import random
import argparse
import os
import json, csv
import torch.backends.cudnn


# Reproducibility
random_seed = 77
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

parser = argparse.ArgumentParser(description='This code is for ECPE task.')

# Training Environment
parser.add_argument('--gpus', default=[0])
parser.add_argument('--num_process', default=int(os.cpu_count() * 0.8)) 
parser.add_argument('--num_worker', default=6)
parser.add_argument('--port', default=1234)

parser.add_argument('--model_name', default='DFC_MoE')
parser.add_argument('--pretrained_model', default="./model/PRG_MoE-binary_cause--original_data_DailyDialog-2023-01-11_202517.pt") # 만들어진 모델 써서
parser.add_argument('--test', default=True) # 테스트

parser.add_argument('--split_directory', default=None)
parser.add_argument('--train_data', default="data/dailydialog_train.json")
parser.add_argument('--valid_data', default="data/dailydialog_valid.json")
parser.add_argument('--test_data', default="data/dailydialog_test.json")

parser.add_argument('--log_directory', default=None)
parser.add_argument('--data_label', help='the label that attaches to saved model', default='')

parser.add_argument('--dropout', help='dropout probability', default=0.5)
parser.add_argument('--n_speaker', help='the number of speakers', default=2)
parser.add_argument('--n_emotion', help='the number of emotions', default=7)
parser.add_argument('--n_cause', help='the number of causes', default=2)
parser.add_argument('--n_expert', help='the number of experts', default=4)
parser.add_argument('--guiding_lambda', help='the mixing ratio', default=0.6)


parser.add_argument('--max_seq_len', help='the max length of each tokenized utterance', default=75)
parser.add_argument('--contain_context', help='While tokenizing, previous utterances are contained or not', default=False)

# Hyperparameters
parser.add_argument('--training_iter', default=40)
parser.add_argument('--batch_size', default=5)
parser.add_argument('--learning_rate', default=5e-5)
parser.add_argument('--patience', help='patience for Early Stopping', default=None)

args = parser.parse_args()
args_dict = vars(args)

if args.test:
    assert args.pretrained_model != None, "You should load model for test"


if __name__ == '__main__':
    train_data = './data_fold/data_0/dailydialog_train.json'
     





