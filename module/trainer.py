from numpy import hamming
from .preprocessing import *
import module.model as M
from .evaluation import *
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging, datetime
import sklearn

class learning_env:
    def __init__(self, gpus, train_data, valid_data, test_data, split_directory, max_seq_len, log_directory, model_name, port, contain_context, data_label, **kwargs) -> None:
        self.gpus = gpus
        self.single_gpu = True if len(self.gpus) == 1 else False 
        
        self.train_dataset, self.valid_dataset, self.test_dataset = train_data, valid_data, test_data
        