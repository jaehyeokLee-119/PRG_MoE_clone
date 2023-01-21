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
        self.gpus = gpus # 사용할 GPU 번호들
        self.single_gpu = True if len(self.gpus) == 1 else False # gpu 개수가 1개인지 아닌지를 저장 (여러개면 multiprocessing해야 함)  
        self.train_dataset = train_data
        self.valid_dataset = valid_data
        self.test_dataset = test_data
        
        self.split_directory = split_directory
        self.split_num = None
        
        self.contain_context = contain_context  # ?
        self.max_seq_len = max_seq_len      # 최대 sequence 길이
       
        self.start_time = datetime.datetime.now()
        self.log_directory = log_directory
        
        self.options = kwargs
        self.model_name = model_name
        self.port = port
        
        self.split_performance = None
        self.data_label = data_label
        self.best_performance = [0, 0, 0] # precision, recall, F1-score
    
    
    # 모델을 셋팅 (모델에 들어갈 매개변수를 설정, 모델을 불러오고 정의해서 리턴)
    def __set_model__(self, pretraind_model, dropout, n_speaker, n_emotion, n_cause, n_expert, guiding_lambda, **kwargs):
        self.n_cause = n_cause
        model_args = {'dropout': dropout,
                      'n_speaker': n_speaker,
                      'n_emotion': n_emotion,
                      'n_cause': n_cause,
                      'n_expert': n_expert,
                      'guiding_lambda': guiding_lambda
                      }
        
        if pretraind_model != None:
            model = getattr(M, self.model_name)(**model_args)
            model.load_state_dict(torch.load(pretraind_model))
            return model
        else:
            model = getattr(M, self.model_name)(**model_args)
            return model
    
    def set_model(self, allocated_gpu):
        # single gpu든, multi gpu든 입력한 설정값대로 밖에서 보면 동일하게 동작하도록 나눠서 처리해줌
        if not self.single_gpu:
            pass
            # single_gpu가 아닌 경우를 설정 (나중에)
        
        model = self.__set_model__(**self.options).cuda(allocated_gpu)
        self.model = model
    
    def work(self, test, training_iter, batch_size, learning_rate, patience, num_worker, **kwargs):
        stopper = torch.zeros(1)
        print(stopper)
        
        self.child_process(0, training_iter, batch_size, learning_rate, patience, num_worker, stopper, None, test)
        
    def run(self, **kwargs):
        print("print args")
        print(kwargs)
        self.work(**kwargs)
        