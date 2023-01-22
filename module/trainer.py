from numpy import hamming
from .preprocessing import *
import module.model as M
from .evaluation import *
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging, datetime
import sklearn
import torch

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
    # __는__ 내장함수라는 뜻 (private)
    def __set_model__(self, pretrained_model, dropout, n_speaker, n_emotion, n_cause, n_expert, guiding_lambda, **kwargs):
        self.n_cause = n_cause
        model_args = {'dropout': dropout,
                      'n_speaker': n_speaker,
                      'n_emotion': n_emotion,
                      'n_cause': n_cause,
                      'n_expert': n_expert,
                      'guiding_lambda': guiding_lambda
                      }
        
        if pretrained_model != None:
            model = getattr(M, self.model_name)(**model_args)
            model.load_state_dict(torch.load(pretrained_model))
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
    
    def init_stopper(self):
        self.stopper[0] = 0
        
    def work(self, test, training_iter, batch_size, learning_rate, patience, num_worker, **kwargs):
        stopper = torch.zeros(1) # : tensor([0]) 하나? 왜 쓴건지 모르겠다
        self.child_process(0, training_iter, batch_size, learning_rate, patience, num_worker, stopper, None, test)
    
    def child_process(self, allocated_gpu, training_iter, batch_size, learning_rate, patience, num_worker, stopper, split_performance, test=False):
        print('allocated_gpu: ',allocated_gpu)
        self.set_model(allocated_gpu)
        
        if test:
            self.valid(allocated_gpu, batch_size, num_worker, saver=None, option='test')
        else:
            self.train(allocated_gpu, training_iter, batch_size, learning_rate, patience, num_worker)
            self.valid(allocated_gpu, batch_size, num_worker, saver=None, option='test')
    
    def get_dataloader(self, dataset_file, batch_size, num_worker, shuffle=True, contain_context=False):
        # dataset_file 파일명대로 데이터를 불러와서 DataLoader를 리턴함
        (utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t), /
        speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t = get_data(dataset_file, f"cuda:0", self.max_seq_len, contain_context)
        dataset_ = TensorDataset(utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t, speaker_t, emotion_label_t, pair_cause_label_t, pair_binary_cause_label_t)
        
    
    def train(self, allocated_gpu, training_iter, batch_size, learning_rate, patience, num_worker):
        def get_pad_idx(utterance_input_ids_batch):
            pass
        def get_pair_pad_idx(utterance_input_ids_batch, window_constraint=3, emotion_pred=None):
            pass
        
        if allocated_gpu == 0:
            self.init_stopper()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                lr_lambda=lambda epoch: 0.95 ** epoch,
                                                last_epoch=-1,
                                                verbose=False)    
        train_dataloader = self.get_dataloader(self.train_dataset, batch_size, num_worker)
        
    def run(self, **kwargs):
        print(kwargs)
        self.work(**kwargs)
        