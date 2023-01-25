import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel

class guided_moe_basic(nn.Module):
    def __init__(self, dropout=0.5, n_speaker=2, n_emotion=7, n_cause=2, n_expert=2, guiding_lambda=0, **kwargs):
        super(guided_moe_basic, self).__init__()
        
        self.n_expert = n_expert
        self.guiding_lambda = guiding_lambda
        self.bert = BertModel.from_pretrained('bert-base-cased')
        
        # 감정분류 linear layer, 768->7 
        self.emotion_linear = nn.Linear(self.bert.config.hidden_size, n_emotion)
        # MoE로 Gating하는 Linear, 1552->4
        self.gating_network = nn.Linear(2*(self.bert.config.hidden_size + n_emotion + 1), n_expert) 
            # 여기 input 크기가 왜?
            # gating_network로 입력
        
        # Linear(1552->256) -> Linear(256->2) 짜리가 병렬로 4개  
        self.cause_linear = nn.ModuleList()
        for _ in range(n_expert):
            self.cause_linear.append(nn.Sequential(
                nn.Linear(2 * (self.bert.config.hidden_size + n_emotion +1), 256),
                nn.Linear(256, n_cause)
            )) # Input: Cause Utterance + Emotion Prediction + Speaker Information (Sparse)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, token_type_ids, speaker_ids):
        emotion_pred = self.emotion_classification_task(input_ids, attention_mask, token_type_ids)
        cause_pred = self.binary_cause_classification_task(emotion_pred, input_ids, attention_mask, token_type_ids, speaker_ids)
        
        return (emotion_pred, cause_pred)
    
    def emotion_classification_task(self, input_ids, attention_mask, token_type_ids):
        batch_size, max_doc_len, max_seq_len = input_ids.shape
        breakpoint()

class PRG_MoE(guided_moe_basic):
    def __init__(self, dropout=0.5, n_speaker=2, n_emotion=7, n_cause=2, n_expert=4, guiding_lambda=0, **kwargs):
        super().__init__(dropout=dropout, n_speaker=n_speaker, n_emotion=n_emotion, n_cause=n_cause, n_expert=4, guiding_lambda=guiding_lambda)
