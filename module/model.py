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
        
        # input 3종을 view로 펴서 bert에 넣음
        _, utterance_representation = self.bert(input_ids=input_ids.view(-1, max_seq_len), attention_mask=attention_mask.view(-1, max_seq_len), token_type_ids=token_type_ids.view(-1, max_seq_len), return_dict=False)
        '''
            _는 token representation [31,75,768] (31개의 max_utterances, 75개의 max_tokens, 768의 dimension)
            utternace_representation은 pooled_output (output의 [CLS])
        '''
        
        dropoutted_representation = self.dropout(utterance_representation)
        emotion_representation = self.emotion_linear(dropoutted_representation)
        
        return emotion_representation
        
    def binary_cause_classification_task(self, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids):
        pair_embedding = self.get_pair_embedding(emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids)
        gating_prob = self.gating_network(pair_embedding.view(-1, pair_embedding.shape[-1]).detach())

        # gating_prob combination
        gating_prob = self.guiding_lambda * self.get_subtask_label(input_ids, speaker_ids, emotion_prediction).view(-1, self.n_expert) + (1 - self.guiding_lambda) * gating_prob

        breakpoint()
        pred = list()
        for _ in range(self.n_expert):
            pass

    
    def get_pair_embedding(self, emotion_prediction, input_ids, attention_mask, token_type_ids, speaker_ids):
        '''
        pair_embedding을 구해서 리턴
        pair_embedding: 두 utterance 정보를 이어붙인 것
        1. 
        - concatenated_embedding(776) = 768(bert_hidden) + 7(emotions) + 1(speaker) 
        2. pair_embedding[0][0]: concatenated_embedding(utterance representation) 두 개를 이어 붙인 것 (1552)
        pair_embedding[0]: 0번째 pair embedding
        '''
        batch_size, max_doc_len, max_seq_len = input_ids.shape
        
        _, pooled_output = self.bert(input_ids=input_ids.view(-1, max_seq_len), attention_mask=attention_mask.view(-1, max_seq_len), token_type_ids=token_type_ids.view(-1, max_seq_len), return_dict=False)
        # utterance_representation: pooled_output([CLS]의 representation)
        utterance_representation = self.dropout(pooled_output)
        
        # concate해서 utterance_representation으로 합치기
        concatenated_embedding = torch.cat((utterance_representation, emotion_prediction, speaker_ids.view(-1).unsqueeze(1)), dim=1) 
        
        # pair_embedding
        pair_embedding = list()
        for batch in concatenated_embedding.view(batch_size, max_doc_len, -1): # concatenated_embedding을 펼쳐서
            pair_per_batch = list()
            for end_t in range(max_doc_len): # 모든 경우의 수
                for t in range(end_t + 1):
                    pair_per_batch.append(torch.cat((batch[t], batch[end_t])))
            pair_embedding.append(torch.stack(pair_per_batch))
        
        pair_embedding = torch.stack(pair_embedding).to(input_ids.device)

        return pair_embedding
    
    def get_subtask_label(self, input_ids, speaker_ids, emotion_prediction):
        '''
        utterance pair의 speaker 정보, emotion prediction 정보를 통해서 기계적으로 expert를 결정하는 요소
        input: token sequence[75], speaker 정보[1], emotion 예측 정보[7]
        output: 어떤 Expert를 선택할지의 one-hot[4]
        '''
        batch_size, max_doc_len, max_seq_len = input_ids.shape
        
        pair_info = list()
        for speaker_batch, emotion_batch in zip(speaker_ids.view(batch_size, max_doc_len, -1), emotion_prediction.view(batch_size, max_doc_len, -1)):
            info_pair_per_batch = list()
            for end_t in range(max_doc_len):
                for t in range(end_t + 1):
                    # pair를 간단한 조건문에 넣어 (speaker가 같은가? emotion이 같은가?) condition을 파악함
                    # argmax -> emotion[] 중에 가장 높은 값을 갖는 쪽이 emotion
                    # ex) [ 0.2089, -0.4470, -0.8788,  1.6546,  0.2775,  0.0992, -0.1644] -> 3
                    speaker_condition, emotion_condition = bool(speaker_batch[t] == speaker_batch[end_t]), bool(torch.argmax(emotion_batch[t]) == torch.argmax(emotion_batch[end_t]))
                    
                    if speaker_condition and emotion_condition:
                        info_pair_per_batch.append(torch.Tensor([1, 0, 0, 0]))
                    elif speaker_condition:
                        info_pair_per_batch.append(torch.Tensor([0, 1, 0, 0]))
                    elif emotion_condition:
                        info_pair_per_batch.append(torch.Tensor([0, 0, 1, 0]))
                    else:
                        info_pair_per_batch.append(torch.Tensor([0, 0, 0, 1]))
            pair_info.append(torch.stack(info_pair_per_batch))

        pair_info = torch.stack(pair_info).to(input_ids.device)    
        return pair_info
    
    
    
    
class PRG_MoE(guided_moe_basic):
    def __init__(self, dropout=0.5, n_speaker=2, n_emotion=7, n_cause=2, n_expert=4, guiding_lambda=0, **kwargs):
        super().__init__(dropout=dropout, n_speaker=n_speaker, n_emotion=n_emotion, n_cause=n_cause, n_expert=4, guiding_lambda=guiding_lambda)
