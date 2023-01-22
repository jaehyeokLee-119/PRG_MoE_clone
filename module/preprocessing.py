import json
import numpy as np
from transformers import BertTokenizer
import torch


def get_data(data_file, device, max_seq_len, contain_context=False):
    # 파일을 읽어서 json 형태로 data에 저장
    with open(data_file) as f:
        data = json.load(f) # dict 형태로 데이터를 갖고 있음
    
    emotion_label_policy = {'angry': 0, 'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3, 'happines': 3, 'happiness': 3, 'excited': 3,
        'sad': 4, 'sadness': 4, 'frustrated': 4,
        'surprise': 5, 'surprised': 5, 
        'neutral': 6}
    cause_label_policy = {'no-context':0, 'inter-personal':1, 'self-contagion':2, 'latent':3}
    preprocessed_utterance, max_doc_len, max_seq_len = load_utterance(data_file, device, max_seq_len)
    
def load_utterance(data_file, device, max_seq_len):
    # 파일을 읽어서 json 형태로 data에 저장
    with open(data_file) as f:
        data = json.load(f) # dict 형태로 데이터를 갖고 있음
    
    tokenizer_ = BertTokenizer.from_pretrained('bert-base-cased') # Tokenizer를 불러옴, 끝의 _는 충돌피하기
    
    max_seq_len, max_doc_len = max_seq_len, 0
    
    doc_utterance = list()
    
    # data.items()의 각각의 item은 하나의 대화 전체가 들어있음
    for doc_id, content in data.items():
        content = content[0] # content는 하나의 배열 안에 있는 utterance list로 구성됨, content[0]으로 utterancelist를 꺼낸다
        pair_cause_label = torch.zeros((int(max_doc_len * (max_doc_len + 1) / 2), 4), dtype=torch.long) 
        pair_binary_cause_label = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2), dtype=torch.long)
        
        print("pcl:", pair_cause_label, "pbc: ",pair_binary_cause_label)
              
    return (0,0,0)
    
    
if __name__ == '__main__':
    datafile = './test_datafile.json'
    get_data(datafile, 0, 75, False)
    