import json
import numpy as np
from transformers import BertTokenizer
import torch
    
def get_data(data_file, device, max_seq_len, contain_context=False):
    device = 'cpu'
    # 파일을 읽어서 json 형태로 data에 저장
    with open(data_file) as f:
        data = json.load(f) # dict 형태로 데이터를 갖고 있음
    
    emotion_label_policy = {'angry': 0, 'anger': 0, # policy (문자열로 나타나는 타입마다 번호를 배정)
        'disgust': 1,
        'fear': 2,
        'happy': 3, 'happines': 3, 'happiness': 3, 'excited': 3,
        'sad': 4, 'sadness': 4, 'frustrated': 4,
        'surprise': 5, 'surprised': 5, 
        'neutral': 6}
    
    cause_label_policy = {'no-context':0, 'inter-personal':1, 'self-contagion':2, 'latent':3} # policy (문자열로 나타나는 타입마다 번호를 배정)
    
    preprocessed_utterance, max_doc_len, max_seq_len = load_utterance(data_file, device, max_seq_len)
    
    '''
    doc_speaker, doc_emotion_label, ...
    : 데이터셋[대화[발화, 발화, 발화, ...]]
    '''
    doc_speaker, doc_emotion_label, doc_pair_cause_label, doc_pair_binary_cause_label = [list() for _ in range(4)]
    
    # data.items()의 각각의 item은 하나의 대화 전체가 들어있음
    # 그러므로 for문 속 iteration 1회당 하나씩의 dialogue(item) 처리
    for doc_id, content in data.items():
        
        # 화자, 감정, cause가 있는 turn, cause의 범위(?), cause label에 각각의 empty list를 배정
        # 하나의 대화 전체를 각각의 배열로 표현
        r_speaker, r_emotion_label, r_corresponding_cause_turn, r_corresponding_cause_span, r_corresponding_cause_label = [list() for _ in range(5)]
        
        content = content[0] # content는 하나의 배열 안에 있는 utterance list로 구성됨, content[0]으로 utterancelist를 꺼낸다
        
        # ？❓？
        pair_cause_label = torch.zeros((int(max_doc_len * (max_doc_len + 1) / 2), 4), dtype=torch.long) 
        pair_binary_cause_label = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2), dtype=torch.long)
        
        # dialogue 속 각 turn들을 처리
        for each_turn in content:
            # speaker 추출 (A->0, else->1)
            r_speaker.append(0 if each_turn['speaker'] == "A" else 1) 
            # emotion label 추출 ('감정'->숫자)
            r_emotion_label.append(emotion_label_policy[each_turn['emotion']])
            
            corresponding_cause_label_by_turn = list()
            
            # cause가 존재하는 turn인 경우
            if "expanded emotion cause evidence" in each_turn.keys():
                # cause가 숫자면, 1씩 빼서(turn 0부터 시작하도록) 넣음, cause가 'b'라면 -1로 해서 넣는다
                # cause가 'b'라는 거의 의미는? 'back'(cause type latent)
                corresponding_cause_per_turn = [_ - 1 if type(_) != str else -1 for _ in each_turn["expanded emotion cause evidence"]]
                
                # print(each_turn["expanded emotion cause evidence"], corresponding_cause_per_turn)
                r_corresponding_cause_turn.append(corresponding_cause_per_turn)
                
                # cause_label을 붙이기 (inter-personal, no-context, self-contagion)
                for _ in corresponding_cause_per_turn: 
                    if _ == -1:
                        corresponding_cause_label_by_turn.append(cause_label_policy["latent"])
                    elif _ + 1 == each_turn["turn"]:
                        corresponding_cause_label_by_turn.append(cause_label_policy["no-context"])
                    elif content[_]["speaker"] == each_turn["speaker"]:
                        corresponding_cause_label_by_turn.append(cause_label_policy["self-contagion"])
                    elif content[_]["speaker"] != each_turn["speaker"]:
                        corresponding_cause_label_by_turn.append(cause_label_policy["inter-personal"])
                        
            r_corresponding_cause_label.append(corresponding_cause_label_by_turn)
        
        # dialogue 하나 속의 모든 turn에 대해 처리한 상황
        
        # ?
        for idx, corresponding_cause_per_turn in enumerate(r_corresponding_cause_label):
            # enumerate: [1, 2, 3, 4, 5]를 [(0, 1), (1, 2), ...] 로 변환해줌
            # print('idx: ',idx,', ccpt: ',corresponding_cause_per_turn)
            pair_idx = int(idx * (idx + 1) / 2)
            if corresponding_cause_per_turn:
                for cause_turn, cause in zip(content[idx]['expanded emotion cause evidence'], corresponding_cause_per_turn):
                    if type(cause_turn) == str:
                        continue
                    
                    cause_idx = int(cause_turn) - 1
                    pair_cause_label[pair_idx + cause_idx][cause] = 1
                    pair_binary_cause_label[pair_idx + cause_idx] = 1
        
        pair_cause_label[(torch.sum(pair_cause_label, dim=1) == False).nonzero(as_tuple=True)[0], 3] = 1

        doc_speaker.append(r_speaker)
        doc_emotion_label.append(r_emotion_label)
        doc_pair_cause_label.append(pair_cause_label)
        doc_pair_binary_cause_label.append(pair_binary_cause_label)
        # pair_cause_label[]
        
    out_speaker, out_emotion_label = [list() for _ in range(2)]
    out_pair_cause_label, out_pair_binary_cause_label = torch.stack(doc_pair_cause_label), torch.stack(doc_pair_binary_cause_label)

    for speaker, emotion_label in zip(doc_speaker, doc_emotion_label):
        speaker_t = torch.zeros(max_doc_len, dtype=torch.long)
        speaker_t[:len(speaker)] = torch.tensor(speaker)

        emotion_label_t = torch.zeros(max_doc_len, dtype=torch.long)
        emotion_label_t[:len(speaker)] = torch.tensor(emotion_label)

        out_speaker.append(speaker_t); out_emotion_label.append(emotion_label_t)

    out_speaker, out_emotion_label = torch.stack(out_speaker).type(torch.FloatTensor), torch.stack(out_emotion_label)
    
    result = (preprocessed_utterance, out_speaker.to(device), out_emotion_label.to(device), out_pair_cause_label.to(device), out_pair_binary_cause_label.to(device))
    return result
    
    
    
def load_utterance(data_file, device, max_seq_len):
    '''
    load_utterance의 역할 (아웃풋)
    인풋: 데이터셋 파일 (각 대화가 들어있음)
    아웃풋:  
        1. (out_utterance_input_ids,        : utterance 하나의 tokenized 'token sequence'
            out_utterance_attention_mask,   : utterance 하나의 'attention mask'
            out_utterance_token_type_ids)
                                            : DataSet file의 각 '파일-대화-발화-토큰' 리스트가 담긴 텐서
        2. max_doc_len                      : 데이터 파일 속에서 가장 긴 대화의 턴수 (이보다 짧으면 padding(101,102)가 채워짐)
        3. max_seq_len                      : 한 utterance에서 나올 수 있는 최대 토큰 개수
    '''
    
    # 파일을 읽어서 json 형태로 data에 저장
    with open(data_file) as f:
        data = json.load(f) # dict 형태로 데이터를 갖고 있음
    
    tokenizer_ = BertTokenizer.from_pretrained('bert-base-cased') # Tokenizer를 불러옴, 끝의 _는 충돌피하기
    max_seq_len, max_doc_len = max_seq_len, 0
    
    
    # doc_dialog: 한 데이터셋 속 dialogue들의 배열 (원랜 doc_utterance)
    doc_dialog = list()
    
    # 데이터(data): 여러 개의 대화(dialog)가 들어있음
    # for문: 전체 dialogue에 대해 (각 content는 각각의 dialogue) 
    for doc_id, content in data.items():
        
        # 1. dialog 구하기: (현재 dialogue의 각 utterance를 item으로 갖는 배열)
        dialog = list()
        content = content[0]
        max_doc_len = max(len(content), max_doc_len) # 전체 content 중에서 최대 길이

        for each_turn in content:
            _tokenized = tokenizer_(each_turn["utterance"], padding='max_length', max_length=max_seq_len, truncation=True, return_tensors="pt")
            dialog.append(_tokenized)
            '''
            utterance 문자열을 tokenizer에 넣으면, tokenizer는 문자열을 token(int) sequence로 바꿔준다
            이 때, 처음에는 101, 끝에는 102가 붙는다
            tokenizer는 attention mask도 반환하는데, 이는 토큰이 있는 index만 1를, 나머지는 0을 갖는 배열이다
            '''
        # ~> dialog의 item: 각 utterance의 [input_ids, attention_mask, token_type_ids]
        
        # 2. doc_dialog에 구한 dialog를 넣어주기
        doc_dialog.append(dialog)
        
        # output할 [input_ids], [attention_mask], [token_type_ids]를 리스트로 return하기 위해 각각 list를 초기화
        out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = [list() for _ in range(3)]
        
        ### doc_dialog: 문서(데이터셋) 단위 ###
        # item: dialogue 속 각각 발화들을 token sequence로 나타낸 것의 배열 (즉, item 하나가 대화 전체의 token sequence 배열)
        # doc_dialog: 문서[대화[발화, 발화, 발화, ...]] 구조
        # 대화 -> [발화1, 발화2, ..., 발화n] -> [Token sequence 1, token sequence 2, ..., token sequence n]
        for each_dialog in doc_dialog: 
            # 한 전체 대화문 속 각 발화(utterance)에 대해서
            # [101 (Start), 102 (End)]로 이루어진 token sequence 생성 (padding 용도)
            padding_sequence = tokenizer_('', padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt")
            padding_sequence_t = [padding_sequence for _ in range(max_doc_len - len(each_dialog))]
            
            # 각 dialog 뒤에다가 padding_sequence (시작과 끝으로 이뤄진 utterance)를 넣어 
            # 모든 dialog가 같은 개수의 utterance를 갖게 함
            each_dialog = each_dialog + padding_sequence_t # shape: (max_doc_len, max_seq_len)
            
            utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = [list() for _ in range(3)]
            
            for each_utterance in each_dialog:
                # 각 utterance item을 구성하는 [token sequence, attention mask, token type ids]를 펼쳐서
                # 각각 다른 배열에 넣는 코드
                utterance_input_ids_t.append(each_utterance['input_ids'])
                utterance_attention_mask_t.append(each_utterance['attention_mask'])
                utterance_token_type_ids_t.append(each_utterance['token_type_ids'])

            # 배열 속 tensor 속 tensor 배열이었던 원래의 복잡한 형태를 풀어서 합쳐주기 위함
            ''' 
                torch.vstack(): tensor들을 합쳐서 이어주는 메소드 중 하나 (vertical하게 stack)
                기존 utterance_input_ids_t: [ tensor([[101, ..., 102, 0, 0, ..., 0]]), tensor([[]]), ..., tensor([[]]) ]
                torch.stack()랑은 좀 다르다 (horizontal하게 stack)
            '''
            utterance_input_ids_t = torch.vstack(utterance_input_ids_t)
            utterance_attention_mask_t = torch.vstack(utterance_attention_mask_t)
            utterance_token_type_ids_t = torch.vstack(utterance_token_type_ids_t)
            
            # 
            out_utterance_input_ids.append(utterance_input_ids_t)
            out_utterance_attention_mask.append(utterance_attention_mask_t)
            out_utterance_token_type_ids.append(utterance_token_type_ids_t)
    
    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = torch.stack(out_utterance_input_ids), torch.stack(out_utterance_attention_mask), torch.stack(out_utterance_token_type_ids)
    return (out_utterance_input_ids.to(device), out_utterance_attention_mask.to(device), out_utterance_token_type_ids.to(device)), max_doc_len, max_seq_len
    
    
'''
if __name__ == '__main__':
    datafile = './test_datafile.json'
    get_data(datafile, 0, 75, False)
'''
    