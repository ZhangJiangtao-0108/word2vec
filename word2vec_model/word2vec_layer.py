import sys
sys.path.append('../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
import random
from config.config import train_args, model_args




class Word2vec(nn.Module):
    def __init__(self, input_dim, emb_dim, label_len):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.sentence_lstm = nn.LSTM(emb_dim, 64, 2, bidirectional = True, batch_first = True)
        self.sentence_BN = nn.BatchNorm1d(label_len)
        self.fc = nn.Linear(2048, label_len)
        
    def forward(self, sentence_input): 
        batch_size = sentence_input.shape[0]
        sentence_emb = self.embedding(sentence_input.to(torch.int64))
        # print(sentence_emb.shape)
        sentence_out ,(sentence_h_n, sentence_c_n)= self.sentence_lstm(sentence_emb)
        # print(sentence_out.shape)
        sentence_out = self.sentence_BN(sentence_out)
        sentence_out = sentence_out.view(batch_size,-1)
        sentence_pre = self.fc(sentence_out)
        return sentence_emb, sentence_pre


def Word2Vec_model(input_dim, emb_dim, label_len):
    return Word2vec(input_dim, emb_dim, label_len)