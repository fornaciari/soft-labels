# coding=latin-1
import util200818 as ut
import re, os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pylab as plt
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel, BertConfig, AutoModel, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, FlaubertTokenizer, FlaubertModel
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=(FutureWarning, UserWarning))
###################################################################################################


###################################################################################################


def norm(x):
    return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)


###################################################################################################


class KLregular(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(torch.sum(P_targ * torch.log2(P_targ/Q_pred), dim=1))
        # return torch.sum(P_targ * torch.log2(P_targ / Q_pred))


class KLinverse(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(torch.sum(Q_pred * torch.log2(Q_pred/P_targ), dim=1))
        # return torch.sum(Q_pred * torch.log2(Q_pred / P_targ))


class CrossEntropySoft(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        Q_pred = F.softmax(Q_pred, dim=1) # per allinearmi a nn.CrossEntropyLoss, che applica softmax a valori qualsiasi
        return torch.mean(-torch.sum(P_targ * torch.log2(Q_pred), dim=1))
        # return -torch.sum(P_targ * torch.log2(Q_pred))


class CrossEntropy(nn.Module):
    def __init__(self, device='cuda:0'):
        self.device = device
        super().__init__()

    def forward(self, Q_pred, P_targ):
        return torch.mean(-torch.sum(P_targ * torch.log2(Q_pred), dim=1))
        # return -torch.sum(P_targ * torch.log2(Q_pred))


###################################################################################################
# Samuel Lynn-Evans
# https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, device='cuda:0', max_seq_len=512):
        self.device = device
        super().__init__()
        self.d_model = d_model
        # create constant 'pe' matrix with values dependant on pos and i
        # cioè sequence_length (position) ed emb_size
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
# If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer,
# you should register them as buffers. Buffers won?t be returned in model.parameters(), so that the optimizer won?t have a chance to update them.

    def forward(self, x):
        x = x * math.sqrt(self.d_model) # make embeddings relatively larger
        seq_len = x.size(1) # add constant to embedding
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).to(device=self.device) # Variable ammette la back propagation
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, device='cuda:0', dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model).to(device=device)
        self.v_linear = nn.Linear(d_model, d_model).to(device=device)
        self.k_linear = nn.Linear(d_model, d_model).to(device=device)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model).to(device=device)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, device='cuda:0', d_ff=2048, dropout=0.1):
        super().__init__()
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff).to(device=device)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model).to(device=device)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, device='cuda:0', eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size).to(device=device))
        self.bias = nn.Parameter(torch.zeros(self.size).to(device=device))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, device='cuda:0', dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model, device)
        self.norm_2 = Norm(d_model, device)
        self.attn = MultiHeadAttention(heads, d_model, device)
        self.ff = FeedForward(d_model, device)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x2 = self.norm_1.forward(x)
        x = x + self.dropout_1(self.attn.forward(x2, x2, x2, mask))
        x2 = self.norm_2.forward(x)
        x = x + self.dropout_2(self.ff.forward(x2))
        return x


class Encoder(nn.Module):
    def __init__(self, rep_size, n_heads, n_layers, device='cuda:0'):
        super().__init__()
        self.n_layers = n_layers
        self.pe = PositionalEncoder(rep_size, device)
        self.layers = get_clones(EncoderLayer(rep_size, n_heads, device), n_layers)
        self.norm = Norm(rep_size, device)

    def forward(self, src, mask=None):
        x = self.pe.forward(src)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.norm.forward(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, device='cuda:0', dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).cuda()

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1.forward(x)
        x = x + self.dropout_1(self.attn_1.forward(x2, x2, x2, trg_mask))
        x2 = self.norm_2.forward(x)
        x = x + self.dropout_2(self.attn_2.forward(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3.forward(x)
        x = x + self.dropout_3(self.ff(x2))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, device='cuda:0'):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, device)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed.forward(trg)
        x = self.pe.forward(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm.forward(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder.forward(src, src_mask)
        d_output = self.decoder.forward(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


###################################################################################################


class StlPreembBilstmEnc3(nn.Module):
    def __init__(self, pad1_emb, pad2_emb, y_size, trainable=False, lstm_nrlayer=2, hid_size=100, bidir=True, att_heads=1, att_layers=1, mlp_nrlayer=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()

        pad1_emb = torch.from_numpy(pad1_emb).to(device=device, dtype=floatype)
        self.pad1_emb = nn.Embedding.from_pretrained(pad1_emb)
        if trainable: self.pad1_emb.weight.requires_grad = True

        pad2_emb = torch.from_numpy(pad2_emb).to(device=device, dtype=floatype)
        self.pad2_emb = nn.Embedding.from_pretrained(pad2_emb)
        if trainable: self.pad2_emb.weight.requires_grad = True

        pad1_embsize = self.pad1_emb.embedding_dim
        pad2_embsize = self.pad2_emb.embedding_dim

        self.pad1_lstm = nn.LSTM(pad1_embsize, hid_size, num_layers=lstm_nrlayer, bidirectional=bidir, dropout=droprob).to(device=device, dtype=floatype)
        self.pad2_lstm = nn.LSTM(pad2_embsize, hid_size, num_layers=lstm_nrlayer, bidirectional=bidir, dropout=droprob).to(device=device, dtype=floatype)
        lstm_outsize = hid_size * 2 if bidir else hid_size

        concat_size = lstm_outsize * 2
        self.enc = Encoder(concat_size, att_heads, att_layers, device=device)
        self.hard_enc = Encoder(concat_size, att_heads, att_layers, device=device)

        layersizes_hard = [(concat_size if i == 1 else int(concat_size * ((mlp_nrlayer - i + 1)/(mlp_nrlayer))), int(concat_size * ((mlp_nrlayer - i)/(mlp_nrlayer))) if i != (mlp_nrlayer) else y_size) for i in range(1, mlp_nrlayer+1)]
        self.fc_layers = nn.ModuleList(nn.Linear(layersizes_hard[il][0], layersizes_hard[il][1]) for il in range(mlp_nrlayer)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, pad1, pad2, wantedid1=-1, wantedid2=-1): # di default, l'ultimo id
        pad1_lookup = self.pad1_emb(pad1) # [batsize, padsize, embsize]
        pad1_lookup = pad1_lookup.permute(1, 0, 2) # [padsize, batsize, embsize] come richiesto da nn.LSTM
        pad1_states, (pad1_hidden, pad1_cell) = self.pad1_lstm(pad1_lookup) # states = [padsize, batsize, num_directions * hidsize]
        pad1_states = pad1_states.permute(1, 0, 2) # [batsize, padsize, num_directions * hidsize]
        pad1_state = pad1_states[torch.arange(pad1_states.size(0)), wantedid1]  # [batsize, num_directions * hidsize]

        pad2_lookup = self.pad2_emb(pad2) # [batsize, padsize, embsize]
        pad2_lookup = pad2_lookup.permute(1, 0, 2) # [padsize, batsize, embsize] come richiesto da nn.LSTM
        pad2_states, (pad2_hidden, pad2_cell) = self.pad2_lstm(pad2_lookup) # states = [padsize, batsize, num_directions * hidsize]
        pad2_states = pad2_states.permute(1, 0, 2) # [batsize, padsize, num_directions * hidsize]
        pad2_state = pad2_states[torch.arange(pad2_states.size(0)), wantedid2]

        cat_states = torch.cat([pad1_state, pad2_state], dim=1)
        states_enc = self.enc.forward(cat_states.unsqueeze(1))
        states_enc = self.dropout(states_enc)

        hard_enc = self.hard_enc.forward(states_enc).squeeze(1)
        for layer in self.fc_layers:
            hard_enc = layer(hard_enc)
        return hard_enc


class MtlPreembBilstmEnc3(nn.Module):
    def __init__(self, pad1_emb, pad2_emb, y_size, trainable=False, lstm_nrlayer=2, hid_size=100, bidir=True, att_heads=1, att_layers=1, mlp_nrlayer=1, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()

        pad1_emb = torch.from_numpy(pad1_emb).to(device=device, dtype=floatype)
        self.pad1_emb = nn.Embedding.from_pretrained(pad1_emb)
        if trainable: self.pad1_emb.weight.requires_grad = True

        pad2_emb = torch.from_numpy(pad2_emb).to(device=device, dtype=floatype)
        self.pad2_emb = nn.Embedding.from_pretrained(pad2_emb)
        if trainable: self.pad2_emb.weight.requires_grad = True

        pad1_embsize = self.pad1_emb.embedding_dim
        pad2_embsize = self.pad2_emb.embedding_dim

        self.pad1_lstm = nn.LSTM(pad1_embsize, hid_size, num_layers=lstm_nrlayer, bidirectional=bidir, dropout=droprob).to(device=device, dtype=floatype)
        self.pad2_lstm = nn.LSTM(pad2_embsize, hid_size, num_layers=lstm_nrlayer, bidirectional=bidir, dropout=droprob).to(device=device, dtype=floatype)
        lstm_outsize = hid_size * 2 if bidir else hid_size

        concat_size = lstm_outsize * 2
        self.enc = Encoder(concat_size, att_heads, att_layers, device=device)
        self.hard_enc = Encoder(concat_size, att_heads, att_layers, device=device)
        self.soft_enc = Encoder(concat_size, att_heads, att_layers, device=device)

        layersizes_hard = [(concat_size if i == 1 else int(concat_size * ((mlp_nrlayer - i + 1)/(mlp_nrlayer))), int(concat_size * ((mlp_nrlayer - i)/(mlp_nrlayer))) if i != (mlp_nrlayer) else y_size) for i in range(1, mlp_nrlayer+1)]
        layersizes_soft = [(concat_size if i == 1 else int(concat_size * ((mlp_nrlayer - i + 1)/(mlp_nrlayer))), int(concat_size * ((mlp_nrlayer - i)/(mlp_nrlayer))) if i != (mlp_nrlayer) else y_size) for i in range(1, mlp_nrlayer+1)]
        self.fc_layers_hard = nn.ModuleList(nn.Linear(layersizes_hard[il][0], layersizes_hard[il][1]) for il in range(mlp_nrlayer)).to(device=device)
        self.fc_layers_soft = nn.ModuleList(nn.Linear(layersizes_soft[il][0], layersizes_soft[il][1]) for il in range(mlp_nrlayer)).to(device=device)

        self.dropout = nn.Dropout(droprob)
        # for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f'The model has {sum(p.numel() for p in self.parameters() if p.requires_grad):,} trainable parameters')

    def forward(self, pad1, pad2, wantedid1=-1, wantedid2=-1): # di default, l'ultimo id
        pad1_lookup = self.pad1_emb(pad1) # [batsize, padsize, embsize]
        pad1_lookup = pad1_lookup.permute(1, 0, 2) # [padsize, batsize, embsize] come richiesto da nn.LSTM
        pad1_states, (pad1_hidden, pad1_cell) = self.pad1_lstm(pad1_lookup) # states = [padsize, batsize, num_directions * hidsize]
        pad1_states = pad1_states.permute(1, 0, 2) # [batsize, padsize, num_directions * hidsize]
        pad1_state = pad1_states[torch.arange(pad1_states.size(0)), wantedid1]  # [batsize, num_directions * hidsize]

        pad2_lookup = self.pad2_emb(pad2) # [batsize, padsize, embsize]
        pad2_lookup = pad2_lookup.permute(1, 0, 2) # [padsize, batsize, embsize] come richiesto da nn.LSTM
        pad2_states, (pad2_hidden, pad2_cell) = self.pad2_lstm(pad2_lookup) # states = [padsize, batsize, num_directions * hidsize]
        pad2_states = pad2_states.permute(1, 0, 2) # [batsize, padsize, num_directions * hidsize]
        pad2_state = pad2_states[torch.arange(pad2_states.size(0)), wantedid2]

        cat_states = torch.cat([pad1_state, pad2_state], dim=1)
        states_enc = self.enc.forward(cat_states.unsqueeze(1))
        states_enc = self.dropout(states_enc)

        hard_enc = self.hard_enc.forward(states_enc).squeeze(1)
        for layer_hard in self.fc_layers_hard:
            hard_enc = layer_hard(hard_enc)

        soft_enc = self.soft_enc.forward(states_enc).squeeze(1)
        for layer_soft in self.fc_layers_soft:
            soft_enc = layer_soft(soft_enc)
        soft_enc = F.softmax(soft_enc, dim=1)

        return hard_enc, soft_enc


