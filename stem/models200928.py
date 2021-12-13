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


class StlEmbsConvEnc2Stem(nn.Module):
    def __init__(self, voc_size, emb_size, y_size_hard, conv_channels=[16, 32], filter_sizes=[3, 4, 5], conv_stridesizes=[1, 1], pool_filtersizes=[2, 2], pool_stridesizes=[1, 1], att_heads=2, att_layers=2, nrlayer=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.embs = nn.Embedding(voc_size, emb_size).to(device=device)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.ilastconv = self.nconv - 1

        self.convs_word = nn.ModuleList(nn.ModuleList(nn.Conv1d(in_channels=emb_size, out_channels=conv_channels[iconv], kernel_size=filter_sizes[ifilt], stride=conv_stridesizes[iconv])
                        for ifilt in range(self.nfilt))
                        for iconv in range(self.nconv)).to(device=device, dtype=floatype)
        convout_word_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        self.convs_stem = nn.ModuleList(nn.ModuleList(nn.Conv1d(in_channels=emb_size, out_channels=conv_channels[iconv], kernel_size=filter_sizes[ifilt], stride=conv_stridesizes[iconv])
                        for ifilt in range(self.nfilt))
                        for iconv in range(self.nconv)).to(device=device, dtype=floatype)
        convout_stem_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        self.word_encoder = Encoder(convout_word_size, att_heads, att_layers, device=device)
        self.stem_encoder = Encoder(convout_stem_size, att_heads, att_layers, device=device)

        encout_size = convout_word_size + convout_stem_size

        layersizes_hard = [(encout_size if i == 1 else int(encout_size * ((nrlayer - i + 1)/(nrlayer))), int(encout_size * ((nrlayer - i)/(nrlayer))) if i != (nrlayer) else y_size_hard) for i in range(1, nrlayer+1)]
        self.fc_layers_hard = nn.ModuleList(nn.Linear(layersizes_hard[il][0], layersizes_hard[il][1]) for il in range(nrlayer)).to(device=device)
        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def conv_block(self, embs_lookup, convs):
        embs_lookup = embs_lookup.permute(0, 2, 1)
        conveds = [[F.relu(convs[iconv][ifilt](embs_lookup))
                    for ifilt in range(self.nfilt)]
                    for iconv in range(self.nconv)] # [batsize, nr_filters, sent len - filter_sizes[n] + 1]
        pool_filters = [[self.pool_filtersizes[iconv] if iconv != self.ilastconv else
                        conveds[iconv][ifilter].shape[2] # poiché voglio che esca un vettore, assegno all'ultimo filtro la stessa dim della colonna in input,
                        for ifilter in range(self.nfilt)] # così il nr delle colonne in uscita all'ultima conv sarà 1, ed eliminerò la dimensione con squueze.
                        for iconv in range(self.nconv)]
        pooleds = [[F.max_pool1d(conveds[iconv][ifilt], pool_filters[iconv][ifilt], stride=self.pool_stridesizes[iconv]) if iconv != self.ilastconv else
                    F.max_pool1d(conveds[iconv][ifilt], pool_filters[iconv][ifilt], stride=self.pool_stridesizes[iconv]).squeeze(2)
                    for ifilt in range(self.nfilt)]
                    for iconv in range(self.nconv)] # [batsize, nr_filters]
        concat = self.dropout(torch.cat([pooled for pooled in pooleds[self.ilastconv]], dim=1)) # [batsize, nr_filters * len(filter_sizes)]
        # for iconv in range(len(args.conv_channels)):
        #     for ifilter in range(len(args.filter_sizes)):
        #         print("$$$ conveds", conveds[iconv][ifilter].shape)
        # for iconv in range(len(args.conv_channels)):
        #     for ifilter in range(len(args.filter_sizes)):
        #         print("$$$ pooleds", pooleds[iconv][ifilter].shape)
        # print("$$$ concat", concat.shape)
        return concat

    def forward(self, word, stem):
        lookup_word  = self.embs(word)
        out_conv_word = self.conv_block(lookup_word, self.convs_word)
        out_conv_word = self.dropout(out_conv_word)
        out_enc_word = self.word_encoder.forward(out_conv_word.unsqueeze(1)).squeeze(1)

        lookup_stem  = self.embs(stem)
        out_conv_stem = self.conv_block(lookup_stem, self.convs_stem)
        out_conv_stem = self.dropout(out_conv_stem)
        out_enc_stem = self.stem_encoder.forward(out_conv_stem.unsqueeze(1)).squeeze(1)

        word_stem = torch.cat([out_enc_word, out_enc_stem], dim=1)

        out_hard = word_stem
        for layer_hard in self.fc_layers_hard:
            out_hard = layer_hard(out_hard)
        out_hard = F.sigmoid(out_hard).squeeze(1)

        return out_hard


class MtlEmbsConvEnc2Stem(nn.Module):
    def __init__(self, voc_size, emb_size, y_size_hard, y_size_soft, conv_channels=[16, 32], filter_sizes=[3, 4, 5], conv_stridesizes=[1, 1], pool_filtersizes=[2, 2], pool_stridesizes=[1, 1], att_heads=2, att_layers=2, nrlayer=2, droprob=.1, device='cuda:0', floatype=torch.float32):
        super().__init__()
        self.embs = nn.Embedding(voc_size, emb_size).to(device=device)
        self.pool_filtersizes = pool_filtersizes
        self.pool_stridesizes = pool_stridesizes
        self.nconv = len(conv_channels)
        self.nfilt = len(filter_sizes)
        self.ilastconv = self.nconv - 1

        self.convs_word = nn.ModuleList(nn.ModuleList(nn.Conv1d(in_channels=emb_size, out_channels=conv_channels[iconv], kernel_size=filter_sizes[ifilt], stride=conv_stridesizes[iconv])
                        for ifilt in range(self.nfilt))
                        for iconv in range(self.nconv)).to(device=device, dtype=floatype)
        convout_word_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        self.convs_stem = nn.ModuleList(nn.ModuleList(nn.Conv1d(in_channels=emb_size, out_channels=conv_channels[iconv], kernel_size=filter_sizes[ifilt], stride=conv_stridesizes[iconv])
                        for ifilt in range(self.nfilt))
                        for iconv in range(self.nconv)).to(device=device, dtype=floatype)
        convout_stem_size = conv_channels[-1] * self.nfilt # conv size out = len last conv channel * nr of filters, infatti concatenerò i vettori in uscita di ogni filtro, che hanno il size dell'ultimo channel

        self.word_encoder = Encoder(convout_word_size, att_heads, att_layers, device=device)
        self.stem_encoder = Encoder(convout_stem_size, att_heads, att_layers, device=device)

        encout_size = convout_word_size + convout_stem_size

        layersizes_hard = [(encout_size if i == 1 else int(encout_size * ((nrlayer - i + 1)/(nrlayer))), int(encout_size * ((nrlayer - i)/(nrlayer))) if i != (nrlayer) else y_size_hard) for i in range(1, nrlayer+1)]
        layersizes_soft = [(encout_size if i == 1 else int(encout_size * ((nrlayer - i + 1)/(nrlayer))), int(encout_size * ((nrlayer - i)/(nrlayer))) if i != (nrlayer) else y_size_soft) for i in range(1, nrlayer+1)]
        self.fc_layers_hard = nn.ModuleList(nn.Linear(layersizes_hard[il][0], layersizes_hard[il][1]) for il in range(nrlayer)).to(device=device)
        self.fc_layers_soft = nn.ModuleList(nn.Linear(layersizes_soft[il][0], layersizes_soft[il][1]) for il in range(nrlayer)).to(device=device)
        self.dropout = nn.Dropout(droprob)
        for name_str, param in self.named_parameters(): print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
        print(f"{'trainable parameters':.<25} {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def conv_block(self, embs_lookup, convs):
        embs_lookup = embs_lookup.permute(0, 2, 1)
        conveds = [[F.relu(convs[iconv][ifilt](embs_lookup))
                    for ifilt in range(self.nfilt)]
                    for iconv in range(self.nconv)] # [batsize, nr_filters, sent len - filter_sizes[n] + 1]
        pool_filters = [[self.pool_filtersizes[iconv] if iconv != self.ilastconv else
                        conveds[iconv][ifilter].shape[2] # poiché voglio che esca un vettore, assegno all'ultimo filtro la stessa dim della colonna in input,
                        for ifilter in range(self.nfilt)] # così il nr delle colonne in uscita all'ultima conv sarà 1, ed eliminerò la dimensione con squueze.
                        for iconv in range(self.nconv)]
        pooleds = [[F.max_pool1d(conveds[iconv][ifilt], pool_filters[iconv][ifilt], stride=self.pool_stridesizes[iconv]) if iconv != self.ilastconv else
                    F.max_pool1d(conveds[iconv][ifilt], pool_filters[iconv][ifilt], stride=self.pool_stridesizes[iconv]).squeeze(2)
                    for ifilt in range(self.nfilt)]
                    for iconv in range(self.nconv)] # [batsize, nr_filters]
        concat = self.dropout(torch.cat([pooled for pooled in pooleds[self.ilastconv]], dim=1)) # [batsize, nr_filters * len(filter_sizes)]
        # for iconv in range(len(args.conv_channels)):
        #     for ifilter in range(len(args.filter_sizes)):
        #         print("$$$ conveds", conveds[iconv][ifilter].shape)
        # for iconv in range(len(args.conv_channels)):
        #     for ifilter in range(len(args.filter_sizes)):
        #         print("$$$ pooleds", pooleds[iconv][ifilter].shape)
        # print("$$$ concat", concat.shape)
        return concat

    def forward(self, word, stem):
        lookup_word  = self.embs(word)
        out_conv_word = self.conv_block(lookup_word, self.convs_word)
        out_conv_word = self.dropout(out_conv_word)
        out_enc_word = self.word_encoder.forward(out_conv_word.unsqueeze(1)).squeeze(1)

        lookup_stem  = self.embs(stem)
        out_conv_stem = self.conv_block(lookup_stem, self.convs_stem)
        out_conv_stem = self.dropout(out_conv_stem)
        out_enc_stem = self.stem_encoder.forward(out_conv_stem.unsqueeze(1)).squeeze(1)

        word_stem = torch.cat([out_enc_word, out_enc_stem], dim=1)

        out_hard = word_stem
        for layer_hard in self.fc_layers_hard:
            out_hard = layer_hard(out_hard)
        out_hard = F.sigmoid(out_hard).squeeze(1)

        out_soft = word_stem
        for layer_soft in self.fc_layers_soft:
            out_soft = layer_soft(out_soft)
        out_soft = F.softmax(out_soft, dim=1)

        return out_hard, out_soft



