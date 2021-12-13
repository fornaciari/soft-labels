# coding=latin-1
import util200818 as ut
import step200928 as st
import models200928 as mod
import argparse, os, re, sys, time
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, log_loss, confusion_matrix
import pandas as pd
import random
import torch
import torch.nn as nn
from torch import optim
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=(FutureWarning, UserWarning))
###################################################################################################
parser = argparse.ArgumentParser()
# inputs
parser.add_argument("-char_embs",     type=str, default="/inputs/190801163604/emb_matrix.bin")
parser.add_argument("-word_pad",      type=str, default="/inputs/200520120928/word_pad.npz")
parser.add_argument("-stem_pad",      type=str, default="/inputs/200520120928/stem_pad.npz")
parser.add_argument("-word_stem_pad", type=str, default="/inputs/200520120928/word_stem_pad.npz")
parser.add_argument("-path_xls",      type=str, default='/inputs/200925153230/stemming.xlsx')
# torch settings
parser.add_argument("-seed",   type=int, default=1234)
parser.add_argument("-device", type=str, default='cuda:0')
parser.add_argument("-dtype",  type=int, default=32, choices=[32, 64])
# preproc
parser.add_argument("-embsize",      type=int,  default=300)
parser.add_argument("-trainable",    type=bool, default=True)
# model settings
parser.add_argument("-save",        type=bool,  default=False)
parser.add_argument("-experiments", type=int,   default=10)
parser.add_argument("-epochs",      type=int,   default=10)
parser.add_argument("-splits",      type=int,   default=10, help='almeno 3 o dà un errore, credo dovuto all\'output dello stratified')
parser.add_argument("-batsize",     type=int,   default=1024)
parser.add_argument("-learate",     type=float, default=0.001)
parser.add_argument("-droprob",     type=float, default=0.1)
# conv settings
parser.add_argument("-conv_channels",     type=int, nargs='+', default=[64, 128],  help="nr of channels conv by conv")
parser.add_argument("-conv_filter_sizes", type=int, nargs='+', default=[3, 4, 5], help="sizes of filters: window, in each conv")
parser.add_argument("-conv_stridesizes",  type=int, nargs='+', default=[1, 1, 1],    help="conv stride size, conv by conv")
parser.add_argument("-pool_filtersizes",  type=int, nargs='+', default=[2, 2, 2],    help="pool filter size, conv by conv. in order to have a vector as output, the last value will be substituted with the column size of the last conv, so that the last column size will be 1, then squeezed")
parser.add_argument("-pool_stridesizes",  type=int, nargs='+', default=[1, 1, 1],    help="pool stride size, conv by conv")

# lstm settings
parser.add_argument("-lstm_layers", type=int,   default=1)
parser.add_argument("-lstm_size",   type=int,   default=128)
# attention
parser.add_argument("-att_layers",  type=int, default=1)
parser.add_argument("-att_heads",   type=int, default=1)
# fc
parser.add_argument("-mlp_layers", type=int, default=1)
# bootstrap
parser.add_argument("-bootloop",    type=int, default=10000)
parser.add_argument("-perc_sample", type=float, default=.33)

args = parser.parse_args()
sys.stdout = sys.stderr = log = ut.log(__file__, ut.__file__, st.__file__, mod.__file__)
startime = ut.start()
align_size = ut.print_args(args)
print(f"{'dirout':.<{align_size}} {log.pathtime}")
###################################################################################################
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
dtype_float = torch.float64 if args.dtype == 64 else torch.float32
dtype_int = torch.int64 # if args.dtype == 64 else torch.int32 # o 64 o s'inkazza
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(f"{'GPU in use':.<{align_size}} {device}\n{'#'*80}" if torch.cuda.is_available() else f"No GPU available, using the CPU.\n{'#'*80}")
###################################################################################################
df = pd.read_excel(args.path_xls)
print(df.head())
print(df.shape)

###################################################################################################
# char_emb = ut.readbin(args.char_embs)
# word_pad = load_npz(args.word_pad).toarray()
# stem_pad = load_npz(args.stem_pad).toarray()
# word_stem_pad = load_npz(args.word_stem_pad).toarray()
#
# word_pad_trn = word_pad[df.index[df.set == 'trn']]
# word_pad_dev = word_pad[df.index[df.set == 'dev']]
# word_pad_tst = word_pad[df.index[df.set == 'hard-tst']]
#
# stem_pad_trn = stem_pad[df.index[df.set == 'trn']]
# stem_pad_dev = stem_pad[df.index[df.set == 'dev']]
# stem_pad_tst = stem_pad[df.index[df.set == 'hard-tst']]
#
# word_stem_pad_trn = word_stem_pad[df.index[df.set == 'trn']]
# word_stem_pad_dev = word_stem_pad[df.index[df.set == 'dev']]
# word_stem_pad_tst = word_stem_pad[df.index[df.set == 'hard-tst']]
#
# wantedid = np.array([-1 for i in range(len(word_stem_pad))])
# wantedid_trn = np.array([-1 for i in range(len(word_stem_pad_trn))])
# wantedid_dev = np.array([-1 for i in range(len(word_stem_pad_dev))])
# wantedid_tst = np.array([-1 for i in range(len(word_stem_pad_tst))])
###################################################################################################
proc = st.Processing(log.pathtime, device)

words_stems = np.concatenate((df.word.values, df.stem.values), axis=0)
words_stems_padsize = max([len(item) for item in words_stems])
word_stem_pad, word_stem_mask, word_stem_vocsize = proc.char_preproc(words_stems, words_stems_padsize)
word_pad  = word_stem_pad[:len(df.word.values)]
word_mask = word_stem_mask[:len(df.word.values)]
stem_pad  = word_stem_pad[len(df.word.values):]
stem_mask = word_stem_mask[len(df.word.values):]
# print(word_stem_pad.shape, word_pad.shape, word_pad[:3], df.word[:3], stem_pad.shape, stem_pad[:3], df.stem[:3])

word_pad_trn = word_pad[df.set == 'trn']
word_pad_dev = word_pad[df.set == 'dev']
word_pad_tst = word_pad[df.set == 'hard-tst']

word_mask_trn = word_mask[df.set == 'trn']
word_mask_dev = word_mask[df.set == 'dev']
word_mask_tst = word_mask[df.set == 'hard-tst']

stem_pad_trn = stem_pad[df.set == 'trn']
stem_pad_dev = stem_pad[df.set == 'dev']
stem_pad_tst = stem_pad[df.set == 'hard-tst']

stem_mask_trn = stem_mask[df.set == 'trn']
stem_mask_dev = stem_mask[df.set == 'dev']
stem_mask_tst = stem_mask[df.set == 'hard-tst']


word_pad_mv_trn = word_pad[(df.set == 'trn') & (df.mv != 2)]
word_pad_mv_dev = word_pad[(df.set == 'dev') & (df.mv != 2)]
word_pad_mv_tst = word_pad[(df.set == 'hard-tst') & (df.mv != 2)]

word_mask_mv_trn = word_mask[(df.set == 'trn') & (df.mv != 2)]
word_mask_mv_dev = word_mask[(df.set == 'dev') & (df.mv != 2)]
word_mask_mv_tst = word_mask[(df.set == 'hard-tst') & (df.mv != 2)]

stem_pad_mv_trn = stem_pad[(df.set == 'trn') & (df.mv != 2)]
stem_pad_mv_dev = stem_pad[(df.set == 'dev') & (df.mv != 2)]
stem_pad_mv_tst = stem_pad[(df.set == 'hard-tst') & (df.mv != 2)]

stem_mask_mv_trn = stem_mask[(df.set == 'trn') & (df.mv != 2)]
stem_mask_mv_dev = stem_mask[(df.set == 'dev') & (df.mv != 2)]
stem_mask_mv_tst = stem_mask[(df.set == 'hard-tst') & (df.mv != 2)]

###################################################################################################

y_hard     = df.gold.to_numpy()
y_hard_trn = df.gold[df.set == 'trn'].to_numpy()
y_hard_dev = df.gold[df.set == 'dev'].to_numpy()
y_hard_tst = df.gold[df.set == 'hard-tst'].to_numpy()

y_hard_mv_trn = df.mv[(df.set == 'trn') & (df.mv != 2)].to_numpy()
y_hard_mv_dev = df.mv[(df.set == 'dev') & (df.mv != 2)].to_numpy()
y_hard_mv_tst = df.mv[(df.set == 'hard-tst') & (df.mv != 2)].to_numpy()

y_soft     = df[['soft1', 'soft0']].to_numpy()
y_soft_trn = df[['soft1', 'soft0']][df.set == 'trn'].to_numpy()
y_soft_dev = df[['soft1', 'soft0']][df.set == 'dev'].to_numpy()
y_soft_tst = df[['soft1', 'soft0']][df.set == 'hard-tst'].to_numpy()

y_soft_mv_trn = df[['soft1', 'soft0']][(df.set == 'trn') & (df.mv != 2)].to_numpy()
y_soft_mv_dev = df[['soft1', 'soft0']][(df.set == 'dev') & (df.mv != 2)].to_numpy()
y_soft_mv_tst = df[['soft1', 'soft0']][(df.set == 'hard-tst') & (df.mv != 2)].to_numpy()

print(#f"{'char emb shape':.<28} {char_emb.shape}\n"
      #f"{'word pad shape':.<28} {word_pad.shape}\n{'stem pad shape':.<28} {stem_pad.shape}\n"
      #f"{'word_stem pad shape':.<28} {word_stem_pad.shape}\n"
      #f"{'word_stem pad trn shape':.<28} {word_stem_pad_trn.shape}\n{'word_stem pad dev shape':.<28} {word_stem_pad_dev.shape}\n{'word_stem pad tst shape':.<28} {word_stem_pad_tst.shape}\n"
      #f"{'word pad trn shape':.<28} {word_pad_trn.shape}\n{'word pad dev shape':.<28} {word_pad_dev.shape}\n{'word pad tst shape':.<28} {word_pad_tst.shape}\n"
      #f"{'stem pad trn shape':.<28} {stem_pad_trn.shape}\n{'stem pad dev shape':.<28} {stem_pad_dev.shape}\n{'stem pad tst shape':.<28} {stem_pad_tst.shape}\n"
      f"{'word_pad shape':.<25} {word_pad.shape}\n{'stem_pad shape':.<25} {stem_pad.shape}\n"
      f"{'word_stem_vocsize':.<25} {word_stem_vocsize}\n"
      f"{'y hard mv trn shape':.<28} {y_hard_mv_trn.shape}\n{'y mv dev shape':.<28} {y_hard_mv_dev.shape}\n{'y mv tst shape':.<28} {y_hard_mv_tst.shape}\n"
      f"{'y soft mv trn shape':.<28} {y_soft_mv_trn.shape}\n{'y mv dev shape':.<28} {y_soft_mv_dev.shape}\n{'y mv tst shape':.<28} {y_soft_mv_tst.shape}\n"
      f"{'y hard trn shape':.<28} {y_hard_trn.shape}\n{'y hard dev shape':.<28} {y_hard_dev.shape}\n{'y hard tst shape':.<28} {y_hard_tst.shape}\n"
      f"{'y soft trn shape':.<28} {y_soft_trn.shape}\n{'y soft dev shape':.<28} {y_soft_dev.shape}\n{'y soft tst shape':.<28} {y_soft_tst.shape}"
)
###################################################################################################
# proc = st.Processing(log.pathtime, device)
y_size_soft = y_soft_trn.shape[1]
###################################################################################################

bootdata_gold = {'holdout': # defaultdict contiene la condizione sperimentale, le liste gli output di ogni esperimento
                       {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                        'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})},
             'crossval':
                       {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                        'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})}}

bootdata_mv = {'holdout': # defaultdict contiene la condizione sperimentale, le liste gli output di ogni esperimento
                       {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                        'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})},
             'crossval':
                       {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                        'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})}}

targs_all_goldhard_stl,     preds_all_goldhard_stl     = list(), list()
targs_all_goldhard_mtl_klr, preds_all_goldhard_mtl_klr = list(), list()
targs_all_goldhard_mtl_kli, preds_all_goldhard_mtl_kli = list(), list()
targs_all_goldhard_mtl_ce,  preds_all_goldhard_mtl_ce  = list(), list()
targs_all_goldmv_stl,       preds_all_goldmv_stl       = list(), list()
targs_all_goldmv_mtl_klr,   preds_all_goldmv_mtl_klr   = list(), list()
targs_all_goldmv_mtl_kli,   preds_all_goldmv_mtl_kli   = list(), list()
targs_all_goldmv_mtl_ce,    preds_all_goldmv_mtl_ce    = list(), list()
for iexp in range(args.experiments):
    seed = (iexp + 1) * 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    str_info = f"exp{str(iexp)}_"

    # model = mod.StlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    # lossfunc_hard = nn.BCELoss().to(device=device)
    # optimizer = optim.Adam(model.parameters(), lr=args.learate)
    # dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard],
    #                        [word_pad_trn, stem_pad_trn], [word_pad_dev, stem_pad_dev], [word_pad_tst, stem_pad_tst],
    #                        [y_hard_trn], [y_hard_dev], [y_hard_tst], [dtype_float], args.batsize, args.epochs, str_info=str_info + '_hardgold_')
    # for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata['holdout']['treatment'][f"{model.__class__.__name__}_"][k].append(v)
    #
    # model = mod.MtlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, y_size_soft, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    # lossfunc_hard = nn.BCELoss().to(device=device)
    # lossfunc_soft = mod.KLregular(device=device)
    # # lossfunc_soft = mod.KLinverse(device=device)
    # # lossfunc_soft = nn.KLDivLoss().to(device=device)
    # # lossfunc_soft = nn.CrossEntropyLoss().to(device=device)
    # optimizer = optim.Adam(model.parameters(), lr=args.learate)
    # dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
    #                        [word_pad_trn, stem_pad_trn], [word_pad_dev, stem_pad_dev], [word_pad_tst, stem_pad_tst],
    #                        [y_hard_trn, y_soft_trn], [y_hard_dev, y_soft_dev], [y_hard_tst, y_soft_tst], [dtype_float, dtype_float], args.batsize, args.epochs, str_info=str_info + '_hardgold_')
    # for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}"][k].append(v)
    #
    # model = mod.StlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    # lossfunc_hard = nn.BCELoss().to(device=device)
    # optimizer = optim.Adam(model.parameters(), lr=args.learate)
    # dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard],
    #                        [word_pad_mv_trn, stem_pad_mv_trn], [word_pad_mv_dev, stem_pad_mv_dev], [word_pad_mv_tst, stem_pad_mv_tst],
    #                        [y_hard_mv_trn], [y_hard_mv_dev], [y_hard_mv_tst], [dtype_float], args.batsize, args.epochs, str_info=str_info + '_hardmv_')
    # for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata['holdout']['treatment'][f"{model.__class__.__name__}_"][k].append(v)
    #
    # model = mod.MtlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, y_size_soft, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    # lossfunc_hard = nn.BCELoss().to(device=device)
    # lossfunc_soft = mod.KLregular(device=device)
    # # lossfunc_soft = mod.KLinverse(device=device)
    # # lossfunc_soft = nn.KLDivLoss().to(device=device)
    # # lossfunc_soft = nn.CrossEntropyLoss().to(device=device)
    # optimizer = optim.Adam(model.parameters(), lr=args.learate)
    # dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
    #                        [word_pad_mv_trn, stem_pad_mv_trn], [word_pad_mv_dev, stem_pad_mv_dev], [word_pad_mv_tst, stem_pad_mv_tst],
    #                        [y_hard_mv_trn, y_soft_mv_trn], [y_hard_mv_dev, y_soft_mv_dev], [y_hard_mv_tst, y_soft_mv_tst], [dtype_float, dtype_float], args.batsize, args.epochs, str_info=str_info + '_hardmv_')
    # for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}"][k].append(v)

    targs_exp_goldhard_stl,     preds_exp_goldhard_stl     = list(), list()
    targs_exp_goldhard_mtl_klr, preds_exp_goldhard_mtl_klr = list(), list()
    targs_exp_goldhard_mtl_kli, preds_exp_goldhard_mtl_kli = list(), list()
    targs_exp_goldhard_mtl_ce,  preds_exp_goldhard_mtl_ce  = list(), list()
    targs_exp_goldmv_stl,       preds_exp_goldmv_stl       = list(), list()
    targs_exp_goldmv_mtl_klr,   preds_exp_goldmv_mtl_klr   = list(), list()
    targs_exp_goldmv_mtl_kli,   preds_exp_goldmv_mtl_kli   = list(), list()
    targs_exp_goldmv_mtl_ce,    preds_exp_goldmv_mtl_ce    = list(), list()
    for ifold in range(args.splits):
        nfold = ifold + 1
        fold_tst = ifold
        fold_dev = ifold + 1 if ifold != args.splits - 1 else 0
        folds_trn = [i for i in range(args.splits) if i not in [fold_tst, fold_dev]]

        trn_df_ids = df.index[df.fold10.isin(folds_trn)]
        dev_df_ids = df.index[df.fold10 == fold_dev]
        tst_df_ids = df.index[df.fold10 == fold_tst]

        word_pad_trn = word_pad[trn_df_ids]
        word_pad_dev = word_pad[dev_df_ids]
        word_pad_tst = word_pad[tst_df_ids]

        word_mask_trn = word_mask[trn_df_ids]
        word_mask_dev = word_mask[dev_df_ids]
        word_mask_tst = word_mask[tst_df_ids]

        stem_pad_trn = stem_pad[trn_df_ids]
        stem_pad_dev = stem_pad[dev_df_ids]
        stem_pad_tst = stem_pad[tst_df_ids]

        stem_mask_trn = stem_mask[trn_df_ids]
        stem_mask_dev = stem_mask[dev_df_ids]
        stem_mask_tst = stem_mask[tst_df_ids]

        word_pad_mv_trn = word_pad[(df.index.isin(trn_df_ids)) & (df.mv != 2)]
        word_pad_mv_dev = word_pad[(df.index.isin(dev_df_ids)) & (df.mv != 2)]
        word_pad_mv_tst = word_pad[(df.index.isin(tst_df_ids)) & (df.mv != 2)]

        word_mask_mv_trn = word_mask[(df.index.isin(trn_df_ids)) & (df.mv != 2)]
        word_mask_mv_dev = word_mask[(df.index.isin(dev_df_ids)) & (df.mv != 2)]
        word_mask_mv_tst = word_mask[(df.index.isin(tst_df_ids)) & (df.mv != 2)]

        stem_pad_mv_trn = stem_pad[(df.index.isin(trn_df_ids)) & (df.mv != 2)]
        stem_pad_mv_dev = stem_pad[(df.index.isin(dev_df_ids)) & (df.mv != 2)]
        stem_pad_mv_tst = stem_pad[(df.index.isin(tst_df_ids)) & (df.mv != 2)]

        stem_mask_mv_trn = stem_mask[(df.index.isin(trn_df_ids)) & (df.mv != 2)]
        stem_mask_mv_dev = stem_mask[(df.index.isin(dev_df_ids)) & (df.mv != 2)]
        stem_mask_mv_tst = stem_mask[(df.index.isin(tst_df_ids)) & (df.mv != 2)]

###################################################################################################

        y_hard_trn = df.gold[df.index.isin(trn_df_ids)].to_numpy()
        y_hard_dev = df.gold[df.index.isin(dev_df_ids)].to_numpy()
        y_hard_tst = df.gold[df.index.isin(tst_df_ids)].to_numpy()

        y_hard_mv_trn = df.mv[df.index.isin(trn_df_ids) & (df.mv != 2)].to_numpy()
        y_hard_mv_dev = df.mv[df.index.isin(dev_df_ids) & (df.mv != 2)].to_numpy()
        y_hard_mv_tst = df.mv[df.index.isin(tst_df_ids) & (df.mv != 2)].to_numpy()

        y_soft_trn = df[['soft1', 'soft0']][df.index.isin(trn_df_ids)].to_numpy()
        y_soft_dev = df[['soft1', 'soft0']][df.index.isin(dev_df_ids)].to_numpy()
        y_soft_tst = df[['soft1', 'soft0']][df.index.isin(tst_df_ids)].to_numpy()

        y_soft_mv_trn = df[['soft1', 'soft0']][df.index.isin(trn_df_ids) & (df.mv != 2)].to_numpy()
        y_soft_mv_dev = df[['soft1', 'soft0']][df.index.isin(dev_df_ids) & (df.mv != 2)].to_numpy()
        y_soft_mv_tst = df[['soft1', 'soft0']][df.index.isin(tst_df_ids) & (df.mv != 2)].to_numpy()

        print(f"{'word_pad shape':.<25} {word_pad.shape}\n{'stem_pad shape':.<25} {stem_pad.shape}\n"
              f"{'word_stem_vocsize':.<25} {word_stem_vocsize}\n"
              f"{'y hard mv trn shape':.<28} {y_hard_mv_trn.shape}\n{'y mv dev shape':.<28} {y_hard_mv_dev.shape}\n{'y mv tst shape':.<28} {y_hard_mv_tst.shape}\n"
              f"{'y soft mv trn shape':.<28} {y_soft_mv_trn.shape}\n{'y mv dev shape':.<28} {y_soft_mv_dev.shape}\n{'y mv tst shape':.<28} {y_soft_mv_tst.shape}\n"
              f"{'y hard trn shape':.<28} {y_hard_trn.shape}\n{'y hard dev shape':.<28} {y_hard_dev.shape}\n{'y hard tst shape':.<28} {y_hard_tst.shape}\n"
              f"{'y soft trn shape':.<28} {y_soft_trn.shape}\n{'y soft dev shape':.<28} {y_soft_dev.shape}\n{'y soft tst shape':.<28} {y_soft_tst.shape}"
        )

        model = mod.StlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
        lossfunc_hard = nn.BCELoss().to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.learate)
        dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard],
                               [word_pad_trn, stem_pad_trn], [word_pad_dev, stem_pad_dev], [word_pad_tst, stem_pad_tst],
                               [y_hard_trn], [y_hard_dev], [y_hard_tst], [dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardgold_fold' + str(nfold) + '_')
        for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['holdout']['control'][f"{model.__class__.__name__}_hardgold"][k].append(v)
        preds_exp_goldhard_stl.extend(preds)
        targs_exp_goldhard_stl.extend(targs)

        model = mod.MtlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, y_size_soft, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
        lossfunc_hard = nn.BCELoss().to(device=device)
        lossfunc_soft = mod.KLregular(device=device)
        # lossfunc_soft = mod.KLinverse(device=device)
        # lossfunc_soft = nn.KLDivLoss().to(device=device)
        # lossfunc_soft = nn.CrossEntropyLoss().to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.learate)
        dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                               [word_pad_trn, stem_pad_trn], [word_pad_dev, stem_pad_dev], [word_pad_tst, stem_pad_tst],
                               [y_hard_trn, y_soft_trn], [y_hard_dev, y_soft_dev], [y_hard_tst, y_soft_tst], [dtype_float, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardgold_fold' + str(nfold) + '_')
        for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardgold"][k].append(v)
        preds_exp_goldhard_mtl_klr.extend(preds)
        targs_exp_goldhard_mtl_klr.extend(targs)

        model = mod.MtlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, y_size_soft, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
        lossfunc_hard = nn.BCELoss().to(device=device)
        # lossfunc_soft = mod.KLregular(device=device)
        lossfunc_soft = mod.KLinverse(device=device)
        # lossfunc_soft = nn.KLDivLoss().to(device=device)
        # lossfunc_soft = nn.CrossEntropyLoss().to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.learate)
        dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                               [word_pad_trn, stem_pad_trn], [word_pad_dev, stem_pad_dev], [word_pad_tst, stem_pad_tst],
                               [y_hard_trn, y_soft_trn], [y_hard_dev, y_soft_dev], [y_hard_tst, y_soft_tst], [dtype_float, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardgold_fold' + str(nfold) + '_')
        for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardgold"][k].append(v)
        preds_exp_goldhard_mtl_kli.extend(preds)
        targs_exp_goldhard_mtl_kli.extend(targs)

        model = mod.MtlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, y_size_soft, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
        lossfunc_hard = nn.BCELoss().to(device=device)
        lossfunc_soft = mod.CrossEntropy(device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.learate)
        dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                               [word_pad_trn, stem_pad_trn], [word_pad_dev, stem_pad_dev], [word_pad_tst, stem_pad_tst],
                               [y_hard_trn, y_soft_trn], [y_hard_dev, y_soft_dev], [y_hard_tst, y_soft_tst], [dtype_float, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardgold_fold' + str(nfold) + '_')
        for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardgold"][k].append(v)
        preds_exp_goldhard_mtl_ce.extend(preds)
        targs_exp_goldhard_mtl_ce.extend(targs)



        model = mod.StlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
        lossfunc_hard = nn.BCELoss().to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.learate)
        dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard],
                               [word_pad_mv_trn, stem_pad_mv_trn], [word_pad_mv_dev, stem_pad_mv_dev], [word_pad_mv_tst, stem_pad_mv_tst],
                               [y_hard_mv_trn], [y_hard_mv_dev], [y_hard_mv_tst], [dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardmv_fold' + str(nfold) + '_')
        for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['holdout']['control'][f"{model.__class__.__name__}_hardmv"][k].append(v)
        preds_exp_goldmv_stl.extend(preds)
        targs_exp_goldmv_stl.extend(targs)

        model = mod.MtlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, y_size_soft, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
        lossfunc_hard = nn.BCELoss().to(device=device)
        lossfunc_soft = mod.KLregular(device=device)
        # lossfunc_soft = mod.KLinverse(device=device)
        # lossfunc_soft = nn.KLDivLoss().to(device=device)
        # lossfunc_soft = nn.CrossEntropyLoss().to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.learate)
        dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                               [word_pad_mv_trn, stem_pad_mv_trn], [word_pad_mv_dev, stem_pad_mv_dev], [word_pad_mv_tst, stem_pad_mv_tst],
                               [y_hard_mv_trn, y_soft_mv_trn], [y_hard_mv_dev, y_soft_mv_dev], [y_hard_mv_tst, y_soft_mv_tst], [dtype_float, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardmv_fold' + str(nfold) + '_')
        for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardmv"][k].append(v)
        preds_exp_goldmv_mtl_klr.extend(preds)
        targs_exp_goldmv_mtl_klr.extend(targs)

        model = mod.MtlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, y_size_soft, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
        lossfunc_hard = nn.BCELoss().to(device=device)
        # lossfunc_soft = mod.KLregular(device=device)
        lossfunc_soft = mod.KLinverse(device=device)
        # lossfunc_soft = nn.KLDivLoss().to(device=device)
        # lossfunc_soft = nn.CrossEntropyLoss().to(device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.learate)
        dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                               [word_pad_mv_trn, stem_pad_mv_trn], [word_pad_mv_dev, stem_pad_mv_dev], [word_pad_mv_tst, stem_pad_mv_tst],
                               [y_hard_mv_trn, y_soft_mv_trn], [y_hard_mv_dev, y_soft_mv_dev], [y_hard_mv_tst, y_soft_mv_tst], [dtype_float, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardmv_fold' + str(nfold) + '_')
        for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardmv"][k].append(v)
        preds_exp_goldmv_mtl_kli.extend(preds)
        targs_exp_goldmv_mtl_kli.extend(targs)

        model = mod.MtlEmbsConvEnc2Stem(word_stem_vocsize, args.embsize, 1, y_size_soft, conv_channels=args.conv_channels, filter_sizes=args.conv_filter_sizes, conv_stridesizes=args.conv_stridesizes, pool_filtersizes=args.pool_filtersizes, pool_stridesizes=args.pool_stridesizes, att_heads=args.att_heads, att_layers=args.att_layers, nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
        lossfunc_hard = nn.BCELoss().to(device=device)
        lossfunc_soft = mod.CrossEntropy(device=device)
        optimizer = optim.Adam(model.parameters(), lr=args.learate)
        dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                               [word_pad_mv_trn, stem_pad_mv_trn], [word_pad_mv_dev, stem_pad_mv_dev], [word_pad_mv_tst, stem_pad_mv_tst],
                               [y_hard_mv_trn, y_soft_mv_trn], [y_hard_mv_dev, y_soft_mv_dev], [y_hard_mv_tst, y_soft_mv_tst], [dtype_float, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardmv_fold' + str(nfold) + '_')
        for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardmv"][k].append(v)
        preds_exp_goldmv_mtl_ce.extend(preds)
        targs_exp_goldmv_mtl_ce.extend(targs)

    proc.metrics(targs_exp_goldhard_stl,     preds_exp_goldhard_stl,     'goldhard_stl')
    proc.metrics(targs_exp_goldhard_mtl_klr, preds_exp_goldhard_mtl_klr, 'goldhard_mtl_klr')
    proc.metrics(targs_exp_goldhard_mtl_kli, preds_exp_goldhard_mtl_kli, 'goldhard_mtl_kli')
    proc.metrics(targs_exp_goldhard_mtl_ce,  preds_exp_goldhard_mtl_ce,  'goldhard_mtl_ce')
    proc.metrics(targs_exp_goldmv_stl,       preds_exp_goldmv_stl,       'goldmv_stl')
    proc.metrics(targs_exp_goldmv_mtl_klr,   preds_exp_goldmv_mtl_klr,   'goldv_mtl_klr')
    proc.metrics(targs_exp_goldmv_mtl_kli,   preds_exp_goldmv_mtl_kli,   'goldmv_mtl_kli')
    proc.metrics(targs_exp_goldmv_mtl_ce,    preds_exp_goldmv_mtl_ce,    'goldmv_mtl_ce')
    preds_all_goldhard_stl.extend(preds_exp_goldhard_stl)
    targs_all_goldhard_stl.extend(targs_exp_goldhard_stl)
    preds_all_goldhard_mtl_klr.extend(preds_exp_goldhard_mtl_klr)
    targs_all_goldhard_mtl_klr.extend(targs_exp_goldhard_mtl_klr)
    preds_all_goldhard_mtl_kli.extend(preds_exp_goldhard_mtl_kli)
    targs_all_goldhard_mtl_kli.extend(targs_exp_goldhard_mtl_kli)
    preds_all_goldhard_mtl_ce.extend(preds_exp_goldhard_mtl_ce)
    targs_all_goldhard_mtl_ce.extend(targs_exp_goldhard_mtl_ce)
    preds_all_goldmv_stl.extend(preds_exp_goldmv_stl)
    targs_all_goldmv_stl.extend(targs_exp_goldmv_stl)
    preds_all_goldmv_mtl_klr.extend(preds_exp_goldmv_mtl_klr)
    targs_all_goldmv_mtl_klr.extend(targs_exp_goldmv_mtl_klr)
    preds_all_goldmv_mtl_kli.extend(preds_exp_goldmv_mtl_kli)
    targs_all_goldmv_mtl_kli.extend(targs_exp_goldmv_mtl_kli)
    preds_all_goldmv_mtl_ce.extend(preds_exp_goldmv_mtl_ce)
    targs_all_goldmv_mtl_ce.extend(targs_exp_goldmv_mtl_ce)

print(f"{'#'*80}\nall experiments results:\n")
proc.metrics(targs_all_goldhard_stl,     preds_all_goldhard_stl,     'goldhard_stl')
proc.metrics(targs_all_goldhard_mtl_klr, preds_all_goldhard_mtl_klr, 'goldhard_mtl_klr')
proc.metrics(targs_all_goldhard_mtl_kli, preds_all_goldhard_mtl_kli, 'goldhard_mtl_kli')
proc.metrics(targs_all_goldhard_mtl_ce,  preds_all_goldhard_mtl_ce,  'goldhard_mtl_ce')
proc.metrics(targs_all_goldmv_stl,       preds_all_goldmv_stl,       'goldmv_stl')
proc.metrics(targs_all_goldmv_mtl_klr,   preds_all_goldmv_mtl_klr,   'goldv_mtl_klr')
proc.metrics(targs_all_goldmv_mtl_kli,   preds_all_goldmv_mtl_kli,   'goldmv_mtl_kli')
proc.metrics(targs_all_goldmv_mtl_ce,    preds_all_goldmv_mtl_ce,    'goldmv_mtl_ce')

print(f"{'#'*80}\nbootstraps:\n")
ut.writejson(bootdata_gold, log.pathtime + 'bootstrap_gold_input.json')
proc.bootstrap(bootdata_gold, args.bootloop, args.perc_sample)
ut.writejson(bootdata_mv, log.pathtime + 'bootstrap_mv_input.json')
proc.bootstrap(bootdata_mv, args.bootloop, args.perc_sample)

ut.end(startime)



