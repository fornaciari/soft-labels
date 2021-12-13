# coding=latin-1
import util200818 as ut
import step200928 as st
import models200928 as mod
import argparse, os, re, sys, time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, load_npz
import pandas as pd
import random
import torch
import torch.nn as nn
from torch import optim
# import warnings filter
from warnings import simplefilter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, log_loss, confusion_matrix
# ignore all future warnings
simplefilter(action='ignore', category=(FutureWarning, UserWarning))
###################################################################################################
parser = argparse.ArgumentParser()
# inputs
parser.add_argument("-word_emb_matrix", type=str, default="/inputs/190801161330/dh_emb_matrix.bin")
parser.add_argument("-word_pad_matrix", type=str, default="/inputs/190801161330/dh_pad_csr.bin")
parser.add_argument("-word_id_in_sent", type=str, default="/inputs/190801161330/dh_i_word_in_sent.bin")
parser.add_argument("-char_emb_matrix", type=str, default="/inputs/190801163604/emb_matrix.bin")
parser.add_argument("-char_pad_matrix", type=str, default="/inputs/190801163604/dh_pad_csr.bin")
parser.add_argument("-path_xls", type=str, default='/inputs/200512130637/dh.xlsx')
parser.add_argument("-y_name",   type=str, default='gold')
# torch settings
parser.add_argument("-seed",   type=int, default=1234)
parser.add_argument("-device", type=str, default='cuda:2')
parser.add_argument("-dtype",  type=int, default=32, choices=[32, 64])
# preproc
parser.add_argument("-word_padsize", type=int,  default=100)
parser.add_argument("-char_padsize", type=int,  default=15)
parser.add_argument("-word_vocsize", type=int,  default=25000)
parser.add_argument("-word_embsize", type=int,  default=300)
parser.add_argument("-char_embsize", type=int,  default=100)
parser.add_argument("-trainable",    type=bool, default=True)
# model settings
parser.add_argument("-save",        type=bool,  default=False)
parser.add_argument("-experiments", type=int,   default=10)
parser.add_argument("-epochs",      type=int,   default=10)
parser.add_argument("-splits",      type=int,   default=5, help='almeno 3 o dà un errore, credo dovuto all\'output dello stratified')
parser.add_argument("-batsize",     type=int,   default=1024)
parser.add_argument("-learate",     type=float, default=0.001)
parser.add_argument("-droprob",     type=float, default=0.1)
# lstm settings
parser.add_argument("-lstm_layers", type=int,   default=1)
parser.add_argument("-lstm_size",   type=int,   default=256)
# attention
parser.add_argument("-att_layers",  type=int, default=2)
parser.add_argument("-att_heads",   type=int, default=2)
# fc
parser.add_argument("-mlp_layers", type=int, default=2)
# bootstrap
parser.add_argument("-bootloop",    type=int, default=1000)
parser.add_argument("-perc_sample", type=float, default=.33)

args = parser.parse_args()
sys.stdout = sys.stderr = log = ut.log(__file__, ut.__file__, st.__file__, mod.__file__)
startime = ut.start()
align_size = ut.print_args(args)
print(f"{'dirout':.<{align_size}} {log.pathtime}")
###################################################################################################
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
dtype_float = torch.float64 if args.dtype == 64 else torch.float32
dtype_int = torch.int64 # if args.dtype == 64 else torch.int32 # o 64 o s'inkazza
device = torch.device(args.device if torch.cuda.is_available() else "cpu")
print(f"{'GPU in use':.<{align_size}} {device}\n{'#'*80}" if torch.cuda.is_available() else f"No GPU available, using the CPU.\n{'#'*80}")
###################################################################################################
proc = st.Processing(log.pathtime, device)
###################################################################################################
df = pd.read_excel(args.path_xls)
df = df[df.id_word_in_sent < args.word_padsize] # taglio le frasi più lunghe di padsize, o andrà out of range
print(df.iloc[:5, :15])
print(df.shape)
# print(df.columns.values)
# print(set(df.set))
soft_cols = [col for col in df.columns if re.match('soft_', col)]

###################################################################################################
# inputs ACL
word_emb = ut.readbin(args.word_emb_matrix)
word_pad = ut.readbin(args.word_pad_matrix)
char_emb = ut.readbin(args.char_emb_matrix)
char_pad = ut.readbin(args.char_pad_matrix)
print('word_emb shape', word_emb.shape)#, type(word_emb))
print('word_pad shape', word_pad.shape)#, type(word_pad))
print('char_emb shape', char_emb.shape)#, type(char_emb))
print('char_pad shape', char_pad.shape)#, type(char_pad))

first_tst = 7877 # 7877 è il nr di righe in annotation/soft-tst per ACL, ora  mi rusulta 7876. non importa perché non lo uso.
first_trn = 7877 + 3064 # 3064 è lowlands/hard-tst,
first_dev = 7877 + 3064 + 14439 - 1439 # sto usando un dev set diverso da prima: prima erano le prime 2439 righe del trn, ora le ultime 1439

word_pad_tst    = word_pad[first_tst:first_trn].toarray()
word_pad_trn    = word_pad[first_trn:first_dev].toarray()
word_pad_dev    = word_pad[first_dev:].toarray()

char_pad_tst    = char_pad[first_tst:first_trn].toarray()
char_pad_trn    = char_pad[first_trn:first_dev].toarray()
char_pad_dev    = char_pad[first_dev:].toarray()

print('word_pad_tst shape', word_pad_tst.shape)#, type(word_emb))
print('word_pad_trn shape', word_pad_trn.shape)#, type(word_pad))
print('word_pad_dev shape', word_pad_dev.shape)#, type(word_pad))
print('char_pad_tst shape', char_pad_tst.shape)#, type(char_emb))
print('char_pad_trn shape', char_pad_trn.shape)#, type(char_pad))
print('char_pad_dev shape', char_pad_dev.shape)#, type(char_pad))
###################################################################################################
# inputs xls
y_mv_trn = df['mv'][df.set == 'trn'].to_numpy()
y_gold_trn = df['gold'][df.set == 'trn'].to_numpy()
y_gold_dev = df['gold'][df.set == 'dev'].to_numpy()
y_gold_tst = df['gold'][df.set == 'hard-tst'].to_numpy()

y_soft_trn = df[soft_cols][df.set == 'trn'].to_numpy()
y_soft_dev = df[soft_cols][df.set == 'dev'].to_numpy()
y_soft_tst = df[soft_cols][df.set == 'hard-tst'].to_numpy()

id_word_in_sent_trn = df.id_word_in_sent[df.set == 'trn'].to_numpy()
id_word_in_sent_dev = df.id_word_in_sent[df.set == 'dev'].to_numpy()
id_word_in_sent_tst = df.id_word_in_sent[df.set == 'hard-tst'].to_numpy()

# mask_trn = np.array([[1 if ipad <= id_word_in_sent_trn[irow] else 0 for ipad in range(word_pad_trn.shape[1])] for irow in range(len(id_word_in_sent_trn))])
# mask_dev = np.array([[1 if ipad <= id_word_in_sent_dev[irow] else 0 for ipad in range(word_pad_trn.shape[1])] for irow in range(len(id_word_in_sent_dev))])
# mask_tst = np.array([[1 if ipad <= id_word_in_sent_tst[irow] else 0 for ipad in range(word_pad_trn.shape[1])] for irow in range(len(id_word_in_sent_tst))])

print(f"{'y_gold_trn shape':.<25} {y_gold_trn.shape}")
print(f"{'y_gold_dev shape':.<25} {y_gold_dev.shape}")
print(f"{'y_gold_tst shape':.<25} {y_gold_tst.shape}")
print(f"{'y_soft_trn shape':.<25} {y_soft_trn.shape}")
print(f"{'y_soft_dev shape':.<25} {y_soft_dev.shape}")
print(f"{'y_soft_tst shape':.<25} {y_soft_tst.shape}")
print(f"{'id_word_in_sent_trn shape':.<25} {id_word_in_sent_trn.shape}")
print(f"{'id_word_in_sent_dev shape':.<25} {id_word_in_sent_dev.shape}")
print(f"{'id_word_in_sent_tst shape':.<25} {id_word_in_sent_tst.shape}")

###################################################################################################
df_crossval = df[df.set.isin(['trn', 'dev'])]
# df_crossval.reset_index(inplace=True) # in realtà non necessario perché 'trn' e 'dev' sono i primi set
print(df_crossval.iloc[-5:, :15])
print(df_crossval.shape)# texts_ids, texts_masks, y, y_size = token_preproc(df_crossval.sent, df_crossval.gold, args.word_vocsize, args.word_padsize)

# word_pad, word_mask, _ = proc.realigned_token_preproc(df_crossval.sent, args.word_vocsize, args.word_padsize)
# char_pad, char_mask, chars_vocsize = proc.char_preproc(df_crossval.word, args.char_padsize) # chars_vocsize in teoria potrebbe essere più piccolo di holdout, perché non ha tst-hard
word_pad = word_pad[first_trn:].toarray()
char_pad = char_pad[first_trn:].toarray()
id_word_in_sent = df_crossval.id_word_in_sent.to_numpy()
print(f"{'word_pad shape':.<25} {word_pad.shape}")
print(f"{'char_pad shape':.<25} {char_pad.shape}")
print(f"{'id_word_in_sent shape':.<25} {id_word_in_sent.shape}")

y_mv   = df_crossval.mv.to_numpy()
y_gold = df_crossval.gold.to_numpy()
y_soft = df_crossval[soft_cols].to_numpy()
y_size = len(set(y_gold))
print(f"{'y soft shape':.<25} {y_soft.shape}")
print(f"{'y hard shape':.<25} {y_gold.shape}")
print(f"{'y hard size':.<25} {y_size}")
###################################################################################################


def batches(step, model, optimizer, lossfuncs, x_inputs_step, y_inputs_step, y_dtypes, batsize):
    preds = list()
    losss = list()
    model.train() if step == 'trn' else model.eval()
    # for x in x_inputs_step: print(x.shape)
    # for y in y_inputs_step: print(y.shape)
    for ifir_bat in tqdm(range(0, len(y_inputs_step[0]), batsize), desc=step, ncols=80): # desc='training' # prefix
        nlas_bat = ifir_bat + batsize
        x_inputs_bat = [torch.from_numpy(x[ifir_bat: nlas_bat]).to(device=device, dtype=torch.int64) for x in x_inputs_step]
        y_inputs_bat = [torch.from_numpy(y[ifir_bat: nlas_bat]).to(device=device, dtype=dt) for y, dt in zip(y_inputs_step, y_dtypes)]
        # for x in x_inputs_bat: print(x.shape)
        # for y in y_inputs_bat: print(y.shape)
        pred_bat = model(*x_inputs_bat)
        if len(lossfuncs) > 1:
            # for lossf, pred, y in zip(lossfuncs, pred_bat, y_inputs_bat): print(lossf, pred, y, pred.shape, y.shape)
            loss_bat = [lossf(pred, y) for lossf, pred, y in zip(lossfuncs, pred_bat, y_inputs_bat)]
            # print(loss_bat)
            pred_bat = pred_bat[0].argmax(1).data.tolist() if len(pred_bat[0].shape) > 1 else [0 if v < .5 else 1 for v in pred_bat[0].data.tolist()] # se le pred hanno righe e colonne, le trasformo in int. la tupla diventa un solo vettore, ma tanto il secondo che perdo non viene usato per misurare la performance, e l'ho già usato nella lossfunc
            loss_bat4list = [loss.item() for loss in loss_bat]
        else:
            loss_bat = lossfuncs[0](pred_bat, y_inputs_bat[0])
            pred_bat = pred_bat.argmax(1).data.tolist() if len(pred_bat.shape) > 1 else [0 if v < .5 else 1 for v in pred_bat.data.tolist()]
            loss_bat4list = loss_bat.item()
        if step == 'trn':
            if isinstance(loss_bat, list):
                # distinct back propagations
                # for i, loss in enumerate(loss_bat):
                #     loss.backward(retain_graph=True) # retain_graph sarebbe sufficiente solo per la prima back, ma pare che la seconda non faccia danni. E serve sole per nn.KLDivLoss(): le mie loss, con require_grad, potrebbero farne a meno
                #     optimizer.step()
                # overall back propagation
                optimizer.zero_grad()
                sum(loss_bat).backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                loss_bat.backward()
                optimizer.step()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        preds.extend(pred_bat)
        losss.append(loss_bat4list)
    y_inputs = y_inputs_step[0].argmax(1).data.tolist() if len(y_inputs_step[0].shape) > 1 else y_inputs_step[0] # solo per la replica di lalor. normalmente ogni y è int, non un prob dist
    accu = round(accuracy_score(y_inputs, preds) * 100, 2)
    fmea = round(f1_score(y_inputs, preds, average='macro') * 100, 2)
    loss = [round(np.array(losss)[:, 0].mean(), 4), round(np.array(losss)[:, 1].mean(), 4)] if isinstance(losss[0], list) else round(np.mean(losss), 4)
    str_bat = f"{'loss hard':.<12} {loss[0]:<10}{'loss soft':.<12} {loss[1]:<10}{'accuracy':.<11} {accu:<10}{'f1':.<5} {fmea:<10}" if isinstance(loss, list) else \
              f"{'loss hard':.<12} {loss:<10}{'accuracy':.<11} {accu:<10}{'f1':.<5} {fmea}"
    print(str_bat)
    serie_out = pd.Series({'loss_hard': round(loss[0], 4), 'loss_soft': round(loss[1], 4), 'acc': round(accu, 2), 'f1': round(fmea, 2)}, name=step) if isinstance(loss, list) else \
                pd.Series({'loss': round(loss, 4), 'acc': round(accu, 2), 'f1': round(fmea, 2)}, name=step)
    return serie_out, preds


def crossval_mv(model, optimizer, lossfuncs, x_inputs, y_inputs_gold, y_inputs_mv, y_dtypes, batsize, n_epochs=10, n_splits=10, save=False, str_info=''):
    strloss = '' if len(lossfuncs) < 2 else '_' + lossfuncs[1].__class__.__name__
    dir_exp = log.pathtime + str_info + 'epoch' + str(n_epochs) + '_' + crossval_mv.__name__ + '_' + model.__class__.__name__ + strloss
    print(f"{'dir out':.<25} {dir_exp}")
    os.mkdir(dir_exp)
    df_trn_epochs, df_dev_epochs, df_tst_epochs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    # preds_out, targs_out = list(), list()

    path_virgin = f"{log.pathtime}virgin_model.pt"
    torch.save(model.state_dict(), path_virgin)
    preds_all, targs_all = list(), list()
    df_all = pd.DataFrame()
    fold = 0
    skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
    for i_trn_dev, i_tst in skf.split(x_inputs[0], y_inputs_mv[0]):
        fold += 1
        max_metric = 0
        print(f"{'#'*80}\nfold {fold}")
        dir_fold = f"{dir_exp}/fold{fold}/"
        os.mkdir(dir_fold)
        path_model = f"{dir_fold}model.pt"
        path_results = f"{dir_fold}results.pdf"
        dev_rate = int(len(y_inputs_mv[0]) * (1 / n_splits))
        shuffled_indices = torch.randperm(len(i_trn_dev)) # skf mescola, ma anche ordina: rimescolo
        i_dev = i_trn_dev[shuffled_indices][-dev_rate:]
        i_trn = i_trn_dev[shuffled_indices][:-dev_rate]
        x_inputs_trn = [x[i_trn] for x in x_inputs]
        x_inputs_dev = [x[i_dev] for x in x_inputs]
        x_inputs_tst = [x[i_tst] for x in x_inputs]
        y_inputs_trn = [y[i_trn] for y in y_inputs_mv]
        y_inputs_dev = [y[i_dev] for y in y_inputs_mv]
        y_inputs_tst = [y[i_tst] for y in y_inputs_gold]
        # for x in x_inputs_trn: print(x.shape)
        # for y in y_inputs_trn: print(y.shape)

        virgin = torch.load(path_virgin)
        model.load_state_dict(virgin)

        df_fold, df_trn_fold, df_dev_fold, df_tst_fold = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        preds_fold, targs_fold = list(), list()
        for epoch in range(1, n_epochs + 1):
            print(f"epoch {epoch}")

            serie_trn, _,        = batches('trn', model, optimizer, lossfuncs, x_inputs_trn, y_inputs_trn, y_dtypes, batsize)
            serie_dev, _,        = batches('dev', model, optimizer, lossfuncs, x_inputs_dev, y_inputs_dev, y_dtypes, batsize)
            serie_tst, preds_tst = batches('tst', model, optimizer, lossfuncs, x_inputs_tst, y_inputs_tst, y_dtypes, batsize)
            df_trn_fold = df_trn_fold.append(serie_trn)
            df_dev_fold = df_dev_fold.append(serie_dev)
            df_tst_fold = df_tst_fold.append(serie_tst)
            if serie_dev['f1'] >= max_metric:# and epoch > 1:
                if save:
                    torch.save(model.state_dict(), path_model)
                    print('model saved')
                print('results selected')
                df_fold    = serie_tst.to_frame().transpose()
                preds_fold = preds_tst
                targs_fold = y_inputs_tst[0].astype(int).tolist()
                max_metric = serie_dev['f1'].mean()

        preds_all.extend(preds_fold)
        targs_all.extend(targs_fold)
        df_all = df_all.append(df_fold)

    os.system(f"rm {path_virgin}")
    ut.list2file(preds_all, dir_exp + '/preds.txt')
    ut.list2file(targs_all, dir_exp + '/targs.txt')
    str_out = f"{'loss hard':.<12} {round(df_all.loss_hard.mean(), 4):<10}{'loss soft':.<12} {round(df_all.loss_soft.mean(), 4):<10}{'accuracy':.<11} {round(accuracy_score(targs_all, preds_all), 4):<10}{'f1':.<5} {round(f1_score(targs_all, preds_all, average='macro'), 4)}" if 'loss_soft' in df_all.columns else \
              f"{'loss hard':.<12} {round(df_all.loss.mean(), 4):<10}{'accuracy':.<11} {round(accuracy_score(targs_all, preds_all), 4):<10}{'f1':.<5} {round(f1_score(targs_all, preds_all, average='macro'), 4)}"
    proc.metrics(targs_all, preds_all)

    dir_out_results = dir_exp + str(round(accuracy_score(targs_all, preds_all), 2))
    os.rename(dir_exp, dir_out_results)
    print(f"{df_all.to_string()}\n{str_out}\nscp bocconi:mimac/{dir_out_results + '/fold_results.pdf'} ./pdf/{''.join(dir_exp.split('/'))}.pdf")
    ut.sendslack(f"{dir_exp} done\n{str_out}")

    # print(f"macro precision_recall_fscore_support:\n{precision_recall_fscore_support(targs_all, preds_all, average='macro')}")
    # print(f"micro precision_recall_fscore_support:\n{precision_recall_fscore_support(targs_all, preds_all, average='micro')}")
    # print("accuracy", round(accuracy_score(targs_all, preds_all) * 100, rounding_value))

    return dir_out_results, preds_all, targs_all

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

targs_crossval_hardgold_stl,     preds_crossval_hardgold_stl     = list(), list()
targs_crossval_hardgold_mtl_klr, preds_crossval_hardgold_mtl_klr = list(), list()
targs_crossval_hardgold_mtl_kli, preds_crossval_hardgold_mtl_kli = list(), list()
targs_crossval_hardgold_mtl_ce,  preds_crossval_hardgold_mtl_ce  = list(), list()
targs_crossval_hardmv_stl,       preds_crossval_hardmv_stl     = list(), list()
targs_crossval_hardmv_mtl_klr,   preds_crossval_hardmv_mtl_klr = list(), list()
targs_crossval_hardmv_mtl_kli,   preds_crossval_hardmv_mtl_kli = list(), list()
targs_crossval_hardmv_mtl_ce,    preds_crossval_hardmv_mtl_ce  = list(), list()
targs_holdout_hardgold_stl,      preds_holdout_hardgold_stl      = list(), list()
targs_holdout_hardgold_mtl_klr,  preds_holdout_hardgold_mtl_klr  = list(), list()
targs_holdout_hardgold_mtl_kli,  preds_holdout_hardgold_mtl_kli  = list(), list()
targs_holdout_hardgold_mtl_ce,   preds_holdout_hardgold_mtl_ce   = list(), list()
targs_holdout_hardmv_stl,        preds_holdout_hardmv_stl        = list(), list()
targs_holdout_hardmv_mtl_klr,    preds_holdout_hardmv_mtl_klr    = list(), list()
targs_holdout_hardmv_mtl_kli,    preds_holdout_hardmv_mtl_kli    = list(), list()
targs_holdout_hardmv_mtl_ce,     preds_holdout_hardmv_mtl_ce     = list(), list()
for iexp in range(args.experiments):
    seed = (iexp + 1 ) * 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    str_info = f"exp{str(iexp)}_"

# crossval ########################################################################################

    model = mod.StlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.crossval(model, optimizer, [lossfunc_hard],
                           [word_pad, char_pad, id_word_in_sent],
                           [y_gold], [dtype_int], args.batsize, n_epochs=args.epochs, n_splits=args.splits, save=args.save, str_info=str_info + 'hardgold_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['crossval']['control'][f"{model.__class__.__name__}_hardgold"][k].append(v)
    preds_crossval_hardgold_stl.extend(preds)
    targs_crossval_hardgold_stl.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.KLregular(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.crossval(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad, char_pad, id_word_in_sent],
                           [y_gold, y_soft], [dtype_int, dtype_float], args.batsize, n_epochs=args.epochs, n_splits=args.splits, save=args.save, str_info=str_info + 'hardgold_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['crossval']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardgold"][k].append(v)
    preds_crossval_hardgold_mtl_klr.extend(preds)
    targs_crossval_hardgold_mtl_klr.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.KLinverse(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.crossval(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad, char_pad, id_word_in_sent],
                           [y_gold, y_soft], [dtype_int, dtype_float], args.batsize, n_epochs=args.epochs, n_splits=args.splits, save=args.save, str_info=str_info + 'hardgold_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['crossval']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardgold"][k].append(v)
    preds_crossval_hardgold_mtl_kli.extend(preds)
    targs_crossval_hardgold_mtl_kli.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.CrossEntropy(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.crossval(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad, char_pad, id_word_in_sent],
                           [y_gold, y_soft], [dtype_int, dtype_float], args.batsize, n_epochs=args.epochs, n_splits=args.splits, save=args.save, str_info=str_info + 'hardgold_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['crossval']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardgold"][k].append(v)
    preds_crossval_hardgold_mtl_ce.extend(preds)
    targs_crossval_hardgold_mtl_ce.extend(targs)


# crossval mv #####################################################################################

    model = mod.StlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = crossval_mv(model, optimizer, [lossfunc_hard],
                           [word_pad, char_pad, id_word_in_sent],
                           [y_gold], [y_mv], [dtype_int], args.batsize, n_epochs=args.epochs, n_splits=args.splits, save=args.save, str_info=str_info + 'hardmv_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['crossval']['control'][f"{model.__class__.__name__}_hardmv"][k].append(v)
    preds_crossval_hardmv_stl.extend(preds)
    targs_crossval_hardmv_stl.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.KLregular(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = crossval_mv(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad, char_pad, id_word_in_sent],
                           [y_gold, y_soft], [y_mv, y_soft], [dtype_int, dtype_float], args.batsize, n_epochs=args.epochs, n_splits=args.splits, save=args.save, str_info=str_info + 'hardmv_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['crossval']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardmv"][k].append(v)
    preds_crossval_hardmv_mtl_klr.extend(preds)
    targs_crossval_hardmv_mtl_klr.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.KLinverse(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = crossval_mv(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad, char_pad, id_word_in_sent],
                           [y_gold, y_soft], [y_mv, y_soft], [dtype_int, dtype_float], args.batsize, n_epochs=args.epochs, n_splits=args.splits, save=args.save, str_info=str_info + 'hardmv_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['crossval']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardmv"][k].append(v)
    preds_crossval_hardmv_mtl_kli.extend(preds)
    targs_crossval_hardmv_mtl_kli.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.CrossEntropy(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = crossval_mv(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad, char_pad, id_word_in_sent],
                           [y_gold, y_soft], [y_mv, y_soft], [dtype_int, dtype_float], args.batsize, n_epochs=args.epochs, n_splits=args.splits, save=args.save, str_info=str_info + 'hardmv_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['crossval']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardmv"][k].append(v)
    preds_crossval_hardmv_mtl_ce.extend(preds)
    targs_crossval_hardmv_mtl_ce.extend(targs)



# holdout gold ####################################################################################

    model = mod.StlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard],
                           [word_pad_trn, char_pad_trn, id_word_in_sent_trn], [word_pad_dev, char_pad_dev, id_word_in_sent_dev], [word_pad_tst, char_pad_tst, id_word_in_sent_tst],
                           [y_gold_trn], [y_gold_dev], [y_gold_tst], [dtype_int], args.batsize, args.epochs, save=True, str_info=str_info + 'hardgold_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['holdout']['control'][f"{model.__class__.__name__}_hardgold"][k].append(v)
    preds_holdout_hardgold_stl.extend(preds)
    targs_holdout_hardgold_stl.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.KLregular(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad_trn, char_pad_trn, id_word_in_sent_trn], [word_pad_dev, char_pad_dev, id_word_in_sent_dev], [word_pad_tst, char_pad_tst, id_word_in_sent_tst],
                           [y_gold_trn, y_soft_trn], [y_gold_dev, y_soft_dev], [y_gold_tst, y_soft_tst], [dtype_int, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardgold_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardgold"][k].append(v)
    preds_holdout_hardgold_mtl_klr.extend(preds)
    targs_holdout_hardgold_mtl_klr.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.KLinverse(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad_trn, char_pad_trn, id_word_in_sent_trn], [word_pad_dev, char_pad_dev, id_word_in_sent_dev], [word_pad_tst, char_pad_tst, id_word_in_sent_tst],
                           [y_gold_trn, y_soft_trn], [y_gold_dev, y_soft_dev], [y_gold_tst, y_soft_tst], [dtype_int, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardgold_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardgold"][k].append(v)
    preds_holdout_hardgold_mtl_kli.extend(preds)
    targs_holdout_hardgold_mtl_kli.extend(targs)

    model = mod.MtlPreembBilstmEnc(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.CrossEntropy(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad_trn, char_pad_trn, id_word_in_sent_trn], [word_pad_dev, char_pad_dev, id_word_in_sent_dev], [word_pad_tst, char_pad_tst, id_word_in_sent_tst],
                           [y_gold_trn, y_soft_trn], [y_gold_dev, y_soft_dev], [y_gold_tst, y_soft_tst], [dtype_int, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardgold_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_gold['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardgold"][k].append(v)
    preds_holdout_hardgold_mtl_ce.extend(preds)
    targs_holdout_hardgold_mtl_ce.extend(targs)

# holdout mv ######################################################################################

    model = mod.StlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard],
                           [word_pad_trn, char_pad_trn, id_word_in_sent_trn], [word_pad_dev, char_pad_dev, id_word_in_sent_dev], [word_pad_tst, char_pad_tst, id_word_in_sent_tst],
                           [y_mv_trn], [y_gold_dev], [y_gold_tst], [dtype_int], args.batsize, args.epochs, str_info=str_info + 'hardmv_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['holdout']['control'][f"{model.__class__.__name__}_hardmv"][k].append(v)
    preds_holdout_hardmv_stl.extend(preds)
    targs_holdout_hardmv_stl.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.KLregular(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad_trn, char_pad_trn, id_word_in_sent_trn], [word_pad_dev, char_pad_dev, id_word_in_sent_dev], [word_pad_tst, char_pad_tst, id_word_in_sent_tst],
                           [y_mv_trn, y_soft_trn], [y_gold_dev, y_soft_dev], [y_gold_tst, y_soft_tst], [dtype_int, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardmv_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardmv"][k].append(v)
    preds_holdout_hardmv_mtl_klr.extend(preds)
    targs_holdout_hardmv_mtl_klr.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.KLinverse(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad_trn, char_pad_trn, id_word_in_sent_trn], [word_pad_dev, char_pad_dev, id_word_in_sent_dev], [word_pad_tst, char_pad_tst, id_word_in_sent_tst],
                           [y_mv_trn, y_soft_trn], [y_gold_dev, y_soft_dev], [y_gold_tst, y_soft_tst], [dtype_int, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardmv_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardmv"][k].append(v)
    preds_holdout_hardmv_mtl_kli.extend(preds)
    targs_holdout_hardmv_mtl_kli.extend(targs)

    model = mod.MtlPreembBilstmEnc3(word_emb, char_emb, y_size, trainable=args.trainable, lstm_nrlayer=args.lstm_layers, hid_size=args.lstm_size, att_heads=args.att_heads, att_layers=args.att_layers, mlp_nrlayer=args.mlp_layers, droprob=args.droprob, device=device)
    lossfunc_hard = nn.CrossEntropyLoss().to(device=device)
    lossfunc_soft = mod.CrossEntropy(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.learate)
    dirout, preds, targs = proc.holdout(model, optimizer, [lossfunc_hard, lossfunc_soft],
                           [word_pad_trn, char_pad_trn, id_word_in_sent_trn], [word_pad_dev, char_pad_dev, id_word_in_sent_dev], [word_pad_tst, char_pad_tst, id_word_in_sent_tst],
                           [y_mv_trn, y_soft_trn], [y_gold_dev, y_soft_dev], [y_gold_tst, y_soft_tst], [dtype_int, dtype_float], args.batsize, args.epochs, str_info=str_info + 'hardmv_')
    for k, v in {'dirs': dirout, 'preds': preds, 'targs': targs}.items(): bootdata_mv['holdout']['treatment'][f"{model.__class__.__name__}_{lossfunc_soft.__class__.__name__}_hardmv"][k].append(v)
    preds_holdout_hardmv_mtl_ce.extend(preds)
    targs_holdout_hardmv_mtl_ce.extend(targs)

print(f"{'#'*80}\nall experiments results:\n")
proc.metrics(targs_crossval_hardgold_stl,     preds_crossval_hardgold_stl,     'crossval_hardgold_stl')
proc.metrics(targs_crossval_hardgold_mtl_klr, preds_crossval_hardgold_mtl_klr, 'crossval_hardgold_mtl_klr')
proc.metrics(targs_crossval_hardgold_mtl_kli, preds_crossval_hardgold_mtl_kli, 'crossval_hardgold_mtl_kli')
proc.metrics(targs_crossval_hardgold_mtl_ce,  preds_crossval_hardgold_mtl_ce,  'crossval_hardgold_mtl_ce')

proc.metrics(targs_crossval_hardmv_stl,     preds_crossval_hardmv_stl,         'crossval_hardmv_stl')
proc.metrics(targs_crossval_hardmv_mtl_klr, preds_crossval_hardmv_mtl_klr,     'crossval_hardmv_mtl_klr')
proc.metrics(targs_crossval_hardmv_mtl_kli, preds_crossval_hardmv_mtl_kli,     'crossval_hardmv_mtl_kli')
proc.metrics(targs_crossval_hardmv_mtl_ce,  preds_crossval_hardmv_mtl_ce,      'crossval_hardmv_mtl_ce')

proc.metrics(targs_holdout_hardgold_stl,      preds_holdout_hardgold_stl,      'holdout_hardgold_stl')
proc.metrics(targs_holdout_hardgold_mtl_klr,  preds_holdout_hardgold_mtl_klr,  'holdout_hardgold_mtl_klr')
proc.metrics(targs_holdout_hardgold_mtl_kli,  preds_holdout_hardgold_mtl_kli,  'holdout_hardgold_mtl_kli')
proc.metrics(targs_holdout_hardgold_mtl_ce,   preds_holdout_hardgold_mtl_ce,   'holdout_hardgold_mtl_ce')

proc.metrics(targs_holdout_hardmv_stl,        preds_holdout_hardmv_stl,        'holdout_hardmv_stl')
proc.metrics(targs_holdout_hardmv_mtl_klr,    preds_holdout_hardmv_mtl_klr,    'holdout_hardmv_mtl_klr')
proc.metrics(targs_holdout_hardmv_mtl_kli,    preds_holdout_hardmv_mtl_kli,    'holdout_hardmv_mtl_kli')
proc.metrics(targs_holdout_hardmv_mtl_ce,     preds_holdout_hardmv_mtl_ce,     'holdout_hardmv_mtl_ce')

print(f"{'#'*80}\nbootstraps:\n")
ut.writejson(bootdata_gold, log.pathtime + 'bootstrap_gold_input.json')
proc.bootstrap(bootdata_gold, args.bootloop, args.perc_sample)
ut.writejson(bootdata_mv, log.pathtime + 'bootstrap_mv_input.json')
proc.bootstrap(bootdata_mv, args.bootloop, args.perc_sample)


ut.end(startime)




