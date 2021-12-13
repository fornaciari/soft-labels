# coding=latin-1
import util200818 as ut
import models200928 as mod
import os, re
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
from scipy.stats import ttest_rel
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel, Whitespace, CharDelimiterSplit
from tokenizers.trainers import BpeTrainer
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, FlaubertTokenizer, FlaubertModel
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from unidecode import unidecode
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
###################################################################################################


class Processing():
    def __init__(self, dir_root, device):
        self.dir_root = dir_root
        self.device = device

    @staticmethod
    def plotexp_stl(trn, dev, tst, path_pdf):
        fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(18, 5))
        x_range = range(1, trn.shape[0]+1)
        ax = plt.subplot(131)
        line1, = ax.plot(x_range, tst.loss, label='tst hard loss')
        line2, = ax.plot(x_range, dev.loss, label='dev hard loss')
        line3, = ax.plot(x_range, trn.loss, label='trn hard loss')
        ax.legend()
        ax = plt.subplot(132)
        line1, = ax.plot(x_range, tst.acc, label='tst acc')
        line2, = ax.plot(x_range, dev.acc, label='dev acc')
        line3, = ax.plot(x_range, trn.acc, label='trn acc')
        ax.legend()
        ax = plt.subplot(133)
        line1, = ax.plot(x_range, tst.f1, label='tst f1')
        line2, = ax.plot(x_range, dev.f1, label='dev f1')
        line3, = ax.plot(x_range, trn.f1, label='trn f1')
        ax.legend()
        plt.savefig(path_pdf)
        plt.close()
        return 1

    @staticmethod
    def plotexp_mtl(trn, dev, tst, path_pdf):
        x_range = range(1, trn.shape[0]+1)
        plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(10, 10))
        ax = plt.subplot(221)
        ax.plot(x_range, tst.acc, label='tst acc')
        ax.plot(x_range, dev.acc, label='dev acc')
        ax.plot(x_range, trn.acc, label='trn acc')
        ax.legend()
        ax = plt.subplot(222)
        ax.plot(x_range, tst.f1, label='tst f1')
        ax.plot(x_range, dev.f1, label='dev f1')
        ax.plot(x_range, trn.f1, label='trn f1')
        ax.legend()
        ax = plt.subplot(223)
        ax.plot(x_range, tst.loss_hard, label='tst hard loss')
        ax.plot(x_range, dev.loss_hard, label='dev hard loss')
        ax.plot(x_range, trn.loss_hard, label='trn hard loss')
        ax.legend()
        ax = plt.subplot(224)
        ax.plot(x_range, tst.loss_soft, label='tst soft loss')
        ax.plot(x_range, dev.loss_soft, label='dev soft loss')
        ax.plot(x_range, trn.loss_soft, label='trn soft loss')
        ax.legend()
        plt.savefig(path_pdf)
        plt.close()
        return 1

    @staticmethod
    def bert_twosent_preproc(lang, X1, X2, padsize=500):
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')               if lang == 'en' else \
                    AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased') if lang == 'it' else \
                    AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')  if lang == 'de' else \
                    FlaubertTokenizer.from_pretrained('flaubert-base-cased')       if lang == 'fr' else \
                    BertTokenizer.from_pretrained('bert-base-dutch-cased')         if lang == 'nl' else None
        texts_ids = list()
        texts_masks = list()
        for text1, text2 in tqdm(zip(X1, X2), desc='bertokenizing', ncols=80):
            encoded_dict = tokenizer.encode_plus(text1, text2, # Sentence to encode.
                                                 add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                                 max_length=padsize, # Pad & truncate all sentences.
                                                 pad_to_max_length=True,
                                                 return_attention_mask=True, # Construct attn. masks.
                                                 truncation_strategy='only_second', # only_first only_second longest_first
                                                 return_tensors='pt') # Return pytorch tensors.
            texts_ids.append(encoded_dict['input_ids'].squeeze(0).tolist())
            texts_masks.append(encoded_dict['attention_mask'].squeeze(0).tolist())
        texts_ids = np.array(texts_ids)
        texts_masks = np.array(texts_masks)
        return texts_ids, texts_masks

    @staticmethod
    def bert_preproc(lang, X, padsize=500):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")  if lang == 'ml' else \
                    BertTokenizer.from_pretrained('bert-base-cased')               if lang == 'en' else \
                    AutoTokenizer.from_pretrained('dbmdz/bert-base-italian-cased') if lang == 'it' else \
                    AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')  if lang == 'de' else \
                    FlaubertTokenizer.from_pretrained('flaubert-base-cased')       if lang == 'fr' else \
                    BertTokenizer.from_pretrained('bert-base-dutch-cased')         if lang == 'nl' else None
        texts_ids = list()
        texts_masks = list()
        for text in X:
            encoded_dict = tokenizer.encode_plus(text, # Sentence to encode.
                                                 add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                                 max_length=padsize, # Pad & truncate all sentences.
                                                 pad_to_max_length=True,
                                                 return_attention_mask=True, # Construct attn. masks.
                                                 return_tensors='pt') # Return pytorch tensors.
            texts_ids.append(encoded_dict['input_ids'].squeeze(0).tolist())
            texts_masks.append(encoded_dict['attention_mask'].squeeze(0).tolist())
        texts_ids = np.array(texts_ids)
        texts_masks = np.array(texts_masks)
        return texts_ids, texts_masks, tokenizer.vocab_size

    @staticmethod
    def token_preproc(X, vocsize, padsize):
        tokenizer = Tokenizer(BPE.empty()) # empty Byte-Pair Encoding model
        tokenizer.normalizer = Sequence([NFKC(), Lowercase()]) # ordered Sequence of normalizers: unicode-normalization then lower-casing
        tokenizer.pre_tokenizer = ByteLevel()  # pre-tokenizer converting the input to a ByteLevel representation.
        tokenizer.decoder = ByteLevelDecoder() # plug a decoder so we can recover from a tokenized input to the original one
        trainer = BpeTrainer(vocab_size=vocsize, show_progress=True, initial_alphabet=ByteLevel.alphabet(), special_tokens=["<pad>"]) # trainer initialization
        ut.list2file([text for text in X], 'temporary.txt') # stupidamente, il trainer vuole come input solo un(a lista di) file
        tokenizer.train(trainer, ['temporary.txt'])
        os.system('rm temporary.txt')
        print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
        texts_ids = list()
        texts_masks = list()
        tokenizer.enable_padding(max_length=padsize) # https://huggingface.co/transformers/_modules/transformers/tokenization_utils.html
        tokenizer.enable_truncation(max_length=padsize)
        for text in X:
            encoded = tokenizer.encode(text) # Return pytorch tensors.
            texts_ids.append(encoded.ids)
            texts_masks.append(encoded.attention_mask)
        texts_ids = np.array(texts_ids)
        texts_masks = np.array(texts_masks)
        return texts_ids, texts_masks, tokenizer.get_vocab_size()

    @staticmethod
    def char_preproc(X, padsize):
        X = [unidecode(str(x)).lower() for x in X] # str perché ci sono alcuni numeri visti come int. alcuni non ascii char vengono sostituiti con ''
        char2id = {c for x in X for c in x} # set of the seen ascii chars
        char2id = {c: i + 1 for i, c in enumerate(sorted(char2id))} # lasciio lo 0 per pad
        chars_ids = np.array([[char2id[c] for c in x][:padsize] if len(x) > padsize else
                              [char2id[c] for c in x] + [0] * (padsize - len(x))
                              for x in X])
        chars_masks = np.array([[1] * padsize if len(x) > padsize else
                                [1] * len(x) + [0] * (padsize - len(x))
                                for x in X])
        vocsize = len(char2id) + 1 # per pad token
        print(f"{'chars vocab size':.<25} {vocsize}\n{'chars shape':.<25} {chars_ids.shape}")
        return chars_ids, chars_masks, vocsize

    @staticmethod
    def metrics(targs, preds, title=''):
        print(title)
        conf_matrix = confusion_matrix(targs, preds)
        micro_measures = precision_recall_fscore_support(targs, preds, average='micro')
        macro_measures = precision_recall_fscore_support(targs, preds, average='macro')
        rounding_value = 4
        tn1 = conf_matrix[0, 0]
        fn1 = conf_matrix[1, 0]
        fp1 = conf_matrix[0, 1]
        tp1 = conf_matrix[1, 1]
        tn0 = conf_matrix[1, 1]
        fn0 = conf_matrix[0, 1]
        fp0 = conf_matrix[1, 0]
        tp0 = conf_matrix[0, 0]
        print(f"tp1 {tp1}\ntn1 {tn1}\nfp1 {fp1}\nfn1 {fn1}\ntp0 {tp0}\ntn0 {tn0}\nfp0 {fp0}\nfn0 {fn0}")
    
        prec1 = round(tp1 / (tp1 + fp1) * 100, rounding_value)
        prec0 = round(tp0 / (tp0 + fp0) * 100, rounding_value)
        rec1 = round(tp1 / (tp1 + fn1) * 100, rounding_value)
        rec0 = round(tp0 / (tp0 + fn0) * 100, rounding_value)
        f1_0 = round(2 * ((prec0 * rec0)/(prec0 + rec0)), rounding_value)
        f1_1 = round(2 * ((prec1 * rec1)/(prec1 + rec1)), rounding_value)
        print(f"rec0  {rec0}\nrec1  {rec1}\nprec0 {prec0}\nprec1 {prec1}\nf1_0  {f1_0}\nf1_1  {f1_1}")
    
        arrs = precision_recall_fscore_support(targs, preds)
        for arr, met in zip(arrs, ['precs', 'recs ', 'f1s  ', 'supps']): print(met, arr, 'mean', round(arr.mean() * 100, rounding_value))
        print(f"confusion matrix:\n{conf_matrix}\nmicro precision_recall_fscore_support:{micro_measures}\nmacro precision_recall_fscore_support:{macro_measures}")
        accu = round(accuracy_score(targs, preds) * 100, 2)
        fmea = round(f1_score(targs, preds, average='macro') * 100, 2)
        print(f"{'accuracy':.<11} {accu:<10}{'f1':.<5} {fmea:<10}\n\n")
        return 1

###################################################################################################

    def batches(self, step, model, optimizer, lossfuncs, x_inputs_step, y_inputs_step, y_dtypes, batsize):
        preds = list()
        losss = list()
        model.train() if step == 'trn' else model.eval()
        # for x in x_inputs_step: print(x.shape)
        # for y in y_inputs_step: print(y.shape)
        for ifir_bat in tqdm(range(0, len(y_inputs_step[0]), batsize), desc=step, ncols=80): # desc='training' # prefix
            nlas_bat = ifir_bat + batsize
            x_inputs_bat = [torch.from_numpy(x[ifir_bat: nlas_bat]).to(device=self.device, dtype=torch.int64) for x in x_inputs_step]
            y_inputs_bat = [torch.from_numpy(y[ifir_bat: nlas_bat]).to(device=self.device, dtype=dt) for y, dt in zip(y_inputs_step, y_dtypes)]
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

    def holdout(self, model, optimizer, lossfuncs, x_inputs_trn, x_inputs_dev, x_inputs_tst, y_inputs_trn, y_inputs_dev, y_inputs_tst, y_dtypes, batsize, n_epochs=10, save=False, str_info=''):
        strloss = '' if len(lossfuncs) < 2 else '_' + lossfuncs[1].__class__.__name__
        dir_out = self.dir_root + str_info + 'epoch' + str(n_epochs) + '_' + Processing.holdout.__name__ + '_' + model.__class__.__name__ + strloss
        print(f"{'dir out':.<25} {dir_out}")
        path_model = dir_out + '/model.pt'
        path_results = dir_out + '/results.pdf'
        os.mkdir(dir_out)
        df_out, df_trn_all, df_dev_all, df_tst_all = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        preds_out, targs_out, loss_out = list(), list(), list()
        max_metric = 0
        for epoch in range(1, n_epochs + 1):
            print(f"{'#'*80}\nepoch {epoch}")
            serie_trn, _,        = self.batches('trn', model, optimizer, lossfuncs, x_inputs_trn, y_inputs_trn, y_dtypes, batsize)
            serie_dev, _,        = self.batches('dev', model, optimizer, lossfuncs, x_inputs_dev, y_inputs_dev, y_dtypes, batsize)
            serie_tst, preds_tst = self.batches('tst', model, optimizer, lossfuncs, x_inputs_tst, y_inputs_tst, y_dtypes, batsize)
            df_trn_all = df_trn_all.append(serie_trn)
            df_dev_all = df_dev_all.append(serie_dev)
            df_tst_all = df_tst_all.append(serie_tst)
            # ut.sendslack(f"{log.pathtime} epoch {epoch}\n{serie_tst.to_string()}")
            if serie_dev['f1'] >= max_metric:# and epoch > 1:
                if save:
                    torch.save(model.state_dict(), path_model)
                    print('model saved')
                print('results selected')
                df_out    = serie_tst.to_frame().transpose()
                preds_out = preds_tst
                targs_out = y_inputs_tst[0].astype(int).tolist()
                max_metric = serie_dev['f1'].mean()


        # if os.path.isfile(path_model): setup['model'].load_state_dict(torch.load(path_model))
        # do other test
        # if os.path.isfile(path_model) and not save: os.remove(path_model) # se save è False, non lo salva nemmeno

        # df_out.to_csv(dir_out + '/results.csv')
        if 'loss_soft' in df_out.columns:
            self.plotexp_mtl(df_trn_all, df_dev_all, df_tst_all, path_results)
        else:
            self.plotexp_stl(df_trn_all, df_dev_all, df_tst_all, path_results)
        ut.list2file(preds_out, dir_out + '/preds.txt')
        ut.list2file(targs_out, dir_out + '/targs.txt')
        str_out = f"{'loss hard':.<12} {round(df_out.loss_hard.mean(), 4):<10}{'loss soft':.<12} {round(df_out.loss_soft.mean(), 4):<10}{'accuracy':.<11} {round(df_out.acc.mean(), 2):<10}{'f1':.<5} {round(df_out.f1.mean(), 2)}" if 'loss_soft' in df_out.columns else \
                  f"{'loss':.<12} {round(df_out.loss.mean(), 4):<10}{'accuracy':.<11} {round(df_out.acc.mean(), 2):<10}{'f1':.<5} {round(df_out.f1.mean(), 2)}"
        self.metrics(targs_out, preds_out)
        dir_out_results = dir_out + str(round(df_out.acc.mean(), 2))
        os.rename(dir_out, dir_out_results)
        print(f"{str_out}\nscp bocconi:{dir_out_results + '/results.pdf'} ./pdf/{''.join(dir_out.split('/'))}.pdf")
        ut.sendslack(f"{dir_out} done\n{str_out}")
        return dir_out_results, preds_out, targs_out

    def crossval(self, model, optimizer, lossfuncs, x_inputs, y_inputs, y_dtypes, batsize, n_epochs=10, n_splits=10, save=False, str_info=''):
        strloss = '' if len(lossfuncs) < 2 else '_' + lossfuncs[1].__class__.__name__
        dir_exp = self.dir_root + str_info + 'epoch' + str(n_epochs) + '_' + Processing.crossval.__name__ + '_' + model.__class__.__name__ + strloss
        print(f"{'dir out':.<25} {dir_exp}")
        os.mkdir(dir_exp)
        df_trn_epochs, df_dev_epochs, df_tst_epochs = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        # preds_out, targs_out = list(), list()

        path_virgin = f"{self.dir_root}virgin_model.pt"
        torch.save(model.state_dict(), path_virgin)
        preds_all, targs_all = list(), list()
        df_all = pd.DataFrame()
        fold = 0
        skf = StratifiedKFold(n_splits=n_splits, random_state=0, shuffle=True)
        for i_trn_dev, i_tst in skf.split(x_inputs[0], y_inputs[0]):
            fold += 1
            max_metric = 0
            print(f"{'#'*80}\nfold {fold}")
            dir_fold = f"{dir_exp}/fold{fold}/"
            os.mkdir(dir_fold)
            path_model = f"{dir_fold}model.pt"
            path_results = f"{dir_fold}results.pdf"
            dev_rate = int(len(y_inputs[0]) * (1 / n_splits))
            shuffled_indices = torch.randperm(len(i_trn_dev)) # skf mescola, ma anche ordina: rimescolo
            i_dev = i_trn_dev[shuffled_indices][-dev_rate:]
            i_trn = i_trn_dev[shuffled_indices][:-dev_rate]
            x_inputs_trn = [x[i_trn] for x in x_inputs]
            x_inputs_dev = [x[i_dev] for x in x_inputs]
            x_inputs_tst = [x[i_tst] for x in x_inputs]
            y_inputs_trn = [y[i_trn] for y in y_inputs]
            y_inputs_dev = [y[i_dev] for y in y_inputs]
            y_inputs_tst = [y[i_tst] for y in y_inputs]
            # for x in x_inputs_trn: print(x.shape)
            # for y in y_inputs_trn: print(y.shape)

            virgin = torch.load(path_virgin)
            model.load_state_dict(virgin)

            df_fold, df_trn_fold, df_dev_fold, df_tst_fold = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            preds_fold, targs_fold = list(), list()
            for epoch in range(1, n_epochs + 1):
                print(f"epoch {epoch}")

                serie_trn, _,        = self.batches('trn', model, optimizer, lossfuncs, x_inputs_trn, y_inputs_trn, y_dtypes, batsize)
                serie_dev, _,        = self.batches('dev', model, optimizer, lossfuncs, x_inputs_dev, y_inputs_dev, y_dtypes, batsize)
                serie_tst, preds_tst = self.batches('tst', model, optimizer, lossfuncs, x_inputs_tst, y_inputs_tst, y_dtypes, batsize)
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

            if 'loss_soft' in df_fold.columns:
                self.plotexp_mtl(df_trn_fold, df_dev_fold, df_tst_fold, path_results)
            else:
                self.plotexp_stl(df_trn_fold, df_dev_fold, df_tst_fold, path_results)
            preds_all.extend(preds_fold)
            targs_all.extend(targs_fold)
            df_all = df_all.append(df_fold)

        os.system(f"rm {path_virgin}")
        ut.list2file(preds_all, dir_exp + '/preds.txt')
        ut.list2file(targs_all, dir_exp + '/targs.txt')
        str_out = f"{'loss hard':.<12} {round(df_all.loss_hard.mean(), 4):<10}{'loss soft':.<12} {round(df_all.loss_soft.mean(), 4):<10}{'accuracy':.<11} {round(accuracy_score(targs_all, preds_all), 4):<10}{'f1':.<5} {round(f1_score(targs_all, preds_all, average='macro'), 4)}" if 'loss_soft' in df_all.columns else \
                  f"{'loss hard':.<12} {round(df_all.loss.mean(), 4):<10}{'accuracy':.<11} {round(accuracy_score(targs_all, preds_all), 4):<10}{'f1':.<5} {round(f1_score(targs_all, preds_all, average='macro'), 4)}"
        self.metrics(targs_all, preds_all)

        dir_out_results = dir_exp + str(round(accuracy_score(targs_all, preds_all), 2))
        os.rename(dir_exp, dir_out_results)
        print(f"{df_all.to_string()}\n{str_out}\nscp bocconi:mimac/{dir_out_results + '/fold_results.pdf'} ./pdf/{''.join(dir_exp.split('/'))}.pdf")
        ut.sendslack(f"{dir_exp} done\n{str_out}")



        # print(f"macro precision_recall_fscore_support:\n{precision_recall_fscore_support(targs_all, preds_all, average='macro')}")
        # print(f"micro precision_recall_fscore_support:\n{precision_recall_fscore_support(targs_all, preds_all, average='micro')}")
        # print("accuracy", round(accuracy_score(targs_all, preds_all) * 100, rounding_value))




        return dir_out_results, preds_all, targs_all



            # df_trn_epochs = df_trn_epochs.append(df_trn_epo.mean(), ignore_index=True)
            # df_dev_epochs = df_dev_epochs.append(df_dev_epo.mean(), ignore_index=True)
            # df_tst_epochs = df_tst_epochs.append(df_tst_epo.mean(), ignore_index=True)
            # str_out = f"{'dev loss hard':.<16} {round(df_dev_epo.loss_hard.mean(), 4):<9}{'loss soft':.<12} {round(df_dev_epo.loss_soft.mean(), 4):<9}{'accuracy':.<11} {round(df_dev_epo.acc.mean(), 2):<9}{'f1':.<5} {round(df_dev_epo.f1.mean(), 2)}\n" \
            #           f"{'tst loss hard':.<16} {round(df_tst_epo.loss_hard.mean(), 4):<9}{'loss soft':.<12} {round(df_tst_epo.loss_soft.mean(), 4):<9}{'accuracy':.<11} {round(df_tst_epo.acc.mean(), 2):<9}{'f1':.<5} {round(df_tst_epo.f1.mean(), 2)}" if 'loss_soft' in df_tst_epo.columns else \
            #           f"{'dev loss hard':.<16} {round(df_dev_epo.loss_hard.mean(), 4):<9}{'accuracy':.<11} {round(df_dev_epo.acc.mean(), 2):<9}{'f1':.<5} {round(df_dev_epo.f1.mean(), 2)}\n" \
            #           f"{'tst loss hard':.<16} {round(df_tst_epo.loss_hard.mean(), 4):<9}{'accuracy':.<11} {round(df_tst_epo.acc.mean(), 2):<9}{'f1':.<5} {round(df_tst_epo.f1.mean(), 2)}"
            # print(str_out)
            # # ut.sendslack(f"{log.pathtime} epoch {epoch}\n{df_tst_epo.to_string()}")
            # if max_acc < df_dev_epo['acc'].mean():# and epoch > 1:
            #     torch.save(model.state_dict(), path_model)
            #     print('model saved')
            #     df_out    = df_tst_epo
            #     preds_out = preds_epo
            #     targs_out = y_epo
            #     max_acc = df_dev_epo['acc'].mean()

        # if os.path.isfile(path_model): setup['model'].load_state_dict(torch.load(path_model))
        # do other test
        # if os.path.isfile(path_model) and not save: os.remove(path_model)
        #
        # # df_out.to_csv(dir_out + '/results.csv')
        # if 'loss_soft' in df_out.columns:
        #     self.plotexp_mtl(df_trn_epochs, df_dev_epochs, df_tst_epochs, path_results)
        # else:
        #     self.plotexp_stl(df_trn_epochs, df_dev_epochs, df_tst_epochs, path_results)
        # ut.list2file(preds_out, dir_out + '/preds.txt')
        # ut.list2file(targs_out, dir_out + '/targs.txt')
        # str_out = f"{'accuracy':.<15} {round(accuracy_score(targs_out, preds_out), 2)}\n{'f1':.<15} {round(f1_score(targs_out, preds_out, average='macro'), 2)}\n{'loss hard':.<15} {round(df_out.loss_hard.mean(), 4)}\n{'loss soft':.<15} {round(df_out.loss_soft.mean(), 4)}" if 'loss_soft' in df_out.columns else \
        #           f"{'accuracy':.<15} {round(accuracy_score(targs_out, preds_out), 2)}\n{'f1':.<15} {round(f1_score(targs_out, preds_out, average='macro'), 2)}\n{'loss hard':.<15} {round(df_out.loss_hard.mean(), 4)}"
        # conf_matrix = confusion_matrix(targs_out, preds_out)
        # micro_measures = precision_recall_fscore_support(targs_out, preds_out, average='micro')
        # macro_measures = precision_recall_fscore_support(targs_out, preds_out, average='macro')
        # print(f"confusion matrix:\n{conf_matrix}\nmicro precision_recall_fscore_support:{micro_measures}\nmacro precision_recall_fscore_support:{macro_measures}\n")
        # dir_out_results = dir_out + str(round(accuracy_score(targs_out, preds_out), 2))
        # os.rename(dir_out, dir_out_results)
        # print(f"{df_out.to_string()}\n{str_out}\nscp bocconi:anaphora/pos/{dir_out_results + '/results.pdf'} ./pdf/{''.join(dir_out.split('/'))}.pdf")
        # ut.sendslack(f"{dir_out} done\n{str_out}")
        # return dir_out_results, preds_out, targs_out

###################################################################################################

    @staticmethod
    def plotloss_stl(trn, dev, tst, label, path_pdf):
        plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(5, 5))
        x_range = range(1, trn.shape[0]+1)
        ax = plt.subplot(111)
        ax.plot(x_range, tst, label='tst ' + label)
        ax.plot(x_range, dev, label='dev ' + label)
        ax.plot(x_range, trn, label='trn ' + label)
        ax.legend()
        plt.savefig(path_pdf)
        plt.close()
        return 1

###################################################################################################

    @staticmethod
    def bootstrap(data, n_loops, perc_sample=.33, verbose=False):
        """
        :param data:

          bootinput = {'holdout':
                                {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                                 'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})},
                       'crossval':
                                {'control':   defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()}),
                                 'treatment': defaultdict(lambda: {'dirs': list(), 'preds': list(), 'targs': list()})}}
        """

        startime = ut.start()

        def metrics(targs, control_preds, treatment_preds):
            control_acc   = round(accuracy_score(targs, control_preds) * 100, 2)
            control_f1    = round(f1_score(targs, control_preds, average='macro') * 100, 2)
            treatment_acc = round(accuracy_score(targs, treatment_preds) * 100, 2)
            treatment_f1  = round(f1_score(targs, treatment_preds, average='macro') * 100, 2)
            diff_acc      = round(treatment_acc - control_acc, 2)
            diff_f1       = round(treatment_f1  - control_f1, 2)
            return control_acc, treatment_acc, diff_acc, control_f1, treatment_f1, diff_f1

        for val in data:
            for control_cond in data[val]['control']:
                print('#'*120)
                control_preds_all, control_targs_all, control_acc_all, control_f1_all = list(), list(), list(), list()
                for dire, preds, targs in zip(data[val]['control'][control_cond]['dirs'], data[val]['control'][control_cond]['preds'], data[val]['control'][control_cond]['targs']):
                    acc = round(accuracy_score(targs, preds) * 100, 2)
                    f1  = round(f1_score(targs, preds, average='macro') * 100, 2)
                    control_preds_all.extend(preds)
                    control_targs_all.extend(targs)
                    control_acc_all.append(acc)
                    control_f1_all.append(f1)
                    if verbose:  print(f"{'control dir':.<25} {dire:.<120} accuracy {acc:<8} F-measure {f1}")
                for treatment_cond in data[val]['treatment']:
                    print(f"{'#'*80}\n{val:<12}{control_cond}   vs   {treatment_cond}")
                    treatment_preds_all, treatment_targs_all, treatment_acc_all, treatment_f1_all = list(), list(), list(), list()
                    for dire, preds, targs in zip(data[val]['treatment'][treatment_cond]['dirs'],
                                                  data[val]['treatment'][treatment_cond]['preds'],
                                                  data[val]['treatment'][treatment_cond]['targs']):
                        acc = round(accuracy_score(targs, preds) * 100, 2)
                        f1  = round(f1_score(targs, preds, average='macro') * 100, 2)
                        treatment_preds_all.extend(preds)
                        treatment_targs_all.extend(targs)
                        treatment_acc_all.append(acc)
                        treatment_f1_all.append(f1)
                        if verbose: print(f"{'treatment dir':.<25} {dire:.<120} accuracy {acc:<8} F-measure {f1}")
                    assert control_targs_all == treatment_targs_all
                    targs_all = control_targs_all
                    tot_control_acc, tot_treatment_acc, tot_diff_acc, tot_control_f1, tot_treatment_f1, tot_diff_f1 = metrics(targs_all, control_preds_all, treatment_preds_all)
                    print(f"{'control total accuracy':.<25} {tot_control_acc:<8} {'treatment total accuracy':.<30} {tot_treatment_acc:<8} {'diff':.<7} {tot_diff_acc}")
                    print(f"{'control total F-measure':.<25} {tot_control_f1:<8} {'treatment total F-measure':.<30} {tot_treatment_f1:<8} {'diff':.<7} {tot_diff_f1}")

                    tst_overall_size = len(targs_all)
                    # estraggo l'equivalente di un esperimento. Più è piccolo il numero, più è facile avere significatività. In altre parole, più esperimenti si fanno più è facile
                    # samplesize = int(len(targs_all) / len(data[val]['control'][control_cond]['dirs']))
                    samplesize = int(len(targs_all) * perc_sample)
                    print(f"{'tot experiments size':.<25} {tst_overall_size}\n{'sample size':.<25} {samplesize}")
                    twice_diff_acc = 0
                    twice_diff_f1  = 0
                    for loop in tqdm(range(n_loops), desc='bootstrap', ncols=80):
                        i_sample = np.random.choice(range(tst_overall_size), size=samplesize, replace=True)
                        sample_control_preds   = [control_preds_all[i]   for i in i_sample]
                        sample_treatment_preds = [treatment_preds_all[i] for i in i_sample]
                        sample_targs           = [targs_all[i]           for i in i_sample]
                        _, _, sample_diff_acc, _, _, sample_diff_f1 = metrics(sample_targs, sample_control_preds, sample_treatment_preds)
                        if sample_diff_acc > 2 * tot_diff_acc: twice_diff_acc += 1
                        if sample_diff_f1  > 2 * tot_diff_f1:  twice_diff_f1 += 1
                    str_out = f"{'count sample diff acc is twice tot diff acc':.<50} {twice_diff_acc:<5}/ {n_loops:<8}p < {round((twice_diff_acc / n_loops), 4):<6} {ut.bcolors.red}{'**' if twice_diff_acc / n_loops < 0.01 else '*' if twice_diff_acc / n_loops < 0.05 else ''}{ut.bcolors.end}\n" \
                              f"{'count sample diff f1  is twice tot diff f1':.<50} {twice_diff_f1:<5}/ {n_loops:<8}p < {round((twice_diff_f1 / n_loops), 4):<6} {ut.bcolors.red}{'**' if twice_diff_f1 / n_loops < 0.01 else '*' if twice_diff_f1 / n_loops < 0.05 else ''}{ut.bcolors.end}"
                    print(str_out)
                    ut.sendslack(f"{val:<12}{control_cond}   vs   {treatment_cond}\n{str_out}")
        ut.end(startime)
        return 1

