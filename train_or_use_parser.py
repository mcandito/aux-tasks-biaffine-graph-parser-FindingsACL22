#!/usr/bin/env python
# -*- coding: utf-8 -*-

# reimplementation of Dozat et al. 2017 / 2018 parser and graph parser
# Marie Candito

import torch
import argparse
import os
import sys
import transformers
from transformers import AutoModel, AutoTokenizer

from data import *
from indexing import *
from indexed_data import *
from biaffine_parser import *

# taken from https://gist.github.com/okomarov/c0d9fd0718f6f9b40c701e61523dfed1
import psutil
import humanize
import GPUtil


# TODO get least used gpu
# check how to link cuda retrieval of device, and the id used in GPUtil (same?) 
#gpu = None
#if len(GPUs) > 0:
#  gpu = GPUs[0]

def printm(gpu):
  if gpu is not None:
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    GPUtil.showUtilization()

        
# ---------- MAIN ------------------
if __name__ == "__main__":


    usage = """ Biaffine graph parser (Dozat et al. 2018)"""

    # read arguments
    argparser = argparse.ArgumentParser(usage = usage)
    argparser.add_argument('mode', choices=['train','test'], help="In train mode: train model + check performance on validation set. In Test mode (not implemented yet), loads a model and parses a file using it. Default='train'", default="train")
    argparser.add_argument('conll_file', help='Contains either train/dev/test sentences (if split_info_file is provided), or should contain training sentences for train mode, or sentences to parse in test mode.', default=None)
    argparser.add_argument('model_dir', help='if mode is train, trained model will be saved to model_dir/model, otherwise model will be loaded from it', default=None)
    argparser.add_argument('--split_info_file', help='split info file (each line = sentence id, tab, corpus type (train, dev, test)', default=None)
    argparser.add_argument('-v', '--validation_conll_file', help='pertains only if split_info_file is not provided: validation sentences, will be used to evaluate model during training, and for early stopping')
    argparser.add_argument('--data_name', help='short name of data: ftb or sequoia etc... Default=ftb', default='ftb')
    argparser.add_argument('-g', '--graph_mode', action="store_true", help='If set, Graph version of the parser, otherwise Tree version. Default=True', default=True)
    argparser.add_argument('-p', '--w_emb_file', help='pre-trained word embeddings file. NB: first line should contain nbwords embedding size', default='None')
    argparser.add_argument('-w', '--w_emb_size', help='size of word embeddings. Default=100', type=int, default=100)
    argparser.add_argument('-l', '--l_emb_size', help='size of lemma embeddings. Default=100', type=int, default=100)
    argparser.add_argument('-c', '--p_emb_size', help='size of POS embeddings. Default=20', type=int, default=20)
    argparser.add_argument('--lstm_h_size', help='size of lstm embeddings. Default=300', type=int, default=300)
    argparser.add_argument('--reduced_bert_size', help='If set to > 0, the bert embeddings will be linearly reduced before concat to non contextual embeddings. Default=0', type=int, default=0)    
    argparser.add_argument('--mlp_arc_o_size', help='size of arc mlp after lstm. Default=300', type=int, default=300)
    argparser.add_argument('--mlp_lab_o_size', help='size of lab mlp after lstm. Default=300', type=int, default=300)
    argparser.add_argument('--bert_name', help='huggingface *bert model name. If not "", will be used as pretrained-LM. Default:flaubert/flaubert_base_cased', default="flaubert/flaubert_base_cased")
    argparser.add_argument('--use_bias', action="store_true", help='Whether to add bias in all internal MLPs. Default=True', default=True)
    argparser.add_argument('-b', '--batch_size', help='batch size. Default=16', type=int, default=16)
    argparser.add_argument('-e', '--early_stopping', action="store_true", help='if set, training will stop as soon as validation loss increases. Note that IN ANY CASE, THE SAVED MODEL will be that with MINIMUM LOSS on validation set, early stopping is just to use if training should be stopped at the first loss increase. Default=True', default=True)
#    argparser.add_argument('-o', '--optimizer', help='The optimization algo (from pytorch optim module). Default=SGD', default='SGD')
    argparser.add_argument('-r', '--learning_rate', help='learning rate, default=0.1', type=float, default=0.00001)
    argparser.add_argument('-d', '--lex_dropout', help='lexical dropout rate, default=0.3', type=float, default=0.3)
    argparser.add_argument('-i', '--lab_loss_weight', help='label loss weight w (arc loss will be 1 - w). Default=0.3', type=float, default=0.3)
    argparser.add_argument('--pos_arc_weight', help='(for graph mode only) weight for positive arcs in binary cross-entropy. Default=1.5', type=float, default=1.5)
    argparser.add_argument('-n', '--nb_epochs', help='Max nb epochs. Default=40', type=int, default=40)
    argparser.add_argument('--nb_epochs_arc_only', help='Nb epocs to run using arc loss only. Default=0', type=int, default=0)
    argparser.add_argument('--device_id', help='GPU cuda device id. Default=0', type=int, default=0)
    argparser.add_argument('-t', '--trace', action="store_true", help='print some traces. Default=False', default=False)
    args = argparser.parse_args()


    # --------------------- DEVICE ---------------------
    # si un GPU est disponible on l'utilisera
    if torch.cuda.is_available():
        # objet torch.device          
        DEVICE = torch.device("cuda:"+str(args.device_id))
        
        print('There are %d GPU(s) available.' % torch.cuda.device_count())    
        device_id = args.device_id #torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print("We will use GPU %d (%s) of compute capability %d.%d with "
              "%.2fGb total memory.\n" % 
              (device_id,
               gpu_properties.name,
               gpu_properties.major,
               gpu_properties.minor,
               gpu_properties.total_memory / 1e9))

    else:
        print('No GPU available, using the CPU instead.')
        DEVICE = torch.device("cpu")

    logstream = open(args.model_dir+'/log', 'w')
    model_file = args.model_dir+'/model'

    if args.bert_name:
      bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    else:
      bert_tokenizer = None

    # train model on train data and check performance on validation set
    if args.mode == 'train':

        # before anything: check whether we will be able to dump the model
        pdir = os.path.dirname(model_file)
        if not pdir: pdir = '.'
        # if parent dir is writable
        if not os.access(pdir, os.W_OK):
            exit("Model file %s will not be writable!\n" % model_file)

        # ------------- DATA (WITHOUT INDICES YET) ------------------------------
        split_info_file = args.split_info_file
        if args.graph_mode:
            load_fn = 'load_dep_graphs'
        else:
            load_fn = 'load_dep_trees'
        if split_info_file:
            sentences = eval(load_fn)(args.conll_file, split_info_file=split_info_file)
        else:
            sentences = eval(load_fn)(args.conll_file, corpus_type='train')
            v = eval(load_fn)(args.validation_conll_file, corpus_type='dev')
            sentences['dev'] = v['dev']
                            
        if args.w_emb_file != 'None':
            w_emb_file = args.w_emb_file
            use_pretrained_w_emb = True
        else:
            w_emb_file = None
            use_pretrained_w_emb = False

        # ------------- INDICES ------------------------------
        # indices are defined on train sentences only
        indices = Indices(sentences['train'], w_emb_file=w_emb_file, bert_tokenizer=bert_tokenizer)

        # ------------- INDEXED DATA -------------------------
        data = {}
        for part in sentences.keys():
            if args.graph_mode:
                data[part] = DepGraphDataSet(part, sentences[part], indices, DEVICE)
            else:
                data[part] = DepTreeDataSet(part, sentences[part], indices, DEVICE)

        # ------------- THE PARSER ---------------------------
        biaffineparser = BiAffineParser(indices, DEVICE, 
                                        w_emb_size=args.w_emb_size,
                                        l_emb_size=args.l_emb_size,
                                        p_emb_size=args.p_emb_size,
                                        lstm_h_size=args.lstm_h_size,
                                        mlp_arc_o_size=args.mlp_arc_o_size, 
                                        mlp_lab_o_size=args.mlp_lab_o_size, 
                                        use_pretrained_w_emb=use_pretrained_w_emb, 
                                        use_bias=args.use_bias,
                                        bert_name=args.bert_name,
                                        reduced_bert_size=args.reduced_bert_size)

        # pour tests plus rapides: utiliser training sur val
        #train_data = data['val'] # data['train']
        train_data = data['train']
        val_data = data['dev']


        biaffineparser.train_model(train_data, val_data, args.data_name, model_file, logstream,
                                   args.nb_epochs,
                                   args.batch_size,
                                   args.learning_rate,
                                   args.lab_loss_weight, 
                                   args.lex_dropout,
                                   nb_epochs_arc_only = args.nb_epochs_arc_only,
                                   pos_arc_weight = args.pos_arc_weight,
                                   graph_mode = args.graph_mode)
                
        # TODO: use this opt...
        opt = {
            'logstream':logstream,
            'w_emb_size':args.w_emb_size,
            'w_emb_file':args.w_emb_file,
            'l_emb_size':args.l_emb_size,
            'p_emb_size':args.p_emb_size,
            'lstm_h_size':args.lstm_h_size,
            'mlp_arc_o_size':args.mlp_arc_o_size,
            'use_bias':args.use_bias,
            'bert_name':args.bert_name,

            'model_file':args.model_dir+'/model',
        
            'graph_mode':args.graph_mode,
            'lr':args.learning_rate,
            'lab_loss_weight':args.lab_loss_weight,
            'lex_dropout':args.lex_dropout,
            'nb_epochs':args.nb_epochs,
            'nb_epochs_arc_only':args.nb_epochs_arc_only,
            'early_stopping':args.early_stopping,
        }

    # parsing mode
    else:
        sys.stderr.write("Parsing %s with %s ...\n" % (args.conll_file, model_file))
        biaffine_parser = torch.load(model_file)
        sys.stderr.write("loaded model %s, nb epochs=%d\n" % (model_file, classifier.current_nb_epochs))
        indices = biaffine_parser.indices
        if args.graph_mode:
            sentences = load_dep_graphs(args.conll_file, corpus_type='toparse')
            dataset = DepGraphDataSet('toparse', sentences[part], indices, DEVICE)
        else:
            sentences = load_dep_trees(args.conll_file, corpus_type='toparse')
            dataset = DepGraphDataSet('toparse', sentences[part], indices, DEVICE)
        # TODO parsing mode
