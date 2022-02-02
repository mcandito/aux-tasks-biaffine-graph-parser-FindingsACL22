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
#@ import psutil
#@ import humanize
#@ import GPUtil


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
    argparser.add_argument('--tasks', help='dot-separated list of tasks, among a (biaffine arcs), l (biaffine labeled arcs), h (nb of heads), d (nb of deps), s (sorted lab sequence), b (bag of labels). Default=a.l ', default='a.l')
    argparser.add_argument('--split_info_file', help='split info file (each line = sentence id, tab, corpus type (train, dev, test)', default=None)
    argparser.add_argument('-v', '--validation_conll_file', help='pertains only if split_info_file is not provided: validation sentences, will be used to evaluate model during training, and for early stopping')
    argparser.add_argument('--data_name', help='short name of data: ftb or sequoia etc... Default=ftb', default='ftb')
    argparser.add_argument('-g', '--data_format', choices=['tree', 'sdp', 'deep'], help='type of input and format: "tree"= conllu dependency trees, "deep"=conllu compact dependency graphs (pipe-separated governors), "sdp" = sdp 2015 dependency graphs. Default=deep', default='deep')
    argparser.add_argument('-p', '--w_emb_file', help='If not "None", pre-trained word embeddings file. NB: first line should contain nbwords embedding size', default='None')
    argparser.add_argument('--l_emb_file', help='If not "None", pre-trained lemma embeddings file. NB: first line should contain nbwords embedding size', default='None')
    argparser.add_argument('-w', '--w_emb_size', help='size of word embeddings. Default=100', type=int, default=100)
    argparser.add_argument('-l', '--l_emb_size', help='size of lemma embeddings. Default=100', type=int, default=100)
    argparser.add_argument('-c', '--p_emb_size', help='size of POS embeddings. Default=20', type=int, default=20)
    argparser.add_argument('--lstm_h_size', help='size of lstm embeddings. Default=300', type=int, default=300)
    argparser.add_argument('--lstm_dropout', help='lstm dropout. Default=0.33', type=float, default=0.33)
    argparser.add_argument('--reduced_bert_size', help='If set to > 0, the bert embeddings will be linearly reduced before concat to non contextual embeddings. Default=0', type=int, default=0)    
    argparser.add_argument('--mlp_arc_o_size', help='size of arc mlp after lstm. Default=300', type=int, default=300)
    argparser.add_argument('--mlp_lab_o_size', help='size of lab mlp after lstm. Default=300', type=int, default=300)
    argparser.add_argument('--aux_hidden_size', help='size of hidden layers of aux tasks. Default=300', type=int, default=300)
    argparser.add_argument('--bert_name', help='huggingface *bert model name. If not "None", will be used as pretrained-LM. Default:flaubert/flaubert_base_cased', default="flaubert/flaubert_base_cased")
    argparser.add_argument('-f', '--freeze_bert', action="store_true", help='Whether to freeze *bert parameters. Default=False', default=False)
    argparser.add_argument('--use_bias', action="store_true", help='Whether to add bias in all internal MLPs. Default=True', default=True)
    argparser.add_argument('-b', '--batch_size', help='batch size. Default=16', type=int, default=16)
    argparser.add_argument('-e', '--early_stopping_style', choices=['L*acc','Lacc','loss'], help='Pertains if validation file is provided. Early stop using validation "loss", or all L accuracies ("L*acc"), or L accuracy ("Lacc"). Default=L*acc', default='L*acc')
#    argparser.add_argument('-o', '--optimizer', help='The optimization algo (from pytorch optim module). Default=SGD', default='SGD')
    argparser.add_argument('-r', '--learning_rate', help='learning rate, default=0.1', type=float, default=0.00001)
    argparser.add_argument('-d', '--lex_dropout', help='lexical dropout rate, default=0.3', type=float, default=0.3)
#    argparser.add_argument('-i', '--lab_loss_weight', help='label loss weight w (arc loss will be 1 - w). Default=0.3', type=float, default=0.3)
#    argparser.add_argument('--pos_arc_weight', help='(for graph mode only) weight for positive arcs in binary cross-entropy. Default=1.5', type=float, default=1.5)
    argparser.add_argument('-n', '--nb_epochs', help='Max nb epochs. Default=40', type=int, default=40)
    argparser.add_argument('--mtl_sharing_level', help='Sharing level for MTL. 1: shared until bilstm, 2:extra MLP shared. Default: 1', type=int, default=1)
    argparser.add_argument('--coeff_aux_task_as_input', help='"None" or dot-separated list of task:coeff pairs, with task s and/or h. Example: "s:5.h15", Default=""', default='None')
    argparser.add_argument('--coeff_aux_task_stack_propag', help='"None" or dot-separated list of task:coeff pairs, with task s and/or h. Example: "b:1.h:10.s:2", Default=""', default='None')
#    argparser.add_argument('--nb_epochs_arc_only', help='Nb epocs to run using arc loss only. Default=0', type=int, default=0)
    argparser.add_argument('--device_id', help='in train mode only: GPU cuda device id (in test mode: device is read in model). Default=0', type=int, default=0)
    argparser.add_argument('-t', '--trace', action="store_true", help='print some traces. Default=False', default=False)
    argparser.add_argument('--out_parsed_file', help='Pertains in test mode only. If set to non None, filename into which predictions will be dumped', default=None)
    argparser.add_argument('--study_scores', action="store_true", help='Pertains in test mode only. Is set, print study of arc scores.', default=False)
    argparser.add_argument('--arc_loss', choices=['bce', 'hinge', 'dyn_hinge'], help='Which loss to use for the arcs. Default = bce (for binary_cross_entropy). For dyn_hinge: at least one auxiliary task must predict the nb of arcs of each dependent.', default='bce')
    argparser.add_argument('--margin', help='Minimum margin required for hinge losses (cf. arc_loss). Default=1.0', type=float, default=1.0)
    argparser.add_argument('--margin_alpha', help='Power for difference to minimum margin for hinge losses (cf. arc_loss). Default=1.0', type=float, default=1.0)
    argparser.add_argument('--use_dyn_weights_pos_neg', action="store_true", help='If set, dynamic weights for pos and negative arc examples will be used.', default=False)
    
    args = argparser.parse_args()



    model_file = args.model_dir+'/model'

    if args.bert_name != 'None':
      bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
    else:
      bert_tokenizer = None
      args.bert_name = None

    # parsing mode
    if args.mode == 'test':
        sys.stderr.write("Parsing %s with %s ...\n" % (args.conll_file, model_file))
        parser = torch.load(model_file)
        DEVICE = parser.device
        sys.stderr.write("loaded model %s\n" % (model_file))
        indices = parser.indices
        if args.data_format in ['deep', 'sdp']:
            if args.data_format == 'deep':
              sentences = load_dep_graphs(args.conll_file, corpus_type='toparse')
            else:
              sentences = load_dep_graphs_sdp_format(args.conll_file, corpus_type='toparse')
            graph_mode = True
            dataset = DepGraphDataSet('toparse', sentences['toparse'], indices, DEVICE)
        else:
            graph_mode = False
            sentences = load_dep_trees(args.conll_file, corpus_type='toparse')
            dataset = DepTreeDataSet('toparse', sentences['toparse'], indices, DEVICE)
            
        sys.stderr.write("parsing and evaluating conll file %s\n" % args.conll_file)

        logstream = open(args.model_dir+'/log_parse', 'w')
        
        task2nbcorrect, task2acc = parser.predict_and_evaluate(dataset, logstream, out_file=args.out_parsed_file, study_scores=args.study_scores)
        # print TODO
        for stream in [sys.stderr, logstream]:
          print(task2nbcorrect)
          print(task2acc)
          #r, p, f = rec_prec_fscore(nb_correct_u, nb_gold, nb_pred)
          #stream.write("U R = %5.2f P = %5.2f Fscore = %5.2f on %s\n" % (r, p, f, args.conll_file))
          #r, p, f = rec_prec_fscore(nb_correct_l, nb_gold, nb_pred)
          #stream.write("L R = %5.2f P = %5.2f Fscore = %5.2f on %s\n" % (r, p, f, args.conll_file))

    # train model on train data and check performance on validation set
    else:

        # before anything: check whether we will be able to dump the model
        pdir = os.path.dirname(model_file)
        if not pdir: pdir = '.'
        # if parent dir is writable
        if not os.access(pdir, os.W_OK):
            exit("Model file %s will not be writable!\n" % model_file)

        # --------------------- DEVICE ---------------------
        # si un GPU est disponible on l'utilisera
        if torch.cuda.is_available():
          # objet torch.device          
          DEVICE = torch.device("cuda:"+str(args.device_id))
        
          print('There are %d GPU(s) available.' % torch.cuda.device_count())    
          device_id = args.device_id #torch.cuda.current_device()
          gpu_properties = torch.cuda.get_device_properties(device_id)
          print("We will use GPU %d (%s) of compute capability %d.%d with %.2fGb total memory.\n" % 
                (device_id,
                 gpu_properties.name,
                 gpu_properties.major,
                 gpu_properties.minor,
                 gpu_properties.total_memory / 1e9))

        else:
          print('No GPU available, using the CPU instead.')
          DEVICE = torch.device("cpu")


        # -----  task definition ----------------------------
        tasks = args.tasks.lower().split('.')
        if sum([ int(t not in ['a','l','h','d','s','b','dpa']) for t in tasks ]) > 0:
          exit("ERROR: tasks should be among a l h d s b")

        if args.arc_loss == 'dyn_hinge' and sum([ int(t in ['h','b']) for t in tasks ]) == 0:
          exit("ERROR: h or b task is required when using dyn_hinge loss")

        # in dyn_hinge loss, the L* perfs are not reliable in first epochs
        #        because direct prediction of nb heads not reliable in the beginning?
        # (because relying on H task)
        # => early stopping using loss
        if args.arc_loss == 'dyn_hinge' and args.early_stopping_style != 'loss':
          exit("ERROR: early stopping style should be 'loss' when using dyn_hinge loss")
          
        coeff_aux_task_as_input = {}
        if args.coeff_aux_task_as_input != 'None':
          c = args.coeff_aux_task_as_input
          for x in c.strip().split('.'):
            (t, v) = x.split(':')
            v = int(v)
            if t in tasks and t in ['s', 'h', 'b']:
                coeff_aux_task_as_input[t] = v
            else:
                print("WARNING coeff_aux_task_as_input %s incoherent, skipping %s : %d " % (c, t, v))

        coeff_aux_task_stack_propag = {}
        if args.coeff_aux_task_stack_propag != 'None':
          c = args.coeff_aux_task_stack_propag
          for x in c.strip().split('.'):
            (t, v) = x.split(':')
            v = int(v)
            if t in tasks:
                coeff_aux_task_stack_propag[t] = v
            else:
                print("WARNING coeff_aux_task_stack_propag %s incoherent with tasks, skipping %s : %d " % (c, t, v))
          
        # ------------- DATA (WITHOUT INDICES YET) ------------------------------
        print('loading sentences...')
        split_info_file = args.split_info_file
        if args.data_format in ['deep', 'sdp']:
            graph_mode = True
            if args.data_format == 'deep':
              load_fn = 'load_dep_graphs'
            else:
              load_fn = 'load_dep_graphs_sdp_format'
        else:
            graph_mode = False
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

        if args.l_emb_file != 'None':
            l_emb_file = args.l_emb_file
            use_pretrained_l_emb = True
        else:
            l_emb_file = None
            use_pretrained_l_emb = False
          

        # ------------- INDICES ------------------------------
        # indices are defined on train sentences only
        print('indices...')
        indices = Indices(sentences['train'], w_emb_file=w_emb_file, l_emb_file=l_emb_file, bert_tokenizer=bert_tokenizer)

        # ------------- INDEXED DATA -------------------------
        print('indexing data...')
        data = {}
        for part in sentences.keys():
            if graph_mode:
                data[part] = DepGraphDataSet(part, sentences[part], indices, DEVICE)
            else:
                data[part] = DepTreeDataSet(part, sentences[part], indices, DEVICE)

        # ------------- THE PARSER ---------------------------
        biaffineparser = BiAffineParser(indices, DEVICE, tasks,
                                        w_emb_size=args.w_emb_size,
                                        l_emb_size=args.l_emb_size,
                                        p_emb_size=args.p_emb_size,
                                        lstm_h_size=args.lstm_h_size,
                                        lstm_dropout=args.lstm_dropout,
                                        mlp_arc_o_size=args.mlp_arc_o_size, 
                                        mlp_lab_o_size=args.mlp_lab_o_size,
                                        aux_hidden_size=args.aux_hidden_size,
                                        use_pretrained_w_emb=use_pretrained_w_emb, 
                                        use_pretrained_l_emb=use_pretrained_l_emb,
                                        use_bias=args.use_bias,
                                        bert_name=args.bert_name,
                                        reduced_bert_size=args.reduced_bert_size,
                                        freeze_bert=args.freeze_bert,
                                        mtl_sharing_level=args.mtl_sharing_level,
                                        coeff_aux_task_as_input=coeff_aux_task_as_input,
                                        coeff_aux_task_stack_propag=coeff_aux_task_stack_propag,
                                        use_dyn_weights_pos_neg=args.use_dyn_weights_pos_neg,
        )

        # pour tests plus rapides: utiliser training sur val
        #train_data = data['dev'] # data['train']
        train_data = data['train']
        val_data = data['dev']

        logstream = open(args.model_dir+'/log_train', 'w')

        biaffineparser.train_model(train_data, val_data, args.data_name, model_file, logstream,
                                   args.nb_epochs,
                                   args.batch_size,
                                   args.learning_rate,
                                   args.lex_dropout,
                                   early_stopping_style=args.early_stopping_style,
                                   arc_loss=args.arc_loss,
                                   margin=args.margin,
                                   margin_alpha=args.margin_alpha,
                                   graph_mode = graph_mode)
                
        # TODO: use this opt...
        opt = {
          'logstream':logstream,
          'w_emb_size':args.w_emb_size,
          'w_emb_file':args.w_emb_file,
          'l_emb_size':args.l_emb_size,
          'p_emb_size':args.p_emb_size,
          'lstm_h_size':args.lstm_h_size,
          'mlp_arc_o_size':args.mlp_arc_o_size,
          'mlp_lab_o_size':args.mlp_lab_o_size,
          'use_bias':args.use_bias,
          'bert_name':args.bert_name,
          'tasks':args.tasks,
          'model_file':args.model_dir+'/model',
          'graph_mode':graph_mode,
          'lr':args.learning_rate,
          'lex_dropout':args.lex_dropout,
          'nb_epochs':args.nb_epochs,
          'early_stopping_style':args.early_stopping_style,
        }

