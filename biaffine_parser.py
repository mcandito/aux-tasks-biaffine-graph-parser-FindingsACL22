#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from tqdm import tqdm  
#from tqdm.notebook import tqdm # for progress bars in notebooks
from random import shuffle
import sys
from collections import defaultdict

from modules import *
from data import *
from transformers import AutoModel

"""## The parser"""


def fscore(nb_correct, nb_gold, nb_pred):
    if (nb_correct * nb_gold * nb_pred == 0):
        return 0
    r = nb_correct / nb_gold
    p = nb_correct / nb_pred
    return 100*2*r*p/(p+r)

def rec_prec_fscore(nb_correct, nb_gold, nb_pred):
    if (nb_correct * nb_gold * nb_pred == 0):
        return 0
    r = nb_correct / nb_gold
    p = nb_correct / nb_pred
    return 100*r, 100*p, 100*2*r*p/(p+r)

class BiAffineParser(nn.Module):
    """
dozat and manning 2018 hyperparameters: (table 2 p4)
lstm_layers = 3
p_emb_size = l_emb_size = char = 100
w_emb_size = 125
lstm_layers = 3
lstm_h_size = 600

mlp_arc_o_size = 600
mlp_lab_o_size = 600

dozat conll 2017

lexical dropout (word and pos dropout) = 1/3
we use same- mask dropout (Gal and Ghahramani, 2015) in the LSTM, ReLU layers, and classifiers, dropping in- put and recurrent connections with 33% probability
lstm_layers = 3
lstm_h_size = 200

mlp_arc_o_size = 400
mlp_lab_o_size = 400

    """
    # TODO replicate from dozat conll 2017:
    # "we drop word and tag embeddings inde- pendently with 33% probability
    # When only one is dropped, we scale the other by a factor of two"
    def __init__(self, indices, device, tasks,  # task list (letters a, l, h, d, b, s)
                 w_emb_size=10, #125
                 l_emb_size=None, 
                 p_emb_size=None, # 100
                 use_pretrained_w_emb=False,
                 lstm_dropout=0.33, 
                 lstm_h_size=20, # 600
                 lstm_num_layers=3, 
                 mlp_arc_o_size=25, # 600
                 mlp_lab_o_size=10, # 600
                 mlp_arc_dropout=0.33, 
                 mlp_lab_dropout=0.33,
                 use_bias=False,
                 bert_name=None,   # caution: should match with indices.bert_tokenizer
                 reduced_bert_size=0,
                 freeze_bert=False,
                 mtl_sharing_level=1, # levels of parameter sharing in mtl 1: bert+lstm only, 2: best+lstm+mlp
                 coeff_aux_task_as_input={}, # {'s':5, 'h':20},
    ):
        super(BiAffineParser, self).__init__()

        self.indices = indices
        self.device = device
        self.use_pretrained_w_emb = use_pretrained_w_emb
        # coefficients to multiply output from aux task to serve as input for a / l tasks
        self.coeff_aux_task_as_input = coeff_aux_task_as_input

        self.bert_name = bert_name
        self.reduced_bert_size = reduced_bert_size
        self.freeze_bert = freeze_bert
        if bert_name:
            bert_model = AutoModel.from_pretrained(bert_name,return_dict=True)


        # indices for tasks
        self.tasks = sorted(tasks)
        self.nb_tasks = len(self.tasks)
        self.task2i = dict( [ [self.tasks[i],i ] for i in range(self.nb_tasks) ] )
        self.mtl_sharing_level = mtl_sharing_level

        if 'l' in self.task2i and 'a' not in self.task2i:
          exit("ERROR: task a is required for task l")

        if 'g' in self.task2i and not('d' in self.task2i and 'h' in self.task2i):
          exit("ERROR: tasks d and h are required for task g")

        # ------------ dynamic weights for subtasks -----------------------
        # Kendal et al. 2018 https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
        # log of variance set to zero (<=> variance == 1)
        # (important to put the tensor on the right device BEFORE instantiating a nn.Parameter)
        self.log_sigma2 = nn.Parameter(torch.zeros(self.nb_tasks).to(device))
        #print(self.nb_tasks)
        #print(self.named_parameters())
            
        # ------------ Encoding of sequences ------------------------------
        self.lexical_emb_size = w_emb_size
        w_vocab_size = indices.get_vocab_size('w')

        self.num_labels = indices.get_vocab_size('label')
        self.w_emb_size = w_emb_size
        self.p_emb_size = p_emb_size
        self.l_emb_size = l_emb_size
        self.lstm_h_size = lstm_h_size
        self.mlp_arc_o_size = mlp_arc_o_size
        self.mlp_lab_o_size = mlp_lab_o_size
        self.mlp_arc_dropout = mlp_arc_dropout
        self.mlp_lab_dropout = mlp_lab_dropout
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout

        self.use_bias = use_bias # whether to add bias in all biaffine transformations
    
        # -------------------------
        # word form embedding layer
        if not use_pretrained_w_emb:
            self.w_embs = nn.Embedding(w_vocab_size, w_emb_size).to(self.device)
        else:
            matrix = indices.w_emb_matrix
            
            #if w_emb_size != matrix.shape[1]:
            #    sys.stderr.write("Error, pretrained embeddings are of size %d whereas %d is expected"
            #                     % (matrix.shape[1], w_emb_size))
            if w_vocab_size != matrix.shape[0]:
                sys.stderr.write("Error, pretrained embeddings have a %d vocab size while indices have %d" % (matrix.shape[0], w_vocab_size))
            self.w_embs = nn.Embedding.from_pretrained(matrix, freeze = False).to(self.device)
            # linear transformation of the pre-trained embeddings (dozat 2018)
            self.w_emb_linear_reduction = nn.Linear(matrix.shape[1],w_emb_size).to(self.device)
            print("w_embs done")
        # -------------------------
        # pos tag embedding layer
        if p_emb_size:
            p_vocab_size = indices.get_vocab_size('p')
            # concatenation of embeddings hence +
            self.lexical_emb_size += p_emb_size
            self.p_embs = nn.Embedding(p_vocab_size, p_emb_size).to(self.device)
        else:
            self.p_embs = None

        # -------------------------
        # lemma embedding layer
        if l_emb_size:
            l_vocab_size = indices.get_vocab_size('l')
            self.lexical_emb_size += l_emb_size
            self.l_embs = nn.Embedding(l_vocab_size, l_emb_size).to(self.device)
        else:
            self.l_embs = None

        # -------------------------
        # bert embedding layer
        if bert_name:
            self.bert_layer = bert_model.to(self.device)
            if self.freeze_bert:
              for p in self.bert_layer.parameters():
                p.requires_grad = False

            bert_size = self.bert_layer.config.emb_dim
            if reduced_bert_size > 0:
                self.bert_linear_reduction = nn.Linear(bert_size, reduced_bert_size).to(self.device)
                self.lexical_emb_size += reduced_bert_size
            else:
                self.lexical_emb_size += bert_size
        else:
            self.bert_layer = None
            
        # -------------------------
        # recurrent LSTM bidirectional layers
        #   TODO: same mask dropout across time-steps ("locked dropout")
        self.lstm = nn.LSTM(input_size = self.lexical_emb_size, 
                            hidden_size = lstm_h_size, 
                            num_layers = lstm_num_layers, 
                            batch_first = True,
                            bidirectional = True,
                            dropout = lstm_dropout).to(self.device)

        # -------------------------
        # specialized MLP applied to biLSTM output
        #   rem: here hidden sizes = output sizes
        #   for readability:
        s = 2 * lstm_h_size
        a = mlp_arc_o_size
        l = mlp_lab_o_size
        
        self.arc_d_mlp = MLP(s, a, a, dropout=mlp_arc_dropout).to(device)  
        self.arc_h_mlp = MLP(s, a, a, dropout=mlp_arc_dropout).to(device)  

        self.lab_d_mlp = MLP(s, l, l, dropout=mlp_lab_dropout).to(device)  
        self.lab_h_mlp = MLP(s, l, l, dropout=mlp_lab_dropout).to(device)

        # ------ output of auxiliary tasks included as input for dependents -------
        self.aux_insize = 0
        self.aux_outsize = 0
        if self.coeff_aux_task_as_input:
          if 's' in self.coeff_aux_task_as_input:             
            self.aux_insize += 10
            self.s_embs = nn.Embedding(self.indices.get_vocab_size('slabseq'), 10).to(self.device)
          if 'h' in self.coeff_aux_task_as_input:
            self.aux_insize += 1
          self.aux_outsize = round(mlp_arc_o_size / 8) # part that will be concatenated to arc_d (representation of dependents)
          self.aux_task_as_input_linear_layer = nn.Linear(self.aux_insize, self.aux_outsize).to(self.device)
          print("aux linear layer:", self.aux_insize, self.aux_outsize)
        

        # ---- BiAffine scores for arcs and labels --------
        # biaffine matrix size is num_label x d x d, with d the output size of the MLPs
        if 'a' in self.task2i:
          self.biaffine_arc = BiAffine(device, a, a + self.aux_outsize, use_bias=self.use_bias)
          if 'l' in self.task2i:
            self.biaffine_lab = BiAffine(device, l, l + self.aux_outsize, num_scores_per_arc=self.num_labels, use_bias=self.use_bias)

        # ----- final layers for the sub tasks ------------

        if self.mtl_sharing_level == 1:
          d = s  # final mlps applied to bi-lstm output
        elif self.mtl_sharing_level == 2:
          d = a  # final mlps applied to arc_d / arc_h specialized representations

        #NB: in any case, hidden layers of size a
        
        # final layer to get a single real value for nb heads / nb deps
        # (more precisely : will be interpreted as log(1+nbheads))
        if 'h' in self.task2i:
          #self.final_layer_nbheads = nn.Linear(a, 1).to(self.device)
          self.final_layer_nbheads = MLP(d, a, 1).to(self.device)

        if 'd' in self.task2i:
          #self.final_layer_nbdeps = nn.Linear(a, 1).to(self.device)
          self.final_layer_nbdeps = MLP(d, a, 1).to(self.device)

        # final layer to get a bag of labels vector, of size num_labels + 1 (for an additional "NOLABEL" label) useless in the end
        #@@self.final_layer_bag_of_labels = nn.Linear(a,self.num_labels + 1).to(self.device)
        if 'b' in self.task2i:
          #self.final_layer_bag_of_labels = nn.Linear(a, self.num_labels).to(self.device)
          self.final_layer_bag_of_labels = MLP(d, a, self.num_labels).to(self.device)

        # final layer to get a "sorted label sequence", seen as an atomic symbol
        if 's' in self.task2i:
          #self.final_layer_slabseqs = nn.Linear(a, self.indices.get_vocab_size('slabseq')).to(self.device)
          self.final_layer_slabseqs = MLP(d, a, self.indices.get_vocab_size('slabseq')).to(self.device)

        #for name, param in self.named_parameters():
        #  if name.startswith("final"):
        #    print(name, param.requires_grad)

        
    def forward(self, w_id_seqs, l_id_seqs, p_id_seqs, bert_tid_seqs, bert_ftid_rkss, b_pad_masks, lengths=None):
        """
        Inputs:
         - id sequences for word forms, lemmas and parts-of-speech for a batch of sentences
             = 3 tensors of shape [ batch_size , max_word_seq_length ]
         - bert_tid_seqs : sequences of *bert token ids (=subword ids) 
                           shape [ batch_size, max_token_seq_len ]
         - bert_ftid_rkss : ranks of first subword of each word [batch_size, max_WORD_seq_len +1] (-1+2=1 (no root, but 2 special bert tokens)
         - b_pad_masks : 0 or 1 tensor of shape batch_size , max_word_seq_len , max_word_seq_len 
                         cell [b,i,j] equals 1 iff both i and j are not padded positions in batch instance b
        If lengths is provided : (tensor) list of real lengths of sequences in batch
                                 (for packing in lstm)
        """
        w_embs = self.w_embs(w_id_seqs)
        if self.use_pretrained_w_emb:
            w_embs = self.w_emb_linear_reduction(w_embs)
            
        if self.p_embs:
            p_embs = self.p_embs(p_id_seqs)
            w_embs = torch.cat((w_embs, p_embs), dim=-1)
        if self.l_embs:
            l_embs = self.l_embs(l_id_seqs)
            w_embs = torch.cat((w_embs, l_embs), dim=-1)
        
        if bert_tid_seqs is not None:
            bert_emb_size = self.bert_layer.config.emb_dim
            bert_embs = self.bert_layer(bert_tid_seqs).last_hidden_state
            # select among the subword bert embedddings only the embeddings of the first subword of words
            #   - modify bert_ftid_rkss to serve as indices for gather:
            #     - unsqueeze to add the bert_emb dimension
            #     - repeat the token ranks index along the bert_emb dimension (expand better for memory)
            #     - gather : from bert_embs[batch_sample, all tid ranks, bert_emb_dim]
            #                to bert_embs[batch_sample, only relevant tid ranks, bert_emb_dim]
            #bert_embs = torch.gather(bert_embs, 1, bert_ftid_rkss.unsqueeze(2).repeat(1,1,bert_emb_size))
            bert_embs = torch.gather(bert_embs, 1, bert_ftid_rkss.unsqueeze(2).expand(-1,-1,bert_emb_size))
            if self.reduced_bert_size > 0:
                bert_embs = self.bert_linear_reduction(bert_embs)
            w_embs = torch.cat((w_embs, bert_embs), dim=-1)
            
        # h0, c0 vectors are 0 vectors by default (shape batch_size, num_layers*2, lstm_h_size)

        # pack_padded_sequence to save computations
        #     (compute real length of sequences in batch)
        #     see https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
        #     NB:batch must be sorted in sequence length descending order
        if lengths is not None:
            lengths=lengths.cpu()
            w_embs = pack_padded_sequence(w_embs, lengths, batch_first=True)
        lstm_hidden_seq, _ = self.lstm(w_embs)
        if lengths is not None:
            lstm_hidden_seq, _ = pad_packed_sequence(lstm_hidden_seq, batch_first=True)
        
        # MLPs
        arc_h = self.arc_h_mlp(lstm_hidden_seq) # [b, max_seq_len, mlp_arc_o_size]
        arc_d = self.arc_d_mlp(lstm_hidden_seq)
        lab_h = self.lab_h_mlp(lstm_hidden_seq) # [b, max_seq_len, mlp_lab_o_size]
        lab_d = self.lab_d_mlp(lstm_hidden_seq)

        S_arc = S_lab = log_nbheads = log_nbdeps = bols = S_slabseqs = None

        # ------------- auxiliary tasks ------------------
        # decide the input for the mlps of auxiliary tasks, depending on the sharing level
        if self.mtl_sharing_level == 2:
          input_d = arc_d
          input_h = arc_h
        else:
          input_d = lstm_hidden_seq
          input_h = lstm_hidden_seq
          
        # nb heads / nb deps (actually output will be interpreted as log(1+nb))
        if 'h' in self.task2i:
          log_nbheads = self.final_layer_nbheads(input_d).squeeze(2) # [b, max_seq_len]

        if 'd' in self.task2i:
          log_nbdeps = self.final_layer_nbdeps(input_h).squeeze(2)   # [b, max_seq_len]

        # bag of labels
        if 'b' in self.task2i:
          bols = self.final_layer_bag_of_labels(input_d) # [b, max_seq_len, num_labels + 1] (+1 for NOLABEL)

        # sorted lab sequences (equivalent to bag of labels, but seen as a symbol for the whole BOL)
        if 's' in self.task2i:
          S_slabseqs = self.final_layer_slabseqs(input_d)

        # Biaffine scores for arcs
        if 'a' in self.task2i:
          # if use output of aux tasks s and binh as input rep for dependents
          # => modify arc_d
          if self.coeff_aux_task_as_input:
            if 's' in self.coeff_aux_task_as_input:
              # predict the slabseqs
              pred_slabseqs = torch.argmax(S_slabseqs, dim=2) # [b, d]              
              # convert them to embeddings
              aux_input = self.coeff_aux_task_as_input['s'] * self.s_embs(pred_slabseqs)
              if 'h' in self.coeff_aux_task_as_input:
                # take the scores of each of the bin nb of heads (0, 1, or more than 1 (=2))
                ##pred_binnbheads = torch.clamp(torch.round(torch.exp(log_binnbheads) - 1), 0, 2)
                c = self.coeff_aux_task_as_input['h']
                aux_input = torch.cat((aux_input, c * log_nbheads.unsqueeze(2)), dim=-1)
            elif 'h' in self.coeff_aux_task_as_input:
              aux_input = self.coeff_aux_task_as_input['h'] * log_nbheads.unsqueeze(2)
            #print("AUX INPUT", aux_input.shape)
            aux_hidden = self.aux_task_as_input_linear_layer(aux_input)
            arc_d = torch.cat((arc_d, aux_hidden), dim=-1)
            
          S_arc = self.biaffine_arc(arc_h, arc_d) # S(k, i, j) = score of sample k, head word i, dep word j

        # Biaffine scores for labeled arcs
        if 'l' in self.task2i:
          if self.aux_outsize:
            # aux_hidden already computed in task 'a'
            lab_d = torch.cat((lab_d, aux_hidden), dim=-1)
          S_lab = self.biaffine_lab(lab_h, lab_d) # S(k, l, i, j) = score of sample k, label l, head word i, dep word j
        

        return S_arc, S_lab, log_nbheads, log_nbdeps, bols, S_slabseqs
    
    def batch_forward_and_loss(self, batch, trace_first=False, make_alt_preds=False):
        """
        - batch of sentences (output of make_batches)

        - make_alt_preds : study alternative ways of make predictions
        
        NB: pertains in graph mode only !!
          - in arc_adja (resp. lab_adja), cells equal 1 for gold arcs
             (0 cells for non gold or padded)
          - pad_masks : 0 cells if head OR dep is padded token and 1 otherwise
        """
        lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja, bols, slabseqs = batch
            
        # forward 
        S_arc, S_lab, log_pred_nbheads, log_pred_nbdeps, log_pred_bols, scores_slabseqs = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, lengths=lengths)

        # pad_masks is [b, m, m]
        # => build a simple [b, m] mask
        linear_pad_mask = pad_masks[:,0,:] 
        nb_toks = linear_pad_mask.sum().item()
        batch_size = forms.shape[0]

        task2loss = defaultdict(int)

        loss = torch.zeros(()).to(self.device)
        dyn_loss_weights = torch.exp( - self.log_sigma2 ) # if lsig2 is log(sigma2), then exp(-lsig2) = 1/sigma2

        # NB: all sents in batch start with the <root> tok (not padded)
        if 'a' in self.task2i:
          arc_loss = self.bce_loss_with_mask(S_arc, arc_adja, pad_masks)
          ti = self.task2i['a']
          task2loss['a'] = arc_loss.item()
          loss +=  (dyn_loss_weights[ti] * arc_loss) + self.log_sigma2[ti]

        if 'l' in self.task2i:
          # --- Label loss -------------------------
          # label scores : rearrange into a batch in which each sample is 
          #                - one head and dep token pair
          #                - label scores for such arc (NB: non-gold arcs will be masked)
          # S_lab is [batch, label, head, dep]
          s_labels = S_lab.transpose(2,1).transpose(3,2)             # b , h , d, l
          s_labels = torch.flatten(s_labels, start_dim=0, end_dim=2) # b * h * d, l
        
          # same for gold labels
          g_labels = torch.flatten(lab_adja) # [b, h, d] ==> [b * h * d]

          # loss with ignore_index == 0 (=> ignore padded arcs and non-gold arcs, which don't have gold labels anyway)
          # cf. Dozat et al. 2018 "back-propagating error to the labeler only through edges with a non-null gold label"
          lab_loss = self.ce_loss(s_labels, g_labels) 
          ti = self.task2i['l']
          task2loss['l'] = lab_loss.item()
          loss +=  (dyn_loss_weights[ti] * lab_loss) + self.log_sigma2[ti]
        
            
        # auxiliary tasks
        if 'h' in self.task2i:
          gold_nbheads = arc_adja.sum(dim=1).float() # [b, h, d] => [b, d]
          log_gold_nbheads = torch.log(1 + gold_nbheads)
          loss_h = self.mse_loss_with_mask(log_pred_nbheads, log_gold_nbheads, linear_pad_mask)
          task2loss['h'] = loss_h.item()
          ti = self.task2i['h']
          loss +=  (dyn_loss_weights[ti] * loss_h) + self.log_sigma2[ti]
        else:
          gold_nbheads = None

        if 'd' in self.task2i:
          gold_nbdeps = arc_adja.sum(dim=2).float()  # [b, h, d] => [b, h]
          log_gold_nbdeps = torch.log(1 + gold_nbdeps)
          loss_d = self.mse_loss_with_mask(log_pred_nbdeps, log_gold_nbdeps, linear_pad_mask)
          task2loss['d'] = loss_d.item()
          ti = self.task2i['d']
          loss +=  (dyn_loss_weights[ti] * loss_d) + self.log_sigma2[ti]
        else:
          gold_nbdeps = None

#        # predicted global balance in each sentence, between the predicted nbheads and the predicted nbdeps
#        # which should be 0
#        if 'g' in self.task2i:
#          # for each sent, total nb heads minus total nb deps
#          pred_h_d_per_sentence = (log_pred_nbheads * linear_pad_mask).sum(dim=1) - (log_pred_nbdeps * linear_pad_mask).sum(dim=1)
#          gold_h_d_per_sentence = torch.zeros(batch_size).to(self.device)
#          # rescaling the loss : global loss is for all sentences => rescale to approx nb of tokens
#          loss_global = (nb_toks / batch_size) * self.mse_loss(pred_h_d_per_sentence, gold_h_d_per_sentence)
#          task2loss['g'] = loss_global.item()
#          ti = self.task2i['g']
#          loss +=  (dyn_loss_weights[ti] * loss_global) + self.log_sigma2[ti]

        if 'b' in self.task2i:
          # unfortunately, bincount on 1-d tensors only 
          # so computing the gold BOLs in make_batches rather
          #torch.bincount(arc_lab, minlength=self.num_labels) # +1
          # bols are [b, d, num_labels]
          log_gold_bols = torch.log(1+bols)
          #loss_bol = self.cosine_loss(log_pred_bols, log_gold_bols, linear_pad_mask)
          loss_bol = self.l2dist_loss(torch.flatten(log_pred_bols, start_dim=0, end_dim=1), # flatten from [b, d, l] to [b*d, l]
                                      torch.flatten(log_gold_bols, start_dim=0, end_dim=1),
                                      torch.flatten(linear_pad_mask, start_dim=0, end_dim=1))
          task2loss['b'] = loss_bol.item()
          ti = self.task2i['b']
          loss +=  (dyn_loss_weights[ti] * loss_bol) + self.log_sigma2[ti]

        if 's' in self.task2i:
          loss_slabseq = self.ce_loss(torch.flatten(scores_slabseqs, start_dim=0, end_dim=1),
                                      torch.flatten(slabseqs, start_dim=0, end_dim=1))
          task2loss['s'] = loss_slabseq.item()
          ti = self.task2i['s']
          loss +=  (dyn_loss_weights[ti] * loss_slabseq) + self.log_sigma2[ti]

        if trace_first:
          for ti, task in enumerate(self.tasks):
            print("dyn_loss_w of task %s : %f" %(task.upper(), dyn_loss_weights[ti]))

        # --- Prediction and evaluation --------------------------
        # provide the batch, and all the output of the forward pass
        task2nbcorrect, _, _, _, _ = self.batch_predict_and_evaluate(batch, gold_nbheads, gold_nbdeps, linear_pad_mask, 
                                                                     S_arc, S_lab, log_pred_nbheads, log_pred_nbdeps, log_pred_bols, scores_slabseqs,
                                                                     make_alt_preds)
 
        return loss, task2loss, task2nbcorrect, nb_toks

    #@@ OBSOLETE - TO UPDATE for TREE MODE
        # # tree mode
        # else:
        #     lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, heads, labels = batch
                    
        #     S_arc, S_lab = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, lengths=lengths)
            
        #     # --- Arc loss -------------------------
        #     if self.lab_loss_only:
        #       arc_loss = 0
        #     else:
        #       # predicted scores for arcs:
        #       #   rearrange into a batch containing the seq_len scores for each dependent token
        #       s_heads = torch.flatten(S_arc.transpose(-1,-2), start_dim=0, end_dim=1) # [b, h, d] => [b * d, h]
        #       g_heads = torch.flatten(heads)                                          # [b, d] = gold h => [b * d] = gold h
        #       arc_loss = self.ce_loss_fn_arc(s_heads, g_heads) # ignore -1 gold values (padded)

        #     # Predicted heads (needed for label loss: consider label scores of predicted heads)
        #     #    here simply predict head with max score, no MST, no cycle checking...
        #     pred_heads = torch.argmax(S_arc, dim=1) # [b, h, d ] ==> [b, d] = predicted head for d

        #     # --- Label loss -------------------------
        #     # Get the label scores of gold heads only
        #     # heads is [b, d] = gold h
        #     #           
        #     # prepare heads to serve as index for gather
        #     #   - unsqueeze: add dim for label
        #     #   - expand : repeat gold head values for all labels
        #     #   - unsqueeze : add dim for gold head (size 1)
        #     if lab_loss_weight > 0:
        #       num_labels = S_lab.shape[1]
        #       ## OBS g_heads = heads.unsqueeze(1).repeat(1,num_labels,1).unsqueeze(2)

        #       # Predicted label scores of PREDICTED heads (dozat et al. 2017):
        #       #    prepare pred_heads to serve as index for gather
        #       #    [b, d] = pred h => [b, 1, d] => [b, l, d] => [b, l, 1, d] = pred h
        #       i_pred_heads = pred_heads.unsqueeze(1).expand(-1,num_labels,-1).unsqueeze(2) 
        #       # gather     ==> s_labels[b, l, h, d] = S_lab[b, l, i_pred_heads[b,l,1,d], d]
        #       # squeeze(2) ==> a[b, l, d] score of label l for gold head of d
        #       # transpose(-2,-1) ==> a[b, d, l]
        #       s_labels = torch.gather(S_lab,2,i_pred_heads).squeeze(2).transpose(-2,-1)

        #       # rearrange into a batch containing the labels' scores for each (batch sample, dependent token) pair
        #       # [b, d, l] ==> [b * d, l] = pred score of label l for arc from pred-h-->d
        #       s_labels_flat = torch.flatten(s_labels, start_dim=0, end_dim=1) 

        #       # Gold labels of gold heads: 
        #       # same rearrangement from [b , d] to [b * d] = gold label
        #       g_labels_flat = torch.flatten(labels)
            
        #       lab_loss = self.ce_loss_fn_label(s_labels_flat, g_labels_flat) # cells with PAD_ID=0 gold label will be ignored

        #       # --- Predicted labels -------------------
        #       with torch.no_grad():
        #           # Predicted labels for the predicted arcs:
        #           #  reuse s_labels : [b, d, l] predicted label scores of predicted heads
        #           pred_labels = torch.argmax(s_labels, dim = 2)

        #     else:
        #       lab_loss = 0
        #       pred_labels = None


        #     # --- Evaluation -------------------------
        #     nb_gold, nb_pred, nb_correct_u, nb_correct_u_and_l = self.evaluate_tree_mode(batch, pred_heads, pred_labels)

    
    def batch_predict_and_evaluate(self, batch, 
                                   gold_nbheads, gold_nbdeps, linear_pad_mask, # computed in batch_forward_and_loss
                                   S_arc, S_lab, log_pred_nbheads, log_pred_nbdeps, log_pred_bols, scores_slabseqs, # output by forward pass
                                   make_alt_preds=False # whether to study other prediction algorithms
                                   ):

      lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja, bols, slabseqs = batch

      task2nbcorrect = defaultdict(int)
      task2preds = {}

      # --- Prediction and evaluation --------------------------
      with torch.no_grad():
        if 'a' in self.task2i:
          pred_arcs = (S_arc > 0).int() * pad_masks  # b, h, d
          nb_correct_u = torch.sum((pred_arcs * arc_adja).int()).item()
          nb_gold = torch.sum(arc_adja).item()
          nb_pred = torch.sum(pred_arcs).item()
          task2nbcorrect['a'] = (nb_correct_u, nb_gold, nb_pred)

        if 'l' in self.task2i:
            # labeled
            pred_labels = torch.argmax(S_lab, dim=1) # for all arcs (not only the predicted arcs)
            # count correct labels for the predicted arcs only
            nb_correct_u_and_l = torch.sum((pred_labels == lab_adja).float() * pred_arcs).item()
            task2nbcorrect['l'] = (nb_correct_u_and_l, nb_gold, nb_pred)

        # NB: round predicted numbers of heads / deps for evaluation only
        if 'h' in self.task2i:
          pred_nbheads = torch.round(torch.exp(log_pred_nbheads) - 1).int()
          task2nbcorrect['h'] = torch.sum((pred_nbheads == gold_nbheads).int() * linear_pad_mask).item()
          task2preds['h'] = pred_nbheads

        if 'd' in self.task2i:
          pred_nbdeps = torch.round(torch.exp(log_pred_nbdeps) - 1).int()
          task2nbcorrect['d'] = torch.sum((pred_nbdeps == gold_nbdeps).int() * linear_pad_mask).item()
          task2preds['d'] = pred_nbdeps

        if 'b' in self.task2i:
          pred_bols = torch.round(torch.exp(log_pred_bols) - 1).int() # [b, d, num_labels+1]
          # nb of b , d pairs (token d in batch instance b) for which the full predicted bol is correct
          #   i.e. nb_toks minus the number of b,d pairs for which 
          #        there is at least (torch.any) one label dim differing (!=) between gold and predicted
          # from mask [b,d] to mask [b,d,num_labels+1]
          bol_pad_mask = linear_pad_mask.unsqueeze(2).expand(-1,-1,self.num_labels) #@@ +1)
          nb_incorrect = torch.sum(torch.any(((pred_bols != bols).int() * bol_pad_mask).bool(), dim=2).int())
          nb_toks = linear_pad_mask.sum().item()          
          task2nbcorrect['b'] = nb_toks - nb_incorrect.item()
          task2preds['b'] = pred_bols

        if 's' in self.task2i:
          pred_slabseqs = torch.argmax(scores_slabseqs, dim=2) # [b, d]
          task2nbcorrect['s'] = torch.sum((pred_slabseqs == slabseqs).int() * linear_pad_mask).item()
          # count the unk slabseq as incorrect
          task2nbcorrect['sknown'] = task2nbcorrect['s'] - torch.sum((pred_slabseqs == UNK_ID).int() * linear_pad_mask).item()
          task2preds['s'] = pred_slabseqs

        # alternative ways to predict arcs
        alt_pred_arcs = {}
        if make_alt_preds:
          # tensors for other ways to predict arcs, 
          #           according to best xxx scores for each dependent d
          #           with xxx being the nbheads predicted using tasks h, s, or v
          if 'h' in self.tasks:
            # pred_nbheads already computed
            alt_pred_arcs['h'] = torch.zeros(S_arc.shape)
          if 'a' in self.tasks:
            # recompute the nb of heads predicted using the 'a' task
            nbheads_from_a = torch.sum(pred_arcs, dim=1) # b, d
            # alt_pred_arcs['a'] useless, cf. this is already the 'a' task
          if 'b' in self.tasks:
            nbheads_from_b = torch.sum(pred_bols, dim=2) # b, d, num_labels => b, d
            alt_pred_arcs['b'] = torch.zeros(S_arc.shape)
          if 's' in self.tasks:
            nbheads_from_s, bols_from_s = self.indices.interpret_slabseqs(pred_slabseqs)
            nbheads_from_s = nbheads_from_s.to(self.device)
            alt_pred_arcs['s'] = torch.zeros(S_arc.shape)
          if 'a' in self.tasks and 'h' in self.tasks and 's' in self.tasks:
            uninterpretable = (nbheads_from_s == -1).int() # cases for which output slabseq is not interpretable
            interpretable = (nbheads_from_s != -1).int()
            # majority vote on a, h, s
            if 'b' not in self.tasks:
              nbheads_from_v = torch.round(((nbheads_from_a + nbheads_from_s + pred_nbheads) * interpretable / 3) # will yield 0 if score 0 or 1, and 1 if score 2 or 3
                                            + (pred_nbheads * uninterpretable)).int() # when s in unavailable, use nbheads from task h
              task2nbcorrect['v'] = torch.sum((nbheads_from_v == gold_nbheads).int() * linear_pad_mask).item()
              alt_pred_arcs['v'] = torch.zeros(S_arc.shape)
            # else majority vote on b, h, s if s is known, else on b, h, a
            else:
              nbheads_from_v = torch.round(((nbheads_from_b + nbheads_from_s + pred_nbheads) * interpretable / 3) # will yield 0 if score 0 or 1, and 1 if score 2 or 3
                                            + ((nbheads_from_a + nbheads_from_s + pred_nbheads) * uninterpretable / 3)).int() # when s in unavailable, use nbheads from task h
              task2nbcorrect['v'] = torch.sum((nbheads_from_v == gold_nbheads).int() * linear_pad_mask).item()
              alt_pred_arcs['v'] = torch.zeros(S_arc.shape)

          # --- predict the top most arcs according to various nbheads ---
          # sort the scores of the arcs in ascending order
          # before sorting:
          # set the padded cells to min score in all the batch
          min_arc_score = torch.min(S_arc).item() - 1
          S_arc_padded = (S_arc * pad_masks) + (1 - pad_masks) * min_arc_score 
          s, indices = torch.sort(S_arc_padded, dim=1) 
          (bs, m, m) = S_arc.shape
          for b in range(bs):
              # get the number of heads for each d, according to the various nbheads
              for d in range(m): # d
                  int_nbheads_list = {}
                  if 'h' in alt_pred_arcs:
                    int_nbheads_list['h'] = pred_nbheads[b,d].item()
                  if 'b' in alt_pred_arcs:
                    int_nbheads_list['b'] = nbheads_from_b[b,d].item()
                  if 's' in alt_pred_arcs:
                    # fall back on task h if unknown slabseq
                    if nbheads_from_s[b,d] == -1 and 'h' in alt_pred_arcs:
                      int_nbheads_list['s'] = pred_nbheads[b,d].item()
                    else:
                      int_nbheads_list['s'] = nbheads_from_s[b,d].item()
                  if 'v' in alt_pred_arcs:
                    int_nbheads_list['v'] = nbheads_from_v[b,d].item()
                  # => we keep the *last* int_nbheads_list[t] heads in the sorted heads
                  for t in int_nbheads_list: # different ways to get the nbheads
                      nbheads = int_nbheads_list[t]
                      for i in range(nbheads):
                          h = m - i - 1 # get the m-i th last score
                          alt_pred_arcs[t][b, indices[b, h, d], d] = 1
                  #for h in range(m): 
                  #    # arc is predicted if
                  #      if h < (m - int_nbheads_list[t]):
                  #        alt_pred_arcs[t][b, indices[b, h, d], d] = 0
                  #      else: 
                  #        alt_pred_arcs[t][b, indices[b, h, d], d] = 1

        
          # evaluate the alternative arc predictions
          for t in alt_pred_arcs.keys(): 
            alt_pred_arcs[t] = alt_pred_arcs[t].to(self.device) * pad_masks
            nb_correct_u = torch.sum((alt_pred_arcs[t] * arc_adja).int()).item()
            nb_pred = torch.sum(alt_pred_arcs[t]).item()
            task2nbcorrect['a' + t] = (nb_correct_u, nb_gold, nb_pred)
            # pred_labels contains the best label for all (h,d) pairs, 
            # and thus are common to any arc prediction style
            nb_correct_u_and_l = torch.sum((pred_labels == lab_adja).float() * alt_pred_arcs[t]).item()
            task2nbcorrect['l' + t] = (nb_correct_u_and_l, nb_gold, nb_pred)

      return task2nbcorrect, pred_arcs, pred_labels, alt_pred_arcs, task2preds

    def train_model(self, train_data, val_data, data_name, out_model_file, log_stream, nb_epochs, batch_size, lr, lex_dropout, graph_mode=True):
        """
        CAUTION: does not work in tree mode anymore
        # TODO: recode the tree mode
        """
        self.graph_mode = graph_mode
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.lex_dropout = lex_dropout # proba of word / lemma / pos tag dropout
        self.batch_size = batch_size
        self.beta1 = 0.9
        self.beta2 = 0.9
        #optimizer = optim.SGD(biaffineparser.parameters(), lr=LR)
        #optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0., 0.95), eps=1e-09)
        optimizer = optim.Adam(self.parameters(), lr=lr, betas=(self.beta1, self.beta2), eps=1e-09)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

        # loss function for nbheads / nbdeps (per token)
        self.mse_loss_with_mask = MSELoss_with_mask(reduction='sum') # 'h', 'd'
        # loss function for global_h_d (per sentence)
        self.mse_loss = nn.MSELoss(reduction='sum')                  # 'g'
        #self.cosine_loss = CosineLoss_with_mask(reduction='sum')
        self.l2dist_loss =  L2DistanceLoss_with_mask(reduction='sum') # 'b'
        # for label loss, the label for padded deps is PAD_ID=0 
        #   ignoring padded dep tokens (i.e. whose label id equals PAD_ID)
        # used both for arc labels and for sorted label sequences (seen as atoms)
        self.ce_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_ID) # 'l', 's'
        # for graph mode arcs
        self.bce_loss_with_mask = BCEWithLogitsLoss_with_mask(reduction='sum') # 'a'
        # for tree mode arcs (and tree mode labels??)
        #   (CrossEnt cf. softmax not applied yet in BiAffine output)
        #   ignoring padded dep tokens (i.e. whose head equals PAD_HEAD_RK)
        #self.ce_loss_fn_arc = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_HEAD_RK)
                
        #log_stream = open(out_log_file, 'w')

        self.data_name = data_name
        self.build_log_suff()
        self.log_train_hyper(sys.stdout)
        self.log_train_hyper(log_stream)
        
        # losses and scores at each epoch (on train / validation)
        train_losses = []
        val_losses = []
        val_task2losses = defaultdict(list) 
        train_task2accs = defaultdict(list)
        val_task2accs = defaultdict(list)  

        # word / pos / lemma dropout of training data only
        # NB: re-drop at each epoch seems ok
        #train_data.lex_dropout(lex_dropout)

        for epoch in range(1,nb_epochs+1):
   
            i = 0
            train_loss = 0
            train_task2loss = defaultdict(int)
            train_task2nbcorrect = defaultdict(int)
            train_nb_toks = 0
            val_loss = 0
            val_task2loss = defaultdict(int)
            val_task2nbcorrect = defaultdict(int)
            val_nb_toks = 0
            
            if 'a' in self.task2i:
              train_task2nbcorrect['a'] = [0,0,0] # tasks with fscore metric we have a triplet nbcorrect, nbgold, nbpred
              val_task2nbcorrect['a'] = [0,0,0] 
            if 'l' in self.task2i:
              train_task2nbcorrect['l'] = [0,0,0]
              val_task2nbcorrect['l'] = [0,0,0] 

            # training mode (certain modules behave differently in train / eval mode)
            self.train()
            train_data.lex_dropout(lex_dropout) 
            bid = 0
            trace_first = True
            for batch in train_data.make_batches(self.batch_size, shuffle_data=True, sort_dec_length=True, shuffle_batches=True):        
                self.zero_grad()
                bid += 1
                if bid % 2000 == 0:
                  print("BATCH SHAPE:", batch[2].shape, batch[5].shape)
                  print("MEMORY BEFORE BATCH FORWARD AND LOSS")
                  printm()

                loss, task2loss, task2nbcorrect, nb_toks = self.batch_forward_and_loss(batch, trace_first=trace_first)
                trace_first = False
                
                loss.backward()
                optimizer.step() 
                loss.detach()           

                train_loss += loss.item()
                train_nb_toks += nb_toks
                for k in self.tasks:
                  train_task2loss[k] += task2loss[k]
                  if k in ['a','l']: 
                    for i in [0,1,2]:
                      train_task2nbcorrect[k][i] += task2nbcorrect[k][i]
                  elif k != 'g':
                    train_task2nbcorrect[k] += task2nbcorrect[k]

            # for one epoch              
            print("Train: nb toks " + str(train_nb_toks) + "/ " + " / ".join([t.upper()+":"+str(train_task2nbcorrect[t]) for t in self.tasks]))              
            assert train_nb_toks == train_data.nb_words, "train_nb_toks %d should equal train_data.nb_words %d" %(train_nb_toks, train_data.nb_words)

            for k in self.tasks:
              train_task2loss[k] /= train_data.nb_words
              if k in ['a', 'l']:
                train_task2accs[k].append(fscore(*train_task2nbcorrect[k]))
              elif k != 'g':
                train_task2accs[k].append( 100 * train_task2nbcorrect[k] / train_nb_toks )            
            train_loss = train_loss/train_data.nb_words
            train_losses.append(train_loss)

            self.log_perf(log_stream, epoch, 'Train', train_loss, train_task2loss, train_task2accs)

            if val_data:
                self.eval()
                # calcul de la perte sur le validation set
                with torch.no_grad():
                    trace_first = True
                    for batch in val_data.make_batches(self.batch_size, sort_dec_length=True):
                        loss, task2loss, task2nbcorrect, nb_toks = self.batch_forward_and_loss(batch, trace_first=trace_first, make_alt_preds=True)
                        val_loss += loss.item()
                        val_nb_toks += nb_toks
                        for k in task2nbcorrect.keys():
                          if k in task2loss:
                            val_task2loss[k] += task2loss[k]
                          if type(task2nbcorrect[k]) != int: # tuple or list 
                            if k not in val_task2nbcorrect: # those that are not registered yet are the make_alt_preds, and are only fscore-like
                              val_task2nbcorrect[k] = [0,0,0]
                            for i in [0,1,2]:
                              val_task2nbcorrect[k][i] += task2nbcorrect[k][i]
                          elif k != 'g':
                            val_task2nbcorrect[k] += task2nbcorrect[k]
                        trace_first = False
                        
                    # for one epoch
                    print("Val: nb toks " + str(val_nb_toks) + "/ " + " / ".join([t.upper()+":"+str(val_task2nbcorrect[t]) for t in self.tasks]))              
                    assert val_nb_toks == val_data.nb_words, "val_nb_toks %d should equal val_data.nb_words %d" %(val_nb_toks, val_data.nb_words)
                    for k in task2nbcorrect.keys():#self.tasks:
                      if k in val_task2loss:
                        val_task2loss[k] /= val_data.nb_words
                      # if task if fscore-like
                      if type(val_task2nbcorrect[k]) == list:
                        val_task2accs[k].append(fscore(*val_task2nbcorrect[k]))
                      elif k != 'g':
                        val_task2accs[k].append( 100 * val_task2nbcorrect[k] / val_nb_toks )            

                    val_loss = val_loss/val_data.nb_words
                    val_losses.append(val_loss)

                self.log_perf(log_stream, epoch, '\tValid', val_loss, val_task2loss, val_task2accs)
    
                if epoch == 1:
                    print("saving model after first epoch\n")
                    torch.save(self, out_model_file)
                # if validation loss has decreased: save model
                # nb: when label loss comes into play, it might artificially increase the overall loss
                #     => we don't early stop at this stage 
                #elif (val_losses[-1] < val_losses[-2]) or (epoch == self.nb_epochs_arc_only) :
                # early stopping if all the L* perfs have decreased
                else:
                  stop = True
                  for t in [ x for x in val_task2accs.keys() if x.startswith("l")]:
                    if val_task2accs[t][-1] > val_task2accs[t][-2] :
                      stop = False
                      break
                  if not stop:
                    for stream in [sys.stdout, log_stream]:
                        #stream.write("Validation loss has decreased, saving model, current nb epochs = %d\n" % epoch)
                        stream.write("Validation L* perf has increased, saving model, current nb epochs = %d\n" % epoch)
                    torch.save(self, out_model_file)
                # otherwise: early stopping, stop training, reload previous model
                # NB: the model at last epoch was not saved yet
                # => we can reload the model from the previous storage
                  else:
                    #print("Validation loss has increased, reloading previous model, and stop training\n")
                    print("Validation L* perf has decreased, reloading previous model, and stop training\n")
                    self.log_best_perf(log_stream, 'val', epoch - 1, val_task2accs)
                    # reload (on the appropriate device)
                    # cf. https://pytorch.org/docs/stable/generated/torch.load.html#torch-load
                    self = torch.load(out_model_file)
                    # stop loop on epochs
                    break
                
            scheduler.step()
        # if no early stopping
        else:
            print("Max nb epochs reached\n")
            self.log_best_perf(log_stream, 'val', epoch , val_task2accs)
        # end loop on epochs

        for stream in [sys.stdout, log_stream]:
          stream.write("train losses: %s\n" % ' / '.join([ "%.4f" % x for x in train_losses]))
          stream.write("val   losses: %s\n" % ' / '.join([ "%.4f" % x for x in val_losses]))
          for k in sorted(val_task2accs.keys()):
            if k in train_task2accs:
              stream.write("train %s accs: %s\n" % (k.upper(), ' / '.join([ "%.2f" % x for x in train_task2accs[k] ])))
            stream.write("val   %s accs: %s\n" % (k.upper(), ' / '.join([ "%.2f" % x for x in val_task2accs[k] ])))

    def log_perf(self, outstream, epoch, ctype, l, task2loss, task2accs):
        for stream in [sys.stdout, outstream]:
          stream.write("%s   Loss  for epoch %d: %.4f\n" % (ctype, epoch, l))
          for k in sorted(task2loss.keys()):
            stream.write("%s %s Loss  for epoch %d: %.4f\n" % (ctype, k.upper(), epoch, task2loss[k]))
          for k in sorted(task2accs.keys()):
            stream.write("%s %s ACC after epoch %d : %.2f\n" % (ctype, k.upper(), epoch, task2accs[k][-1]))
          #stream.write("Loss / arc loss / lab loss for epoch %2d on %s: %12.2f / %12.2f / %12.2f\n" % (epoch, ctype, l, arc_l, lab_l))
          #stream.write("            Fscore U / L after epoch %2d on %s :     U %5.2f / L %5.2f\n" % (epoch, ctype, f_u, f_l))

    def log_best_perf(self, outstream, ctype, epoch, task2accs):
        """
        task2accs is a dict from task keys to list of performances for each epoch
        """
        # see build_log_suff for the headings
        outstream.write(self.log_heading_suff)

        perfs = []
        for t in ['a', 'l', 'ah', 'lh', 'as', 'ls', 'av', 'lv']:
            if t in task2accs:
                perfs.append("%5.2f" % task2accs[t][epoch - 1]) # epoch -1 cf. epochs start at 1, but rank start at 0
            else:
                perfs.append('NA')

        s = '\t'.join( [ 'RESULT' , ctype ] + perfs + [ str(epoch) ] ) + '\t' + self.log_values_suff
        outstream.write(s)
        
          
    def build_log_suff(self):
        # Fscore for tasks a, l, ah, lh (ah = n best-scored arcs, n computed with nbheads task (h))
        self.log_heading_suff = '\t'.join([ 'RESULT', 'corpus', 'Fa', 'Fl', 'Fah', 'Flh', 'Fas', 'Fls', 'Fav', 'Flv', 'effective nb epochs', 'g or t'] )
        if self.graph_mode:
            self.log_values_suff = 'graph\t'
        else:
            self.log_values_suff = 'tree\t'
        featnames = ['data_name', 'w_emb_size', 'use_pretrained_w_emb', 'l_emb_size', 'p_emb_size', 'bert_name', 'reduced_bert_size', 'freeze_bert', 'lstm_h_size','mlp_arc_o_size','mlp_arc_dropout', 'batch_size', 'beta1','beta2','lr', 'nb_epochs', 'lex_dropout']

        featvals = [ str(self.__dict__[f]) for f in featnames ]

        t = '.'.join(sorted(self.tasks))
        featvals = [t, str(self.coeff_aux_task_as_input)] + featvals
        featnames = ['tasks', 'coeff_aux_task_as_input'] + featnames

        config_str = '_'.join(featvals) # get a compact name for the hyperparameter config
        featvals = [config_str] + featvals
        featnames = ['config_str'] + featnames

        self.log_heading_suff += '\t' + '\t'.join( featnames ) + '\n'
        self.log_values_suff += '\t'.join (featvals) + '\n'


    def log_train_hyper(self, outstream):
        for h in ['w_emb_size', 'use_pretrained_w_emb', 'l_emb_size', 'p_emb_size', 'bert_name', 'reduced_bert_size', 'lstm_h_size','mlp_arc_o_size','mlp_arc_dropout', 'mlp_lab_o_size', 'mlp_lab_dropout']:
          outstream.write("# %s : %s\n" %(h, str(self.__dict__[h])))
        outstream.write("\n")
        for h in ['graph_mode', 'batch_size', 'beta1','beta2','lr','lex_dropout', 'freeze_bert']:
          outstream.write("# %s : %s\n" %(h, str(self.__dict__[h])))
        for k in self.tasks:
          outstream.write("task %s\n" % k)          
        outstream.write("\n")

    def predict_and_evaluate(self, test_data, log_stream, out_file=None):
      """ predict on test data and evaluate 
      if out_file is set, prediction will be dumped in readable format in out_file
      """
      # TODO: tree mode

      # potentially several outputs, using different prediction modes
      task2stream = {}
      if out_file != None:
        make_alt_preds = True
        task2stream['l'] = open(out_file + '.l', 'w')
      else:
        make_alt_preds = False
          
      
      self.eval()
      test_nb_toks = 0
      test_task2nbcorrect = defaultdict(int)
      test_task2acc = defaultdict(float)
      if 'a' in self.task2i:
        test_task2nbcorrect['a'] = [0,0,0] 
      if 'l' in self.task2i:
        test_task2nbcorrect['l'] = [0,0,0] 

      with torch.no_grad():
        for batch in test_data.make_batches(self.batch_size, sort_dec_length=True):
          lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja, bols, slabseqs = batch

          # forward 
          S_arc, S_lab, log_pred_nbheads, log_pred_nbdeps, log_pred_bols, scores_slabseqs = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, lengths=lengths)

          linear_pad_mask = pad_masks[:,0,:] # from [b, m, m] to [b, m]
          test_nb_toks += linear_pad_mask.sum().item()

          #@@useless in the end
          #if 'h' in self.task2i:
          #  pred_nbheads = torch.round(torch.exp(log_pred_nbheads) - 1)

          #if 'd' in self.task2i:
          #  pred_nbdeps = torch.round(torch.exp(log_pred_nbdeps) - 1)

          # --- Prediction and evaluation --------------------------
          # provide the batch, and all the output of the forward pass
          gold_nbheads = arc_adja.sum(dim=1).float() # [b, h, d] => [b, d]
          gold_nbdeps = arc_adja.sum(dim=2).float()  # [b, h, d] => [b, h]
          linear_pad_mask = pad_masks[:,0,:] 
          task2nbcorrect, pred_arcs, pred_labels, alt_pred_arcs, task2preds = self.batch_predict_and_evaluate(batch, gold_nbheads, gold_nbdeps, linear_pad_mask,
                                                                                                              S_arc, S_lab, log_pred_nbheads, log_pred_nbdeps, log_pred_bols, scores_slabseqs, make_alt_preds=make_alt_preds)
          for k in self.tasks:
            if k in ['a','l']: 
              for i in [0,1,2]:
                test_task2nbcorrect[k][i] += task2nbcorrect[k][i]
            elif k != 'g':
              test_task2nbcorrect[k] += task2nbcorrect[k]

          # TODO : update the tree mode
                          # tree mode
                # else:
                #     # forward
                #     lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, heads, labels = batch
                #     S_arc, S_lab = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, lengths=lengths)

                #     # Predicted heads
                #     #    here simply predict head with max score, no MST, no cycle checking...
                #     pred_heads = torch.argmax(S_arc, dim=1) # [b, h, d ] ==> [b, d] = predicted head for d
            
                #     # Predicted labels for the predicted arcs (see comments in batch_forward_and_loss)
                #     num_labels = S_lab.shape[1]
                #     i_pred_heads = pred_heads.unsqueeze(1).expand(-1,num_labels,-1).unsqueeze(2) 
                #     s_labels = torch.gather(S_lab,2,i_pred_heads).squeeze(2).transpose(-2,-1)
                #     pred_labels = torch.argmax(s_labels, dim = 2)
                
                #     # evaluation
                #     nb_gold, nb_pred, nb_correct_u, nb_correct_u_and_l = self.evaluate_tree_mode(batch, pred_heads, pred_labels)

          if out_file:
            if self.graph_mode:
              self.dump_predictions_graph_mode(batch, pred_arcs, pred_labels, task2stream['l'], task2preds)
              for task in alt_pred_arcs:
                if task not in task2stream:
                  task2stream[task] = open(out_file + '.' + task, 'w')
                self.dump_predictions_graph_mode(batch, alt_pred_arcs[task], pred_labels, task2stream[task], task2preds)
            # TODO update, not working currently
            else:
                self.dump_predictions_tree_mode(batch, pred_heads, pred_labels, out_stream)

        # end loop on batches        

        print("Test: nb toks " + str(test_nb_toks) + "/ " + " / ".join([t.upper()+":"+str(test_task2nbcorrect[t]) for t in self.tasks]))              
        for k in self.tasks:
          if k in ['a', 'l']:
            test_task2acc[k] = fscore(*test_task2nbcorrect[k])
          elif k != 'g':
            test_task2acc[k] = 100 * test_task2nbcorrect[k] / test_nb_toks


        return test_task2nbcorrect, test_task2acc

# OBSOLETE        
    def evaluate_tree_mode(self, batch, pred_heads, pred_labels):
        """
        Evaluate predicted trees for a batch
        pred_heads  : [b, d] = predicted head rk (redundant with pred_labels...)
        pred_labels : [b, d] = label id 
        """
        with torch.no_grad():
            lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, heads, labels = batch

            pad_mask = (heads != PAD_HEAD_RK).float()
            nb_gold = torch.sum(pad_mask).item()
            nb_pred = nb_gold
            correct_u = (pred_heads == heads).int() * pad_mask
            nb_correct_u = torch.sum(correct_u).item()

            if pred_labels != None:
                # NB: count correct labels only for deps whose head is correctly predicted
                nb_correct_u_and_l = torch.sum((pred_labels == labels).int() * correct_u).item()
                # for debugging...
                if nb_correct_u_and_l > nb_correct_u:
                    print("BUG! correct_l %d > correct_u %d (nb_gold=%d)")
                    torch.save(S_arc, 'debug_S_arc.save')
                    torch.save(S_lab, 'debug_S_lab.save')
                    torch.save(heads, 'debug_heads.save')
                    torch.save(labels, 'debug_labels.save')
                  
            else:
                nb_correct_u_and_l = 0
        return nb_gold, nb_pred, nb_correct_u, nb_correct_u_and_l

# OBSOLETE
    def evaluate_graph_mode(self, batch, pred_arcs, pred_labels):
        """
        pred_arcs   : [b, h, d ] = 0 if padded or not predicted, 1 otherwise
        pred_labels : [b, h, d ] = label id (0 if padded)
        Caution: pred_labels contains prediction for all arcs, not only for the predicted ones
        """
        lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja = batch

        with torch.no_grad():
            # unlabeled
            nb_correct_u = torch.sum((pred_arcs * arc_adja).int()).item()
            nb_gold = torch.sum(arc_adja).item()
            nb_pred = torch.sum(pred_arcs).item()
            if pred_labels != None:
                # labeled
                # count correct labels for the predicted arcs only
                nb_correct_u_and_l = torch.sum((pred_labels == lab_adja).float() * pred_arcs).item()
            else:
                nb_correct_u_and_l = 0
        return nb_gold, nb_pred, nb_correct_u, nb_correct_u_and_l
    

# TODO UPDATE                    
    def dump_predictions_tree_mode(self, batch, pred_heads, pred_labels, out_stream):
        """ dump gold and predictions into file """
        lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, heads, labels = batch
        
        (batch_size, n) = forms.size() 

        for b in range(batch_size):     # sent in batch
            for d in range(n):          # tok in sent
                if forms[b,d] == PAD_ID:
                    break
                out = [str(d+1), self.indices.i2s('w', forms[b,d])]
                # gold head / label
                out.append(str( heads[b,d].item() + 1 ))
                out.append(self.indices.i2s('label', labels[b,d]))

                out_stream.write('\t'.join(out) + '\n')

            out_stream.write('\n')

    def dump_predictions_graph_mode(self, batch, pred_arcs, pred_labels, out_stream, task2preds=None):
        """ dump gold and predictions into file 

        task2preds : predictions for auxiliary tasks h, s, d, b
        """
        lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja, bols, slabseqs = batch

        (batch_size, n) = forms.size() 

        # whether sentences in batch start with a dummy root token or not
        root_form_id = self.indices.s2i('w', ROOT_FORM)
        if forms[0,0] == root_form_id:
            start = 1
            add = 0
        else:
            start = 0
            add = 1

        if not task2preds:
            task2preds = {}
        elif 's' in task2preds:
            nbheads_from_s, bols_from_s = self.indices.interpret_slabseqs(task2preds['s'])

        for b in range(batch_size):     # sent in batch
            for d in range(start, n):   # tok in sent (skiping root token)
                if forms[b,d] == PAD_ID:
                    break
                out = [str(d+add), self.indices.i2s('w', forms[b,d])] 
                # gold head / label pairs for dependent d
                gpairs = [ [h, self.indices.i2s('label', lab_adja[b,h,d])] for h in range(n) if lab_adja[b,h,d] != 0 ] # PAD_ID or no arc == 0
                # predicted head / label pairs for dependent d, for predicted arcs only
                ppairs = [ [h, self.indices.i2s('label', pred_labels[b,h,d])] for h in range(n) if pred_arcs[b,h,d] != 0 ]

                # marquage bruit / silence
                for pair in gpairs:
                    if pair not in ppairs:
                        pair[1] = 'SIL:' + pair[1]
                for pair in ppairs:
                    if pair not in gpairs:
                        pair[1] = 'NOI:' + pair[1]

                for pairs in [gpairs, ppairs]:
                    if len(pairs):
                        hs, ls = zip(*pairs)
                        out.append('|'.join( [ str(x+add) for x in hs ] ))
                        out.append('|'.join( ls )) #[ self.indices.i2s('label', l) for l in ls ] ))
                    else:
                        out.append('_')
                        out.append('_')
                # nb heads
                nbheads = {}
                nbheads['gold'] = len(gpairs)
                nbheads['a'] = len(ppairs)
                out.append('a:%s%d' % ( '' if nbheads['a'] == nbheads['gold'] else 'WRONG_A:' , nbheads['a']))

                # nb heads from aux task h
                if 'h' in task2preds:
                    nbheads['h'] = task2preds['h'][b,d].item()
                    out.append('h:%s%d' % ( '' if nbheads['h'] == nbheads['gold'] else 'WRONG_H:' , nbheads['h']))
                    # nb heads from aux task s
                    if 's' in task2preds:
                        nbheads['s'] = nbheads_from_s[b, d].item()
                        out.append('s:%s%d' % ( '' if nbheads['s'] == nbheads['gold'] else 'WRONG_S:' , nbheads['s']))

                # slabseq
                if 's' in task2preds:
                    ipred = task2preds['s'][b,d]
                    pred = self.indices.i2s('slabseq', ipred)
                    out.append('slabseq:%s%s' % ('' if ipred == slabseqs[b,d] else 'WRONG_SLABSEQ:', pred))
                    

                out_stream.write('\t'.join(out) + '\n')

            out_stream.write('\n')

# TODO: à tester sur une sortie a.l.s.h


            
        
