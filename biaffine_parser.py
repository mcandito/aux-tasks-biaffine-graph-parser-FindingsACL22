#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from tqdm import tqdm  
#from tqdm.notebook import tqdm # for progress bars in notebooks
from random import shuffle
import sys

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
    # TODO: replicate also:
    #   We used 100-dimensional pretrained GloVe em- beddings
    #   but linearly transformed them to be 125-dimensional
    def __init__(self, indices, device, 
                 w_emb_size=10, #125
                 l_emb_size=None, 
                 p_emb_size=None, # 100
                 use_pretrained_w_emb=False,
                 lstm_dropout=0.33, 
                 lstm_h_size=20, # 600
                 lstm_num_layers=3, 
                 mlp_arc_o_size=25, # 600
                 mlp_arc_dropout=0.25, 
                 mlp_lab_o_size=10, # 600
                 mlp_lab_dropout=0.33,
                 use_bias=False,
                 bert_name=None,   # caution: should match with indices.bert_tokenizer
                 reduced_bert_size=0,
                 freeze_bert=False,
    ):
        super(BiAffineParser, self).__init__()

        self.indices = indices
        self.device = device
        self.use_pretrained_w_emb = use_pretrained_w_emb

        self.bert_name = bert_name
        self.reduced_bert_size = reduced_bert_size
        self.freeze_bert = freeze_bert
        if bert_name:
            bert_model = AutoModel.from_pretrained(bert_name,return_dict=True)

        self.lexical_emb_size = w_emb_size
        w_vocab_size = indices.get_vocab_size('w')

        self.num_labels = indices.get_vocab_size('label')
        self.w_emb_size = w_emb_size
        self.p_emb_size = p_emb_size
        self.l_emb_size = l_emb_size
        self.lstm_h_size = lstm_h_size
        self.mlp_arc_o_size = mlp_arc_o_size
        self.mlp_arc_dropout = mlp_arc_dropout
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

        # ---------------------------
        # BiAffine scores
        # biaffine matrix size is num_label x d x d, with d the output size of the MLPs
        self.biaffine_arc = BiAffine(device, a, use_bias=self.use_bias)
        self.biaffine_lab = BiAffine(device, l, num_scores_per_arc=self.num_labels, use_bias=self.use_bias)
        
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
        arc_h = self.arc_h_mlp(lstm_hidden_seq)
        arc_d = self.arc_d_mlp(lstm_hidden_seq)
        lab_h = self.lab_h_mlp(lstm_hidden_seq)
        lab_d = self.lab_d_mlp(lstm_hidden_seq)

        # Biaffine scores
        S_arc = self.biaffine_arc(arc_h, arc_d) # S(k, i, j) = score of sample k, head word i, dep word j
        S_lab = self.biaffine_lab(lab_h, lab_d) # S(k, l, i, j) = score of sample k, label l, head word i, dep word j
        
        # padded cells get -inf : does not work
        #S_arc[b_pad_masks == 0] = -math.inf
        
        return S_arc, S_lab
    
    def batch_forward_and_loss(self, batch, lab_loss_weight, arc_l0=None, lab_l0=None):
        """
        - batch of sentences (output of make_batches)
        - lab_loss_weight : weight for label loss (arc loss weight is 1 - x)
        - arc_l0 and lab_l0 : arc and lab losses of first batch
          if set to non None,
          actual weight of lab_loss (resp. arc_loss) is lab_loss_weight * (lab_loss/lab_l0)
          (Liu et al. AAAI 19)

        NB: in batch graph mode
          - in arc_adja (resp. lab_adja), 0 cells are either
             - 0 cells for non gold or padded
             - 1 (resp. label id) for gold arcs
          - pad_masks : 0 cells if head or dep is padded token and 1 otherwise
        Used losses:
        for arcs:
           - tree mode  : self.ce_loss_fn_arc : cross-entropy loss (ignore_index = -1)
           - graph mode : self.bce_loss_fn_arc : binary cross-entropy
        for labels (tree and graph mode)                               
          - self.ce_loss_fn_label : cross-entropy loss with ignore_index = 0 MOCHE A CHANGER
        """
        if self.graph_mode:
            lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja = batch
            
            # forward 
            S_arc, S_lab = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, lengths=lengths)

            # --- Arc loss -------------------------
            if self.lab_loss_only:
              arc_loss = 0
            else:
              arc_loss = self.bce_loss_fn_arc(S_arc, arc_adja, pad_masks)
            
            if lab_loss_weight > 0:
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
              lab_loss = self.ce_loss_fn_label(s_labels, g_labels) 
            else:
              lab_loss = 0


            # --- Prediction --------------------------
            with torch.no_grad():
                pred_arcs = (S_arc > 0).int() * pad_masks  # b, h, d
                if lab_loss_weight > 0:
                    pred_labels = torch.argmax(S_lab, dim=1)   # for all arcs (not only the predicted arcs)
                else:
                    pred_labels = None

            # --- Evaluation --------------------------
            nb_gold, nb_pred, nb_correct_u, nb_correct_u_and_l = self.evaluate_graph_mode(batch, pred_arcs, pred_labels)
            
        # tree mode
        else:
            lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, heads, labels = batch
                    
            S_arc, S_lab = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, lengths=lengths)
            
            # --- Arc loss -------------------------
            if self.lab_loss_only:
              arc_loss = 0
            else:
              # predicted scores for arcs:
              #   rearrange into a batch containing the seq_len scores for each dependent token
              s_heads = torch.flatten(S_arc.transpose(-1,-2), start_dim=0, end_dim=1) # [b, h, d] => [b * d, h]
              g_heads = torch.flatten(heads)                                          # [b, d] = gold h => [b * d] = gold h
              arc_loss = self.ce_loss_fn_arc(s_heads, g_heads) # ignore -1 gold values (padded)

            # Predicted heads (needed for label loss: consider label scores of predicted heads)
            #    here simply predict head with max score, no MST, no cycle checking...
            pred_heads = torch.argmax(S_arc, dim=1) # [b, h, d ] ==> [b, d] = predicted head for d

            # --- Label loss -------------------------
            # Get the label scores of gold heads only
            # heads is [b, d] = gold h
            #           
            # prepare heads to serve as index for gather
            #   - unsqueeze: add dim for label
            #   - expand : repeat gold head values for all labels
            #   - unsqueeze : add dim for gold head (size 1)
            if lab_loss_weight > 0:
              num_labels = S_lab.shape[1]
              ## OBS g_heads = heads.unsqueeze(1).repeat(1,num_labels,1).unsqueeze(2)

              # Predicted label scores of PREDICTED heads (dozat et al. 2017):
              # TODO: distinguish training and inference time
              #     - at inference time : predicted label is necessarily best label of predicted head
              #     - at training time : the predicted head might be wrong,
              #                          problem to enforce a gold label that does not pertain to the predicted head
              #                          => try to use predicted label for GOLD heads at training time
              #    prepare pred_heads to serve as index for gather
              #    [b, d] = pred h => [b, 1, d] => [b, l, d] => [b, l, 1, d] = pred h
              i_pred_heads = pred_heads.unsqueeze(1).expand(-1,num_labels,-1).unsqueeze(2) 
              # gather     ==> s_labels[b, l, h, d] = S_lab[b, l, i_pred_heads[b,l,1,d], d]
              # squeeze(2) ==> a[b, l, d] score of label l for gold head of d
              # transpose(-2,-1) ==> a[b, d, l]
              s_labels = torch.gather(S_lab,2,i_pred_heads).squeeze(2).transpose(-2,-1)

              # rearrange into a batch containing the labels' scores for each (batch sample, dependent token) pair
              # [b, d, l] ==> [b * d, l] = pred score of label l for arc from pred-h-->d
              s_labels_flat = torch.flatten(s_labels, start_dim=0, end_dim=1) 

              # Gold labels of gold heads: 
              # same rearrangement from [b , d] to [b * d] = gold label
              g_labels_flat = torch.flatten(labels)
            
              lab_loss = self.ce_loss_fn_label(s_labels_flat, g_labels_flat) # cells with PAD_ID=0 gold label will be ignored

              # --- Predicted labels -------------------
              with torch.no_grad():
                  # Predicted labels for the predicted arcs:
                  #  reuse s_labels : [b, d, l] predicted label scores of predicted heads
                  pred_labels = torch.argmax(s_labels, dim = 2)

            else:
              lab_loss = 0
              pred_labels = None


            # --- Evaluation -------------------------
            nb_gold, nb_pred, nb_correct_u, nb_correct_u_and_l = self.evaluate_tree_mode(batch, pred_heads, pred_labels)

        a = 1 - lab_loss_weight
        l = lab_loss_weight
        # loss balancing : reduce weight for already well-performing task (loss a lot smaller than for first batch)
        if lab_l0 is not None:
            a = a * (arc_loss.item() / arc_l0)**self.alpha
            l = l * (lab_loss.item() / lab_l0)**self.alpha
        loss = (l * lab_loss) + (a * arc_loss)
        # returning the sub-losses too for trace purpose
        return loss, arc_loss.item(), lab_loss.item(), nb_correct_u, nb_correct_u_and_l, nb_gold, nb_pred    
    
    def train_model(self, train_data, val_data, data_name, out_model_file, log_stream, nb_epochs, batch_size, lr, lab_loss_weight, lex_dropout, alpha=0.1, nb_epochs_arc_only=0, graph_mode=True, pos_arc_weight=None):
        """
                
        For graph mode only:
        - pos_arc_weight : weight used in binary cross-entropy loss for positive examples, i.e. for gold arcs
        """
        self.graph_mode = graph_mode
        self.lr = lr
        self.lab_loss_weight = lab_loss_weight # interpolation between arc_loss and label_loss
        self.alpha = 0.1 # exponent for loss-balancing # not used currently
        self.nb_epochs = nb_epochs
        self.nb_epochs_arc_only = nb_epochs_arc_only # nb of epochs to train the arcs only (no dep labeling)
        self.lex_dropout = lex_dropout # proba of word / lemma / pos tag dropout
        self.batch_size = batch_size
        self.beta1 = 0.9
        self.beta2 = 0.9
        self.lab_loss_weight = lab_loss_weight
        #optimizer = optim.SGD(biaffineparser.parameters(), lr=LR)
        #optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0., 0.95), eps=1e-09)
        optimizer = optim.Adam(self.parameters(), lr=lr, betas=(self.beta1, self.beta2), eps=1e-09)
        # from benoit
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)
        
        # loss functions
        self.pos_arc_weight = pos_arc_weight
        # for graph mode arcs
        self.bce_loss_fn_arc = BCEWithLogitsLoss_with_mask(reduction='sum', pos_weight_scalar=pos_arc_weight)
        # for tree mode arcs and both tree and graph mode labels
        #   (CrossEnt cf. softmax not applied yet in BiAffine output)
        #   ignoring padded dep tokens (i.e. whose head equals PAD_HEAD_RK)
        self.ce_loss_fn_arc = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_HEAD_RK)
        # for label loss, the label for padded deps is PAD_ID=0 
        #   ignoring padded dep tokens (i.e. whose label id equals PAD_ID)
        self.ce_loss_fn_label = nn.CrossEntropyLoss(reduction='sum', ignore_index=PAD_ID) 
        
        #log_stream = open(out_log_file, 'w')

        self.data_name = data_name
        self.build_log_suff()
        self.log_train_hyper(sys.stdout)
        self.log_train_hyper(log_stream)
        
        # losses and scores at each epoch (on train / validation)
        train_losses = []
        train_fscores_u = []
        train_fscores_l = []
        val_losses = []
        val_arc_losses = []
        val_lab_losses = []
        val_fscores_u = []
        val_fscores_l = []
        min_val_loss = None
        max_val_perf = 0

        # word / pos / lemma dropout of training data only
        # NB: re-drop at each epoch seems ok
        #train_data.lex_dropout(lex_dropout)

        # will be set to True if arc loss increases but not lab loss
        self.lab_loss_only = False
        # losses of first batch
        arc_l0 = None
        lab_l0 = None

        for epoch in range(1,nb_epochs+1):
            train_data.lex_dropout(lex_dropout) # don't understand why detrimental here
   
            i = 0
            train_loss = 0
            train_arc_loss = 0
            train_lab_loss = 0

            # arc evaluation on train
            train_nb_correct_u = 0
            train_nb_correct_l = 0
            train_nb_pred = 0
            train_nb_gold = 0
    
            if self.nb_epochs_arc_only and epoch <= self.nb_epochs_arc_only:
                lab_loss_weight = 0
            else:
                lab_loss_weight = self.lab_loss_weight
            # training mode (certain modules behave differently in train / eval mode)
            self.train()
            bid = 0
            for batch in train_data.make_batches(self.batch_size, shuffle_data=True, sort_dec_length=True, shuffle_batches=True):        
                self.zero_grad()
                bid += 1
                if bid % 2000 == 0:
                  print("BATCH SHAPE:", batch[2].shape, batch[5].shape)
                  print("MEMORY BEFORE BATCH FORWARD AND LOSS")
                  printm()

                loss, arc_loss, lab_loss, nb_correct_u, nb_correct_l, nb_gold, nb_pred = self.batch_forward_and_loss(batch, lab_loss_weight, arc_l0=None, lab_l0=None)
                train_loss += loss.item()
                if arc_loss:
                    train_arc_loss += arc_loss
                    if arc_l0 == None:
                      arc_l0 = arc_loss
                if lab_loss:
                    train_lab_loss += lab_loss
                    if lab_l0 == None:
                      lab_l0 = lab_loss
                
                loss.backward()
                optimizer.step() 
                loss.detach()           

                #predictions, nb_correct, nb_gold, nb_pred = self.predict_from_scores_and_evaluate(S_arc, S_lab, batch)    
                train_nb_correct_u += nb_correct_u
                train_nb_correct_l += nb_correct_l
                train_nb_gold += nb_gold
                train_nb_pred += nb_pred

            print(train_nb_correct_u, train_nb_correct_l, train_nb_gold, train_nb_pred)
            train_fscores_u.append( fscore(train_nb_correct_u, train_nb_gold, train_nb_pred) )            
            train_fscores_l.append( fscore(train_nb_correct_l, train_nb_gold, train_nb_pred) )
            train_loss = train_loss/train_data.size
            train_arc_loss = train_arc_loss/train_data.size
            train_lab_loss = train_lab_loss/train_data.size
            train_losses.append(train_loss)

            self.log_perf(log_stream, epoch, 'Train', train_loss, train_arc_loss, train_lab_loss, 
                          train_fscores_u[-1], train_fscores_l[-1])

            if val_data:
                self.eval()
                # arc evaluation on validation
                val_nb_correct_u = 0
                val_nb_correct_l = 0
                val_nb_pred = 0
                val_nb_gold = 0

                # calcul de la perte sur le validation set
                with torch.no_grad():
                    val_loss = 0
                    val_arc_loss = 0
                    val_lab_loss = 0
                    for batch in val_data.make_batches(self.batch_size, sort_dec_length=True):
                        loss, arc_loss, lab_loss, nb_correct_u, nb_correct_l, nb_gold, nb_pred = self.batch_forward_and_loss(batch, lab_loss_weight)
                        val_loss += loss.item()
                        if arc_loss:
                            val_arc_loss += arc_loss
                        if lab_loss:
                            val_lab_loss += lab_loss
                        #predictions, nb_correct, nb_gold, nb_pred = self.predict_from_scores_and_evaluate(S_arc, S_lab, batch)    
                        
                        val_nb_correct_u += nb_correct_u
                        val_nb_correct_l += nb_correct_l
                        val_nb_gold += nb_gold
                        val_nb_pred += nb_pred
                        
                    print(val_nb_correct_u, val_nb_correct_l, val_nb_gold, val_nb_pred)
                    val_fscores_u.append( fscore(val_nb_correct_u, val_nb_gold, val_nb_pred) )            
                    val_fscores_l.append( fscore(val_nb_correct_l, val_nb_gold, val_nb_pred) )
                    val_loss = val_loss / val_data.size
                    val_arc_loss = val_arc_loss / val_data.size
                    val_lab_loss = val_lab_loss / val_data.size
                    val_losses.append(val_loss)
                    val_lab_losses.append(val_lab_loss)

                self.log_perf(log_stream, epoch, 'Valid', val_loss, val_arc_loss, val_lab_loss, val_fscores_u[-1], val_fscores_l[-1])
    
                if epoch == 1:
                    #min_val_loss = val_loss
                    max_val_perf = val_fscores_l[-1]
                    print("saving model after first epoch\n")
                    torch.save(self, out_model_file)
                # if validation loss has decreased: save model
                # nb: when label loss comes into play, it might artificially increase the overall loss
                #     => we don't early stop at this stage 
                #elif (val_losses[-1] < val_losses[-2]) or (epoch == self.nb_epochs_arc_only) :
                elif (val_fscores_l[-1] >= val_fscores_l[-2]) or (epoch == self.nb_epochs_arc_only) :
                    for stream in [sys.stdout, log_stream]:
                        stream.write("Validation perf has increased, saving model, current nb epochs = %d\n" % epoch)
                    torch.save(self, out_model_file)
                # if overall loss increase, but still a decrease for the lab loss
                # desactivated because detrimental
                #elif (val_lab_losses[-1] < val_lab_losses[-2]):
                #    for stream in [sys.stdout, log_stream]:
                #        stream.write("Label Validation loss has decreased, saving model, current nb epochs = %d\n" % epoch)
                #    torch.save(self, out_model_file)
                #    self.lab_loss_only = True
                # otherwise: early stopping, stop training, reload previous model
                # NB: the model at last epoch was not saved yet
                # => we can reload the model from the previous storage
                else:
                    print("Validation perf has decreased, reloading previous model, and stop training\n")
                    self.log_best_perf(log_stream, epoch - 1, val_fscores_u[-2], val_fscores_l[-2])
                    # reload (on the appropriate device)
                    # cf. https://pytorch.org/docs/stable/generated/torch.load.html#torch-load
                    self = torch.load(out_model_file)
                    # stop loop on epochs
                    break
                
            scheduler.step()
        # if no early stopping
        else:
            print("Max nb epochs reached\n")
            self.log_best_perf(log_stream, epoch , val_fscores_u[-1], val_fscores_l[-1])

        for stream in [sys.stdout, log_stream]:
          stream.write("train losses: %s\n" % ' / '.join([ "%.2f" % x for x in train_losses]))
          stream.write("val   losses: %s\n" % ' / '.join([ "%.2f" % x for x in val_losses]))
          stream.write("train unlab fscores: %s\n" % ' / '.join([ "%.4f" % x for x in train_fscores_u]))
          stream.write("val   unlab fscores: %s\n" % ' / '.join([ "%.4f" % x for x in val_fscores_u]))
          stream.write("train   lab fscores: %s\n" % ' / '.join([ "%.4f" % x for x in train_fscores_l]))
          stream.write("val     lab fscores: %s\n" % ' / '.join([ "%.4f" % x for x in val_fscores_l]))

    def log_perf(self, outstream, epoch, ctype, l, arc_l, lab_l, f_u, f_l):
        for stream in [sys.stdout, outstream]:
          stream.write("Loss / arc loss / lab loss for epoch %2d on %s: %12.2f / %12.2f / %12.2f\n" % (epoch, ctype, l, arc_l, lab_l))
          stream.write("            Fscore U / L after epoch %2d on %s :     U %5.2f / L %5.2f\n" % (epoch, ctype, f_u, f_l))

    def log_best_perf(self, outstream, epoch, f_u, f_l):
        # see build_log_suff for the headings
        outstream.write(self.log_heading_suff)
        
        s = '\t'.join( [ 'RESULT', "%5.2f" % f_u, "%5.2f" % f_l, str(epoch) ] ) + '\t' + self.log_values_suff
        outstream.write(s)
        
          
    def build_log_suff(self):

        self.log_heading_suff = '\t'.join([ 'RESULT', 'val UF', 'val LF', 'effective nb epochs', 'g or t' ] )
        if self.graph_mode:
            self.log_values_suff = 'graph\t'
        else:
            self.log_values_suff = 'tree\t' 
        l = ['data_name', 'w_emb_size', 'use_pretrained_w_emb', 'l_emb_size', 'p_emb_size', 'bert_name', 'reduced_bert_size', 'freeze_bert', 'lstm_h_size','mlp_arc_o_size','mlp_arc_dropout', 'batch_size', 'beta1','beta2','lr', 'nb_epochs', 'nb_epochs_arc_only', 'lab_loss_weight', 'lex_dropout', 'pos_arc_weight']

        l_strs = [ str(self.__dict__[f]) for f in l ]
        config_str = '_'.join(l_strs) # get a compact name for the hyperparameter config
        l_strs = [config_str] + l_strs
        l = ['config_str'] + l

        self.log_heading_suff += '\t' + '\t'.join( l ) + '\n'
        self.log_values_suff += '\t'.join (l_strs) + '\n'


    def log_train_hyper(self, outstream):
        for h in ['w_emb_size', 'use_pretrained_w_emb', 'l_emb_size', 'p_emb_size', 'bert_name', 'reduced_bert_size', 'freeze_bert', 'lstm_h_size','mlp_arc_o_size','mlp_arc_dropout']:
          outstream.write("# %s : %s\n" %(h, str(self.__dict__[h])))
        outstream.write("\n")
        for h in ['graph_mode', 'batch_size', 'beta1','beta2','lr','lab_loss_weight', 'nb_epochs_arc_only', 'lex_dropout']:
          outstream.write("# %s : %s\n" %(h, str(self.__dict__[h])))
        if self.graph_mode:
          outstream.write("# pos_arc_weight : %s\n" %(str(self.pos_arc_weight)))
          
        outstream.write("\n")

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
    

    def predict_and_evaluate(self, graph_mode, test_data, log_stream, out_file=None):
        """ predict on test data and evaluate 
        if out_file is set, prediction will be dumped in readable format in out_file
        """
        
        tot_nb_correct_u = 0
        tot_nb_correct_l = 0
        tot_nb_pred = 0
        tot_nb_gold = 0

        if out_file != None:
            out_stream = open(out_file, 'w')
            
        self.eval()
        with torch.no_grad():
            for batch in test_data.make_batches(self.batch_size, sort_dec_length=True):
                if graph_mode:
                    # forward 
                    lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja = batch
                    S_arc, S_lab = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, lengths=lengths)

                    # prediction
                    pred_arcs = (S_arc > 0).int() * pad_masks  # b, h, d
                    pred_labels = torch.argmax(S_lab, dim=1) # for all arcs (not only the predicted arcs)

                    # evaluation
                    nb_gold, nb_pred, nb_correct_u, nb_correct_u_and_l = self.evaluate_graph_mode(batch, pred_arcs, pred_labels)
                # tree mode
                else:
                    # forward
                    lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, heads, labels = batch
                    S_arc, S_lab = self(forms, lemmas, tags, bert_tokens, bert_ftid_rkss, pad_masks, lengths=lengths)

                    # Predicted heads
                    #    here simply predict head with max score, no MST, no cycle checking...
                    pred_heads = torch.argmax(S_arc, dim=1) # [b, h, d ] ==> [b, d] = predicted head for d
            
                    # Predicted labels for the predicted arcs (see comments in batch_forward_and_loss)
                    num_labels = S_lab.shape[1]
                    i_pred_heads = pred_heads.unsqueeze(1).expand(-1,num_labels,-1).unsqueeze(2) 
                    s_labels = torch.gather(S_lab,2,i_pred_heads).squeeze(2).transpose(-2,-1)
                    pred_labels = torch.argmax(s_labels, dim = 2)
                
                    # evaluation
                    nb_gold, nb_pred, nb_correct_u, nb_correct_u_and_l = self.evaluate_tree_mode(batch, pred_heads, pred_labels)
                    
                tot_nb_correct_u += nb_correct_u
                tot_nb_correct_l += nb_correct_u_and_l
                tot_nb_pred += nb_pred
                tot_nb_gold += nb_gold

                if out_file:
                    if graph_mode:
                        self.dump_predictions_graph_mode(batch, pred_arcs, pred_labels, out_stream)
                    else:
                        self.dump_predictions_tree_mode(batch, pred_heads, pred_labels, out_stream)
                        

        if out_stream:
            out_stream.close()
        
        return tot_nb_gold, tot_nb_pred, tot_nb_correct_u, tot_nb_correct_l

                    
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

    def dump_predictions_graph_mode(self, batch, pred_arcs, pred_labels, out_stream):
        """ dump gold and predictions into file """
        lengths, pad_masks, forms, lemmas, tags, bert_tokens, bert_ftid_rkss, arc_adja, lab_adja = batch

        (batch_size, n) = forms.size() 

        # whether sentences in batch start with a dummy root token or not
        root_form_id = self.indices.s2i('w', ROOT_FORM)
        if forms[0,0] == root_form_id:
            start = 1
            add = 0
        else:
            start = 0
            add = 1
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
                    
                out_stream.write('\t'.join(out) + '\n')

            out_stream.write('\n')


            
        
