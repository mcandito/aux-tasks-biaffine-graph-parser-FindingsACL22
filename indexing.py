#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""## Indices"""

from data import *
import numpy as np
import torch
from random import random
import copy
from collections import defaultdict

class Indices:
   
    def __init__(self, known_sentences, w_emb_file=None, l_emb_file=None, bert_tokenizer=None):
        """
        Input:
        known_sentences : list of sentences, 
                          one sentence = list of 5-tuples
                          Each sent has a dummy <root> first token
                          (as output by load_dep_trees_file or load_dep_graphs_file)
                          
        Records correspondance between symbols and indices
        for each type of vocabulary (word form, lemma, pos tag, dependency label)
        """
        self.sentences = known_sentences
        
        self.vocabs = {} # the various vocabs key=vocab type, val=dic i2s and s2i
        
        # compute indices on known sentences 
        #         one tok = a 5-tuple [form, lemma, tag, gov(s), label(s)]
        #     or a 6-tuple            [form, lemma, tag, govs, labels, slabseq]
        train_tokens = [tok for sent in known_sentences for tok in sent]

        if len(train_tokens[0]) == 5:
          (forms, lemmas, tags, heads, labels) = list(zip(*train_tokens))
          add_slabseqs = False
        else:
          (forms, lemmas, tags, heads, labels, slabseqs) = list(zip(*train_tokens))
          add_slabseqs = True
        
        self.bert_tokenizer = bert_tokenizer

        self.emb_size = {} # vocab type to size
        self.i2emb = {}    # key = vocab type, value = list from id to string
        self.emb_matrix = {} # key = vocab type, value = the loaded/random embedding tensor

        # indices for the various vocabularies of symbols
        
        if w_emb_file is not None:
            # add indices for special symbols only: calling index_new_vocab with an empty sequence of tokens
            self.index_new_vocab('w', [], add_pad=True, add_unk=True)
            # and then for known embeddings + additional known forms
            self.load_embeddings_from_scratch('w', w_emb_file, additional_forms=set(forms))
        else:
            self.index_new_vocab('w', forms, add_pad=True, add_unk=True)
            self.i2emb['w'] = None
            self.emb_size['w'] = 0


        if l_emb_file is not None:
            # add indices for special symbols only: calling index_new_vocab with an empty sequence of tokens
            self.index_new_vocab('l', [], add_pad=True, add_unk=True)
            # and then for known embeddings + additional known forms
            self.load_embeddings_from_scratch('l', l_emb_file, additional_forms=set(lemmas))
        else:
            self.index_new_vocab('l', lemmas, add_pad=True, add_unk=True)
            self.i2emb['l'] = None
            self.emb_size['l'] = 0
        
        self.index_new_vocab('p', tags, add_pad=True, add_unk=True)
        # heads are already integers (ranks in sequence), padded dep tokens will get -1 as head
        # NB: important to define the true label ids distinct from the padding label id (==0)
        #     unk useless for labels
        self.index_new_vocab('label', labels, add_pad=True, add_unk=False)

        # indices for the sorted lab sequences (seen as atoms)
        if add_slabseqs:
          # too many symbols, keep only the first 30
          slabseq2occ = defaultdict(int)
          for s in slabseqs:
            slabseq2occ[s] +=1
          known_slabseqs = sorted(slabseq2occ.keys(), key=lambda x: slabseq2occ[x], reverse=True)
          # keep only the 30 most freq (others will get the UNK_ID)
          i2s = [PAD_SYMB, UNK_SYMB] + known_slabseqs[:50]
          # symbols to indices
          s2i = {x:i for i,x in enumerate(i2s)}    
          self.vocabs['slabseq'] = {'i2s': i2s, 's2i': s2i}
        

    def index_new_vocab(self, vocab, symbol_seq, add_pad=True, add_unk=True):
        """
        Input:
        - vocab is the name of the vocabulary
        - symbol_seq is a sequence of symbols in this vocabulary
        - 
        NB: add_unk pertains iff add_pad is True (unkable are necessarily padable objects)
        
        By construction, 
         - if a pad symbol is inserted, it gets id 0
         - if a unk symbol is inserted it gets id 1
        """
        
        # flatten symbol list if this is a list of tuples (occurs when dep_graphs were read)
        if len(symbol_seq) and (isinstance(symbol_seq[0], list) or isinstance(symbol_seq[0],tuple)):
            symbol_seq = [ x for symb_tuple in symbol_seq for x in symb_tuple ]
                                    
        # index to symbols
        i2s = list(set(symbol_seq))
        
        # special symbols
        # NB: by construction pad id SHOULD always be 0
        #     and if existing, unk is 1 and drop is 2
        if add_pad: 
            if add_unk: # add distinct symbols for unk and drop
                i2s = [PAD_SYMB, UNK_SYMB, DROP_SYMB] + i2s
            else:
                i2s = [PAD_SYMB] + i2s

        # symbols to indices
        s2i = {x:i for i,x in enumerate(i2s)}
    
        self.vocabs[vocab] = {'i2s': i2s, 's2i': s2i}
        

    def s2i(self, vocab, s):
        """ Returns the index of symbol s in vocabulary vocab 
        If s in unknown, return the index of unk
        """
        if s not in self.vocabs[vocab]['s2i']:
            return UNK_ID #self.iunk[vocab]
        return self.vocabs[vocab]['s2i'][s]

    def i2s(self, vocab, i):
        return self.vocabs[vocab]['i2s'][i]
    
    def get_vocab_size(self, vocab_type):
        return len(self.vocabs[vocab_type]['i2s'])

    def interpret_slabseqs(self, slabseqs):
      """ From a tensor of sorted sequences of labels to 
         - corresponding nbheads
         - corresponding bag of labels ("bol"s)

      Since the unk slabseq is not interpretable
      - the nb of heads is set to -1 
      - bol is null vector
      
      Input: tensor of shape *
      Output:
        - nbheads (shape *)
        - bols    (shape *, num_labels)
      """
      num_labels = self.get_vocab_size('label')
      flat_slabseqs = slabseqs.view(-1)
      nbtot = flat_slabseqs.shape[0] # total nb of slabseq in input tensor
      flat_bols = torch.zeros(nbtot, num_labels, dtype=torch.int32) #@@ +1 # bag of labels, +1 for NOLABEL
      flat_nbheads = torch.zeros(nbtot, dtype=torch.int32)
      for i in range(nbtot):
        islabseq = flat_slabseqs[i]
        if islabseq == UNK_ID: # if unknown slabseq => cannot interpret the nb of heads
          flat_nbheads[i] = -1
        else:
          slabseq = self.i2s('slabseq', islabseq)
          # if label is not '' (i.e. if the dependent has at least one governor)
          if slabseq:
            ilabels = [ self.s2i('label', label) for label in slabseq.split('|') ]
            for ilab in ilabels:
                flat_bols[i,ilab] += 1
                flat_nbheads[i] += 1
      # reshaping to input shape
      input_size = slabseqs.size()
      return flat_nbheads.view(input_size), flat_bols.view(list(input_size) + [num_labels])

    def convert_tree_tok_to_indices(self, tok):
        """ 
        Input = one token from dep tree (= a 5-tuple)
        Output: same but converted to indices for each vocab type
        """
        return [self.s2i('w', tok[0]),
                self.s2i('l', tok[1]),
                self.s2i('p', tok[2]),
                tok[3], # heads are integers already
                self.s2i('label', tok[4])]
                
    def convert_graph_tok_to_indices(self, tok):
        """ 
        Input = one token from dep graph (= a 6-tuple, with list of labels / heads)
        Output: same but converted to indices for each vocab type
        """
        return [self.s2i('w', tok[0]),
                self.s2i('l', tok[1]),
                self.s2i('p', tok[2]),
                tok[3], # heads are integers already
                [ self.s2i('label', x) for x in tok[4]],
                self.s2i('slabseq', tok[5])
        ]

    def convert_tree_symbols_to_indices(self, sentences):
        return [ [self.convert_tree_tok_to_indices(tok) for tok in sent] for sent in sentences ]

    def convert_graph_symbols_to_indices(self, sentences):
        return [ [self.convert_graph_tok_to_indices(tok) for tok in sent] for sent in sentences ]

    def lex_dropout_itok(self, itok, dropout_rate, drop_to_unk_rate=0.01):
      """
      Input: 
        - a token already converted into indices (5-tuple w, l, p, heads(s), labels(s))
        - a dropout rate
      Output: the token, with ids potentially replaced by IUNK  
      """
      # independent dropout of w / l / p
      #@@ DEBUG
      nitok = copy.deepcopy(itok)
      r = False
      for i in [0,1,2]:
        # drop to unk to learn the unk w / l / p embeddings
        #if random() < drop_to_unk_rate:
        #  itok[i] = UNK_ID
        # dozat 2008 dropped embeddings were replaced with learnt dropped tokens
        #elif random() < dropout_rate:
        if random() < dropout_rate:
          nitok[i] = DROP_ID #UNK_ID
          r = True
      if r:
          return nitok
      return itok

    def lex_dropout_isentences(self, isentences, dropout_rate):
      """
      Input: isentences (list of lists of itok, as output by convert_tree/graph_symbols_to_indices)
      Output: same but with lexical dropout
      """
      return [ [ self.lex_dropout_itok(itok, dropout_rate) for itok in sent ] for sent in isentences ]
     
    def load_embeddings_from_scratch(self, vocab, embeddings_file, additional_forms=None):
        """
        Loads txt file containing lexical (non-contextual) embeddings 
        @param embeddings_file: vectors associated to words (or strings)
        First line contains : nb_words w_emb_size
        
        - fills self.i2emb[vocab] list from id to its pretrained embedding
        - sets self.emb_size[vocab]
    
        """
        instream = open(embeddings_file)
        # reading nb_words and w embedding size from first line
        line = instream.readline()
        line = line[:-1]
        (nb_words, emb_size) = [ int(x) for x in line.split(' ') ]
        self.emb_size[vocab] = emb_size
        self.i2emb[vocab] = []
        
        # NB: when calling load_embeddings_from_scratch,
        # the indices contain the special symbols only, if any (*PAD*, *UNK*, *DROP*)
        for s in self.vocabs[vocab]['i2s']:
            if s == UNK_SYMB or s == DROP_SYMB:
                # random vector for unk token and for drop token(between a=-1 and b=1 : (b-a)*sample + a)
                # rem: apparently for drop token, more stable to learn from a random vector than from a null vec
                self.i2emb[vocab].append( 2 * np.random.random(emb_size) - 1 )
            elif s == PAD_SYMB:
                # null vector for pad token
                self.i2emb[vocab].append(np.zeros(self.emb_size[vocab]))
        
        line = instream.readline()
        i = len(self.vocabs[vocab]['i2s']) - 1
        while line:
            i += 1
            line = line[:-1].strip() # trailing space
            cols = line.split(" ")
            w = cols[0]
            vect = [float(x) for x in cols[1:]]
            self.vocabs[vocab]['s2i'][w] = i
            self.vocabs[vocab]['i2s'].append(w)
            self.i2emb[vocab].append(vect)
            
            line = instream.readline()

        # if additional sentences were provided
        if additional_forms:
            # get the forms that have no pretrained embedding
            additional_forms = list(additional_forms.difference(self.vocabs[vocab]['s2i'].keys()))
            last = len(self.vocabs[vocab]['i2s']) # current size of vocab
            # already defined
            #emb_size = len(self.i2emb[vocab][-1])
            self.vocabs[vocab]['i2s'] += additional_forms
            for i, form in enumerate(additional_forms):
                self.vocabs[vocab]['s2i'][form] = last + i
                # random vector between -1 and 1  (b-a)*sample + a
                self.i2emb[vocab].append( 2 * np.random.random(emb_size) - 1 )

            
        self.emb_matrix[vocab] = torch.tensor(self.i2emb[vocab]).float()
        print("Pretrained %s embeddings shape: %s" % (vocab, str(self.emb_matrix[vocab].shape)))
        
        
    # *BERT tokenization (subwords) and correspondance between 
    # - word rank in a word sequence 
    # - ranks of corresponding tokens in the token sequence output by *BERT tokenization
    def bert_encode(self, sentences):
        """ Input: 
            - a list of sentences (list of tokens, 1 tok = a 5-tuple)
              (each has a <root> dummy first symbol)
            Output:
            - bert tokenization : list of bert-token id sequences 
            - list of list : for each sent s, 
                                 for each word-token w : 
                                    rank, in the *BERT tokenization of s,
                                    of the first bert-token of w
                                    
        """
        # pour chaque mot dans sentence, tokenisé en t1 t2 ... tn, 
        # on calcule les rangs qu'auraient le premier et le dernier token, t1 et tn
        # dans la tokenisation complète de la phrase
    
        tid_seqs = []
        first_tok_rankss = []

        tkz = self.bert_tokenizer

        if tkz.bos_token_id != None:
            bos_token_id = tkz.bos_token_id # flaubert
        else:
            bos_token_id = tkz.cls_token_id # bert
        
        for sent in sentences:
            (forms, lemmas, tags, heads, labels, _) = list(zip(*sent))
            tid_seq = [ bos_token_id ]
            first_tok_ranks = [ 0 ] # rank of bos # will be used for bert embedding of <root> token
            start = 1
            # removing <root> token
            forms = forms[1:]
            for word in forms:
              # tokenization of a single word 
              tid_word = tkz.encode(word, add_special_tokens=False)
              end = start + len(tid_word) - 1 
              first_tok_ranks.append(start)
              tid_seq.extend(tid_word)
              start = end + 1
            # end of sentence symbol
            tid_seq.append( tkz.sep_token_id )  # eos_token_id not defined in FlauBERT tkz !
            # first_tok_ranks.append(start)

            first_tok_rankss.append(first_tok_ranks)
            tid_seqs.append(tid_seq)
        
        return tid_seqs, first_tok_rankss

"""## Test of loading / encoding trees into indices"""

if False:
  # les informations pour le split train / dev / test
  # tel qu'utilisé généralement pour ce corpus
  split_info_file = './deep_french_dep/sequoiaftb_split_info'

  # dep trees
  gold_conll_file = './deep_french_dep/surf.sequoia'
  sentences = load_dep_trees(gold_conll_file, split_info_file)

  # indices defined on training sentences only
  # bert tok, no pretrained embeddings
  indices = Indices(sentences['train'], bert_tokenizer=bert_tokenizer)

  print(indices.vocabs['p']['s2i'])
  print(indices.vocabs['label']['s2i'])

  print(indices.s2i('label', 'suj'))

  for i in [0,1,2]:
      print(indices.vocabs['w']['i2s'][i])
