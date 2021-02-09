#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""## Indices"""

from data import *
import numpy as np
import torch
from random import random

class Indices:
   
    def __init__(self, known_sentences, w_emb_file=None, bert_tokenizer=None):
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
        self.iunk = {}   # indices of unk for the various vocab types
        
        # compute indices on known sentences 
        #         one tok = a 5-tuple [form, lemma, tag, gov(s), label(s)]
        train_tokens = [tok for sent in known_sentences for tok in sent]

        (forms, lemmas, tags, heads, labels) = list(zip(*train_tokens))
        
        self.bert_tokenizer = bert_tokenizer

        if w_emb_file is not None:
            # add indices for special symbols only: calling index_new_vocab with an empty sequence of tokens
            self.index_new_vocab('w', [], add_pad=True, add_unk=True)
            # and then for known embeddings
            self.load_embeddings_from_scratch(w_emb_file)
        else:
            self.index_new_vocab('w', forms, add_pad=True, add_unk=True)
            self.iw2emb = None
            self.w_emb_size = 0

        # indices for the various vocabularies of symbols
        self.index_new_vocab('l', lemmas, add_pad=True, add_unk=True)
        self.index_new_vocab('p', tags, add_pad=True, add_unk=True)
        # heads are already integers (ranks in sequence), padded dep tokens will get -1 as head
        # NB: important to define the true label ids distinct from the padding label id (==0)
        #     unk useless for labels
        self.index_new_vocab('label', labels, add_pad=True, add_unk=False)
        

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
        self.iunk[vocab] = None
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
        Input = one token from dep graph (= a 5-tuple, with list of labels / heads)
        Output: same but converted to indices for each vocab type
        """
        return [self.s2i('w', tok[0]),
                self.s2i('l', tok[1]),
                self.s2i('p', tok[2]),
                tok[3], # heads are integers already
                [ self.s2i('label', x) for x in tok[4]]]

    def convert_tree_symbols_to_indices(self, sentences):
        return [ [self.convert_tree_tok_to_indices(tok) for tok in sent] for sent in sentences ]

    def convert_graph_symbols_to_indices(self, sentences):
        return [ [self.convert_graph_tok_to_indices(tok) for tok in sent] for sent in sentences ]

    def lex_dropout_itok(self, itok, dropout_rate):
      """
      Input: 
        - a token already converted into indices (5-tuple w, l, p, heads(s), labels(s))
        - a dropout rate
      Output: the token, with ids potentially replaced by IUNK  
      """
      # independent dropout of w / l / p
      for i in [0,1,2]:
        if random() < dropout_rate:
          itok[i] = DROP_ID #UNK_ID
      return itok

    def lex_dropout_isentences(self, isentences, dropout_rate):
      """
      Input: isentences (list of lists of itok, as output by convert_tree/graph_symbols_to_indices)
      Output: same but with lexical dropout
      """
      return [ [ self.lex_dropout_itok(itok, dropout_rate) for itok in sent ] for sent in isentences ]
     
    def load_embeddings_from_scratch(self, embeddings_file):
        """
        Loads txt file containing lexical (non-contextual) embeddings 
        @param embeddings_file: vectors associated to words (or strings)
        First line contains : nb_words w_emb_size
        
        - fills self.iw2emb list frow word id to its pretrained embedding
        - sets self.w_emb_size
    
        """
        instream = open(embeddings_file)
        iw2emb = []
        # reading nb_words and w embedding size from first line
        line = instream.readline()
        line = line[:-1]
        (nb_words, w_emb_size) = [ int(x) for x in line.split(' ') ]
        self.w_emb_size = w_emb_size
        self.iw2emb = []
        
        # NB: when calling load_embeddings_from_scratch,
        # the indices contain the special symbols only, if any (*PAD*, *UNK*, *DROP*)
        for s in self.vocabs['w']['i2s']:
            if s == UNK_SYMB or s == DROP_SYMB:
                # random vector for unk token and for drop token(between a=-1 and b=1 : (b-a)*sample + a)
                # rem: apparently for drop token, better to have a random vector than a null vec
                self.iw2emb.append( 2 * np.random.random(self.w_emb_size) - 1 )
            elif s == PAD_SYMB:
                # null vector for pad token
                self.iw2emb.append(np.zeros(self.w_emb_size))
        
        line = instream.readline()
        i = len(self.vocabs['w']['i2s'])
        while line:
            i += 1
            line = line[:-1].strip() # trailing space
            cols = line.split(" ")
            w = cols[0]
            vect = [float(x) for x in cols[1:]]
            self.vocabs['w']['s2i'][w] = i
            self.vocabs['w']['i2s'].append(w)
            self.iw2emb.append(vect)
            
            line = instream.readline()
            
        self.w_emb_matrix = torch.tensor(self.iw2emb).float()
        print("Pretrained word embeddings shape:", self.w_emb_matrix.shape)
        
        
    # *BERT tokenization (subwords) and correspondance between 
    # - word rank in a word sequence 
    # - ranks of corresponding tokens in the token sequence output by *BERT tokenization
    def bert_encode(self, sentences):
        """ Input: 
            - a list of sentences (list of tokens, 1 tok = a 5-tuple)
              (each has a <root> dummy first symbol)
            Output:
            - bert tokenization : list of token id sequences 
              (WITHOUT the dummy root)
            - list of list of token ranks for the first token of each word
                in the *BERT tokenization of each sequence
        """
        # pour chaque mot dans sentence, tokenisé en t1 t2 ... tn, 
        # on calcule les rangs qu'auraient le premier et le dernier token, t1 et tn
        # dans la tokenisation complète de la phrase
    
        tid_seqs = []
        first_tok_rankss = []

        tkz = self.bert_tokenizer
        
        for sent in sentences:
            (forms, lemmas, tags, heads, labels) = list(zip(*sent))
            tid_seq = [ tkz.bos_token_id ]
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
