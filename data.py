"""## Data loader (before conversion to indices)"""

# lecture des données
import re

ROOT_FORM = '<root>'
ROOT_LEMM = '<root>'
ROOT_TAG = '<root>'
ROOT_LABEL = '<root>'
ROOT_RK = 0

PAD_SYMB = '*PAD*'
UNK_SYMB = '*UNK*'
DROP_SYMB = '*DROP*'
# see Indices.index_new_vocab, by construction pad id will always be 0 and UNK will always be 1, drop 2
PAD_ID = 0
UNK_ID = 1
DROP_ID = 2
PAD_HEAD_RK = -1 # for tree mode only: id for head of padded dep token
 

def load_dep_trees(gold_conll_file, corpus_type='all', split_info_file=None, val_proportion=None):
    """
        Inputs: - conll(u) file for dependency trees

                - either provide one of the following:
                    - corpus_type : smthing like 'train', 'dev' etc...
                    - or split_info_file: 
                      file with list of pairs (sentid , corpus type) (corpus types are train/dev/test)
                      (first token of each sentence should then contain a 'sentid' feature)

                    split_info_file overrides corpus_type

                - val_proportion : if set to value > 0 (and <1)
                  the training file is split into train/validation,
                  (the validation part representing the provided proportion 
                  of the original training file)

        Returns 3 dictionaries (whose keys are corpus types (train/dev/test/val))
        - sentences dictionary
          - key = corpus type
          - value = list of sentences, 
                    each sentence is a list of 5-tuples [form, lemma, tag, gov, label]                                
    """
    if split_info_file:
        # lecture du fichier donnant la répartition usuelle des phrases en train / dev / test
        s = open(split_info_file)
        lines = [ l[:-1].split('\t') for l in s.readlines() ]
        split_info_dic = { line[0]:line[1] for line in lines }

        # les phrases de dev / train / test
        sentences = {'dev':[], 'train':[], 'test':[]}
        max_sent_len = {'dev':0, 'train':0, 'test':0}
    else:
        sentences = { corpus_type:[] }
        max_sent_len = { corpus_type:0 } 


    stream = open(gold_conll_file)
    # debug: fake token root gets pad token as head (so its dep won't be counted in loss nor evaluation)
    #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, ROOT_RK, ROOT_LABEL]]
    sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, PAD_HEAD_RK, PAD_SYMB]]
    sent_rk = 0
    for line in stream.readlines():
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            if split_info_file:
                part = split_info_dic[sentid]
            else:
                part = corpus_type
            sentences[part].append(sent)
            l = len(sent)
            if max_sent_len[part] < l: 
                max_sent_len[part] = l 

            #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, ROOT_RK, ROOT_LABEL]]
            sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, PAD_HEAD_RK, PAD_SYMB]]
        else:
            cols = line.split('\t')
            # skip conllu multi-word tokens
            if '-' in cols[0]:
                continue
            form  = cols[1]
            lemma = cols[2]
            tag   = cols[4]
            gov   = int(cols[6])
            label = cols[7]
            if label == '':
                print("PROBLEM", line)
            # sentid attribute on first token
            if cols[0] == '1':
                m = re.search('sentid=([^\|=]+)', cols[5])
                if m:
                    sentid = m.group(1)
                else:
                    sentid = sent_rk
            sent.append([form, lemma, tag, gov, label])

    print("Max sentence length:", max_sent_len)
    
    # decoupage du train en train + validation
    # (pour réglage du nombre d'époques)
    if val_proportion:
        if 'train' in sentences.keys():
            (sentences['val'], sentences['train']) = split_list(sentences['train'], proportion=val_proportion)
        else:
            exit("PB val_proportion used but no training sentences")
            
    return sentences

def load_dep_graphs(gold_conll_file, corpus_type='all', split_info_file=None, use_canonical_gf=True, val_proportion=None):
    """
        Inputs: - conll(u) file with dependency graphs
                    (columns HEAD and GOV are pipe-separated values)

                - either provide one of the following:
                    - corpus_type : smthing like 'train', 'dev' etc...
                    - or split_info_file: 
                      file with list of pairs (sentid , corpus type) (corpus types are train/dev/test)
                      (first token of each sentence should then contain a 'sentid' feature)

                    split_info_file overrides corpus_type

                - use_canonical_gf: if set, canonical grammatical functions are used (final gf are discarded)

                - val_proportion : if set to value > 0 (and <1)
                  the training file is split into train/validation,
                  (the validation part representing the provided proportion 
                  of the original training file)

        Returns 3 dictionaries (whose keys are corpus types (train/dev/test/val))
        - sentences dictionary
          - key = corpus type
          - value = list of sentences, 
                    each sentence is a list of 5-tuples :
                    [form, lemma, tag, list of govs, list of labels]                                
    """
    if split_info_file:
        # lecture du fichier donnant la répartition usuelle des phrases en train / dev / test
        s = open(split_info_file)
        lines = [ l[:-1].split('\t') for l in s.readlines() ]
        split_info_dic = { line[0]:line[1] for line in lines }

        # les phrases de dev / train / test
        sentences = {'dev':[], 'train':[], 'test':[]}
        max_sent_len = {'dev':0, 'train':0, 'test':0}
    else:
        sentences = { corpus_type:[] }
        max_sent_len = { corpus_type:0 }


    stream = open(gold_conll_file)
    # debug: fake token root should get an empty list of governors (no reflexive link to itself!)
    #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [ROOT_RK], [ROOT_LABEL]]]
    sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [], []]]
    sent_rk = 0
    for line in stream.readlines():
        if line.startswith('#'):
            continue
        line = line.strip()
        if not line:
            if split_info_file:
                part = split_info_dic[sentid]
            else:
                part = corpus_type
            sentences[part].append(sent)
            l = len(sent)
            if max_sent_len[part] < l: 
                max_sent_len[part] = l 

            #sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [ROOT_RK], [ROOT_LABEL]]]
            sent = [[ROOT_FORM, ROOT_LEMM, ROOT_TAG, [], []]]
        else:
            cols = line.split('\t')
            # skip conllu multi-word tokens
            if '-' in cols[0]:
                continue
            form  = cols[1]
            lemma = cols[2]
            tag   = cols[4]
            govs   = cols[6]
            labels = cols[7]
            (govs, labels) = get_deep_govs(govs, labels, use_canonical_gf)
            if labels == '':
                print("PROBLEM", line)
            # sentid attribute on first token
            if cols[0] == '1':
                m = re.search('sentid=([^\|=]+)', cols[5])
                if m:
                    sentid = m.group(1)
                else:
                    sentid = sent_rk
            sent.append([form, lemma, tag, govs, labels])

    print("Max sentence length:", max_sent_len)
    
    # decoupage du train en train + validation
    # (pour réglage du nombre d'époques)
    if val_proportion:
        if 'train' in sentences.keys():
            (sentences['val'], sentences['train']) = split_list(sentences['train'], proportion=val_proportion)
        else:
            exit("PB val_proportion used but no training sentences")
    return sentences

def get_label(label, use_canonical_gf=True):
    if label.startswith("S:") or label.startswith("I:"):
        return ''
    if label.startswith('D:'):
        label = label[2:]
    if ':' in label:
        (label, cano) = label.split(':')
        if use_canonical_gf:
            return cano
    return label
    
def get_deep_govs(govs, labels, use_canonical_gf=True):
    """ works out the governors / labels in the deep_and_surf sequoia format
    - S: and I: arcs are discarded
    - get canonical function if use_canonical_gf is set, otherwise final functions
    
    Returns list of gov linear indices, list of corresponding labels
    
    Examples:
        input : "16|15", "S:obj.p|D:de_obj:de_obj" 
        output : [15], ["de_obj"]
       
        input : "3|6", "suj:suj|D:suj:obj" => [3,6], ['suj','obj']
    """
    govs = [int(x) for x in govs.split("|")]
    labels = [get_label(x, use_canonical_gf) for x in labels.split("|")]
    filtered = filter(lambda x: x[1], zip(govs, labels))
    f = list(zip(*filtered))
    if not(f):
        return [],[]
    return f[0],f[1]

def split_list(inlist, proportion=0.1, shuffle=False):
     """ partitions the input list of items (of any kind) into 2 lists, 
     the first one representing @proportion of the whole 
     
     If shuffle is not set, the partition takes one item every xxx items
     otherwise, the split is random"""
     n = len(inlist)
     size1 = int(n * proportion)
     if not(size1):
          size1 = 1
     print("SPLIT %d items into %d and %d" % (n, n-size1, size1))
     # if shuffle : simply shuffle and return slices
     if shuffle:
          # shuffle inlist (without changing the original external list
          # use of random.sample instead of random.shuffle
          inlist = sample(inlist, n)
          return (inlist[:size1], inlist[size1:])
     # otherwise, return validation set as one out of xxx items
     else:
          divisor = int(n / size1)
          l1 = []
          l2 = []
          for (i,x) in enumerate(inlist):
               if i % divisor or len(l1) >= size1:
                    l2.append(x)
               else:
                    l1.append(x)
          return (l1,l2)
