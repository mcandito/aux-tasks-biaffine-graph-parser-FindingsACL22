#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Marie Candito

import argparse
import os
import sys
from collections import defaultdict
import numpy as np
import math

def mean_mad(serie):
    n = len(serie)
    if n == 0:
        return 0, 0
    mean = sum(serie) / n
    # mean absolute deviation 
    mean_abs_dev = sum( [ abs(x - mean) for x in serie ] ) / n
    return mean, mean_abs_dev

def squared_loss_of_logs(g, p):
    return (math.log(1+g) - math.log(1+p))**2

NOLABEL = '**NOLABEL**'

def dict_like_bag_of_labels(label_seq):
    if len(label_seq) == 0:
        return { NOLABEL:1 } # add special label NOLABEL if no label...
    bol = {} 
    for l in label_seq:
        if l in bol:
            bol[l] += 1
        else:
            bol[l] = 1
    return bol

def dict_norm(dvec):
    return (sum([ x**2 for x in dvec.values() ]))**0.5

def dict_cosinus(dvec1, dvec2):
    cos = 0
    for k in dvec1:
        if k in dvec2:
            cos += dvec1[k] * dvec2[k]
    return cos / (dict_norm(dvec1) * dict_norm(dvec2))

# whether 2 dictionary-vectors are equal (for equality of BOL vectors)
def dict_equality(dvec1, dvec2):
    for k in dvec1:
        if k not in dvec2:
            return 0
        if dvec1[k] != dvec2[k]:
            return 0
    return 1

def get_binnednb(nb):
    if nb == 0:
        return 0
    if nb == 1:
        return 1
    return 2

usage = """ eval_dep.py gold and sys file """

# read arguments
argparser = argparse.ArgumentParser(usage = usage)
argparser.add_argument('gold_and_pred_dep_graphs_file', help='Tabulated file with gold and predicted dep graphs', default=None)
argparser.add_argument('--split_info_file', help='split info file (each line = sentence id, tab, corpus type (train, dev, test)', default=None)
argparser.add_argument('--data_name', help='short name of data: ftb or sequoia etc... Default=ftb', default='ftb')
argparser.add_argument('-g', '--graph_mode', action="store_true", help='If set, Graph version of the parser, otherwise Tree version. Default=True', default=True)
argparser.add_argument('-p', '--w_emb_file', help='If not "None", pre-trained word embeddings file. NB: first line should contain nbwords embedding size', default='None')
argparser.add_argument('-t', '--trace', action="store_true", help='print some traces. Default=False', default=False)
args = argparser.parse_args()


l2dist = { 'g':defaultdict(list), 'p':defaultdict(list) }
l2direction = { 'g':defaultdict(list), 'p':defaultdict(list) }

# nb occs of each sequence of labels
lseqs2occ = { 'g':defaultdict(int), 'p':defaultdict(int) }
# same but for sorted sequences of labels
slseqs2occ = { 'g':defaultdict(int), 'p':defaultdict(int) }

# bag of labels
bols = {'g':[], 'p':[]}

# nb of heads gold/pred
nbheads = {'g':[], 'p':[]}
nbdeps  = {'g':[], 'p':[]}
binnednbheads = {'g':[], 'p':[]}

# total nb gold, nb pred, nb correct unlab, nb correct labeled
all_g_p_cu_cl = [0,0,0,0]
# same, but for long sentences
long_g_p_cu_cl = [0,0,0,0]

# total nb toks in all sentences
nb_toks = 0
# nb of toks for which gold and predicted nbheads is same
nb_correct_nbheads = 0
nb_correct_binnednbheads = 0
# "loss" for nbheads prediction
nbheads_squared_loss = 0
# NB: for nbdeps prediction : computed at the end (cf. need to read all sentence to get the nb of deps)


# for one sentence
sentlen = 0
nb_gold = 0
nb_pred = 0
nb_correct_u = 0
nb_correct_l = 0
sent_nbdeps = {'g':defaultdict(int), 'p':defaultdict(int)} # nb of dependents of each head in sent

instream = open(args.gold_and_pred_dep_graphs_file)
for line in instream.readlines():
    if line.startswith('#'):
        continue
    line = line.strip()
    line = line.replace('NOI:','').replace('SIL:','') # marques bruit silence eventuellement existantes
    if not line: # end of sent
        local = [nb_gold, nb_pred, nb_correct_u, nb_correct_l]
        all_g_p_cu_cl = [ all_g_p_cu_cl[i] + local[i] for i in range(4) ]
        if sentlen > 40:
            long_g_p_cu_cl = [ long_g_p_cu_cl[i] + local[i] for i in range(4) ]
        # append the numbers of dependents
        # (loop over all tokens, to account for 0 dep cases)
        #@@        for d in range(1, sentlen + 1):
        for d in range(2, sentlen + 2):
            nbdeps['g'].append(sent_nbdeps['g'][d])
            nbdeps['p'].append(sent_nbdeps['p'][d])
        # reset sentence-based variables
        sentlen = 0
        nb_gold = 0
        nb_pred = 0
        nb_correct_u = 0
        nb_correct_l = 0
        sent_nbdeps = {'g':defaultdict(int), 'p':defaultdict(int)} # nb of dependents of each head in sent
        continue
    else:
        sentlen += 1
        cols = line.split('\t')
        dep = int(cols[0])
        if cols[2] != '_':
            gheads = cols[2].split('|')  # 2|3
            glabels = cols[3].split('|') # suj|obj
            gbol = dict_like_bag_of_labels(glabels)
        else:
            gheads = []
            glabels = []
            
        if cols[4] != '_':
            pheads = cols[4].split('|')  # 2|3
            plabels = cols[5].split('|') # suj|obj
            pbol = dict_like_bag_of_labels(plabels)
        else:
            pheads = []
            plabels = []

        gnbheads = len(gheads)
        pnbheads = len(pheads)
        nbheads['g'].append(gnbheads)
        nbheads['p'].append(pnbheads)
        gbinnednbheads = get_binnednb(gnbheads)
        pbinnednbheads = get_binnednb(pnbheads)
        binnednbheads['g'].append(gbinnednbheads)
        binnednbheads['p'].append(pbinnednbheads)
        bols['g'].append(gbol)
        bols['p'].append(pbol)
        for h in gheads:
            sent_nbdeps['g'][int(h)] += 1
        for h in pheads:
            sent_nbdeps['p'][int(h)] += 1

        nb_toks += 1
        if (gnbheads == pnbheads):
            nb_correct_nbheads += 1
        if gbinnednbheads == pbinnednbheads:
            nb_correct_binnednbheads += 1
        nbheads_squared_loss += squared_loss_of_logs(gnbheads, pnbheads)
            
        ugold = set(gheads)
        upred = set(pheads)
    
        lgold = set(zip(gheads, glabels))
        lpred = set(zip(pheads, plabels))

        nb_pred += len(pheads)
        nb_gold += len(gheads)
        nb_correct_u += len(ugold.intersection(upred))
        nb_correct_l += len(lgold.intersection(lpred))


        # distances
        for (head, lab) in lgold:
            dist = int(head) - dep
            direction = np.sign(dist)
            l2dist['g'][lab].append(dist)
            l2dist['g']['all'].append(dist)
            l2direction['g'][lab].append(direction)
            l2direction['g']['all'].append(direction)

        # directions
        for (head, lab) in lpred:
            dist = int(head) - dep
            direction = np.sign(dist)
            l2dist['p'][lab].append(dist)
            l2dist['p']['all'].append(dist)
            l2direction['p'][lab].append(direction)
            l2direction['p']['all'].append(direction)
            
        # label sequences
        glseq = '|'.join(glabels)
        lseqs2occ['g'][glseq] += 1
        plseq = '|'.join(plabels)
        lseqs2occ['p'][plseq] += 1
        # sorted label sequences
        gslseq = '|'.join(sorted(glabels))
        slseqs2occ['g'][gslseq] += 1
        pslseq = '|'.join(sorted(plabels))
        slseqs2occ['p'][pslseq] += 1



nb_correct_nbdeps = sum([ 1 for i in range(nb_toks) if nbdeps['g'][i] == nbdeps['p'][i] ])
nbdeps_squared_loss = sum([ squared_loss_of_logs(nbdeps['g'][i], nbdeps['p'][i]) for i in range(nb_toks) ])

# ---- for all sentences ------------
nb_gold, nb_pred, nb_correct_u, nb_correct_l = all_g_p_cu_cl
UR = 100 * nb_correct_u / nb_gold  
UP = 100 * nb_correct_u / nb_pred  
LR = 100 * nb_correct_l / nb_gold  
LP = 100 * nb_correct_l / nb_pred

UF = 2 * UP * UR / (UP + UR)
LF = 2 * LP * LR / (LP + LR)

print("ALL UR %5.2f UP %5.2f UF %5.2f" % (UR, UP, UF))
print("ALL LR %5.2f LP %5.2f LF %5.2f" % (LR, LP, LF))

# ---- for long sentences ------------
nb_gold, nb_pred, nb_correct_u, nb_correct_l = long_g_p_cu_cl
UR = 100 * nb_correct_u / nb_gold  
UP = 100 * nb_correct_u / nb_pred  
LR = 100 * nb_correct_l / nb_gold  
LP = 100 * nb_correct_l / nb_pred

UF = 2 * UP * UR / (UP + UR)
LF = 2 * LP * LR / (LP + LR)

print(">40 UR %5.2f UP %5.2f UF %5.2f" % (UR, UP, UF))
print(">40 LR %5.2f LP %5.2f LF %5.2f" % (LR, LP, LF))


# average distances per label
for l in sorted(l2dist['g'].keys()):
    gmean, gmad = mean_mad(l2dist['g'][l])
    pmean, pmad = mean_mad(l2dist['p'][l])
    print("LABEL %13s GOLD DIST: %5.2f (MAD %5.2f) PRED DIST: %5.2f (MAD %5.2f)" % (l,  gmean, gmad, pmean, pmad))
    gmean, gmad = mean_mad(l2direction['g'][l])
    pmean, pmad = mean_mad(l2direction['p'][l])
    print("LABEL %13s GOLD DIR : %5.2f (MAD %5.2f) PRED DIR : %5.2f (MAD %5.2f)" % (l,  gmean, gmad, pmean, pmad))
    # on compte la proportion d'arcs prédits avec ce label, ayant une direction atypique
    if gmean > 0.95 or gmean < -0.95:
        s = np.sign(gmean)
        a = len([ dist for dist in l2dist['p'][l] if np.sign(dist) != s ])
        t = len(l2dist['p'][l])
        print("LABEL %13s : proportion of odd direction in predicted arcs : %5.2f (%d / %d)" % (l, 100 * a / t, a , t))

glseqs = set(lseqs2occ['g'].keys())
plseqs = set(lseqs2occ['p'].keys())
print("\nLABELS SEQS GOLD: %d PRED %d COMMON %d NOT_IN_PRED %d NOT_IN_GOLD %d" % (len(glseqs), len(plseqs), len(glseqs.intersection(plseqs)), len(glseqs.difference(plseqs)), len(plseqs.difference(glseqs))))

gslseqs = set(slseqs2occ['g'].keys())
pslseqs = set(slseqs2occ['p'].keys())
print("\nSORTED LABELS SEQS GOLD: %d PRED %d COMMON %d NOT_IN_PRED %d NOT_IN_GOLD %d" % (len(gslseqs), len(pslseqs), len(gslseqs.intersection(pslseqs)), len(gslseqs.difference(pslseqs)), len(pslseqs.difference(gslseqs))))

# --------- label sequences ---------------------
print("\n")
# predicted label sequences too much predicted
for (lseq,pocc) in sorted(lseqs2occ['p'].items(), key=lambda x: x[1], reverse=True):
    gocc = lseqs2occ['g'][lseq]
    prop = (pocc - gocc) / pocc
    if prop > 0.2:
        print('TOO MUCH PRED LAB SEQ %15s \tPRED: %d GOLD: %d PROP: %f' % (lseq, pocc, gocc, prop))

print("\n")

# -------- most frequent label sequences ---------------------
# most freq label sequences in gold, with difference to freq in pred
s = sorted(lseqs2occ['g'].items(), key=lambda x: x[1], reverse=True)
for (lseq,o) in s[:50]:
    #if o == 0: # cf. 0-valued keys added in previous loop
    #    continue
    print('LAB SEQ %15s \tGOLD: %d PRED: %d (DIFF: %d)' % (lseq, o, lseqs2occ['p'][lseq], o - lseqs2occ['p'][lseq]))
print("\n")

for limit in [20, 30, 40, 50, 70, 100]:
    print("total nb occs for the first %d most frequent label seqs: %d" % (limit, sum([x[1] for x in s[:limit]])))
    print("total nb occs for label seqs after the first %d most freq lab seq: %d" % (limit, sum([x[1] for x in s[limit:]])))
    print("among which %d are hapaxes" % sum([x[1] for x in s if x[1] == 1]))
    print("\n")

# ---- same for sorted label sequences ------------------------

# most freq label sequences in gold, with difference to freq in pred
s = sorted(slseqs2occ['g'].items(), key=lambda x: x[1], reverse=True)
for (slseq,o) in s[:50]:
    #if o == 0: # cf. 0-valued keys added in previous loop
    #    continue
    print('SORTED LAB SEQ %15s \tGOLD: %d PRED: %d (DIFF: %d)' % (slseq, o, slseqs2occ['p'][slseq], o - slseqs2occ['p'][slseq]))
print("\n")

for limit in [20, 30, 40, 50, 70, 100]:
    print("total nb occs for the first %d most frequent sorted label seqs: %d" % (limit, sum([x[1] for x in s[:limit]])))
    print("total nb occs for label seqs after the first %d most freq sorted lab seqs: %d" % (limit, sum([x[1] for x in s[limit:]])))
    print("among which %d are hapaxes" % sum([x[1] for x in s if x[1] == 1]))
    print("\n")

# --------- similarity of bag of labels -----------------------
nbtoks = len(bols['g'])
bol_similarities = [ dict_cosinus(bols['g'][i], bols['p'][i]) for i in range(nbtoks) ]
mean, mad = mean_mad(bol_similarities)
print("AVG COS SIMILARITY of BAG OF LABELS: %5.2f (MAD %5.2f)\n" % (mean, mad))

bol_acc = 100 * sum([ dict_equality(bols['g'][i], bols['p'][i]) for i in range(nbtoks) ]) / len(bols['g'])
print("ACCURACY          for BAG OF LABELS: %5.2f\n" % bol_acc)


# --------- differences in nb of heads / nb of deps -----------
gmean, gmad = mean_mad(nbheads['g'])
pmean, pmad = mean_mad(nbheads['p'])
print("     AVG NB HEADS : GOLD %5.2f (MAD %5.2f) PRED %5.2f (MAD %5.2f)" % (gmean, gmad, pmean, pmad))
print("ACCURACY NB HEADS: %5.2f  SQUARED_LOSS: %f" % ( 100 * nb_correct_nbheads / nb_toks, nbheads_squared_loss))


gmean, gmad = mean_mad(binnednbheads['g'])
pmean, pmad = mean_mad(binnednbheads['p'])
print("     AVG BINNED NB HEADS : GOLD %5.2f (MAD %5.2f) PRED %5.2f (MAD %5.2f)" % (gmean, gmad, pmean, pmad))
print("ACCURACY BINNED NB HEADS: %5.2f  " % ( 100 * nb_correct_binnednbheads / nb_toks))

# differences in nb of deps / nb of deps ...
gmean, gmad = mean_mad(nbdeps['g'])
pmean, pmad = mean_mad(nbdeps['p'])
print("     AVG NB DEPS : GOLD %5.2f (MAD %5.2f) PRED %5.2f (MAD %5.2f)" % (gmean, gmad, pmean, pmad))
print("ACCURACY NB DEPS: %5.2f  SQUARED_LOSS: %f" % ( 100 * nb_correct_nbdeps / nb_toks, nbdeps_squared_loss))

#print(nbdeps)
