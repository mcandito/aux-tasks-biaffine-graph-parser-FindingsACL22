
set -x


PROJH=$HOME/Documents/A_PROJETS/GRAPH_PARSER/github/biaffine-graph-parser
# on CLC
PROJH=/home/mcandito/A_PROJETS/GRAPH_PARSER/github/biaffine-graph-parser


D=$PROJH/../../git/marie-parsing/resources

# TREES
TRAIN_FILE=$D/FRENCH_SPMRL_dep_trees/train.French.predmorph.conll
DEV_FILE=$D/FRENCH_SPMRL_dep_trees/dev.French.predmorph.conll

# GRAPHS (predmorph version)
TRAIN_FILE=$D/deep_french_dep_graphs/compacte.ftb.predmorph.dev
#TRAIN_FILE=$D/deep_french_dep_graphs/compacte.ftb.predmorph.train
DEV_FILE=$D/deep_french_dep_graphs/compacte.ftb.predmorph.dev

DATA_NAME='ftb_predmorph'

timestamp=$(date "+%Y-%m-%d-%H-%M-%S")
O=$PROJH/../OUTPUT/output-$timestamp
mkdir $O

DEVICE_ID=1

EMB_FILE=None #$D/vecs100-linear-frwiki
W_EMB_SIZE=100
L_EMB_SIZE=0 #100
P_EMB_SIZE=0 #50
R_BERT_SIZE=0 # no reduction
LSTM_H_SIZE=300 #600
MLP_ARC_O_SIZE=200 #400
MLP_LAB_O_SIZE=200 #400

# obsolete options:
#NB_EPOCHS_ARC_ONLY=0
#LAB_LOSS_WEIGHT=0.5
#POS_ARC_WEIGHT=1.5

NB_EPOCHS=2 #40 #
LR=0.00002
BATCH_SIZE=12
LEX_DROPOUT=0.4
BERT_NAME=flaubert/flaubert_base_cased
#BERT_NAME=None
TASKS=a.l.h

python $PROJH/train_or_use_parser.py train $TRAIN_FILE $O -g --tasks $TASKS --data_name $DATA_NAME -v $DEV_FILE -p $EMB_FILE -w $W_EMB_SIZE -l $L_EMB_SIZE -c $P_EMB_SIZE --reduced_bert_size $R_BERT_SIZE --lstm_h_size $LSTM_H_SIZE --mlp_arc_o_size $MLP_ARC_O_SIZE --mlp_lab_o_size $MLP_LAB_O_SIZE -b $BATCH_SIZE -r $LR -d $LEX_DROPOUT  -n $NB_EPOCHS --device_id $DEVICE_ID --bert_name $BERT_NAME


