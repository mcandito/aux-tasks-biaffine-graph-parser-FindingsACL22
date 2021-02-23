
set -x


PROJH=$HOME/Documents/A_PROJETS/GRAPH_PARSER/git/marie-parsing
# on CLC
PROJH=/home/mcandito/A_PROJETS/GRAPH_PARSER/github/biaffine-graph-parser


D=$PROJH/../../git/marie-parsing/resources

# TREES
TRAIN_FILE=$D/FRENCH_SPMRL_dep_trees/train.French.predmorph.conll
DEV_FILE=$D/FRENCH_SPMRL_dep_trees/dev.French.predmorph.conll

# GRAPHS (TODO: predmorph version)
TRAIN_FILE=$D/deep_french_dep_graphs/compacte.ftb.gold.dev
#TRAIN_FILE=$D/deep_french_dep_graphs/compacte.ftb.gold.train
DEV_FILE=$D/deep_french_dep_graphs/compacte.ftb.gold.dev

timestamp=$(date "+%Y-%m-%d-%H-%M-%S")
O=$PROJH/../OUTPUT/output-$timestamp
mkdir $O

DEVICE_ID=0

EMB_FILE=$D/vecs100-linear-frwiki
W_EMB_SIZE=100
L_EMB_SIZE=100
P_EMB_SIZE=100
R_BERT_SIZE=300
LSTM_H_SIZE=600
MLP_ARC_O_SIZE=200
MLP_LAB_O_SIZE=200

NB_EPOCHS=2 #40 #
NB_EPOCHS_ARC_ONLY=0
LR=0.00002
BATCH_SIZE=16
LEX_DROPOUT=0.4
LAB_LOSS_WEIGHT=0.5
POS_ARC_WEIGHT=1.5
BERT_NAME=flaubert/flaubert_base_cased
#BERT_NAME=''


python $PROJH/biaffine_graph_parser/train_or_use_parser.py train $TRAIN_FILE $O -g -v $DEV_FILE -p $EMB_FILE -w $W_EMB_SIZE -l $L_EMB_SIZE -c $P_EMB_SIZE --reduced_bert_size $R_BERT_SIZE --lstm_h_size $LSTM_H_SIZE --mlp_arc_o_size $MLP_ARC_O_SIZE -b $BATCH_SIZE -r $LR -d $LEX_DROPOUT -i $LAB_LOSS_WEIGHT --pos_arc_weight $POS_ARC_WEIGHT -n $NB_EPOCHS --nb_epochs_arc_only $NB_EPOCHS_ARC_ONLY --device_id $DEVICE_ID --bert_name $BERT_NAME

# sans embeddings
#python $PROJH/biaffine_graph_parser/train_or_use_parser.py train $TRAIN_FILE $O -g -v $DEV_FILE -w $W_EMB_SIZE -l $L_EMB_SIZE -c $P_EMB_SIZE --reduced_bert_size $R_BERT_SIZE --lstm_h_size $LSTM_H_SIZE --mlp_arc_o_size $MLP_ARC_O_SIZE -b $BATCH_SIZE -r $LR -d $LEX_DROPOUT -i $LAB_LOSS_WEIGHT --pos_arc_weight $POS_ARC_WEIGHT -n $NB_EPOCHS --nb_epochs_arc_only $NB_EPOCHS_ARC_ONLY --device_id $DEVICE_ID --bert_name $BERT_NAME
