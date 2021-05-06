
#set -x


PROJH=$HOME/Documents/A_PROJETS/GRAPH_PARSER/git/marie-parsing
# on CLC
PROJH=/home/mcandito/A_PROJETS/GRAPH_PARSER/github/biaffine-graph-parser


D=$PROJH/../../git/marie-parsing/resources

# TREES
TRAIN_FILE=$D/FRENCH_SPMRL_dep_trees/train.French.predmorph.conll
DEV_FILE=$D/FRENCH_SPMRL_dep_trees/dev.French.predmorph.conll

# GRAPHS (predmorph version)
#TRAIN_FILE=$D/deep_french_dep_graphs/compacte.ftb.gold.dev
TRAIN_FILE=$D/deep_french_dep_graphs/compacte.ftb.predmorph.train
DEV_FILE=$D/deep_french_dep_graphs/compacte.ftb.predmorph.dev
DATA_NAME='ftb_predmorph'


META_LOG=$PROJH/../OUTPUT/meta_log
#META_LOG=$PROJH/biaffine_graph_parser/meta_log
rm $META_LOG
touch $META_LOG

DEVICE_ID=1

W_EMB_SIZE=100
L_EMB_SIZE=100
P_EMB_SIZE=50

NB_EPOCHS=30 #

BATCH_SIZE=8 #12

BERT_NAME=flaubert/flaubert_base_cased
#BERT_NAME=''

#set -x

LSTM_H_SIZE=600
LEX_DROPOUT=0.4

# rerun after debug of unk when using pretrained-embeddings

for FREEZE_BERT in ''; #'-f ' 
do for LR in 0.00002 0.00001;
   do for MLP_ARC_O_SIZE in 400;# 300; # used for ARC and LAB # 600 is too big
      do for P_EMB_SIZE in 0;# 50;# with or without POS
			   #do for EMB_FILE in None $D/vecs100-linear-frwiki; # with or without embeddings
	 do for TASKS in a.l.h a.l.h.s a.l
	    do for i in 1 2 3; # 3 runs with same config
	       do
		   timestamp=$(date "+%Y-%m-%d-%H-%M-%S");
		   O=$PROJH/../OUTPUT/output-$timestamp;
		   mkdir $O;
		   echo $O >> $META_LOG ;
		   echo python $PROJH/train_or_use_parser.py train $TRAIN_FILE $O -g --data_name $DATA_NAME -v $DEV_FILE -p $EMB_FILE -w $W_EMB_SIZE -l $L_EMB_SIZE -c $P_EMB_SIZE --lstm_h_size $LSTM_H_SIZE --mlp_arc_o_size $MLP_ARC_O_SIZE --mlp_lab_o_size $MLP_ARC_O_SIZE -b $BATCH_SIZE -r $LR -d $LEX_DROPOUT -n $NB_EPOCHS --device_id $DEVICE_ID --bert_name $BERT_NAME $FREEZE_BERT >> $META_LOG ;
		   python $PROJH/train_or_use_parser.py train $TRAIN_FILE $O -g --data_name $DATA_NAME -v $DEV_FILE -p $EMB_FILE -w $W_EMB_SIZE -l $L_EMB_SIZE -c $P_EMB_SIZE --lstm_h_size $LSTM_H_SIZE --mlp_arc_o_size $MLP_ARC_O_SIZE --mlp_lab_o_size $MLP_ARC_O_SIZE -b $BATCH_SIZE -r $LR -d $LEX_DROPOUT -n $NB_EPOCHS --device_id $DEVICE_ID --bert_name $BERT_NAME $FREEZE_BERT >> $META_LOG ;
	       done
	    done
	 done
      done
   done
done
       
