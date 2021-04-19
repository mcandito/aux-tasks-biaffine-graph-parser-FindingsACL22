
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

DEVICE_ID=2

W_EMB_SIZE=100
#L_EMB_SIZE=100
P_EMB_SIZE=50

NB_EPOCHS=40 #
NB_EPOCHS_ARC_ONLY=0

BATCH_SIZE=8 #12

BERT_NAME=flaubert/flaubert_base_cased
#BERT_NAME=''

#set -x

LSTM_H_SIZE=600
LAB_LOSS_WEIGHT=0.5
LEX_DROPOUT=0.4

for FREEZE_BERT in '-f ' '';
do for LR in 0.00002 0.00001;
   do for POS_ARC_WEIGHT in 1.5 1;
      do for MLP_ARC_O_SIZE in 400;# 300; # used for ARC and LAB # 600 is too big
	 do  for R_BERT_SIZE in 0; # 300;
	     do for L_EMB_SIZE in 0 100;
		do for EMB_FILE in $D/vecs100-linear-frwiki ; #None
		   do
		       timestamp=$(date "+%Y-%m-%d-%H-%M-%S");
		       O=$PROJH/../OUTPUT/output-$timestamp;
		       mkdir $O;
		       echo $O >> $META_LOG ;
		       echo python $PROJH/train_or_use_parser.py train $TRAIN_FILE $O -g --data_name $DATA_NAME -v $DEV_FILE -p $EMB_FILE -w $W_EMB_SIZE -l $L_EMB_SIZE -c $P_EMB_SIZE --reduced_bert_size $R_BERT_SIZE --lstm_h_size $LSTM_H_SIZE --mlp_arc_o_size $MLP_ARC_O_SIZE --mlp_lab_o_size $MLP_ARC_O_SIZE -b $BATCH_SIZE -r $LR -d $LEX_DROPOUT -i $LAB_LOSS_WEIGHT --pos_arc_weight $POS_ARC_WEIGHT -n $NB_EPOCHS --nb_epochs_arc_only $NB_EPOCHS_ARC_ONLY --device_id $DEVICE_ID --bert_name $BERT_NAME $FREEZE_BERT >> $META_LOG ;
		       python $PROJH/train_or_use_parser.py train $TRAIN_FILE $O -g --data_name $DATA_NAME -v $DEV_FILE -p $EMB_FILE -w $W_EMB_SIZE -l $L_EMB_SIZE -c $P_EMB_SIZE --reduced_bert_size $R_BERT_SIZE --lstm_h_size $LSTM_H_SIZE --mlp_arc_o_size $MLP_ARC_O_SIZE --mlp_lab_o_size $MLP_ARC_O_SIZE -b $BATCH_SIZE -r $LR -d $LEX_DROPOUT -i $LAB_LOSS_WEIGHT --pos_arc_weight $POS_ARC_WEIGHT -n $NB_EPOCHS --nb_epochs_arc_only $NB_EPOCHS_ARC_ONLY --device_id $DEVICE_ID --bert_name $BERT_NAME $FREEZE_BERT;
		   done
		done
	     done
	  done
       done
    done
done

       
