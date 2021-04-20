
set -x


PROJH=$HOME/Documents/A_PROJETS/GRAPH_PARSER/github/biaffine-graph-parser
# on CLC
PROJH=/home/mcandito/A_PROJETS/GRAPH_PARSER/github/biaffine-graph-parser


D=$PROJH/../../git/marie-parsing/resources

MODEL_DIR=$PROJH/../OUTPUT/output-2021-04-20-17-33-03

# TREES
DEV_FILE=$D/FRENCH_SPMRL_dep_trees/dev.French.predmorph.conll

# GRAPHS (predmorph version)
DEV_FILE=$D/deep_french_dep_graphs/compacte.ftb.predmorph.dev

OUT_PARSED_FILE=$MODEL_DIR/compacte.ftb.predbiaff.dev

DEVICE_ID=2


BATCH_SIZE=12
BERT_NAME=flaubert/flaubert_base_cased
#BERT_NAME=None


python $PROJH/train_or_use_parser.py test $TEST_FILE $DEV_FILE $MODEL_DIR -g -b $BATCH_SIZE --out_parsed_file $OUT_PARSED_FILE --device_id $DEVICE_ID --bert_name $BERT_NAME


