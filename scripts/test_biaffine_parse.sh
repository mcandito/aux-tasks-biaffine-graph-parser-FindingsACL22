
set -x

# eg : 2021-04-21-12-28-41
MODEL_STAMP=$1

PROJH=$HOME/Documents/A_PROJETS/GRAPH_PARSER/github/biaffine-graph-parser
# on CLC
PROJH=/home/mcandito/A_PROJETS/GRAPH_PARSER/github/biaffine-graph-parser


D=$PROJH/../../git/marie-parsing/resources

MODEL_DIR=$PROJH/../OUTPUT/output-$1

# TREES
DEV_FILE=$D/FRENCH_SPMRL_dep_trees/dev.French.predmorph.conll

# GRAPHS (predmorph version)
DEV_FILE=$D/deep_french_dep_graphs/compacte.ftb.predmorph.dev
OUT_PARSED_DEV_FILE=$MODEL_DIR/compacte.ftb.predbiaff.dev

#!! test predmorh is corrupt
#@@ temporarily using gold morpho (not used in parsing model)
#TEST_FILE=$D/deep_french_dep_graphs/compacte.ftb.predmorph.test
TEST_FILE=$D/deep_french_dep_graphs/compacte.ftb.gold.test
OUT_PARSED_TEST_FILE=$MODEL_DIR/compacte.ftb.predbiaff.test

DEVICE_ID=1


BATCH_SIZE=12
BERT_NAME=flaubert/flaubert_base_cased
#BERT_NAME=None


python $PROJH/train_or_use_parser.py test $TEST_FILE $MODEL_DIR -g -b $BATCH_SIZE --out_parsed_file $OUT_PARSED_TEST_FILE --device_id $DEVICE_ID --bert_name $BERT_NAME
python $PROJH/train_or_use_parser.py test $DEV_FILE $MODEL_DIR -g -b $BATCH_SIZE --out_parsed_file $OUT_PARSED_DEV_FILE --device_id $DEVICE_ID --bert_name $BERT_NAME


