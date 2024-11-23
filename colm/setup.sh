export MM_ROOT=`pwd`
export PYTHONPATH=$MM_ROOT:$PYTHONPATH
export PYTHON_EXEC=python

PHATGOOSE_CASHE_DIR=/scratch2/workspace/wenlongzhao_umass_edu-SKD/code_wenlong/phatgoose_cache/
mkdir -p $PHATGOOSE_CASHE_DIR
export HUGGINGFACE_HUB_CACHE=$PHATGOOSE_CASHE_DIR
export TRANSFORMERS_CACHE=$PHATGOOSE_CASHE_DIR
export HF_HOME=$PHATGOOSE_CASHE_DIR
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=2024-seqkd4da-phatgoose
