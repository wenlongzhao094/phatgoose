export MM_ROOT=`pwd`
export PYTHONPATH=$MM_ROOT:$PYTHONPATH

PHATGOOSE_CACHE_DIR=/scratch3/workspace/wenlongzhao_umass_edu-metakd/cache_phatgoose/
mkdir -p PHATGOOSE_CACHE_DIR
export HUGGINGFACE_HUB_CACHE=PHATGOOSE_CACHE_DIR
export HF_HOME=PHATGOOSE_CACHE_DIR

export WANDB_PROJECT=2024-seqkd4da-phatgoose
