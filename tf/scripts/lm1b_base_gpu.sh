#!/bin/bash

# Data
DATA_ROOT=../data/one-billion-words/

# Model
DIV_VAL=4
N_LAYER=18
D_MODEL=1024
D_EMBED=1024
N_HEAD=8
D_HEAD=128
D_INNER=4096

# Training
TGT_LEN=256
MEM_LEN=256

BSZ=256
NUM_CORE=4

# Testing
TEST_TGT_LEN=32
TEST_MEM_LEN=128
TEST_CLAMP_LEN=-1

TEST_BSZ=16
TEST_NUM_CORE=1


if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
      --data_dir=${DATA_ROOT}/ \
      --dataset=lm1b \
      --tgt_len=${TGT_LEN} \
      --per_host_train_bsz=${BSZ} \
      --per_host_valid_bsz=${BSZ} \
      --num_passes=1 \
      --use_tpu=False \
      ${@:2}
elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
      --data_dir=${DATA_ROOT}/ \
      --dataset=lm1b \
      --tgt_len=${TEST_TGT_LEN} \
      --per_host_test_bsz=${TEST_BSZ} \
      --num_passes=1 \
      --use_tpu=False \
      ${@:2}
elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train_gpu.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=EXP-lm1b \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=False \
        --proj_same_dim=False \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.1 \
        --dropatt=0.0 \
        --learning_rate=0.00025 \
        --warmup_steps=0 \
        --train_steps=400000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=200 \
        --save_steps=4000 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train_gpu.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=EXP-lm1b \
        --div_val=${DIV_VAL} \
        --untie_r=True \
        --proj_share_all_but_first=False \
        --proj_same_dim=False \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.0 \
        --dropatt=0.0 \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --clamp_len=${TEST_CLAMP_LEN} \
        --same_length=True \
        --eval_batch_size=${TEST_BSZ} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --do_train=False \
        --do_eval=True \
        --eval_split=test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
