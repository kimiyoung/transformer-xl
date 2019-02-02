#!/bin/bash

# Data
DATA_ROOT=../data/enwik8/

# Model
N_LAYER=12
D_MODEL=512
D_EMBED=512
N_HEAD=8
D_HEAD=64
D_INNER=2048

# Training
TGT_LEN=512
MEM_LEN=512

BSZ=24
NUM_CORE=4

# Testing
TEST_TGT_LEN=80
TEST_MEM_LEN=2100
TEST_CLAMP_LEN=820

TEST_BSZ=10
TEST_NUM_CORE=1

if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=enwik8 \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${BSZ} \
        --per_host_valid_bsz=${BSZ} \
        --num_passes=1 \
        --use_tpu=False \
        ${@:2}
elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
        --data_dir=${DATA_ROOT}/ \
        --dataset=enwik8 \
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
        --model_dir=EXP-enwik8 \
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
        --do_train=True \
        --do_eval=False \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train_gpu.py \
        --data_dir=${DATA_ROOT}/tfrecords \
        --record_info_dir=${DATA_ROOT}/tfrecords/ \
        --corpus_info_path=${DATA_ROOT}/corpus-info.json \
        --model_dir=EXP-enwik8 \
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