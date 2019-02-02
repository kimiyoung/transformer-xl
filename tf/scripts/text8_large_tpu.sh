#!/bin/bash

# Path
LOCAL_DIR=../data/text8/
GSDATA=
GSEXP=

# TPU setting
NUM_HOST=2
NUM_CORE=16 # TPUv2 -> 8 | TPUv3 -> 16

TEST_NUM_HOST=1
TEST_NUM_CORE=8 # TPUv2 -> 8 | TPUv3 -> 16

# Model
N_LAYER=24
D_MODEL=1024
D_EMBED=1024
N_HEAD=8
D_HEAD=128
D_INNER=3072

# Training
TGT_LEN=768
MEM_LEN=768
TRAIN_BSZ=64
VALID_BSZ=64

# Testing
TEST_TGT_LEN=128
TEST_MEM_LEN=3800
TEST_CLAMP_LEN=1000
TEST_BSZ=16

if [[ $1 == 'train_data' ]]; then
    python data_utils.py \
        --data_dir=${LOCAL_DIR}/ \
        --dataset=text8 \
        --tgt_len=${TGT_LEN} \
        --per_host_train_bsz=${TRAIN_BSZ} \
        --per_host_valid_bsz=${VALID_BSZ} \
        --num_core_per_host=${NUM_CORE} \
        --num_passes=10 \
        --use_tpu=True \
        ${@:2}

    SRC_PATTERN=train.bsz-${TRAIN_BSZ}.tlen-${TGT_LEN}.core-${NUM_CORE}*
    gsutil cp ${LOCAL_DIR}/tfrecords/${SRC_PATTERN} ${GSDATA}/text8-tfrecords/

    SRC_PATTERN=valid.bsz-${VALID_BSZ}.tlen-${TGT_LEN}.core-${NUM_CORE}*
    gsutil cp ${LOCAL_DIR}/tfrecords/${SRC_PATTERN} ${GSDATA}/text8-tfrecords/

elif [[ $1 == 'test_data' ]]; then
    python data_utils.py \
        --data_dir=${LOCAL_DIR}/ \
        --dataset=text8 \
        --tgt_len=${TEST_TGT_LEN} \
        --per_host_test_bsz=${TEST_BSZ} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --num_passes=1 \
        --use_tpu=True \
        ${@:2}

    SRC_PATTERN=test.bsz-${TEST_BSZ}.tlen-${TEST_TGT_LEN}.core-${TEST_NUM_CORE}*
    gsutil cp ${LOCAL_DIR}/tfrecords/${SRC_PATTERN} ${GSDATA}/text8-tfrecords/

elif [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --data_dir=${GSDATA}/text8-tfrecords \
        --record_info_dir=${LOCAL_DIR}/tfrecords/ \
        --corpus_info_path=${LOCAL_DIR}/corpus-info.json \
        --model_dir=${GSEXP}/text8 \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --dropout=0.15 \
        --dropatt=0.15 \
        --learning_rate=0.00025 \
        --warmup_steps=4000 \
        --train_steps=400000 \
        --tgt_len=${TGT_LEN} \
        --mem_len=${MEM_LEN} \
        --train_batch_size=${TRAIN_BSZ} \
        --use_tpu=True \
        --num_host=${NUM_HOST} \
        --num_core_per_host=${NUM_CORE} \
        --iterations=1000 \
        --save_steps=10000 \
        --do_train=True \
        --do_eval=False \
        ${@:2}

elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python train.py \
        --data_dir=${GSDATA}/text8-tfrecords \
        --record_info_dir=${LOCAL_DIR}/tfrecords/ \
        --corpus_info_path=${LOCAL_DIR}/corpus-info.json \
        --model_dir=${GSEXP}/text8 \
        --n_layer=${N_LAYER} \
        --d_model=${D_MODEL} \
        --d_embed=${D_EMBED} \
        --n_head=${N_HEAD} \
        --d_head=${D_HEAD} \
        --d_inner=${D_INNER} \
        --tgt_len=${TEST_TGT_LEN} \
        --mem_len=${TEST_MEM_LEN} \
        --eval_batch_size=${TEST_BSZ} \
        --num_host=${TEST_NUM_HOST} \
        --num_core_per_host=${TEST_NUM_CORE} \
        --use_tpu=True \
        --do_train=False \
        --do_eval_only=True \
        --eval_split=test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
