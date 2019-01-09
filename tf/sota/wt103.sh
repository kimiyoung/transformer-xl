#!/bin/bash

# Data
DATA_ROOT=./
DATA_DIR=${DATA_ROOT}/pretrained_xl/tf_wt103/data
MODEL_DIR=${DATA_ROOT}/pretrained_xl/tf_wt103/model

# Model
DIV_VAL=4
N_LAYER=18
D_MODEL=1024
D_EMBED=1024
N_HEAD=16
D_HEAD=64
D_INNER=4096

# Training
TGT_LEN=256
MEM_LEN=256

BSZ=16
NUM_CORE=2

# Testing
TEST_TGT_LEN=128
TEST_MEM_LEN=1600
TEST_CLAMP_LEN=1000

TEST_CKPT_PATH=${MODEL_DIR}/model.ckpt-0
TEST_BSZ=16
TEST_NUM_CORE=1


echo 'Preprocess test set...'
python data_utils.py \
    --data_dir=${DATA_DIR}/ \
    --dataset=enwik8 \
    --tgt_len=${TEST_TGT_LEN} \
    --per_host_test_bsz=${TEST_BSZ} \
    --num_passes=1 \
    --use_tpu=False


echo 'Run evaluation on test set...'
python train_gpu.py \
    --data_dir=${DATA_DIR}/tfrecords \
    --record_info_dir=${DATA_DIR}/tfrecords/ \
    --corpus_info_path=${DATA_DIR}/corpus-info.json \
    --eval_ckpt_path=${TEST_CKPT_PATH} \
    --model_dir=EXP-wt103 \
    --div_val=${DIV_VAL} \
    --untie_r=True \
    --proj_share_all_but_first=True \
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
    --eval_split=test

