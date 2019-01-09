#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --n_layer 12 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 2048 \
        --dropout 0.2 \
        --optim adam \
        --lr 0.00025 \
        --tgt_len 256 \
        --mem_len 256 \
        --eval_tgt_len 256 \
        --batch_size 16 \
        --max_step 1000000 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --tgt_len 256 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
