#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data /ncluster/data/transformer-xl-data/wikitext-103 \
        --dataset wt103 \
        --log-interval 20 \
        --n_layer 12 \
        --d_model 768 \
        --n_head 12 \
        --d_head 41 \
        --d_inner 2100 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 600000 \
        --tgt_len 150 \
        --mem_len 150 \
        --eval_tgt_len 150 \
        --adaptive \
        --batch_size 60 \
        --multi_gpu \
        --gpu0_bsz 4 \
        --run_name large1 \
        --work_dir /ncluster/transformer-xl/workdir \
    ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/wikitext-103/ \
        --dataset wt103 \
        --tgt_len 64 \
        --mem_len 640 \
        --clamp_len 400 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argment 1'
fi
