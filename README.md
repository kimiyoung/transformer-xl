# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

This repository contains the code in both **PyTorch** and **TensorFlow** for our paper
>[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](http://arxiv.org/abs/1901.02860)

>Zihang Dai\*, Zhilin Yang\*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov (*: equal contribution)

>Preprint 2018

## TensorFlow

- The source code is in the `tf/` folder, supporting (1) single-node multi-gpu training, and (2) multi-host TPU training.
- Besides the source code, we also provide pretrained "TensorFlow" models with state-of-the-art (SoTA) performances reported in the paper.
- Please refer to `tf/README.md` for details.

## PyTorch

- The source code is in the `pytorch/` folder, supporting single-node multi-gpu training via the module `nn.DataParallel`.
- Please refer to `pytorch/README.md` for details.

## Results

Transformer-XL achieves new state-of-the-art results on multiple language modeling benchmarks. Transformer-XL is also the first to break through the 1.0 barrier on char-level language modeling. Below is a summary.

Method | enwiki8 | text8 | One Billion Word | WT-103 | PTB (w/o finetuning)
-- | -- | -- | -- | -- | -- 
Previous Best | 1.06 | 1.13 | 23.7 | 20.5 | 55.5
Transformer-XL | **0.99** | **1.08** | **21.8** | **18.3** | **54.5**



## Acknowledgement

A large portion of the `getdata.sh` script comes from the [awd-lstm](https://github.com/salesforce/awd-lstm-lm/) repo. Happy Language Modeling :)
