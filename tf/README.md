## Prerequisite

- Python 2.7
- Tensorflow 0.1.12



## Obtain and evaluate pretrained SoTA models

#### 1. Download preprocessed data (vocab) & pretrained models

(a) Set your own `DATA_ROOT` in `sota/download.sh` (default to `./`), which will be the root diretory of downloaded model.

(b) Then, download the model & data by `bash sota/download.sh`. After downloading, the expected directory structure is as follows

```markdown
pretrained_xl
  tf_enwik8/
    data/
      cache.pkl
      corpus-info.json
    model/
      checkpoint
      model.ckpt*
  tf_wt103/
  	...
  ...
```

**Note**: we include preprocessed data in the download files to make sure the **same vocabulary** is used. Please see the code `tf/data_utils.py` to understand the data structure.



#### 2. Run evaluation scripts to replicate SoTA results on GPUs

- **enwik8**: modify the script `sota/enwik8.sh` accordingly (see below)
  - set `DATA_ROOT` to the same folder used in the download step (default to `./`)
  - set `TEST_NUM_CORE ` (number of GPUs to use): we recommend 2 GPUs => about 60 mins
  - run the script: `bash sota/enwik8.sh`

- **lm1b**: modify the script `sota/lm1b.sh` accordingly  (see below)
  - set `DATA_ROOT` to the same folder used in the download step (default to `./`)
  - set `TEST_NUM_CORE ` (number of GPUs to use): we recommend 1 GPUs => less than 5 mins
  - run the script: `bash sota/lm1b.sh`

- **wt103**:  modify the script `sota/wt103.sh` accordingly  (see below)
  - set `DATA_ROOT` to the same folder used in the download step (default to `./`)
  - set `TEST_NUM_CORE ` (number of GPUs to use): we recommend 1 GPUs => less than 5 mins
  - run the script: `bash sota/wt103.sh`

- **text8**:  modify the script `sota/text8.sh` accordingly  (see below)
  - set `DATA_ROOT` to the same folder used in the download step (default to `./`)
  - set `TEST_NUM_CORE ` (number of GPUs to use): we recommend 2 GPUs => about 60 mins
  - run the script: `bash sota/text8.sh`



## Train "Transformer-XL" from scratch with GPUs or TPUs

### 1. Download raw data

`bash getdata.sh`



### 2. Preprocess, training and evaluation

For `dataset` in `[enwik8, lm1b, wt103, text8]`:

- check out `scripts/dataset_gpu.sh` for GPU training and evaluation
- check out `scripts/dataset_tpu.sh` for TPU training and evaluation



#### (1) Preprocess raw data and create tfrecords

**NOTE**: The preprocessing for GPU and TPU are different. So, you have to run them separately.

GPU:

- create training and validation data: `bash scripts/dataset_gpu.sh train_data`
- create test data: `bash scripts/dataset_gpu.sh test_data`

TPU:

- Set the Google storage URL  in `scripts/dataset_tpu.sh`:
  - `GSDATA`: data URL
  - `GSEXP`: experiment URL
- create training and validation data: `bash scripts/dataset_tpu.sh train_data`
- create test data: `bash scripts/dataset_tpu.sh test_data`



#### (2) Run training

GPU:

- Modify the configurations in `scripts/dataset_gpu.sh`  according to your needs.
- `bash scripts/dataset_gpu.sh train`

TPU:

- Modify the configurations in `scripts/dataset_tpu.sh`  according to your needs.
- `bash scripts/dataset_tpu.sh train`



#### (3) Run evaluation

GPU:

- `bash scripts/dataset_gpu.sh eval --eval_ckpt_path PATH_TO_CKPT`

TPU:

- `bash scripts/dataset_tpu.sh eval --eval_ckpt_path PATH_TO_CKPT`
