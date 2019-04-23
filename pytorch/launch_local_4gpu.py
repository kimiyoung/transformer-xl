#!/usr/bin/env python
# launch training locally with 4 GPU training

import argparse
import ncluster
import os
import sys

IMAGE_NAME = 'Deep Learning AMI (Ubuntu) Version 22.0'
INSTANCE_TYPE = 'p3.8xlarge'
NUM_GPUS = 4
NUM_NODES = 1

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl-4gpu-local',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
parser.add_argument('--aws', type=int, default=0,
                    help="whether to launch on AWS")
args = parser.parse_args()

if args.aws:
  ncluster.set_backend('aws')
else:
  ncluster.set_backend('local')


# routines to build NCCL ring orders
def get_nccl_params(num_tasks, num_gpus):
  return 'NCCL_DEBUG=VERSION'


def main():

  # todo: figure out why run is not set properly
  # TODO: logdir root is set to ncluster/runs instead of /ncluster/runs
  ncluster.set_logdir_root('/ncluster/runs.new')
  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          image_name=IMAGE_NAME,
                          instance_type=INSTANCE_TYPE)

  job.upload('*')
  job.run('source activate pytorch_p36')
  # The following is needed, but kills launcher script if run locally
  #  job.run('killall python || echo failed')  # kill previous run
  
  # Training script args
  default_params = [
    '--logdir', job.logdir,
    '--distributed',
  ]

  # todo: consistency with - and _ in args
  # taken run_wt103_base.sh
  training_params = [
    '--seed', 1,
    '--cuda', 
    '--data', '/ncluster/data/transformer-xl-data/wikitext-103', # source of train.txt
    '--dataset', 'wt103',
      '--dist-backend', 'nccl',
      # TODO(y), remove adaptive, since it uses sparse tensors?
    '--adaptive',
    '--log-interval', 10,
    '--n_layer', 16,
    '--d_model', 410,
    '--n_head', 10,
    '--d_head', 41,
    '--d_inner', 2100,
    '--dropout', 0.1,
    '--dropatt', 0.0,
    '--optim', 'adam',
    '--lr', 0.00025,
    '--warmup_step', 0,
    '--max_step', 200000,
    '--tgt_len', 150,
    '--mem_len', 150,
    '--eval_tgt_len', 150,
    '--batch_size', 15,  # per-gpu batch size
    '--gpu0_bsz', 4,
#    '--work_dir', job.logdir+'/', # legacy code logs to logdir/-wt103/{ts}
  ]

  training_params = default_params + training_params
  training_params = ' '.join(str(p) for p in training_params)
  nccl_params = get_nccl_params(1, NUM_GPUS)
  i = 0
  task = job.tasks[i]
  dist_params = f'--nproc_per_node={NUM_GPUS} --nnodes={NUM_NODES} --node_rank={i} --master_addr={job.tasks[0].ip} --master_port={6006}'
  cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} train.py {training_params}'
  
  task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
  task.run(cmd, non_blocking=True)

  print(f"Logging to {job.logdir}")


if __name__ == '__main__':
  main()
