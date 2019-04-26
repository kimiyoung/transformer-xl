#!/usr/bin/env python
# launch training locally with 4 GPU training

import argparse
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl-4gpu',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default="p3.8xlarge",
                    help="how many machines to use")
parser.add_argument('--image_name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 22.0',
                    help="name of AMI to use ")
parser.add_argument('--aws', type=int, default=1,
                    help="whether to launch on AWS")
args = parser.parse_args()

ncluster.set_backend('aws')


# routines to build NCCL ring orders
def get_nccl_params(_num_tasks, _num_gpus):
  return 'NCCL_DEBUG=VERSION'


def main():
  ncluster.set_logdir_root('/ncluster/runs.new')
  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}",
                          num_tasks=args.machines,
                          image_name=args.image_name,
                          instance_type=args.instance_type)
  job.rsync('.')
  # These are all run separately because job.run can't handle &&
  job.run('killall python || echo failed')  # kill previous run
  job.run('source activate pytorch_p36')
  job.run('pip install -r requirements.txt')
  # workaround for https://github.com/tensorflow/models/issues/3995
  job.run('pip install -U protobuf')
  
  # Training script args
  default_params = [
    '--logdir', job.logdir,
    '--distributed',
  ]

  # todo(y): consistency with - and _ in args
  # taken run_wt103_base.sh
  base_lr = 0.00025
  training_params = [
    '--seed', 1,
    '--cuda', 
    '--data', '/ncluster/data/transformer-xl-data/wikitext-103',
    '--dataset', 'wt103',
    '--dist-backend', 'nccl',
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
    '--lr', base_lr,
    '--warmup_step', 0,
    '--max_step', 200000,
    '--tgt_len', 150,
    '--mem_len', 150,
    '--eval_tgt_len', 150,
    '--batch_size', 15,  # per-gpu batch size
    '--eval-interval', 1000,
  ]

  if args.instance_type == 'p3.8xlarge':
      num_gpus = 4
  elif args.instance_type == 'p3.16xlarge':
      num_gpus = 8
  else:
      assert False, f"Unknown instance type {args.instance_type}"

  training_params = default_params + training_params
  training_params = ' '.join(str(p) for p in training_params)
  nccl_params = get_nccl_params(args.machines, num_gpus)

  for i, task in enumerate(job.tasks):
      dist_params = \
          f'--nproc_per_node={num_gpus} ' \
          f'--nnodes={args.machines} --node_rank={i} ' \
          f'--master_addr={job.tasks[0].ip} --master_port={6006}'
      cmd = f'{nccl_params} python -m torch.distributed.launch {dist_params} train.py {training_params}'
      task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
      task.run(cmd, non_blocking=True)

  print(f"Logging to {job.logdir}")


if __name__ == '__main__':
  main()
