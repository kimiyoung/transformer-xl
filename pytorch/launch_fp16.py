#!/usr/bin/env python
"""Launch training on AWS with 8 GPUs."""

import argparse
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default="p3.16xlarge",
                    help="how many machines to use")
parser.add_argument('--image_name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 22.0',
                    help="name of AMI to use ")
parser.add_argument('--spot', action='store_true',
                    help='Use spot instance')
parser.add_argument('--ncluster_backend', type=str, default='aws',
                    help='Use spot instance')
parser.add_argument('--no_fp16', action='store_true', default=False,
                    help='dont use fp16')
parser.add_argument('--batch_size', default=15,
                    help="model batch size")


# routines to build NCCL ring orders
def get_nccl_params(_num_tasks, _num_gpus):
  return 'NCCL_DEBUG=VERSION'


def main(args):
  ncluster.set_backend(args.ncluster_backend)
  ncluster.set_logdir_root('/ncluster/runs.new')
  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}",
                          num_tasks=args.machines,
                          image_name=args.image_name,
                          instance_type=args.instance_type,
                          spot=args.spot)

  job.rsync('.')
  job.run('killall python || echo failed && '  # kill previous run
          'source activate pytorch_p36 && ' +
          'pip install -r requirements.txt && ' +
          # workaround for https://github.com/tensorflow/models/issues/3995
          'pip install -U protobuf')

  # Training script args
  default_params = [
    '--logdir', job.logdir,
    '--distributed',
  ]

  num_gpus = ncluster.aws_backend.INSTANCE_INFO[args.instance_type]['gpus']

  # todo(y): consistency with - and _ in args
  # Based on run_wt103_base.sh
  bs = args.batch_size

  training_params = [
    '--seed', 1111,
    '--data', '/ncluster/data/transformer-xl-data/wikitext-103',
    '--dataset', 'wt103',
    '--dist-backend', 'nccl',
    '--adaptive',
    '--log-interval', 100,
    '--n_layer', 18,
    '--d_model', 1024,
    '--n_head', 16,
    '--d_head', 64,
    '--d_inner', 4096,
    '--dropout', 0.2,
    '--dropatt', 0.2,
    '--optim', 'adam',
    '--lr', .00025 * num_gpus / 32,
    '--warmup_tokens', int(3e6), #from 3e7
    '--max_tokens', int(6e9), #from 1.8e9 (18 epochs)
    '--tgt_len', 384,
    '--mem_len', 384,
    '--eval_tgt_len', 128,
    '--num_gpu', num_gpus,
    '--batch_size', bs,  # per-gpu batch size
    #'--scheduler', 'finder', # Use max_tokens 2e7 and log-interval 10
  ]

  if not args.no_fp16:
    training_params.append('--fp16')
    training_params.append('--dynamic_loss_scale')

  print(training_params)


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
  args = parser.parse_args()
  main(args)
