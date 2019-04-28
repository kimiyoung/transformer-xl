#!/usr/bin/env python
# launch training locally with 4 GPU training

import argparse
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl-1',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default="p3.2xlarge",
                    help="how many machines to use")
parser.add_argument('--image_name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 22.0',
                    help="name of AMI to use ")
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
  job.run('killall python || echo failed && '  # kill previous run
          'source activate pytorch_p36 && ' +
          'pip install -r requirements.txt && ' +
          # workaround for https://github.com/tensorflow/models/issues/3995
          'pip install -U protobuf')


if __name__ == '__main__':
  main()
