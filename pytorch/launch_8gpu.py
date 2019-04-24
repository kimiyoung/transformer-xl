#!/usr/bin/env python
# launch training locally with 4 GPU training

import argparse
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='txl-1',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=1,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default="p3.16xlarge",
                    help="how many machines to use")
parser.add_argument('--image_name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 22.0',
                    help="name of AMI to use ")
parser.add_argument('--nccl_rings', action='store_true', default=False,
                    help='use special nccl ring setup')
parser.add_argument('--nccl_8rings', action='store_true', default=False,
                    help='use special nccl ring setup')
args = parser.parse_args()

ncluster.set_backend('aws')


def get_nccl_rings(num_tasks, num_gpus):
  ring = build_ring_order(range(num_tasks), range(num_gpus))
  ring_rev = build_ring_order(reversed(range(num_tasks)),
                              reversed(range(num_gpus)))
  rotated_gpu_order = [3, 2, 1, 0, 7, 6, 5, 4]
  skip_gpu_order = get_skip_order(num_gpus)
  if (num_tasks >= 4) and (num_gpus == 8):
    assert ((num_tasks % 4) == 0)
    skip_machine_order = get_skip_order(num_tasks)
    ring_skip = build_ring_order(skip_machine_order, rotated_gpu_order)
    ring_skip_rev = build_ring_order(reversed(skip_machine_order),
                                     skip_gpu_order)
    rings_arr = [ring, ring_rev, ring_skip, ring_skip_rev]
    # rings_arr = [ring, ring_rev, ring_skip]
  else:
    rings_arr = [ring, ring_rev]
  return ' | '.join(rings_arr)


def build_ring_order(machine_order, gpu_order):
  gpu_order = list(gpu_order)
  machine_order = list(machine_order)
  ngpus = len(gpu_order)
  r_order = [(x * ngpus) + y for x in machine_order for y in gpu_order]
  return ' '.join(map(str, r_order))


def get_skip_order(size):
  if size == 4:
    return [0, 2, 1, 3]
  skip_step = 5 if size == 16 else 3
  # step size of 3 yields - [0,3,6,1,4,7,2,5]
  return [(i * skip_step) % size for i in range(size)]


# routines to build NCCL ring orders
def get_nccl_params(num_tasks, num_gpus):
  if num_tasks <= 1 or not args.nccl_rings:
    return 'NCCL_DEBUG=VERSION'
  nccl_rings = get_nccl_rings(num_tasks, num_gpus)
  return f'NCCL_RINGS="{nccl_rings}" NCCL_SINGLE_RING_THRESHOLD=10 NCCL_DEBUG=VERSION'


def main():
  ncluster.set_logdir_root('/ncluster/runs.new')
  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}",
                          num_tasks=args.machines,
                          image_name=args.image_name,
                          instance_type=args.instance_type)

  job.upload('*')
  job.run('killall python || echo failed')  # kill previous run
  job.run('source activate pytorch_p36')
  job.run('export NCCL_SOCKET_IFNAME=ens5')  # tip from cakarak@amazon.com
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
  base_lr = 0.00025 / 4  # from original 4-GPU transformer XL
  # lr = base_lr*8*args.machines  # linear scaling (# of gpus) txl-1.01 fail
  # lr = lr * 0.7  # sqrt scaling txl-1.01 fail
  lr = base_lr * 4  # use same lr as original 4-GPU version txl-1.02
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
    '--lr', lr,
    '--warmup_step', 0,
    '--max_step', 200000,
    '--tgt_len', 150,
    '--mem_len', 150,
    '--eval_tgt_len', 150,
    '--batch_size', 15,  # per-gpu batch size
  ]

  num_gpus = 8
  assert args.instance_type in ['p3.16xlarge', 'p3dn.24xlarge'], f"{args.instance_type} is not 8-gpu"

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
