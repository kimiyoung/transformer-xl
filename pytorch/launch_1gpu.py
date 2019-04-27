#!/usr/bin/env python
#
# Launch a single GPU instance with jupyter notebook

import argparse
import os
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='gpubox',
                    help="instance name")
parser.add_argument('--image-name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 22.0',
                    help="name of AMI to use ")
parser.add_argument('--instance-type', type=str, default='p3.2xlarge',
                    help="type of instance")

args = parser.parse_args()
module_path = os.path.dirname(os.path.abspath(__file__))

ncluster.set_backend('aws')

def main():
  task = ncluster.make_task(name=args.name,
                            instance_type=args.instance_type,
                            disk_size=1000,
                            image_name=args.image_name)

  task.run('source activate pytorch_p36')
  task.run('pip install -U protobuf')
  task.run('pip install -r requirements.txt')

def _replace_lines(fn, startswith, new_line):
  """Replace lines starting with starts_with in fn with new_line."""
  new_lines = []
  for line in open(fn):
    if line.startswith(startswith):
      new_lines.append(new_line)
    else:
      new_lines.append(line)
  with open(fn, 'w') as f:
    f.write('\n'.join(new_lines))


if __name__ == '__main__':
  main()
