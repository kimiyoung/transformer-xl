#!/usr/bin/env python
# shard dataset for transformer-xl codebase
# TODO(y): smarrer sharding to avoid breaking within article?

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser(description='shard utility')
parser.add_argument('--target', type=str,
                    default='/ncluster/data/transformer-xl-data/wikitext-103/train.txt',
                    help='location of train.txt')
parser.add_argument('--shards', type=int, default=4, help='how many ways to shard')
args = parser.parse_args()


def main():
    assert os.path.exists(args.target), args.target
    assert args.target.endswith('train.txt')

    if args.shards < 2:
        print(f'args.shards is {args.shards}, doing nothing')
        sys.exit()

    corpus = open(args.target).read()
    shard_length = int(math.ceil(len(corpus) / args.shards))

    offset = 0
    for i in range(args.shards):
        original_location = os.path.dirname(args.target)
        new_location = f"{original_location}-{i:05d}-of-{args.shards:05d}"
        if os.path.exists(new_location):
            print(f"{new_location} exists, deleting")
            #            os.system(f'rm -Rf {new_location}')

        os.system(f'mkdir {new_location}')
        shard = corpus[offset:offset + shard_length]
        offset += shard_length
        with open(new_location + '/train.txt', 'w') as f:
            print(f"{new_location}: {len(shard)}")
            f.write(shard)


if __name__ == '__main__':
  main()
