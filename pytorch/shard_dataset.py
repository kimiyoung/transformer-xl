#!/usr/bin/env python
# shard dataset for transformer-xl codebase
# TODO(y): smarrer sharding to avoid breaking within articles?

import argparse
import math
import os
import sys

parser = argparse.ArgumentParser(description='shard utility')
parser.add_argument('--datadir', type=str,
                    default='/ncluster/data/transformer-xl-data/wikitext-103',
                    help='location of train.txt')
parser.add_argument('--shards', type=int, default=4, help='how many ways to shard')
args = parser.parse_args()


def shard(fn):
    assert os.path.exists(args.datadir), args.target

    if args.shards < 2:
        print(f'args.shards is {args.shards}, doing nothing')
        sys.exit()

    corpus = open(f'{args.datadir}/{fn}').read()
    shard_length = int(math.ceil(len(corpus) / args.shards))

    offset = 0
    for i in range(args.shards):
        new_location = f"{args.datadir}-{i:05d}-of-{args.shards:05d}"
        if os.path.exists(new_location):
            pass
        else:
            os.system(f'mkdir {new_location}')

        shard = corpus[offset:offset + shard_length]
        offset += shard_length
        target = f'{new_location}/{fn}'
        with open(f'{target}', 'w') as f:
            print(f"{target}: {len(shard)}")
            f.write(shard)


def main():
    shard('train.txt')
    shard('valid.txt')
    shard('test.txt')
    
if __name__ == '__main__':
  main()
