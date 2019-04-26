# coding: utf-8
#
# To run locally:
#
# download/untar s3://yaroslavvb2/data/txl-wikitext-2.tar to /ncluster/data/wikitext-2, then
#
# python train.py --log-interval=1 --eval-interval=5 --max_step=50 --batch_size=1 --work_dir=/tmp/checkpoints --dataset=wt2 --data=../data/wikitext-2 --n_layer=1 --n_head=1 --d_head=1 --d_model=2 --d_inner=2  --dataset wt2 --max_eval_steps 1 --data=/ncluster/data/wikitext-2 --lr 0.025
#
# Tensorboard results go to /ncluster/runs
#
# To run remotely:
# cp -R /ncluster/data/transformer-xl-data ../data
# bash run_wt103_base.sh train --work_dir ~/workdir
import argparse
import datetime
import itertools
import logging
import math
import os
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import pytz
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import tqdm
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--logdir', type=str, default='/temp/default', help="where logs and events go")
parser.add_argument('--run_name', type=str, default='txl', help="name of run")

parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'wt2'],
                    help='dataset name')
parser.add_argument('--n_layer', type=int, default=12,
                    help='number of total layers')
parser.add_argument('--n_head', type=int, default=10,
                    help='number of heads')
parser.add_argument('--d_head', type=int, default=50,
                    help='head dimension')
parser.add_argument('--d_embed', type=int, default=-1,
                    help='embedding dimension')
parser.add_argument('--d_model', type=int, default=500,
                    help='model dimension')
parser.add_argument('--d_inner', type=int, default=1000,
                    help='inner dimension in FF')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='global dropout rate')
parser.add_argument('--dropatt', type=float, default=0.0,
                    help='attention probability dropout rate')
parser.add_argument('--init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--emb_init', default='normal', type=str,
                    help='parameter initializer to use.')
parser.add_argument('--init_range', type=float, default=0.1,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--emb_init_range', type=float, default=0.01,
                    help='parameters initialized by U(-init_range, init_range)')
parser.add_argument('--init_std', type=float, default=0.02,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--proj_init_std', type=float, default=0.01,
                    help='parameters initialized by N(0, init_std)')
parser.add_argument('--optim', default='adam', type=str,
                    choices=['adam', 'sgd', 'adagrad'],
                    help='optimizer to use.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='initial learning rate (0.00025|5 for adam|sgd)')
parser.add_argument('--mom', type=float, default=0.0,
                    help='momentum for sgd')
parser.add_argument('--scheduler', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                    help='lr scheduler to use.')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5,
                    help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--lr_min', type=float, default=0.0,
                    help='minimum learning rate during annealing')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--clip_nonemb', action='store_true',
                    help='only clip the gradient of non-embedding params')
parser.add_argument('--max_step', type=int, default=100000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=60,
                    help='batch size')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='split batch into chunks to save memory')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--not_tied', action='store_true',
                    help='do not tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
# TODO: remove this since it's automatic now
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--adaptive', action='store_true',
                    help='use adaptive softmax')
parser.add_argument('--div_val', type=int, default=1,
                    help='divident value for adapative input and softmax')
parser.add_argument('--pre_lnorm', action='store_true',
                    help='apply LayerNorm to the input instead of the output')
parser.add_argument('--varlen', action='store_true',
                    help='use variable length')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--eval-interval', type=int, default=4000,
                    help='evaluation interval')
parser.add_argument('--work_dir', default=None, type=str,
                    help='Experiment directory. Defaults to logdir')
parser.add_argument('--restart', action='store_true',
                    help='restart training from the saved checkpoint')
parser.add_argument('--restart_dir', type=str, default='',
                    help='restart dir')
parser.add_argument('--debug', action='store_true',
                    help='run in debug mode (do not create exp dir)')
parser.add_argument('--same_length', action='store_true',
                    help='use the same attn length for all tokens')
parser.add_argument('--attn_type', type=int, default=0,
                    help='attention type. 0 for ours, 1 for Shaw et al,'
                    '2 for Vaswani et al, 3 for Al Rfou et al.')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='use the same pos embeddings after clamp_len')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--gpu0_bsz', type=int, default=-1,
                    help='batch size on gpu 0')
parser.add_argument('--max_eval_steps', type=int, default=-1,
                    help='max eval steps')
parser.add_argument('--sample_softmax', type=int, default=-1,
                    help='number of samples in sampled softmax')
parser.add_argument('--patience', type=int, default=0,
                    help='patience')
parser.add_argument('--finetune_v2', action='store_true',
                    help='finetune v2')
parser.add_argument('--finetune_v3', action='store_true',
                    help='finetune v3')
parser.add_argument('--fp16', action='store_true',
                    help='Run in pseudo-fp16 mode (fp16 storage fp32 math).')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can '
                    'improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument'
                    ' supersedes --static-loss-scale.')

# distributed training flags
parser.add_argument('--distributed', action='store_true', help='Run distributed training. Default True')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--local_rank', default=0, type=int,
                    help='Used for multi-process training. Can either be manually set ' +
                    'or automatically set by using \'python -m multiproc\'.')

# infra flags
parser.add_argument('--skip-auto-shutdown', action='store_true',
                    help='skip shutdown at the end of training or failure')
parser.add_argument('--auto-shutdown-success-delay-mins', default=10, type=int,
                    help='how long to wait until shutting down on success')
parser.add_argument('--auto-shutdown-failure-delay-mins', default=60, type=int,
                    help='how long to wait before shutting down on error')


def env_world_size(): return int(os.environ['WORLD_SIZE'])


def env_rank(): return int(os.environ['RANK'])


def sum_tensor(tensor):
    rt = tensor.clone()
    # TODO(y): fix UserWarning: torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead
    #  warnings.warn("torch.distributed.reduce_op is deprecated, please use "
    # /home/ubuntu/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/torch/distributed/distributed_c10d.py:86: U

    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    return rt

# no_op method/object that accept every signature
class NoOp:
  def __getattr__(self, *args):
    def no_op(*args, **kwargs): pass
    return no_op


args = parser.parse_args()
args.tied = not args.not_tied

# global variables
global_timeit_dict = OrderedDict()
global_example_count = 0
global_token_count = 0
event_writer = NoOp()
logdir = None

# TODO(y): replace print's with log.console

local_rank = args.local_rank
global_rank = env_rank()
max_rank = env_world_size()
torch.cuda.set_device(args.local_rank)



# install pdb handler on error
if global_rank == 0:
    def info(type, value, tb):
        if hasattr(sys, 'ps1') or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            import traceback, pdb
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type, value, tb)
            print()
            # ...then start the debugger in post-mortem mode.
            # pdb.pm() # deprecated
            pdb.post_mortem(tb) # more "modern"

    sys.excepthook = info


class FileLogger:
  def __init__(self, output_dir, is_master=False, is_rank0=False):
    self.output_dir = output_dir

    # only log on one process per node
    if is_rank0:
      self.logger = self.get_logger(output_dir, log_to_file=is_master)
    else:
      self.logger = NoOp()

  def get_logger(self, output_dir, log_to_file=True):
    logger = logging.getLogger('txl training')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')

    if log_to_file:
      vlog = logging.FileHandler(output_dir+'/info.log')
      vlog.setLevel(logging.INFO)
      vlog.setFormatter(formatter)
      logger.addHandler(vlog)

      eventlog = logging.FileHandler(output_dir+'/warn.log')
      eventlog.setLevel(logging.WARN)
      eventlog.setFormatter(formatter)
      logger.addHandler(eventlog)

      time_formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
      debuglog = logging.FileHandler(output_dir+'/debug.log')
      debuglog.setLevel(logging.DEBUG)
      debuglog.setFormatter(time_formatter)
      logger.addHandler(debuglog)
      
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    return logger

  def debug(self, *args):
    self.logger.debug(*args)

  def warn(self, *args):
    self.logger.warn(*args)

  def info(self, *args):
    self.logger.info(*args)


class timeit:
    """Decorator to measure length of time spent in the block in millis and log
  it to TensorBoard."""

    def __init__(self, tag=""):
        self.tag = tag

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        interval_ms = 1000 * (self.end - self.start)
        global_timeit_dict.setdefault(self.tag, []).append(interval_ms)
        newtag = 'times/' + self.tag
        log_tb(newtag, interval_ms)


def log_tb(tag, val):
    """Log value to tensorboard (relies on global_example_count rather than step count to give comparable graphs across
    batch sizes)"""
    global global_token_count, event_writer
    event_writer.add_scalar(tag, val, global_token_count)

PT_TZ = pytz.timezone('America/Los_Angeles')
def current_timestamp() -> str:
    # timestamp format like 2019-04-15_11-29-51
    # correct to local timezone (PDT) if running on AWS (which is UTC)
    localtime = pytz.utc.localize(datetime.datetime.now(), is_dst=None).astimezone(PT_TZ)
    return localtime.strftime('%Y-%m-%d_%H-%M-%S')

    
if args.d_embed < 0:
    args.d_embed = args.d_model

assert args.ext_len >= 0, 'extended context length must be non-negative'
assert args.batch_size % args.batch_chunk == 0

if not args.work_dir:
    args.work_dir = args.logdir
#args.work_dir = '{}-{}'.format(args.work_dir, args.dataset)
#args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
#logging = create_exp_dir(args.work_dir,
#    scripts_to_save=['train.py', 'mem_transformer.py'], debug=args.debug)

is_master = (not args.distributed) or (global_rank==0)
is_rank0 = args.local_rank == 0
logger = FileLogger(args.logdir, is_master=is_master, is_rank0=is_rank0)


# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Validate `--fp16` option
# if args.fp16:
#     if not args.cuda:
#         print('WARNING: --fp16 requires --cuda, ignoring --fp16 option')
#         args.fp16 = False
#     else:
#         try:
#             from apex.fp16_utils import FP16_Optimizer
#         except:
#             print('WARNING: apex not installed, ignoring --fp16 option')
#             args.fp16 = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###############################################################################
# Load data
###############################################################################
corpus = get_lm_corpus(args.data, args.dataset)
ntokens = len(corpus.vocab)
args.n_token = ntokens

eval_batch_size = 10
tr_iter = corpus.get_iterator('train', args.batch_size, args.tgt_len,
    device=device, ext_len=args.ext_len)
va_iter = corpus.get_iterator('valid', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)
te_iter = corpus.get_iterator('test', eval_batch_size, args.eval_tgt_len,
    device=device, ext_len=args.ext_len)

# adaptive softmax / embedding
cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ['wt103', 'lm1b']
    if args.dataset == 'wt103' or args.dataset == 'wt2':
        cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)
    elif args.dataset == 'lm1b':
        cutoffs = [60000, 100000, 640000]
        tie_projs += [False] * len(cutoffs)

###############################################################################
# Build the model
###############################################################################
def init_weight(weight):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)

def init_bias(bias):
    nn.init.constant_(bias, 0.0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)

def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout

def update_dropatt(m):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt

if args.restart:
    with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    if not args.fp16:
        model = model.float()
    model.apply(update_dropout)
    model.apply(update_dropatt)
else:
    model = MemTransformerLM(ntokens, args.n_layer, args.n_head, args.d_model,
        args.d_head, args.d_inner, args.dropout, args.dropatt,
        tie_weight=args.tied, d_embed=args.d_embed, div_val=args.div_val,
        tie_projs=tie_projs, pre_lnorm=args.pre_lnorm, tgt_len=args.tgt_len,
        ext_len=args.ext_len, mem_len=args.mem_len, cutoffs=cutoffs,
        same_length=args.same_length, attn_type=args.attn_type,
        clamp_len=args.clamp_len, sample_softmax=args.sample_softmax)
    model.apply(weights_init)
    model.word_emb.apply(weights_init) # ensure embedding init is not overridden by out_layer in case of weight sharing
args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

if args.fp16:
    model = model.half()
model = model.to(device)

#### optimizer
optimizer_sparse = None
if args.optim.lower() == 'sgd':
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SGD(sparse_params, lr=args.lr * 2)
        optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
            momentum=args.mom)
else:
    assert args.optim.lower() == 'adam'
    if args.sample_softmax > 0:
        dense_params, sparse_params = [], []
        for param in model.parameters():
            if param.size() == model.word_emb.weight.size():
                sparse_params.append(param)
            else:
                dense_params.append(param)
        optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
        optimizer = optim.Adam(dense_params, lr=args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

#### scheduler
if args.scheduler == 'cosine':
    # here we do not set eta_min to lr_min to be backward compatible
    # because in previous versions eta_min is default to 0
    # rather than the default value of lr_min 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
        args.max_step, eta_min=args.eta_min) # should use eta_min arg
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(optimizer_sparse,
            args.max_step, eta_min=args.eta_min) # should use eta_min arg
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step \
                   else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
    if args.sample_softmax > 0:
        scheduler_sparse = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sparse,
            factor=args.decay_rate, patience=args.patience, min_lr=args.lr_min)
elif args.scheduler == 'constant':
    pass

# if args.cuda and args.fp16:
#     # If args.dynamic_loss_scale is False, static_loss_scale will be used.
#     # If args.dynamic_loss_scale is True, it will take precedence over static_loss_scale.
#     optimizer = FP16_Optimizer(optimizer,
#                                static_loss_scale = args.static_loss_scale,
#                                dynamic_loss_scale = args.dynamic_loss_scale,
#                                dynamic_loss_args = {'init_scale': 2 ** 16})

if args.restart:
    if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
        with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
            opt_state_dict = torch.load(f)
            optimizer.load_state_dict(opt_state_dict)
    else:
        print('Optimizer was not saved. Start from scratch.')

# todo(y): move into main()
logger.info("Torch version: {}", str(torch.__version__))
logger.info('=' * 100)
for k, v in args.__dict__.items():
    logger.info('    - {} : {}'.format(k, v))
logger.info('=' * 100)
logger.info('#params = {}'.format(args.n_all_param))
logger.info('#non emb params = {}'.format(args.n_nonemb_param))
#print('model')
#print(model)


###############################################################################
# Training code
###############################################################################

def evaluate(eval_iter):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # If the model does not use memory at all, make the ext_len longer.
    # Otherwise, make the mem_len longer and keep the ext_len the same.
    if args.mem_len == 0:
        model_transformer.reset_length(args.eval_tgt_len,
            args.ext_len+args.tgt_len-args.eval_tgt_len, args.mem_len)
    else:
        model_transformer.reset_length(args.eval_tgt_len,
            args.ext_len, args.mem_len+args.tgt_len-args.eval_tgt_len)

    # Evaluation
    total_len, total_loss = 0, 0.
    with torch.no_grad():
        mems = tuple()
        bar = tqdm.tqdm(eval_iter, leave=False, desc="Eval")
        for i, (data, target, seq_len) in enumerate(bar):
            if args.max_eval_steps > 0 and i >= args.max_eval_steps:
                break
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            bar.set_description(f'Eval loss {loss:.2f}')
            total_loss += seq_len * loss.float().item()
            total_len += seq_len

    # Switch back to the training mode
    model_transformer.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len

def train():
    global global_example_count, global_token_count, event_writer, logdir, train_step, train_loss, best_val_loss, eval_start_time, log_start_time
    # Turn on training mode which enables dropout.
    model.train()

    log_tb('sizes/batch_size', args.batch_size)
    log_tb('sizes/seq_size', args.tgt_len)

    log_tb('sizes/params', args.n_all_param)
    log_tb('sizes/non_emb_params', args.n_nonemb_param)

    # TODO(y): get rid of this (who chunks batches anyway?)
    if args.batch_chunk > 1:
        mems = [tuple() for _ in range(args.batch_chunk)]
    else:
        mems = tuple()
    # TODO(b): fix varlen iter
    #train_iter = tr_iter.get_varlen_iter() if args.varlen else tr_iter
    train_iter = tr_iter.get_dist_iter(global_rank, max_rank)
    for batch, (data, target, seq_len) in enumerate(train_iter):	
        # TODO(y): batch is dimension 1, why?
        assert seq_len == data.shape[0]

        batch_total = torch.tensor(data.shape[1]).to(device)
        batch_total = batch_total.to(device)       # needed for NCCL sync
        batch_total = sum_tensor(batch_total)      # global batch size
        total_tokens = batch_total.item()*seq_len
        
        global_token_count += total_tokens
        model.zero_grad()
        if args.batch_chunk > 1:
            data_chunks = torch.chunk(data, args.batch_chunk, 1)
            target_chunks = torch.chunk(target, args.batch_chunk, 1)
            for i in range(args.batch_chunk):
                data_i = data_chunks[i].contiguous()
                target_i = target_chunks[i].contiguous()
                with timeit('model'):
                    ret = model(data_i, target_i, *mems[i])
                loss, mems[i] = ret[0], ret[1:]
                loss = loss.float().mean().type_as(loss) / args.batch_chunk
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                train_loss += loss.float().item()
        else:
            ret = model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            with timeit('backwards'):
              if args.fp16:
                  optimizer.backward(loss)
              else:
                  loss.backward()
            train_loss += loss.float().item()

        if args.fp16:
            optimizer.clip_master_grads(args.clip)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        # step-wise learning rate annealing
        train_step += 1
        if args.scheduler in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if train_step < args.warmup_step:
                curr_lr = args.lr * train_step / args.warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
                if args.sample_softmax > 0:
                    optimizer_sparse.param_groups[0]['lr'] = curr_lr * 2
            else:
                if args.scheduler == 'cosine':
                    scheduler.step(train_step)
                    if args.sample_softmax > 0:
                        scheduler_sparse.step(train_step)
        elif args.scheduler == 'inv_sqrt':
            scheduler.step(train_step)
            
        log_tb('lr', optimizer.param_groups[0]['lr'])
        
        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                      '| ms/batch {:5.2f} | loss {:5.2f}'.format(
                epoch, train_step, batch+1, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(cur_loss / math.log(2))
            else:
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_loss))
            logger.info(log_str)
            log_tb('loss/epoch', epoch)
            log_tb('loss/loss', cur_loss)
            log_tb('loss/ppl', math.exp(cur_loss))
            log_tb('times/step', 1000*elapsed/args.log_interval)

            time_per_batch = elapsed / args.log_interval
            time_per_sample = time_per_batch / args.batch_size
            time_per_token = time_per_sample / args.tgt_len
            
            log_tb('times/batches_per_sec', 1 / time_per_batch)
            log_tb('times/samples_per_sec', 1 / time_per_sample)
            log_tb('times/tokens_per_sec', 1 / time_per_token)

            if str(device) == 'cuda':
                log_tb("memory/allocated_gb", torch.cuda.memory_allocated() / 1e9)
                log_tb("memory/max_allocated_gb", torch.cuda.max_memory_allocated() / 1e9)
                log_tb("memory/cached_gb", torch.cuda.memory_cached() / 1e9)
                log_tb("memory/max_cached_gb", torch.cuda.max_memory_cached() / 1e9)

            # todo(y): refactor to init loss at the top
            train_loss = 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_loss = evaluate(va_iter)
            if not best_val_loss or val_loss < best_val_loss:
                if not args.debug and is_master:
                    logger.info('Saving checkpoint')
                    with open(os.path.join(args.work_dir, 'model.pt'), 'wb') as f:
                        with timeit('save'):
                            torch.save(model, f)
                    with open(os.path.join(args.work_dir, 'optimizer.pt'), 'wb') as f:
                        torch.save(optimizer.state_dict(), f)
                best_val_loss = val_loss

            logger.info('-' * 100)
            log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                      '| valid loss {:5.2f}'.format(
                train_step // args.eval_interval, train_step,
                (time.time() - eval_start_time), val_loss)
            if args.dataset in ['enwik8', 'text8']:
                log_str += ' | bpc {:9.5f}'.format(val_loss / math.log(2))
            else:
                log_str += ' | valid ppl {:9.3f}'.format(math.exp(val_loss))
            logger.info(log_str)
            logger.info('-' * 100)
            log_tb('loss/val_loss', val_loss)
                   
            eval_start_time = time.time()

        if train_step == args.max_step:
            break



def main():
    global global_example_count, global_token_count, event_writer, logdir, train_step, train_loss, best_val_loss, eval_start_time, log_start_time, epoch, model, model_transformer

    os.system('shutdown -c')  # cancel previous shutdown command
    
    # global global_example_count, global_token_count, event_writer, logdir
    #    logdir = f'{args.logdir_root}/{args.run_name}-{current_timestamp()}'
    logdir = args.logdir
    assert os.path.exists(logdir)
    #    os.system(f'mkdir -p {logdir}')

    #### distributed setup
    is_master = (not args.distributed) or (global_rank==0)
    if is_master:
        event_writer = SummaryWriter(logdir)
    else:
        event_writer = NoOp()

    if args.distributed:
        logger.info(f'Distributed initializing process group with {args.dist_backend}, {args.dist_url}, {env_world_size()}')
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=env_world_size())
        assert(env_world_size() == dist.get_world_size())
        logger.info("Distributed: success (%d/%d)"%(args.local_rank, dist.get_world_size()))

        # save original model before wrapping it into DDP because
        # model.reset_length is not propagated
        model_transformer = model
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank)

    log_tb("first", time.time())

    # Loop over epochs.
    train_step = 0
    train_loss = 0
    best_val_loss = None

    log_start_time = time.time()
    eval_start_time = time.time()

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in itertools.count(start=1):
            train()
            if train_step == args.max_step:
                logger.info('-' * 100)
                logger.info('End of training')
                break
    except KeyboardInterrupt:
        logger.info('-' * 100)
        logger.info('Exiting from training early')

    # Load the best saved model.
    logger.info("Loading checkpoint")
    with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
        with timeit('load'):
            model = torch.load(f, map_location = lambda storage,
                               loc: storage.cuda(args.local_rank))
            model = model.to(device)


    # Run on test data.
    test_loss = evaluate(te_iter)
    logger.info('=' * 100)
    if args.dataset in ['enwik8', 'text8']:
        logger.info('| End of training | test loss {:5.2f} | test bpc {:9.5f}'.format(
            test_loss, test_loss / math.log(2)))
    else:
        logger.info('| End of training | test loss {:5.2f} | test ppl {:9.3f}'.format(
            test_loss, math.exp(test_loss)))
    log_tb('loss/test_loss', test_loss)

    logger.info('=' * 100)


if __name__ == '__main__':
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            main()
        if not args.skip_auto_shutdown:
            os.system(f'sudo shutdown -h -P +{args.auto_shutdown_success_delay_mins}')
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        import traceback
        traceback.print_tb(exc_traceback, file=sys.stdout)
        # TODO(y): console log exception
        # logger.event(e)
        # in case of exception, wait 2 hours before shutting down
        if not args.skip_auto_shutdown:
            os.system(f'sudo shutdown -h -P +{args.auto_shutdown_failure_delay_mins}')
    # todo(y): close/flush all the loggers
