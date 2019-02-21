from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.gfile import Exists as exists
import model
import data_utils
import tpu_estimator

import numpy as np
from time import sleep


# TPU parameters
flags.DEFINE_string("master", default=None,
                    help="master")
flags.DEFINE_string("tpu", default=None,
      help="The Cloud TPU to use for training. This should be either the name "
      "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
flags.DEFINE_string("gcp_project", default=None,
      help="Project name for the Cloud TPU-enabled project. If not specified, "
      "we will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string("tpu_zone",default=None,
      help="GCE zone where the Cloud TPU is located in. If not specified, we "
      "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_bool("use_tpu", default=True,
      help="Use TPUs rather than plain CPUs.")
flags.DEFINE_integer("num_hosts", default=1,
      help="number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="number of cores per host")

# Experiment (data/checkpoint/directory) parameters
flags.DEFINE_string("data_dir", default="",
      help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir", default="",
      help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="",
      help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_bool("do_eval", default=False,
      help="Whether to run eval on the dev set.")
flags.DEFINE_bool("track_mean", default=True,
      help="Trace mean loss during training.")
flags.DEFINE_string("eval_ckpt_path", None,
      help="Checkpoint path for evaluation."
           "If set, model_dir will be ignored."
           "If unset, will use the latest ckpt in model_dir.")
flags.DEFINE_string("warm_start_path", None,
      help="Checkpoint path for warm start."
           "If set, will clear Adam states."
           "Note that the new model_dir should be different"
           " from warm_start_path.")

# Optimization paramenters
flags.DEFINE_float("learning_rate", default=2.5e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
      help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.01,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")

# Training parameters
flags.DEFINE_integer("train_batch_size", default=60,
      help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=60,
      help="Size of valid batch.")
flags.DEFINE_integer("train_steps", default=100000,
      help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=500,
      help="Number of iterations per repeat loop.")
flags.DEFINE_integer("save_steps", default=10000,
      help="number of steps for model checkpointing.")

# Evaluation parameters
flags.DEFINE_integer("max_eval_batch", default=-1,
      help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_bool("do_eval_only", default=False,
      help="Run evaluation only.")
flags.DEFINE_integer("start_eval_steps", default=10000,
      help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_string("eval_split", "valid",
      help="Which data split to evaluate.")

# Model paramenters
flags.DEFINE_integer("tgt_len", default=70,
      help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=70,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=False,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=-1,
      help="Clamp length")

flags.DEFINE_integer("n_layer", default=6,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=500,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=500,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=10,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=50,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=1000,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.1,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
      help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val", default=1,
      help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
      help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
      help="Project the bin with the same dimension.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
      help="Initialization std for embedding projection.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")


FLAGS = flags.FLAGS

def metric_fn(loss):
  """Evaluation metric Fn which runs on CPU."""
  perplexity = tf.exp(tf.reduce_mean(loss))
  bpc = tf.reduce_mean(loss) / tf.constant(math.log(2))
  return {
      "perplexity": tf.metrics.mean(perplexity),
      "bpc": tf.metrics.mean(bpc),
  }


def get_model_fn(n_token, cutoffs, train_bin_sizes, eval_bin_sizes):
  def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)


    batch_size = params["batch_size"]

    mems = params["cache"]
    inp = tf.transpose(features["inputs"], [1, 0])
    tgt = tf.transpose(features["labels"], [1, 0])

    bin_sizes = train_bin_sizes if is_training else eval_bin_sizes
    if bin_sizes:
      inp_perms = [tf.transpose(features["inp_mask"], [1, 0])]
      tgt_perms = [tf.transpose(features["tgt_mask"], [1, 0])]

      head_tgt = tf.transpose(features["head_labels"], [1, 0])

      for b in range(len(bin_sizes)):
        inp_perm = tf.transpose(features["inp_perm_{}".format(b)], [1, 0, 2])
        tgt_perm = tf.transpose(features["tgt_perm_{}".format(b)], [1, 0, 2])

        inp_perms.append(inp_perm)
        tgt_perms.append(tgt_perm)
    else:
      inp_perms, tgt_perms, head_tgt = None, None, None

    if FLAGS.init == "uniform":
      initializer = tf.initializers.random_uniform(
          minval=-FLAGS.init_range,
          maxval=FLAGS.init_range,
          seed=None)
    elif FLAGS.init == "normal":
      initializer = tf.initializers.random_normal(
          stddev=FLAGS.init_std,
          seed=None)
      proj_initializer = tf.initializers.random_normal(
          stddev=FLAGS.proj_init_std,
          seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if FLAGS.proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    tf.logging.info("Vocab size : {}".format(n_token))
    tf.logging.info("Batch size : {}".format(batch_size))

    loss, new_mems = model.transformer(
        dec_inp=inp,
        target=tgt,
        mems=mems,
        n_token=n_token,
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=FLAGS.dropout,
        dropatt=FLAGS.dropatt,
        initializer=initializer,
        is_training=is_training,
        mem_len=FLAGS.mem_len,
        cutoffs=cutoffs,
        div_val=FLAGS.div_val,
        tie_projs=tie_projs,
        input_perms=inp_perms,
        target_perms=tgt_perms,
        head_target=head_tgt,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        use_tpu=FLAGS.use_tpu,
        untie_r=FLAGS.untie_r,
        proj_same_dim=FLAGS.proj_same_dim)

    total_loss = tf.reduce_mean(loss)

    if mode == tf.estimator.ModeKeys.EVAL:
      if FLAGS.use_tpu:
        with tf.colocate_with(total_loss):
          total_loss = tf.contrib.tpu.cross_replica_sum(total_loss) \
                     / FLAGS.num_hosts / FLAGS.num_core_per_host
      metric_loss = tf.tile(tf.reshape(total_loss, [1, 1]), [batch_size, 1])
      eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=(metric_fn, [metric_loss]))

      eval_spec.cache = new_mems

      return eval_spec

    # Configuring the optimization step.
    global_step = tf.train.get_global_step()

    # increase the learning rate linearly
    if FLAGS.warmup_steps > 0:
      warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                  * FLAGS.learning_rate
    else:
      warmup_lr = 0.0

    # number of parameters
    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: {}".format(num_params))

    # format_str = '{{:<{0}s}}\t{{}}'.format(
    #     max([len(v.name) for v in tf.trainable_variables()]))
    # for v in tf.trainable_variables():
    #   tf.logging.info(format_str.format(v.name, v.get_shape()))


    # decay the learning rate using the cosine schedule
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step-FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps-FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)

    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)

    if FLAGS.use_tpu:
      optimizer = tf.contrib.tpu.CrossShardOptimizer(
          tf.train.AdamOptimizer(learning_rate=learning_rate))
      #GradientDescentOptimizer
    else:
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grads_and_vars = optimizer.compute_gradients(total_loss)
    gradients, variables = zip(*grads_and_vars)
    clipped, _ = tf.clip_by_global_norm(gradients, FLAGS.clip)
    train_op = optimizer.apply_gradients(
        zip(clipped, variables), global_step=tf.train.get_global_step())

    # Constucting TPUEstimatorSpec with cache.
    train_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op)

    if FLAGS.mem_len < FLAGS.tgt_len:
      new_mems = [new_mems[: FLAGS.mem_len] for mem_t in new_mems]
    train_spec.cache = new_mems

    return train_spec

  return model_fn


def get_cache_fn(mem_len):

  def cache_fn(batch_size):
    mems = []
    for l in xrange(FLAGS.n_layer):
      if mem_len > 0:
        mems.append(
          tf.zeros([mem_len, batch_size, FLAGS.d_model], dtype=tf.float32))
      else:
        mems.append(tf.zeros([mem_len], dtype=tf.float32))

    return mems

  return cache_fn


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  # Get corpus info
  corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
  n_token = corpus_info["vocab_size"]
  cutoffs = corpus_info["cutoffs"][1:-1]

  if FLAGS.save_steps == 0:
    FLAGS.save_steps = None

  if not FLAGS.do_eval_only:
    # Get train input function
    train_input_fn, train_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split="train",
        per_host_bsz=FLAGS.train_batch_size // FLAGS.num_hosts,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=FLAGS.num_hosts,
        use_tpu=FLAGS.use_tpu)
    train_bin_sizes = train_record_info["bin_sizes"]
    num_train_batch = train_record_info["num_batch"]

    # Get train cache function
    train_cache_fn = get_cache_fn(FLAGS.mem_len)
  else:
    train_bin_sizes = []
    num_train_batch = None
    train_cache_fn = None

  if FLAGS.do_eval or FLAGS.do_eval_only:
    assert FLAGS.num_hosts == 1
    # Get eval input function
    eval_input_fn, eval_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split=FLAGS.eval_split,
        per_host_bsz=FLAGS.eval_batch_size // FLAGS.num_hosts,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=FLAGS.num_hosts,
        use_tpu=FLAGS.use_tpu)
    eval_bin_sizes = eval_record_info["bin_sizes"]
    num_eval_batch = eval_record_info["num_batch"]

    if FLAGS.max_eval_batch > 0:
      num_eval_batch = min(FLAGS.max_eval_batch, num_eval_batch)

    # Get eval cache function
    eval_cache_fn = get_cache_fn(FLAGS.mem_len)
    model_fn = get_model_fn(n_token, cutoffs, train_bin_sizes, eval_bin_sizes)
  else:
    eval_cache_fn = None
    model_fn = get_model_fn(n_token, cutoffs, train_bin_sizes, [])

  ##### Create estimator
  # TPU Configuration
  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  per_host_input = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations,
          num_shards=FLAGS.num_core_per_host * FLAGS.num_hosts,
          per_host_input_for_training=per_host_input),
      keep_checkpoint_max=100000, # effectively save all checkpoints
      save_checkpoints_secs=None,
      save_checkpoints_steps=FLAGS.save_steps
  )

  # warm start
  warm_start_from = None
  if FLAGS.warm_start_path is not None:
    warm_start_from = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=FLAGS.warm_start_path)

  # TPU Estimator
  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      train_cache_fn=train_cache_fn,
      eval_cache_fn=eval_cache_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params={"data_dir":FLAGS.data_dir, "track_mean":FLAGS.track_mean},
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      warm_start_from=warm_start_from)

  if FLAGS.do_eval_only:
    if FLAGS.eval_ckpt_path is not None:
      ret = estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_batch,
                               checkpoint_path=FLAGS.eval_ckpt_path)
      tf.logging.info("=" * 200)
      log_str = "Eval results | "
      for key, val in ret.items():
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)
      tf.logging.info("=" * 200)
    else:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.model_dir)
      eval_results = []
      for eval_checkpoint in ckpt_state.all_model_checkpoint_paths:
        if not exists(eval_checkpoint + ".index"): continue
        global_step = int(eval_checkpoint.split("-")[-1])
        if global_step < FLAGS.start_eval_steps or global_step > FLAGS.train_steps:
          continue
        ret = estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_batch,
                                 checkpoint_path=eval_checkpoint)
        eval_results.append(ret)

      eval_results.sort(key = lambda x: x["perplexity"])

      tf.logging.info("=" * 200)
      log_str = "Best results | "
      for key, val in eval_results[0].items():
        log_str += "{} {} | ".format(key, val)
      tf.logging.info(log_str)
      tf.logging.info("=" * 200)
  else:
    if not FLAGS.do_eval:
      estimator.train(input_fn=train_input_fn, steps=FLAGS.train_steps)
    else:
      for step in range(0, FLAGS.train_steps, num_train_batch):
        train_steps = min(FLAGS.train_steps - step, num_train_batch)
        estimator.train(input_fn=train_input_fn, steps=train_steps)
        estimator.evaluate(input_fn=eval_input_fn, steps=num_eval_batch)


if __name__ == "__main__":
  tf.app.run()
