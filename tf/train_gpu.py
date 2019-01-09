from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

import tensorflow as tf
import model
import data_utils

from gpu_utils import assign_to_gpu, average_grads_and_vars

import numpy as np

# GPU config
flags.DEFINE_integer("num_hosts", default=1,
      help="Number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="Number of cores per host")

# Experiment (data/checkpoint/directory) config
flags.DEFINE_string("data_dir", default="",
      help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir", default="",
      help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="",
      help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default=None,
      help="Estimator model_dir.")
flags.DEFINE_bool("do_train", default=True,
      help="Whether to run training.")
flags.DEFINE_bool("do_eval", default=False,
      help="Whether to run eval on the dev set.")
flags.DEFINE_string("eval_ckpt_path", None,
      help="Checkpoint path for do_test evaluation."
           "If set, model_dir will be ignored."
           "If unset, will use the latest ckpt in model_dir.")
flags.DEFINE_string("warm_start_path", None,
      help="Checkpoint path for warm start."
           "If set, will clear Adam states."
           "Note that the new model_dir should be different"
           " from warm_start_path.")

# Optimization config
flags.DEFINE_float("learning_rate", default=2.5e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
      help="Gradient clipping value.")
# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.004,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")

# Training config
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

# Evaluation config
flags.DEFINE_bool("do_test", default=False,
      help="Run on the test set.")
flags.DEFINE_integer("max_eval_batch", default=-1,
      help="Set -1 to turn off. Only used in test mode.")
flags.DEFINE_bool("do_eval_only", default=False,
      help="Run evaluation only.")
flags.DEFINE_integer("start_eval_steps", default=10000,
      help="Which checkpoint to start with in `do_eval_only` mode.")
flags.DEFINE_string("eval_split", "valid",
      help="Which data split to evaluate.")

# Model config
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

def get_model_fn(n_token, cutoffs):
  def model_fn(inp, tgt, mems, is_training):
    inp = tf.transpose(inp, [1, 0])
    tgt = tf.transpose(tgt, [1, 0])

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
        proj_initializer=proj_initializer,
        is_training=is_training,
        mem_len=FLAGS.mem_len,
        cutoffs=cutoffs,
        div_val=FLAGS.div_val,
        tie_projs=tie_projs,
        input_perms=None,
        target_perms=None,
        head_target=None,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        use_tpu=False,
        untie_r=FLAGS.untie_r,
        proj_same_dim=FLAGS.proj_same_dim)

    # number of parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    # format_str = '{{:<{0}s}}\t{{}}'.format(
    #     max([len(v.name) for v in tf.trainable_variables()]))
    # for v in tf.trainable_variables():
    #   tf.logging.info(format_str.format(v.name, v.get_shape()))

    if is_training:
      all_vars = tf.trainable_variables()
      grads = tf.gradients(loss, all_vars)
      grads_and_vars = list(zip(grads, all_vars))

      return loss, new_mems, grads_and_vars
    else:
      return loss, new_mems

  return model_fn


def single_core_graph(n_token, cutoffs, is_training, inp, tgt, mems):
  model_fn = get_model_fn(
      n_token=n_token,
      cutoffs=cutoffs)

  model_ret = model_fn(
      inp=inp,
      tgt=tgt,
      mems=mems,
      is_training=is_training)

  return model_ret


def train(n_token, cutoffs, ps_device):
  ##### Get input function and model function
  train_input_fn, train_record_info = data_utils.get_input_fn(
      record_info_dir=FLAGS.record_info_dir,
      split="train",
      per_host_bsz=FLAGS.train_batch_size,
      tgt_len=FLAGS.tgt_len,
      num_core_per_host=FLAGS.num_core_per_host,
      num_hosts=1,
      use_tpu=False)

  tf.logging.info("num of batches {}".format(train_record_info["num_batch"]))

  ##### Create computational graph
  train_set = train_input_fn({
      "batch_size": FLAGS.train_batch_size,
      "data_dir": FLAGS.data_dir})

  input_feed, label_feed = train_set.make_one_shot_iterator().get_next()

  inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
  labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

  per_core_bsz = FLAGS.train_batch_size // FLAGS.num_core_per_host

  tower_mems, tower_losses, tower_new_mems, tower_grads_and_vars = [], [], [], []

  for i in range(FLAGS.num_core_per_host):
    reuse = True if i > 0 else None
    with tf.device(assign_to_gpu(i, ps_device)), \
        tf.variable_scope(tf.get_variable_scope(), reuse=reuse):

      mems_i = [tf.placeholder(tf.float32,
                               [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                for _ in range(FLAGS.n_layer)]

      loss_i, new_mems_i, grads_and_vars_i = single_core_graph(
          n_token=n_token,
          cutoffs=cutoffs,
          is_training=True,
          inp=inputs[i],
          tgt=labels[i],
          mems=mems_i)

      tower_mems.append(mems_i)
      tower_losses.append(loss_i)
      tower_new_mems.append(new_mems_i)
      tower_grads_and_vars.append(grads_and_vars_i)

  ## average losses and gradients across towers
  if len(tower_losses) > 1:
    loss = tf.add_n(tower_losses) / len(tower_losses)
    grads_and_vars = average_grads_and_vars(tower_grads_and_vars)
  else:
    loss = tower_losses[0]
    grads_and_vars = tower_grads_and_vars[0]
  grads, all_vars = zip(*grads_and_vars)

  ## clip gradient
  clipped, gnorm = tf.clip_by_global_norm(grads, FLAGS.clip)
  grads_and_vars = list(zip(clipped, all_vars))

  ## configure the optimizer
  global_step = tf.train.get_or_create_global_step()

  # warmup stage: increase the learning rate linearly
  if FLAGS.warmup_steps > 0:
    warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                * FLAGS.learning_rate
  else:
    warmup_lr = 0.0

  # decay stage: decay the learning rate using the cosine schedule
  decay_lr = tf.train.cosine_decay(
      FLAGS.learning_rate,
      global_step=global_step-FLAGS.warmup_steps,
      decay_steps=FLAGS.train_steps-FLAGS.warmup_steps,
      alpha=FLAGS.min_lr_ratio)

  # choose warmup or decay
  learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                           warmup_lr, decay_lr)

  # get the train op
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.apply_gradients(grads_and_vars, global_step)

  ##### Training loop
  tower_mems_np = [
      [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
          for layer in range(FLAGS.n_layer)]
      for core in range(FLAGS.num_core_per_host)
  ]

  saver = tf.train.Saver()

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    if FLAGS.warm_start_path is not None:
      tf.logging.info("warm start from {}".format(FLAGS.warm_start_path))
      saver.restore(sess, FLAGS.warm_start_path)

    fetches = [loss, tower_new_mems, global_step, gnorm, learning_rate, train_op]

    total_loss, prev_step = 0., -1
    while True:
      feed_dict = {}
      for i in range(FLAGS.num_core_per_host):
        for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
          feed_dict[m] = m_np

      fetched = sess.run(fetches, feed_dict=feed_dict)

      loss_np, tower_mems_np, curr_step = fetched[:3]
      total_loss += loss_np

      if curr_step > 0 and curr_step % FLAGS.iterations == 0:
        curr_loss = total_loss / (curr_step - prev_step)
        tf.logging.info("[{}] | gnorm {:.2f} lr {:8.6f} "
            "| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
            curr_step, fetched[-3], fetched[-2],
            curr_loss, math.exp(curr_loss), curr_loss / math.log(2)))
        total_loss, prev_step = 0., curr_step

      if curr_step > 0 and curr_step % FLAGS.save_steps == 0:
        save_path = os.path.join(FLAGS.model_dir, "model.ckpt")
        saver.save(sess, save_path)
        tf.logging.info("Model saved in path: {}".format(save_path))

      if curr_step == FLAGS.train_steps:
        break


def evaluate(n_token, cutoffs, ps_device):
  ##### Get input function and model function
  eval_input_fn, eval_record_info = data_utils.get_input_fn(
      record_info_dir=FLAGS.record_info_dir,
      split=FLAGS.eval_split,
      per_host_bsz=FLAGS.eval_batch_size,
      tgt_len=FLAGS.tgt_len,
      num_core_per_host=FLAGS.num_core_per_host,
      num_hosts=1,
      use_tpu=False)

  num_batch = eval_record_info["num_batch"]
  if FLAGS.max_eval_batch > 0:
      num_batch = FLAGS.max_eval_batch
  tf.logging.info("num of batches {}".format(num_batch))

  ##### Create computational graph
  eval_set = eval_input_fn({
      "batch_size": FLAGS.eval_batch_size,
      "data_dir": FLAGS.data_dir})

  input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()

  inputs = tf.split(input_feed, FLAGS.num_core_per_host, 0)
  labels = tf.split(label_feed, FLAGS.num_core_per_host, 0)

  per_core_bsz = FLAGS.eval_batch_size // FLAGS.num_core_per_host
  tower_mems, tower_losses, tower_new_mems = [], [], []

  for i in range(FLAGS.num_core_per_host):
    with tf.device(assign_to_gpu(i, ps_device)), \
        tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):

      mems_i = [tf.placeholder(tf.float32,
                    [FLAGS.mem_len, per_core_bsz, FLAGS.d_model])
                for _ in range(FLAGS.n_layer)]

      loss_i, new_mems_i = single_core_graph(
          n_token=n_token,
          cutoffs=cutoffs,
          is_training=False,
          inp=inputs[i],
          tgt=labels[i],
          mems=mems_i)

      tower_mems.append(mems_i)
      tower_losses.append(loss_i)
      tower_new_mems.append(new_mems_i)

  ## sum losses across towers
  if len(tower_losses) > 1:
    loss = tf.add_n(tower_losses) / len(tower_losses)
  else:
    loss = tower_losses[0]

  ##### Evaluation loop
  tower_mems_np = [
      [np.zeros([FLAGS.mem_len, per_core_bsz, FLAGS.d_model], dtype=np.float32)
          for layer in range(FLAGS.n_layer)]
      for core in range(FLAGS.num_core_per_host)
  ]

  saver = tf.train.Saver()

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    if FLAGS.eval_ckpt_path is None:
      eval_ckpt_path = tf.train.latest_checkpoint(FLAGS.model_dir)
    else:
      eval_ckpt_path = FLAGS.eval_ckpt_path
    tf.logging.info("Evaluate {}".format(eval_ckpt_path))
    saver.restore(sess, eval_ckpt_path)

    fetches = [loss, tower_new_mems, tf.size(label_feed)]

    format_str = "  >> processing batch {{:{0}d}}/{{:{0}d}} ..".format(
        len(str(num_batch)))

    total_loss, total_cnt = 0, 0
    for step in range(num_batch):
      if step % (num_batch // 10) == 0:
        tf.logging.info(format_str.format(step, num_batch))

      feed_dict = {}
      for i in range(FLAGS.num_core_per_host):
        for m, m_np in zip(tower_mems[i], tower_mems_np[i]):
          feed_dict[m] = m_np

      fetched = sess.run(fetches, feed_dict=feed_dict)

      loss_np, tower_mems_np, cnt_np = fetched[:3]
      total_loss += loss_np * cnt_np
      total_cnt += cnt_np

    avg_loss = total_loss / total_cnt
    tf.logging.info("| loss {:.2f} | pplx {:>7.2f}, bpc {:>7.4f}".format(
        avg_loss, math.exp(avg_loss), avg_loss / math.log(2)))


def main(unused_argv):
  del unused_argv  # Unused

  tf.logging.set_verbosity(tf.logging.INFO)

  # Get corpus info
  corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
  n_token = corpus_info["vocab_size"]
  cutoffs = corpus_info["cutoffs"][1:-1]
  tf.logging.info("n_token {}".format(n_token))

  if FLAGS.do_train:
    train(n_token, cutoffs, "/gpu:0")
  if FLAGS.do_eval:
    evaluate(n_token, cutoffs, "/gpu:0")


if __name__ == "__main__":
  tf.app.run()
