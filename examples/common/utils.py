"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import time
import datetime
import json
import logging
import os
import tarfile
import resource
from os import path as osp
from pathlib import Path

import tensorflow as tf

from nncf.utils.logger import logger as nncf_logger
from nncf.utils.save import FROZEN_GRAPH_FORMAT, SAVEDMODEL_FORMAT, KERAS_H5_FORMAT
from examples.common.logger import logger as default_logger

GENERAL_LOG_FILE_NAME = "output.log"
NNCF_LOG_FILE_NAME = "nncf_output.log"


def get_name(config):
    dataset = config.get('dataset', 'imagenet')
    if dataset is None:
        dataset = 'imagenet'
    retval = config["model"] + "_" + dataset
    compression_config = config.get('compression', [])
    if not isinstance(compression_config, list):
        compression_config = [compression_config, ]
    for algo_dict in compression_config:
        algo_name = algo_dict["algorithm"]
        if algo_name == "quantization":
            initializer = algo_dict.get("initializer", {})
            precision = initializer.get("precision", {})
            if precision:
                retval += "_mixed_int"
            else:
                activations = algo_dict.get('activations', {})
                a_bits = activations.get('bits', 8)
                weights = algo_dict.get('weights', {})
                w_bits = weights.get('bits', 8)
                if a_bits == w_bits:
                    retval += "_int{}".format(a_bits)
                else:
                    retval += "_a_int{}_w_int{}".format(a_bits, w_bits)
        else:
            retval += "_{}".format(algo_name)
    return retval


def configure_paths(config):
    d = datetime.datetime.now()
    run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
    config.name = get_name(config)
    config.log_dir = osp.join(config.log_dir, "{}/{}".format(config.name, run_id))
    os.makedirs(config.log_dir)

    compression_config = config.get('compression', [])
    if not isinstance(compression_config, list):
        compression_config = [compression_config, ]
    for algo_config in compression_config:
        algo_config.log_dir = config.log_dir

    if config.checkpoint_save_dir is None:
        config.checkpoint_save_dir = config.log_dir

    # create aux dirs
    config.intermediate_checkpoints_path = config.log_dir + '/intermediate_checkpoints'
    os.makedirs(config.intermediate_checkpoints_path)
    os.makedirs(config.checkpoint_save_dir, exist_ok=True)


def configure_logging(sample_logger, config):
    training_pipeline_log_file_handler = logging.FileHandler(osp.join(config.log_dir, GENERAL_LOG_FILE_NAME))
    training_pipeline_log_file_handler.setFormatter(logging.Formatter("%(message)s"))
    sample_logger.addHandler(training_pipeline_log_file_handler)

    nncf_log_file_handler = logging.FileHandler(osp.join(config.log_dir, NNCF_LOG_FILE_NAME))
    nncf_log_file_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    nncf_logger.addHandler(nncf_log_file_handler)


def create_code_snapshot(root, dst_path, extensions=(".py", ".json", ".cpp", ".cu")):
    """Creates tarball with the source code"""
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


def print_args(config, logger=default_logger):
    for arg in sorted(config):
        logger.info("{: <27s}: {}".format(arg, config.get(arg)))


def serialize_config(config, log_dir):
    with open(osp.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f)


def get_saving_parameters(config):
    if config.to_frozen_graph is not None:
        return config.to_frozen_graph, FROZEN_GRAPH_FORMAT
    if config.to_saved_model is not None:
        return config.to_saved_model, SAVEDMODEL_FORMAT
    if config.to_h5 is not None:
        return config.to_h5, KERAS_H5_FORMAT
    save_path = os.path.join(config.log_dir, 'frozen_model.pb')
    return save_path, FROZEN_GRAPH_FORMAT


class BatchTimestamp:
    """A structure to store batch time stamp."""

    def __init__(self, batch_index, timestamp):
        self.batch_index = batch_index
        self.timestamp = timestamp

    def __repr__(self):
        return "'BatchTimestamp<batch_index: {}, timestamp: {}>'".format(
            self.batch_index, self.timestamp)


class TimeHistory(tf.keras.callbacks.Callback):
    """Callback for Keras models."""

    def __init__(self, batch_size, log_steps, logdir=None):
        """Callback for logging performance.

        Args:
          batch_size: Total batch size.
          log_steps: Interval of steps between logging of batch level stats.
          logdir: Optional directory to write TensorBoard summaries.
        """
        # TODO(wcromar): remove this parameter and rely on `logs` parameter of
        # on_train_batch_end()
        self.batch_size = batch_size
        super().__init__()
        self.log_steps = log_steps
        self.last_log_step = 0
        self.steps_before_epoch = 0
        self.steps_in_epoch = 0
        self.start_time = None
        self.train_finish_time = None
        self.epoch_start = None

        if logdir:
            self.summary_writer = tf.summary.create_file_writer(logdir) # pylint: disable=E1101
        else:
            self.summary_writer = None

        # Logs start of step 1 then end of each step based on log_steps interval.
        self.timestamp_log = []

        # Records the time each epoch takes to run from start to finish of epoch.
        self.epoch_runtime_log = []

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    @property
    def average_steps_per_second(self):
        """The average training steps per second across all epochs."""
        return self.global_steps / sum(self.epoch_runtime_log)

    @property
    def average_examples_per_second(self):
        """The average number of training examples per second across all epochs."""
        return self.average_steps_per_second * self.batch_size

    def on_train_end(self, logs=None):
        self.train_finish_time = time.time()

        if self.summary_writer:
            self.summary_writer.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_train_batch_begin(self, batch, logs=None):
        if not self.start_time:
            self.start_time = time.time()

        # Record the timestamp of the first global step
        if not self.timestamp_log:
            self.timestamp_log.append(BatchTimestamp(self.global_steps,
                                                     self.start_time))

    def on_train_batch_end(self, batch, logs=None):
        """Records elapse time of the batch and calculates examples per second."""
        self.steps_in_epoch = batch + 1
        steps_since_last_log = self.global_steps - self.last_log_step
        if steps_since_last_log >= self.log_steps:
            now = time.time()
            elapsed_time = now - self.start_time
            steps_per_second = steps_since_last_log / elapsed_time
            examples_per_second = steps_per_second * self.batch_size

            self.timestamp_log.append(BatchTimestamp(self.global_steps, now))
            logging.info(
                'TimeHistory: %.2f seconds, %.2f examples/second between steps %d '
                'and %d', elapsed_time, examples_per_second, self.last_log_step,
                self.global_steps)

            if self.summary_writer:
                with self.summary_writer.as_default():  # pylint: disable=E1129
                    tf.summary.scalar('global_step/sec', steps_per_second,
                                      self.global_steps)
                    tf.summary.scalar('examples/sec', examples_per_second,
                                      self.global_steps)

            self.last_log_step = self.global_steps
            self.start_time = None

    def on_epoch_end(self, epoch, logs=None):
        epoch_run_time = time.time() - self.epoch_start
        self.epoch_runtime_log.append(epoch_run_time)

        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0


def set_hard_limit_num_open_files():
    _, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


class SummaryWriter:
    """Simple SummaryWriter for writing dictionary of metrics

    Attributes:
        writer: tf.SummaryWriter
    """

    def __init__(self, log_dir, name):
        """Inits SummaryWriter with paths

        Arguments:
            log_dir: the model folder path
            name: the summary subfolder name
        """
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, name)) # pylint: disable=E1101

    def __call__(self, metrics, step):
        """Write metrics to summary with the given writer

        Args:
            metrics: a dictionary of metrics values
            step: integer. The training step
        """

        with self.writer.as_default(): # pylint: disable=E1129
            for metric_name, value in metrics.items():
                tf.summary.scalar(metric_name, value, step=step)
        self.writer.flush()

    def close(self):
        self.writer.close()
