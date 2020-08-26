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

import tensorflow as tf


class UpdateMask(tf.keras.callbacks.Callback):
    def __init__(self, scheduler):
        super().__init__()
        self._scheduler = scheduler

    def on_train_batch_begin(self, batch, logs=None):
        self._scheduler.step()
        super().on_train_batch_begin(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self._scheduler.epoch_step()
        super().on_epoch_end(epoch, logs)


class SparsityStatistics(tf.keras.callbacks.TensorBoard):
    def __init__(self, statistics_fn, log_dir, update_freq='epoch', **kwargs):
        super().__init__(
            log_dir=log_dir, update_freq=update_freq, **kwargs)
        self._statistics_fn = statistics_fn

    def _log_sparsity_statistics(self, logs, step):
        log_dir = self.log_dir + '/metrics'

        # pylint: disable=no-member
        file_writer = tf.summary.create_file_writer(log_dir)
        file_writer.set_as_default()

        for name, value in logs.items():
            tf.summary.scalar(name, value, step=step)

        file_writer.flush()

    def on_epoch_begin(self, epoch, logs=None):
        if logs is not None:
            super().on_epoch_begin(epoch, logs)

        sparsity_statistics = self._statistics_fn()

        iteration = self.model.optimizer.iterations.numpy()
        self._log_sparsity_statistics(sparsity_statistics, iteration)
