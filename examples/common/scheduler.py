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

from typing import List

import tensorflow as tf

from examples.common.logger import logger


class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """A wrapper for LearningRateSchedule that includes warmup steps."""

    def __init__(
            self,
            lr_schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
            warmup_steps: int):
        """Add warmup decay to a learning rate schedule.

        Args:
          lr_schedule: base learning rate scheduler
          warmup_steps: number of warmup steps

        """
        super(WarmupDecaySchedule, self).__init__()
        self._lr_schedule = lr_schedule
        self._warmup_steps = warmup_steps

    def __call__(self, step: int):
        lr = self._lr_schedule(step)
        if self._warmup_steps:
            initial_learning_rate = tf.convert_to_tensor(
                self._lr_schedule.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            global_step_recomp = tf.cast(step, dtype)
            warmup_steps = tf.cast(self._warmup_steps, dtype)
            warmup_lr = initial_learning_rate * global_step_recomp / warmup_steps
            lr = tf.cond(global_step_recomp < warmup_steps,
                         lambda: warmup_lr,
                         lambda: lr)
        return lr

    def get_config(self):
        config = self._lr_schedule.get_config()
        config.update({
            "warmup_steps": self._warmup_steps,
        })
        return config


# tf.keras.optimizers.schedules.PiecewiseConstantDecay + WarmupDecaySchedule.
class PiecewiseConstantDecayWithWarmup(
    tf.keras.optimizers.schedules.LearningRateSchedule):
    """Piecewise constant decay with warmup schedule."""

    def __init__(self,
                 batch_size: int,
                 epoch_size: int,
                 warmup_epochs: int,
                 boundaries: List[int],
                 multipliers: List[float]):
        """Piecewise constant decay with warmup.

        Args:
          batch_size: The training batch size used in the experiment.
          epoch_size: The size of an epoch, or the number of examples in an epoch.
          warmup_epochs: The number of warmup epochs to apply.
          boundaries: The list of floats with strictly increasing entries.
          multipliers: The list of multipliers/learning rates to use for the
            piecewise portion. The length must be 1 less than that of boundaries.

        """
        super(PiecewiseConstantDecayWithWarmup, self).__init__()
        if len(boundaries) != len(multipliers) - 1:
            raise ValueError("The length of boundaries must be 1 less than the "
                             "length of multipliers")

        steps_per_epoch = epoch_size // batch_size

        self._step_boundaries = [float(steps_per_epoch) * x for x in boundaries]
        self._lr_values = [m for m in multipliers]
        self._warmup_steps = warmup_epochs * steps_per_epoch

    def __call__(self, step: int):
        """Compute learning rate at given step."""

        def warmup_lr():
            return step / tf.cast(self._warmup_steps, tf.float32)

        def piecewise_lr():
            return tf.compat.v1.train.piecewise_constant(
                tf.cast(step, tf.float32), self._step_boundaries, self._lr_values)

        return tf.cond(step < self._warmup_steps, warmup_lr, piecewise_lr)

    def get_config(self):
        return {
            "rescaled_lr": self._rescaled_lr,
            "step_boundaries": self._step_boundaries,
            "lr_values": self._lr_values,
            "warmup_steps": self._warmup_steps,
        }


def build_scheduler(config, epoch_size, batch_size, steps):
    optimizer_config = config.get('optimizer', {})
    schedule_type = optimizer_config.get('schedule_type', 'exponential').lower()
    schedule_params = optimizer_config.get("schedule_params", {})

    warmup_epochs = schedule_params.get('warmup_epochs', 0)

    if schedule_type == 'exponential':
        decay_rate = schedule_params.get('decay_rate', None)
        if decay_rate is None:
            raise ValueError('decay_rate parameter must be specified '
                             'for the exponential scheduler')

        initial_lr = schedule_params.get('initial_lr', None)
        if initial_lr is None:
            raise ValueError('initial_lr parameter must be specified '
                             'for the exponential scheduler')

        decay_epochs = schedule_params.get('decay_epochs', None)
        decay_steps = decay_epochs * steps if decay_epochs is not None else 0

        logger.info('Using exponential learning rate with: '
                    'initial_learning_rate: {initial_lr}, decay_steps: {decay_steps}, '
                    'decay_rate: {decay_rate}'.format(initial_lr=initial_lr,
                                                      decay_steps=decay_steps,
                                                      decay_rate=decay_rate))
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate)
    elif schedule_type == 'piecewise_constant_with_warmup':
        boundaries = schedule_params.get('boundaries', None)
        if boundaries is None:
            raise ValueError('boundaries parameter must be specified '
                             'for the piecewise_constant_with_warmup scheduler')

        multipliers = schedule_params.get('multipliers', None)
        if multipliers is None:
            raise ValueError('multipliers parameter must be specified '
                             'for the piecewise_constant_with_warmup scheduler')

        logger.info('Using Piecewise constant decay with warmup. '
                    'Parameters: batch_size: {batch_size}, epoch_size: {epoch_size}, '
                    'warmup_epochs: {warmup_epochs}, boundaries: {boundaries}, '
                    'multipliers: {multipliers}'.format(batch_size=batch_size,
                                                        epoch_size=epoch_size,
                                                        warmup_epochs=warmup_epochs,
                                                        boundaries=boundaries,
                                                        multipliers=multipliers))
        lr = PiecewiseConstantDecayWithWarmup(
            batch_size=batch_size,
            epoch_size=epoch_size,
            warmup_epochs=warmup_epochs,
            boundaries=boundaries,
            multipliers=multipliers)

    warmup_steps = warmup_epochs * steps if warmup_epochs is not None else 0
    if warmup_steps > 0:
        if schedule_type != 'piecewise_constant_with_warmup':
            logger.info('Applying {} warmup steps to the learning rate'.format(warmup_steps))
            lr = WarmupDecaySchedule(lr, warmup_steps)

    return lr