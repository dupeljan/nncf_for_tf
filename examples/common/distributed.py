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

DISABLE_DISTR = False

def get_distribution_strategy(config):
    if not DISABLE_DISTR:
        if config.get('cpu_only', False):
            return tf.distribute.OneDeviceStrategy('device:CPU:0')

        gpu_id = config.get('gpu_id', None)
        if gpu_id is not None:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            gpu_name = "device:GPU:{}".format(gpu_id)
            gpu = [gpu for gpu in gpus if gpu.name.endswith(gpu_name)][0]
            tf.config.experimental.set_visible_devices(gpu, 'GPU')
            return tf.distribute.OneDeviceStrategy(gpu_name)

        devices = config.get('devices', None)
        if devices:
            gpus = [gpu for gpu in tf.config.experimental.list_physical_devices('GPU')
                    if any(gpu.name.lower().endswith(device.lower()[-5:]) for device in devices)]
            tf.config.experimental.set_visible_devices(gpus, 'GPU')
            return tf.distribute.MirroredStrategy()

        distributed = config.get('distributed', False)
        if distributed:
            return tf.distribute.MirroredStrategy()

    return None


def get_strategy_scope(strategy):
    return strategy.scope() if strategy else DummyContextManager()


class DummyContextManager:
    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass
