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

import functools

import tensorflow as tf
import tensorflow_datasets as tfds

import examples.common.datasets.tfrecords as records_dataset
from examples.common.logger import logger
from examples.common.datasets.augment import create_augmenter
from examples.common.datasets.preprocessing import preprocess_for_train, preprocess_for_eval, get_preprocess_fn
from examples.common.utils import set_hard_limit_num_open_files


class DatasetBuilder:
    def __init__(self, config, num_channels, image_size, num_devices, one_hot, is_train):
        self.config = config

        self._dataset_name = config.get('dataset', 'imagenet2012')
        self._dataset_type = config.get('dataset_type', 'tfrecords')
        self._dataset_dir = config.dataset_dir
        self._num_devices = num_devices
        self._batch_size = config.batch_size
        self._dtype = config.get('dtype', 'float32')
        self._num_preprocess_workers = config.get('workers', tf.data.experimental.AUTOTUNE)

        self._split = 'train' if is_train else 'test'
        self._image_size = image_size
        self._num_channels = num_channels
        self._is_train = is_train
        self._one_hot = one_hot

        self._cache = False
        self._builder = None
        self._skip_decoding = True
        self._shuffle_buffer_size = 10000
        self._deterministic_train = False
        self._use_slack = True

        self._mean_subtract = False
        self._standardize = False

        augmenter_config = self.config.get('augmenter', None)
        if augmenter_config is not None:
            logger.info('Using augmentation: %s', augmenter_config.name)
            self._augmenter = create_augmenter(augmenter_config.name, augmenter_config.get('params', {}))
        else:
            self._augmenter = None

    @property
    def is_train(self):
        return self._is_train

    @property
    def batch_size(self):
        return self._batch_size // self._num_devices

    @property
    def global_batch_size(self):
        return self.batch_size * self._num_devices

    @property
    def num_steps(self):
        return self.num_examples // self.global_batch_size

    @property
    def dtype(self):
        dtype_map = {
            'float32': tf.float32,
            'bfloat16': tf.bfloat16,
            'float16': tf.float16,
            'fp32': tf.float32,
            'bf16': tf.bfloat16,
        }
        try:
            return dtype_map[self._dtype]
        except Exception as exc:
            raise ValueError('Invalid DType provided. Supported types: {}'.format(
                dtype_map.keys())) from exc

    @property
    def num_examples(self):
        if self._dataset_type == 'tfds':
            return self._builder.info.splits[self._split].num_examples
        if self._dataset_type == 'tfrecords':
            return self._builder.num_examples
        return None

    @property
    def num_classes(self):
        if self._dataset_type == 'tfds':
            return self._builder.info.features['label'].num_classes
        if self._dataset_type == 'tfrecords':
            return self._builder.num_classes
        return None

    def build(self):
        builders = {
            'tfds': self.load_tfds,
            'tfrecords': self.load_tfrecords,
        }

        builder = builders.get(self._dataset_type, None)

        if builder is None:
            raise ValueError('Unknown builder type {}'.format(self._builder))

        dataset = builder()
        dataset = self.pipeline(dataset, self.config.model)

        return dataset

    def load_tfds(self):
        logger.info('Using TFDS to load data.')

        set_hard_limit_num_open_files()

        self._builder = tfds.builder(self._dataset_name,
                                     data_dir=self._dataset_dir)

        self._builder.download_and_prepare()

        decoders = {}

        if self._skip_decoding:
            decoders['image'] = tfds.decode.SkipDecoding()

        read_config = tfds.ReadConfig(
            interleave_cycle_length=64,
            interleave_block_length=1)

        dataset = self._builder.as_dataset(
            split=self._split,
            as_supervised=True,
            shuffle_files=True,
            decoders=decoders,
            read_config=read_config)

        return dataset

    def load_tfrecords(self) -> tf.data.Dataset:
        logger.info('Using TFRecords to load data.')

        if self._dataset_name in records_dataset.__dict__:
            self._builder = records_dataset.__dict__[self._dataset_name](
                config=self.config, is_train=self.is_train)
        else:
            raise Exception('Undefined dataset name: {}'.format(self._dataset_name))

        dataset = self._builder.as_dataset()

        return dataset

    def pipeline(self, dataset, model_name):
        if self.is_train and not self._cache:
            dataset = dataset.repeat()

        if self._dataset_type == 'tfrecords':
            dataset = dataset.prefetch(self.global_batch_size)

        if self._cache:
            dataset = dataset.cache()

        if self.is_train:
            dataset = dataset.shuffle(self._shuffle_buffer_size)
            dataset = dataset.repeat()

        if self._dataset_type == 'tfrecords':
            preprocess = lambda record: self.preprocess(model_name, *(self._builder.decoder(record)))
        else:
            preprocess = functools.partial(self.preprocess, model_name)

        dataset = dataset.map(preprocess,
                              num_parallel_calls=self._num_preprocess_workers)

        dataset = dataset.batch(self.global_batch_size, drop_remainder=self.is_train)

        if self.is_train and self._deterministic_train is not None:
            options = tf.data.Options()
            options.experimental_deterministic = self._deterministic_train
            options.experimental_slack = self._use_slack
            options.experimental_optimization.parallel_batch = True
            options.experimental_optimization.map_fusion = True
            options.experimental_optimization.map_vectorization.enabled = True
            options.experimental_optimization.map_parallelization = True
            dataset = dataset.with_options(options)

        dataset = dataset.prefetch(self._num_devices)

        return dataset

    def preprocess(self, model_name: str, image: tf.Tensor, label: tf.Tensor):
        preprocess_fn = get_preprocess_fn(model_name, self._dataset_name)
        if self.is_train:
            image = preprocess_for_train(
                image,
                image_size=self._image_size,
                mean_subtract=self._mean_subtract,
                standardize=self._standardize,
                dtype=self.dtype,
                augmenter=self._augmenter,
                preprocess_fn=preprocess_fn
            )
        else:
            image = preprocess_for_eval(
                image,
                image_size=self._image_size,
                mean_subtract=self._mean_subtract,
                standardize=self._standardize,
                num_channels=self._num_channels,
                dtype=self.dtype,
                preprocess_fn=preprocess_fn
            )

        label = tf.cast(label, tf.int32)
        if self._one_hot:
            label = tf.one_hot(label, self.num_classes) # pylint: disable=E1120
            label = tf.reshape(label, [self.num_classes])

        return image, label
