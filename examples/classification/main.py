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

import sys
import os.path as osp
import tensorflow as tf
from pathlib import Path

from examples.common.logger import logger
from examples.common.distributed import get_distribution_strategy, get_strategy_scope
from examples.common.utils import serialize_config, create_code_snapshot, configure_paths
from examples.common.argparser import get_common_argument_parser
from examples.common.model_loader import get_model
from examples.common.optimizer import build_optimizer
from examples.common.scheduler import build_scheduler
from examples.common.datasets.builder import DatasetBuilder
from examples.common.callbacks import get_callbacks
from nncf.configs.config import Config
from nncf.algorithm_selector import create_compression_algorithm_builder


def get_argument_parser():
    parser = get_common_argument_parser()
    parser.add_argument(
        "--dataset",
        help="Dataset to use.",
        choices=["imagenet", "cifar100", "cifar10"],
        default=None
    )
    parser.add_argument('--test-every-n-epochs', default=1, type=int,
                        help='Enables running validation every given number of epochs')
    return parser


def get_config_from_argv(argv, parser):
    args = parser.parse_args(args=argv)

    config = Config.from_json(args.config)
    config.update_from_args(args, parser)
    configure_paths(config)
    return config


def get_dataset_builders(config, strategy, one_hot=True):
    num_devices = strategy.num_replicas_in_sync if strategy else 1
    image_size = config.input_info.sample_size[-2]
    num_channels = 3

    train_builder = DatasetBuilder(
        config,
        num_channels=num_channels,
        image_size=image_size,
        num_devices=num_devices,
        one_hot=one_hot,
        is_train=True)

    val_builder = DatasetBuilder(
        config,
        num_channels=num_channels,
        image_size=image_size,
        num_devices=num_devices,
        one_hot=one_hot,
        is_train=False)

    return [train_builder, val_builder]


def get_metrics(one_hot=True):
    if one_hot:
        return [
            tf.keras.metrics.CategoricalAccuracy(name='acc@1'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc@5')
        ]
    else:
        return [
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc@1'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='acc@5')
        ]


def resume_from_checkpoint(model, model_dir, train_steps):
    logger.info('Load from checkpoint is enabled.')
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    logger.info('latest_checkpoint: {}'.format(latest_checkpoint))
    if not latest_checkpoint:
        logger.info('No checkpoint detected.')
        return 0

    logger.info('Checkpoint file {} found and restoring from '
                 'checkpoint'.format(latest_checkpoint))
    model.load_weights(latest_checkpoint)
    initial_epoch = model.optimizer.iterations // train_steps
    logger.info('Completed loading from checkpoint.')
    logger.info('Resuming from epoch %d', initial_epoch)
    return int(initial_epoch)


def main(argv):
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)

    serialize_config(config, config.log_dir)

    nncf_root = Path(__file__).absolute().parents[2]
    create_code_snapshot(nncf_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    strategy = get_distribution_strategy(config)
    strategy_scope = get_strategy_scope(strategy)

    model, model_params = get_model(config.model,
                                    pretrained=config.get('pretrained', True))

    compression_algo_builder = create_compression_algorithm_builder(config)

    builders = get_dataset_builders(config, strategy)
    datasets = [builder.build() for builder in builders]

    train_builder, validation_builder = builders
    train_dataset, validation_dataset = datasets

    train_epochs = config.epochs
    train_steps = train_builder.num_steps
    validation_steps = validation_builder.num_steps

    with strategy_scope:
        model = model(**model_params)
        compress_model, compression_callbacks = compression_algo_builder(model)

        scheduler = build_scheduler(
            config=config,
            epoch_size=train_builder.num_examples,
            batch_size=train_builder.global_batch_size,
            steps=train_steps)
        optimizer = build_optimizer(
            config=config,
            scheduler=scheduler)

        metrics = get_metrics()
        loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        compress_model.compile(optimizer=optimizer,
                               loss=loss_obj,
                               metrics=metrics)

        initial_epoch = 0
        if config.get('resume_checkpoint', False):
            initial_epoch = resume_from_checkpoint(model=compress_model,
                                                   model_dir=config.model_dir,
                                                   train_steps=train_steps)

    callbacks = get_callbacks(
        model_checkpoint=True,
        include_tensorboard=True,
        time_history=True,
        track_lr=True,
        write_model_weights=False,
        initial_step=initial_epoch * train_steps,
        batch_size=train_builder.global_batch_size,
        log_steps=100,
        model_dir=config.log_dir)

    callbacks.extend(compression_callbacks)

    validation_kwargs = {
        'validation_data': validation_dataset,
        'validation_steps': validation_steps,
        'validation_freq': 1,
    }

    if config.mode == 'train':
        compress_model.fit(
            train_dataset,
            epochs=train_epochs,
            steps_per_epoch=train_steps,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            **validation_kwargs)

    compress_model.evaluate(
        validation_dataset,
        steps=validation_steps,
        verbose=2)


if __name__ == '__main__':
    main(sys.argv[1:])
