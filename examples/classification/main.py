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
from pathlib import Path

import tensorflow as tf

from examples.common.logger import logger
from examples.common.distributed import get_distribution_strategy, get_strategy_scope
from examples.common.utils import serialize_config, create_code_snapshot, configure_paths, get_saving_parameters
from examples.common.argparser import get_common_argument_parser
from examples.classification.test_models import get_KerasLayer_model
from examples.classification.test_models import get_model
from examples.classification.test_models import ModelType
from examples.common.optimizer import build_optimizer
from examples.common.scheduler import build_scheduler
from examples.common.datasets.builder import DatasetBuilder
from examples.common.callbacks import get_callbacks
from nncf import create_compressed_model
from nncf.configs.config import Config
from nncf import create_compression_callbacks
import tensorflow_hub as hub


SAVE_MODEL_WORKAROUND = False


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
    parser.add_argument(
        "--model_type",
        choices=[ModelType.KerasLayer, ModelType.FuncModel, ModelType.SubClassModel],
        default=ModelType.KerasLayer,
        help="Type of mobilenetV2 model which should be quantized.")
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
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name='acc@1'),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='acc@5')
    ]


def get_loss(one_hot=True):
    if one_hot:
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    return tf.keras.losses.SparseCategoricalCrossentropy()


def load_checkpoint(model, ckpt_path):
    logger.info('Load from checkpoint is enabled.')
    if tf.io.gfile.isdir(ckpt_path):
        checkpoint = tf.train.latest_checkpoint(ckpt_path)
        logger.info('Latest checkpoint: {}'.format(checkpoint))
    else:
        checkpoint = ckpt_path if tf.io.gfile.exists(ckpt_path + '.index') else None
        logger.info('Provided checkpoint: {}'.format(checkpoint))

    if not checkpoint:
        logger.info('No checkpoint detected.')
        return 0

    logger.info('Checkpoint file {} found and restoring from checkpoint'
                .format(checkpoint))
    model.load_weights(checkpoint).expect_partial()
    logger.info('Completed loading from checkpoint.')
    return None


def resume_from_checkpoint(model, ckpt_path, train_steps):
    if load_checkpoint(model, ckpt_path) == 0:
        return 0
    initial_epoch = model.optimizer.iterations // train_steps
    logger.info('Resuming from epoch %d', initial_epoch)
    return int(initial_epoch)


def train_test_export(config):
    strategy = get_distribution_strategy(config)
    strategy_scope = get_strategy_scope(strategy)


    builders = get_dataset_builders(config, strategy)
    datasets = [builder.build() for builder in builders]

    train_builder, validation_builder = builders
    train_dataset, validation_dataset = datasets

    train_epochs = config.epochs
    train_steps = train_builder.num_steps
    validation_steps = validation_builder.num_steps

    if config.model_type == ModelType.KerasLayer:
        args = get_KerasLayer_model()
    else:
        args = None

    with strategy_scope:
        from op_insertion import NNCFWrapperCustom
        if not args:
            args = get_model(config.model_type)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            NNCFWrapperCustom(*args)
        ])
        if SAVE_MODEL_WORKAROUND:
            path = '/tmp/model.pb'
            model.save(path, save_format='tf')
            model = tf.keras.models.load_model(path)


        compression_ctrl, compress_model = create_compressed_model(model, config)
        compression_callbacks = create_compression_callbacks(compression_ctrl, config.log_dir)

        scheduler = build_scheduler(
            config=config,
            epoch_size=train_builder.num_examples,
            batch_size=train_builder.global_batch_size,
            steps=train_steps)
        config['optimizer'] = {'type': 'sgd'}
        optimizer = build_optimizer(
            config=config,
            scheduler=scheduler)

        metrics = get_metrics()
        loss_obj = get_loss()

        compress_model.compile(optimizer=optimizer,
                               loss=loss_obj,
                               metrics=metrics,
                               run_eagerly=config.get('eager_mode', False))

        compress_model.summary()

        logger.info('initialization...')
        compression_ctrl.initialize(dataset=train_dataset)

        initial_epoch = 0
        if config.ckpt_path is not None:
            initial_epoch = resume_from_checkpoint(model=compress_model,
                                                   ckpt_path=config.ckpt_path,
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

    if 'train' in config.mode:
        logger.info('training...')
        compress_model.fit(
            train_dataset,
            epochs=train_epochs,
            steps_per_epoch=train_steps,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            **validation_kwargs)

    logger.info('evaluation...')
    compress_model.evaluate(
        validation_dataset,
        steps=validation_steps,
        verbose=1)

    if 'export' in config.mode:
        save_path, save_format = get_saving_parameters(config)
        compression_ctrl.export_model(save_path, save_format)
        logger.info("Saved to {}".format(save_path))


def export(config):
    raise NotImplementedError('Experemental code, please use train + export mode, '
                              'don\'t use only export mode')
    model = tf.keras.Sequential(
        hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4",
                       trainable=True))
    model.build([None, 224, 224, 3])

    compression_ctrl, compress_model = create_compressed_model(model, config)

    metrics = get_metrics()
    loss_obj = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

    compress_model.compile(loss=loss_obj,
                           metrics=metrics)
    compress_model.summary()

    if config.ckpt_path is not None:
        load_checkpoint(model=compress_model,
                        ckpt_path=config.ckpt_path)

    save_path, save_format = get_saving_parameters(config)
    compression_ctrl.export_model(save_path, save_format)
    logger.info("Saved to {}".format(save_path))


def main(argv):
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)

    #config['eager_mode'] = True
    serialize_config(config, config.log_dir)
    print('*'*50)
    print(f'Using model type: {config.model_type}')
    print('*'*50)
    nncf_root = Path(__file__).absolute().parents[2]
    create_code_snapshot(nncf_root, osp.join(config.log_dir, "snapshot.tar.gz"))
    if 'train' in config.mode or 'test' in config.mode:
        train_test_export(config)
    elif 'export' in config.mode:
        export(config)


if __name__ == '__main__':
    main(sys.argv[1:])
