import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np

from nncf import create_compressed_model
from nncf.configs.config import Config
from examples.object_detection.dataloader import input_reader
from examples.object_detection.dataloader import mode_keys as ModeKeys
from examples.object_detection.modeling import retinanet_model
from examples.common.logger import logger
from examples.common.utils import serialize_config, create_code_snapshot, configure_paths, get_saving_parameters
from examples.common.argparser import get_common_argument_parser
from examples.common.distributed import get_distribution_strategy, get_strategy_scope
from examples.common.optimizer import build_optimizer
from examples.common.scheduler import build_scheduler
from examples.common.utils import SummaryWriter


def get_argument_parser():
    parser = get_common_argument_parser()
    parser.add_argument('--test-every-n-epochs', default=1, type=int,
                        help='Enables running validation every given number of epochs')
    return parser


def get_config_from_argv(argv, parser):
    args = parser.parse_args(args=argv)

    config = Config.from_json(args.config)
    config.update_from_args(args, parser)
    configure_paths(config)
    return config


def load_checkpoint(checkpoint, ckpt_path):
    logger.info('Load from checkpoint is enabled')
    if tf.io.gfile.isdir(ckpt_path):
        path_to_checkpoint = tf.train.latest_checkpoint(ckpt_path)
        logger.info('Latest checkpoint: {}'.format(path_to_checkpoint))
    else:
        path_to_checkpoint = ckpt_path if tf.io.gfile.exists(ckpt_path + '.index') else None
        logger.info('Provided checkpoint: {}'.format(path_to_checkpoint))

    if not path_to_checkpoint:
        logger.info('No checkpoint detected')
        return 0

    logger.info('Checkpoint file {} found and restoring from checkpoint'.format(path_to_checkpoint))
    status = checkpoint.restore(path_to_checkpoint)
    status.expect_partial()
    logger.info('Completed loading from checkpoint')

    return None


def resume_from_checkpoint(checkpoint_manager, ckpt_path, steps_per_epoch):
    if load_checkpoint(checkpoint_manager.checkpoint, ckpt_path) == 0:
        return 0
    optimizer = checkpoint_manager.checkpoint.optimizer
    initial_epoch = optimizer.iterations.numpy() // steps_per_epoch
    logger.info('Resuming from epoch {}'.format(initial_epoch))
    return int(initial_epoch)


def create_test_step_fn(strategy, model, predict_post_process_fn):
    """Creates a distributed test step"""

    def _test_step_fn(inputs):
        inputs, labels = inputs
        model_outputs = model(inputs, training=False)
        if predict_post_process_fn:
            labels, prediction_outputs = predict_post_process_fn(labels, model_outputs)

        return labels, prediction_outputs

    @tf.function
    def test_step(dataset_inputs):
        labels, outputs = strategy.run(_test_step_fn, args=(dataset_inputs,))
        outputs = tf.nest.map_structure(strategy.experimental_local_results, outputs)
        labels = tf.nest.map_structure(strategy.experimental_local_results, labels)

        return labels, outputs

    return test_step


def create_train_step_fn(strategy, model, loss_fn, optimizer):
    """Creates a distributed training step"""

    def _train_step_fn(inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            all_losses = loss_fn(labels, outputs)
            losses = {}
            for k, v in all_losses.items():
                losses[k] = tf.reduce_mean(v)
            per_replica_loss = losses['total_loss'] / strategy.num_replicas_in_sync

        grads = tape.gradient(per_replica_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return losses

    @tf.function
    def train_step(dataset_inputs):
        per_replica_losses = strategy.run(_train_step_fn, args=(dataset_inputs,))
        losses = tf.nest.map_structure(lambda x: strategy.reduce(tf.distribute.ReduceOp.MEAN, x, axis=None),
                                       per_replica_losses)
        return losses

    return train_step


def train(train_step, test_step, eval_metric, train_dist_dataset, test_dist_dataset, initial_epoch, epochs,
          steps_per_epoch, checkpoint_manager, compression_ctrl, log_dir, optimizer, save_checkpoint_freq=200):

    train_summary_writer = SummaryWriter(log_dir, 'eval_train')
    test_summary_writer = SummaryWriter(log_dir, 'eval_test')

    logger.info('Training started')
    for epoch in range(initial_epoch, epochs):
        logger.info('Epoch {}/{}'.format(epoch, epochs))

        statistics = compression_ctrl.statistics()
        train_summary_writer(metrics=statistics, step=optimizer.iterations.numpy())
        logger.info('Compression statistics = {}'.format(statistics))

        for step, x in enumerate(train_dist_dataset):
            if step == steps_per_epoch:
                save_path = checkpoint_manager.save()
                logger.info('Saved checkpoint for step epoch={} step={}: {}'.format(epoch, step, save_path))
                break

            train_loss = train_step(x)
            train_metric_result = tf.nest.map_structure(lambda s: s.numpy().astype(float), train_loss)

            if np.isnan(train_metric_result['total_loss']):
                raise ValueError('total loss is NaN')

            train_metric_result.update({'learning_rate': optimizer.lr(optimizer.iterations).numpy()})

            train_summary_writer(metrics=train_metric_result, step=optimizer.iterations.numpy())

            if step % 100 == 0:
                logger.info('Step {}/{}'.format(step, steps_per_epoch))
                logger.info('Training metric = {}'.format(train_metric_result))

            if step % save_checkpoint_freq == 0:
                save_path = checkpoint_manager.save()
                logger.info("Saved checkpoint for step epoch={} step={}: {}".format(epoch, step, save_path))

        compression_ctrl.scheduler.epoch_step(epoch)

        logger.info('Evaluation...')
        test_metric_result = evaluate(test_step, eval_metric, test_dist_dataset)
        test_summary_writer(metrics=test_metric_result, step=optimizer.iterations.numpy())
        eval_metric.reset_states()
        logger.info('Validation metric = {}'.format(test_metric_result))

    train_summary_writer.close()
    test_summary_writer.close()


def evaluate(test_step, metric, test_dist_dataset):
    """Runs evaluation steps and aggregate metrics"""
    for x in test_dist_dataset:
        labels, outputs = test_step(x)
        metric.update_state(labels, outputs)

    return metric.result()


def train_test_export(config):
    strategy = get_distribution_strategy(config)
    strategy_scope = get_strategy_scope(strategy)

    # Training parameters
    NUM_EXAMPLES_TRAIN = 118287
    NUM_EXAMPLES_EVAL = 5000
    epochs = config.epochs
    batch_size = config.batch_size # per replica batch size
    num_devices = strategy.num_replicas_in_sync if strategy else 1
    global_batch_size = batch_size * num_devices
    steps_per_epoch = NUM_EXAMPLES_TRAIN // global_batch_size

    # Create Dataset
    train_input_fn = input_reader.InputFn(file_pattern=config.train_file_pattern,
                                          params=config,
                                          mode=input_reader.ModeKeys.TRAIN,
                                          batch_size=global_batch_size)

    eval_input_fn = input_reader.InputFn(file_pattern=config.eval_file_pattern,
                                         params=config,
                                         mode=input_reader.ModeKeys.PREDICT_WITH_GT,
                                         batch_size=global_batch_size,
                                         num_examples=NUM_EXAMPLES_EVAL)

    train_dist_dataset = strategy.experimental_distribute_dataset(train_input_fn())
    test_dist_dataset = strategy.experimental_distribute_dataset(eval_input_fn())

    # Create model builder
    mode = ModeKeys.TRAIN if 'train' in config.mode else ModeKeys.PREDICT_WITH_GT
    model_builder = retinanet_model.RetinanetModel(config)
    eval_metric = model_builder.eval_metrics

    with strategy_scope:
        model = model_builder.build_model(pretrained=config.get('pretrained', True),
                                          weights=config.get('weights', None),
                                          mode=mode)

        compression_ctrl, compress_model = create_compressed_model(model, config)
        # compression_callbacks = create_compression_callbacks(compression_ctrl, config.log_dir)

        scheduler = build_scheduler(
            config=config,
            epoch_size=NUM_EXAMPLES_TRAIN,
            batch_size=global_batch_size,
            steps=steps_per_epoch)

        optimizer = build_optimizer(
            config=config,
            scheduler=scheduler)

        eval_metric = model_builder.eval_metrics()
        loss_fn = model_builder.build_loss_fn()
        predict_post_process_fn = model_builder.post_processing

        checkpoint = tf.train.Checkpoint(model=compress_model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, config.checkpoint_save_dir, max_to_keep=None)

        logger.info('initialization...')
        compression_ctrl.initialize(dataset=train_input_fn())

        initial_epoch = 0
        if config.ckpt_path:
            initial_epoch = resume_from_checkpoint(checkpoint_manager, config.ckpt_path, steps_per_epoch)

    train_step = create_train_step_fn(strategy, compress_model, loss_fn, optimizer)
    test_step = create_test_step_fn(strategy, compress_model, predict_post_process_fn)

    if 'train' in config.mode:
        logger.info('Training...')
        train(train_step, test_step, eval_metric, train_dist_dataset, test_dist_dataset, initial_epoch,
              epochs, steps_per_epoch, checkpoint_manager, compression_ctrl, config.log_dir, optimizer)

    logger.info('Evaluation...')
    metric_result = evaluate(test_step, eval_metric, test_dist_dataset)
    logger.info('Validation metric = {}'.format(metric_result))

    if 'export' in config.mode:
        save_path, save_format = get_saving_parameters(config)
        compression_ctrl.export_model(save_path, save_format)
        logger.info("Saved to {}".format(save_path))


def export(config):
    model_builder = retinanet_model.RetinanetModel(config)
    model = model_builder.build_model(pretrained=config.get('pretrained', True),
                                      weights=config.get('weights', None),
                                      mode=ModeKeys.PREDICT_WITH_GT)

    compression_ctrl, compress_model = create_compressed_model(model, config)

    if config.ckpt_path:
        checkpoint = tf.train.Checkpoint(model=compress_model)
        load_checkpoint(checkpoint, config.ckpt_path)

    save_path, save_format = get_saving_parameters(config)
    compression_ctrl.export_model(save_path, save_format)
    logger.info("Saved to {}".format(save_path))


def main(argv):
    parser = get_argument_parser()
    config = get_config_from_argv(argv, parser)

    serialize_config(config, config.log_dir)

    nncf_root = Path(__file__).absolute().parents[2]
    create_code_snapshot(nncf_root, os.path.join(config.log_dir, "snapshot.tar.gz"))

    if 'train' in config.mode or 'test' in config.mode:
        train_test_export(config)
    elif 'export' in config.mode:
        export(config)


if __name__ == '__main__':
    main(sys.argv[1:])
