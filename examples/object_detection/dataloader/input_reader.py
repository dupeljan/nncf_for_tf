"""Data loader and input processing."""

import tensorflow as tf

from examples.object_detection.dataloader import mode_keys as ModeKeys
from examples.object_detection.dataloader import retinanet_parser


class InputFn:
    """Input function that creates dataset from files."""

    def __init__(self, file_pattern, params, mode, batch_size, num_examples=-1):
        """Initialize.

        Args:
          file_pattern: the file pattern for the data example (TFRecords).
          params: the parameter object for constructing example parser and model.
          mode: ModeKeys.TRAIN or ModeKeys.Eval
          batch_size: the data batch size.
          num_examples: If positive, only takes this number of examples and raise
              tf.errors.OutOfRangeError after that. If non-positive, it will be
              ignored.
        """
        assert file_pattern is not None
        assert mode is not None
        assert batch_size is not None
        self._file_pattern = file_pattern
        self._mode = mode
        self._is_training = (mode == ModeKeys.TRAIN)
        self._batch_size = batch_size
        self._num_examples = num_examples
        self._parser_fn = retinanet_parser.Parser(params, mode)
        self._dataset_fn = tf.data.TFRecordDataset

    def __call__(self):
        """Provides tf.data.Dataset object.
        Returns:
            tf.data.Dataset object.
        """

        dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=self._is_training)
        dataset = dataset.cache()

        if self._is_training:
            dataset = dataset.repeat()

        dataset = dataset.interleave(map_func=self._dataset_fn,
                                     cycle_length=32,
                                     num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self._is_training:
            dataset = dataset.shuffle(1000)
        if self._num_examples > 0:
            dataset = dataset.take(self._num_examples)

        # Parses the fetched records to input tensors for model function.
        dataset = dataset.map(self._parser_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(self._batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
