"""Tensorflow Example proto decoder for object detection.

A decoder to decode string tensors containing serialized tensorflow.Example
protos for object detection.
"""
import tensorflow as tf


class TfExampleDecoder:
    """Tensorflow Example proto decoder."""

    def __init__(self):
        self._keys_to_features = {
            'image/encoded':
                tf.io.FixedLenFeature((), tf.string),
            'image/source_id':
                tf.io.FixedLenFeature((), tf.string),
            'image/height':
                tf.io.FixedLenFeature((), tf.int64),
            'image/width':
                tf.io.FixedLenFeature((), tf.int64),
            'image/object/bbox/xmin':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                tf.io.VarLenFeature(tf.float32),
            'image/object/class/label':
                tf.io.VarLenFeature(tf.int64),
            'image/object/area':
                tf.io.VarLenFeature(tf.float32),
            'image/object/is_crowd':
                tf.io.VarLenFeature(tf.int64),
        }

    def _decode_image(self, parsed_tensors):
        """Decodes the image and set its static shape."""
        image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def _decode_areas(self, parsed_tensors):
        xmin = parsed_tensors['image/object/bbox/xmin']
        xmax = parsed_tensors['image/object/bbox/xmax']
        ymin = parsed_tensors['image/object/bbox/ymin']
        ymax = parsed_tensors['image/object/bbox/ymax']
        return tf.cond(tf.greater(tf.shape(parsed_tensors['image/object/area'])[0], 0),
                       lambda: parsed_tensors['image/object/area'],
                       lambda: (xmax - xmin) * (ymax - ymin))

    def decode(self, serialized_example):
        """Decode the serialized example.

        Args:
            serialized_example: a single serialized tf.Example string.

        Returns:
          decoded_tensors: a dictionary of tensors with the following fields:
              - image: a uint8 tensor of shape [None, None, 3].
              - source_id: a string scalar tensor.
              - height: an integer scalar tensor.
              - width: an integer scalar tensor.
              - groundtruth_classes: a int64 tensor of shape [None].
              - groundtruth_is_crowd: a bool tensor of shape [None].
              - groundtruth_area: a float32 tensor of shape [None].
              - groundtruth_boxes: a float32 tensor of shape [None, 4].
              - groundtruth_instance_masks: a float32 tensor of shape
                  [None, None, None].
              - groundtruth_instance_masks_png: a string tensor of shape [None].
        """
        parsed_tensors = tf.io.parse_single_example(serialized=serialized_example, features=self._keys_to_features)
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(parsed_tensors[k], default_value='')
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(parsed_tensors[k], default_value=0)

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        areas = self._decode_areas(parsed_tensors)
        is_crowds = tf.cond(tf.greater(tf.shape(parsed_tensors['image/object/is_crowd'])[0], 0),
                            lambda: tf.cast(parsed_tensors['image/object/is_crowd'], tf.bool),
                            lambda: tf.zeros_like(parsed_tensors['image/object/class/label'], dtype=tf.bool))

        decoded_tensors = {
            'image': image,
            'source_id': parsed_tensors['image/source_id'],
            'height': parsed_tensors['image/height'],
            'width': parsed_tensors['image/width'],
            'groundtruth_classes': parsed_tensors['image/object/class/label'],
            'groundtruth_is_crowd': is_crowds,
            'groundtruth_area': areas,
            'groundtruth_boxes': boxes,
        }

        return decoded_tensors
