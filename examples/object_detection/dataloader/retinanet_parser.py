"""Data parser and processing.

Parse image and ground truths in a dataset to training targets and package them
into (image, labels) tuple for RetinaNet.

T.-Y. Lin, P. Goyal, R. Girshick, K. He,  and P. Dollar
Focal Loss for Dense Object Detection. arXiv:1708.02002
"""

import tensorflow as tf

from examples.object_detection.dataloader import anchor
from examples.object_detection.dataloader import mode_keys as ModeKeys
from examples.object_detection.dataloader import tf_example_decoder
from examples.object_detection.utils import box_utils
from examples.object_detection.utils import input_utils


def process_source_id(source_id):
    """Processes source_id to the right format."""
    if source_id.dtype == tf.string:
        source_id = tf.cast(tf.strings.to_number(source_id), tf.int32)
    with tf.control_dependencies([source_id]):
        source_id = tf.cond(pred=tf.equal(tf.size(input=source_id), 0),
                            true_fn=lambda: tf.cast(tf.constant(-1), tf.int32),
                            false_fn=lambda: tf.identity(source_id))
    return source_id


def pad_groundtruths_to_fixed_size(gt, n):
    """Pads the first dimension of groundtruths labels to the fixed size."""
    gt['boxes'] = input_utils.pad_to_fixed_size(gt['boxes'], n, -1)
    gt['is_crowds'] = input_utils.pad_to_fixed_size(gt['is_crowds'], n, 0)
    gt['areas'] = input_utils.pad_to_fixed_size(gt['areas'], n, -1)
    gt['classes'] = input_utils.pad_to_fixed_size(gt['classes'], n, -1)
    return gt


class Parser:
    """Parser to parse an image and its annotations into a dictionary of tensors."""

    def __init__(self, config, mode):
        """Initializes parameters for parsing annotations in the dataset.

        Attributes:
          output_size: `Tensor` or `list` for [height, width] of output image. The
              output_size should be divided by the largest feature stride 2^max_level.
          min_level: `int` number of minimum level of the output feature pyramid.
          max_level: `int` number of maximum level of the output feature pyramid.
          num_scales: `int` number representing intermediate scales added on each
              level. For instances, num_scales=2 adds one additional intermediate
              anchor scales [2^0, 2^0.5] on each level.
          aspect_ratios: `list` of float numbers representing the aspect raito
              anchors added on each level. The number indicates the ratio of width to
              height. For instances, aspect_ratios=[1.0, 2.0, 0.5] adds three anchors
              on each scale level.
          anchor_size: `float` number representing the scale of size of the base
              anchor to the feature stride 2^level.
          match_threshold: `float` number between 0 and 1 representing the
              lower-bound threshold to assign positive labels for anchors. An anchor
              with a score over the threshold is labeled positive.
          unmatched_threshold: `float` number between 0 and 1 representing the
              upper-bound threshold to assign negative labels for anchors. An anchor
              with a score below the threshold is labeled negative.
          aug_rand_hflip: `bool`, if True, augment training with random horizontal
              flip.
          aug_scale_min: `float`, the minimum scale applied to `output_size` for
              data augmentation during training.
          aug_scale_max: `float`, the maximum scale applied to `output_size` for
              data augmentation during training.
          skip_crowd_during_training: `bool`, if True, skip annotations labeled with
              `is_crowd` equals to 1.
          max_num_instances: `int` number of maximum number of instances in an
              image. The groundtruth data will be padded to `max_num_instances`.
          mode: a ModeKeys. Specifies if this is training, evaluation, prediction or
              prediction with groundtruths in the outputs.
        """

        self._mode = mode

        self._max_num_instances = config.preprocessing.get('max_num_instances', 100)
        self._skip_crowd_during_training = config.preprocessing.get('skip_crowd_during_training', True)
        self._is_training = (mode == ModeKeys.TRAIN)
        self._example_decoder = tf_example_decoder.TfExampleDecoder()

        # Anchor
        self._output_size = config.preprocessing.output_size
        self._min_level = config.model_params.architecture.min_level
        self._max_level = config.model_params.architecture.max_level
        self._num_scales = config.model_params.anchor.num_scales
        self._aspect_ratios = config.model_params.anchor.aspect_ratios
        self._anchor_size = config.model_params.anchor.anchor_size
        self._match_threshold = config.preprocessing.get('match_threshold', 0.5)
        self._unmatched_threshold = config.preprocessing.get('unmatched_threshold', 0.5)

        # Data augmentation
        self._aug_rand_hflip = config.preprocessing.get('aug_rand_hflip', False)
        self._aug_scale_min = config.preprocessing.get('aug_scale_min', 1.0)
        self._aug_scale_max = config.preprocessing.get('aug_scale_max', 1.0)

        self._use_bfloat16 = False # `bool`, if True, cast output image to tf.bfloat16.

        # Data is parsed depending on the model Modekey.
        if mode == ModeKeys.TRAIN:
            self._parse_fn = self._parse_train_data
        elif mode == ModeKeys.PREDICT_WITH_GT:
            self._parse_fn = self._parse_predict_data
        else:
            raise ValueError('mode is not defined.')

    def __call__(self, value):
        """Parses data to an image and associated training labels.

        Args:
            value: a string tensor holding a serialized tf.Example proto.

        Returns:
            image: image tensor that is preproessed to have normalized value and
                dimension [output_size[0], output_size[1], 3]
            labels:
                cls_targets: ordered dictionary with keys
                    [min_level, min_level+1, ..., max_level]. The values are tensor with
                    shape [height_l, width_l, anchors_per_location]. The height_l and
                    width_l represent the dimension of class logits at l-th level.
                box_targets: ordered dictionary with keys
                    [min_level, min_level+1, ..., max_level]. The values are tensor with
                    shape [height_l, width_l, anchors_per_location * 4]. The height_l and
                    width_l represent the dimension of bounding box regression output at
                    l-th level.
                num_positives: number of positive anchors in the image.
                anchor_boxes: ordered dictionary with keys
                    [min_level, min_level+1, ..., max_level]. The values are tensor with
                    shape [height_l, width_l, 4] representing anchor boxes at each level.
                image_info: a 2D `Tensor` that encodes the information of the image and
                    the applied preprocessing. It is in the format of
                    [[original_height, original_width], [scaled_height, scaled_width],
                    [y_scale, x_scale], [y_offset, x_offset]].
                groundtruths:
                    source_id: source image id. Default value -1 if the source id is empty
                        in the groundtruth annotation.
                    boxes: groundtruth bounding box annotations. The box is represented in
                        [y1, x1, y2, x2] format. The tennsor is padded with -1 to the fixed
                        dimension [self._max_num_instances, 4].
                    classes: groundtruth classes annotations. The tennsor is padded with
                        -1 to the fixed dimension [self._max_num_instances].
                    areas: groundtruth areas annotations. The tennsor is padded with -1
                        to the fixed dimension [self._max_num_instances].
                    is_crowds: groundtruth annotations to indicate if an annotation
                        represents a group of instances by value {0, 1}. The tennsor is
                        padded with 0 to the fixed dimension [self._max_num_instances].
        """

        with tf.name_scope('parser'):
            data = self._example_decoder.decode(value)
            return self._parse_fn(data)

    def _parse_train_data(self, data):
        """Parses data for training and evaluation."""
        classes = data['groundtruth_classes']
        boxes = data['groundtruth_boxes']
        is_crowds = data['groundtruth_is_crowd']

        # Skips annotations with `is_crowd` = True.
        if self._skip_crowd_during_training and self._is_training:
            num_groundtrtuhs = tf.shape(input=classes)[0]
            with tf.control_dependencies([num_groundtrtuhs, is_crowds]):
                indices = tf.cond(pred=tf.greater(tf.size(input=is_crowds), 0),
                                  true_fn=lambda: tf.where(tf.logical_not(is_crowds))[:, 0],
                                  false_fn=lambda: tf.cast(tf.range(num_groundtrtuhs), tf.int64))
            classes = tf.gather(classes, indices, axis=None)
            boxes = tf.gather(boxes, indices, axis=None)

        # Gets original image and its size.
        image = data['image']
        image_shape = tf.shape(input=image)[0:2]

        # Normalizes image with mean and std pixel values.
        image = input_utils.normalize_image(image)

        # Flips image randomly during training.
        if self._aug_rand_hflip:
            image, boxes = input_utils.random_horizontal_flip(image, boxes)

        # Converts boxes from normalized coordinates to pixel coordinates.
        boxes = box_utils.denormalize_boxes(boxes, image_shape)

        # Resizes and crops image.
        image, image_info = input_utils.resize_and_crop_image(
            image,
            self._output_size,
            padded_size=input_utils.compute_padded_size(self._output_size, 2**self._max_level),
            aug_scale_min=self._aug_scale_min,
            aug_scale_max=self._aug_scale_max)
        image_height, image_width, _ = image.get_shape().as_list()

        # Resizes and crops boxes.
        image_scale = image_info[2, :]
        offset = image_info[3, :]
        boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, image_info[1, :], offset)

        # Filters out ground truth boxes that are all zeros.
        indices = box_utils.get_non_empty_box_indices(boxes)
        boxes = tf.gather(boxes, indices, axis=None)
        classes = tf.gather(classes, indices, axis=None)

        # Assigns anchors.
        input_anchor = anchor.Anchor(self._min_level, self._max_level,
                                     self._num_scales, self._aspect_ratios,
                                     self._anchor_size, (image_height, image_width))

        anchor_labeler = anchor.AnchorLabeler(input_anchor, self._match_threshold, self._unmatched_threshold)

        (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(
            boxes, tf.cast(tf.expand_dims(classes, axis=1), tf.float32))

        # If bfloat16 is used, casts input image to tf.bfloat16.
        if self._use_bfloat16:
            image = tf.cast(image, tf.bfloat16)

        # Packs labels for model_fn outputs.
        labels = {
            'cls_targets': cls_targets,
            'box_targets': box_targets,
            'anchor_boxes': input_anchor.multilevel_boxes,
            'num_positives': num_positives,
            'image_info': image_info,
        }

        return image, labels

    def _parse_predict_data(self, data):
        """Parses data for prediction."""
        # Gets original image and its size.
        image = data['image']
        image_shape = tf.shape(input=image)[0:2]

        # Normalizes image with mean and std pixel values.
        image = input_utils.normalize_image(image)

        # Resizes and crops image.
        image, image_info = input_utils.resize_and_crop_image(
            image,
            self._output_size,
            padded_size=input_utils.compute_padded_size(self._output_size, 2**self._max_level),
            aug_scale_min=1.0,
            aug_scale_max=1.0)
        image_height, image_width, _ = image.get_shape().as_list()

        # If bfloat16 is used, casts input image to tf.bfloat16.
        if self._use_bfloat16:
            image = tf.cast(image, tf.bfloat16)

        # Compute Anchor boxes.
        input_anchor = anchor.Anchor(self._min_level, self._max_level,
                                     self._num_scales, self._aspect_ratios,
                                     self._anchor_size, (image_height, image_width))

        labels = {
            'anchor_boxes': input_anchor.multilevel_boxes,
            'image_info': image_info,
        }

        # If mode is PREDICT_WITH_GT, returns groundtruths and training targets
        # in labels.
        if self._mode == ModeKeys.PREDICT_WITH_GT:

            # Converts boxes from normalized coordinates to pixel coordinates.
            boxes = box_utils.denormalize_boxes(data['groundtruth_boxes'], image_shape)
            groundtruths = {
                'source_id': data['source_id'],
                'num_detections': tf.shape(data['groundtruth_classes']),
                'boxes': boxes,
                'classes': data['groundtruth_classes'],
                'areas': data['groundtruth_area'],
                'is_crowds': tf.cast(data['groundtruth_is_crowd'], tf.int32),
            }
            groundtruths['source_id'] = process_source_id(groundtruths['source_id'])
            groundtruths = pad_groundtruths_to_fixed_size(groundtruths, self._max_num_instances)
            labels['groundtruths'] = groundtruths

            # Computes training objective for evaluation loss.
            classes = data['groundtruth_classes']

            image_scale = image_info[2, :]
            offset = image_info[3, :]
            boxes = input_utils.resize_and_crop_boxes(boxes, image_scale, image_info[1, :], offset)

            # Filters out ground truth boxes that are all zeros.
            indices = box_utils.get_non_empty_box_indices(boxes)
            boxes = tf.gather(boxes, indices, axis=None)

            # Assigns anchors.
            anchor_labeler = anchor.AnchorLabeler(input_anchor, self._match_threshold, self._unmatched_threshold)

            (cls_targets, box_targets, num_positives) = anchor_labeler.label_anchors(
                boxes, tf.cast(tf.expand_dims(classes, axis=1), tf.float32))

            labels['cls_targets'] = cls_targets
            labels['box_targets'] = box_targets
            labels['num_positives'] = num_positives

        return image, labels
