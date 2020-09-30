"""Model architecture factory."""

import functools

import numpy as np
import tensorflow as tf

from examples.object_detection.modeling.architecture import nn_ops
from examples.object_detection.modeling.architecture import resnet
from examples.object_detection.modeling.architecture import keras_utils


class Fpn:
    """Feature pyramid networks. Feature Pyramid Networks were proposed in:
    [1] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan, and
    Serge Belongie. Feature Pyramid Networks for Object Detection. CVPR 2017.
    """

    def __init__(self,
                min_level=3,
                max_level=7,
                fpn_feat_dims=256,
                use_separable_conv=False,
                activation='relu',
                use_batch_norm=True,
                norm_activation=nn_ops.norm_activation_builder(activation='relu')):
        """FPN initialization function.

        Args:
            min_level: `int` minimum level in FPN output feature maps.
            max_level: `int` maximum level in FPN output feature maps.
            fpn_feat_dims: `int` number of filters in FPN layers.
            use_separable_conv: `bool`, if True use separable convolution for
                convolution in FPN layers.
            use_batch_norm: 'bool', indicating whether batchnorm layers are added.
            norm_activation: an operation that includes a normalization layer
                followed by an optional activation layer.
        """

        self._min_level = min_level
        self._max_level = max_level
        self._fpn_feat_dims = fpn_feat_dims

        if use_separable_conv:
            self._conv2d_op = functools.partial(tf.keras.layers.SeparableConv2D, depth_multiplier=1)
        else:
            self._conv2d_op = tf.keras.layers.Conv2D
        if activation == 'relu':
            self._activation_op = tf.nn.relu
        elif activation == 'swish':
            self._activation_op = tf.nn.swish
        else:
            raise ValueError('Unsupported activation `{}`.'.format(activation))

        self._use_batch_norm = use_batch_norm
        self._norm_activation = norm_activation

        self._norm_activations = {}
        self._lateral_conv2d_op = {}
        self._post_hoc_conv2d_op = {}
        self._coarse_conv2d_op = {}

        for level in range(self._min_level, self._max_level + 1):
            if self._use_batch_norm:
                self._norm_activations[level] = norm_activation(use_activation=False,
                                                                name='p%d-bn' % level)

            self._lateral_conv2d_op[level] = self._conv2d_op(filters=self._fpn_feat_dims,
                                                             kernel_size=(1, 1),
                                                             padding='same',
                                                             name='l%d' % level)

            self._post_hoc_conv2d_op[level] = self._conv2d_op(filters=self._fpn_feat_dims,
                                                              strides=(1, 1),
                                                              kernel_size=(3, 3),
                                                              padding='same',
                                                              name='post_hoc_d%d' % level)

            self._coarse_conv2d_op[level] = self._conv2d_op(filters=self._fpn_feat_dims,
                                                            strides=(2, 2),
                                                            kernel_size=(3, 3),
                                                            padding='same',
                                                            name='p%d' % level)

    def __call__(self, multilevel_features, is_training=None):
        """Returns the FPN features for a given multilevel features.

        Args:
          multilevel_features: a `dict` containing `int` keys for continuous feature
            levels, e.g., [2, 3, 4, 5]. The values are corresponding features with
            shape [batch_size, height_l, width_l, num_filters].
          is_training: `bool` if True, the model is in training mode.

        Returns:
          a `dict` containing `int` keys for continuous feature levels
          [min_level, min_level + 1, ..., max_level]. The values are corresponding
          FPN features with shape [batch_size, height_l, width_l, fpn_feat_dims].
        """

        input_levels = list(multilevel_features.keys())
        if min(input_levels) > self._min_level:
            raise ValueError('The minimum backbone level {} should be '.format(min(input_levels)) +
                             'less or equal to FPN minimum level {}.'.format(self._min_level))

        backbone_max_level = min(max(input_levels), self._max_level)
        with keras_utils.maybe_enter_backend_graph(), tf.name_scope('fpn'):
            # Adds lateral connections.
            feats_lateral = {}
            for level in range(self._min_level, backbone_max_level + 1):
                feats_lateral[level] = self._lateral_conv2d_op[level](multilevel_features[level])

            # Adds top-down path.
            feats = {backbone_max_level: feats_lateral[backbone_max_level]}
            for level in range(backbone_max_level - 1, self._min_level - 1, -1):
                feats[level] = tf.keras.layers.UpSampling2D()(feats[level + 1]) + feats_lateral[level]

            # Adds post-hoc 3x3 convolution kernel.
            for level in range(self._min_level, backbone_max_level + 1):
                feats[level] = self._post_hoc_conv2d_op[level](feats[level])

            # Adds coarser FPN levels introduced for RetinaNet.
            for level in range(backbone_max_level + 1, self._max_level + 1):
                feats_in = feats[level - 1]
                if level > backbone_max_level + 1:
                    feats_in = self._activation_op(feats_in)
                feats[level] = self._coarse_conv2d_op[level](feats_in)

            if self._use_batch_norm:
                # Adds batch_norm layer.
                for level in range(self._min_level, self._max_level + 1):
                    feats[level] = self._norm_activations[level](feats[level], is_training=is_training)

        return feats


class RetinanetHead:
    """RetinaNet head."""

    def __init__(self,
                 min_level,
                 max_level,
                 num_classes,
                 anchors_per_location,
                 num_convs=4,
                 num_filters=256,
                 use_separable_conv=False,
                 norm_activation=nn_ops.norm_activation_builder(activation='relu')):

        """Initialize params to build RetinaNet head.

        Args:
          min_level: `int` number of minimum feature level.
          max_level: `int` number of maximum feature level.
          num_classes: `int` number of classification categories.
          anchors_per_location: `int` number of anchors per pixel location.
          num_convs: `int` number of stacked convolution before the last prediction
            layer.
          num_filters: `int` number of filters used in the head architecture.
          use_separable_conv: `bool` to indicate whether to use separable
            convoluation.
          norm_activation: an operation that includes a normalization layer followed
            by an optional activation layer.
        """
        self._min_level = min_level
        self._max_level = max_level

        self._num_classes = num_classes
        self._anchors_per_location = anchors_per_location

        self._num_convs = num_convs
        self._num_filters = num_filters
        self._use_separable_conv = use_separable_conv
        with tf.name_scope('class_net') as scope_name:
            self._class_name_scope = tf.name_scope(scope_name)
        with tf.name_scope('box_net') as scope_name:
            self._box_name_scope = tf.name_scope(scope_name)
        self._build_class_net_layers(norm_activation)
        self._build_box_net_layers(norm_activation)

    def _class_net_batch_norm_name(self, i, level):
        return 'class-%d-%d' % (i, level)

    def _box_net_batch_norm_name(self, i, level):
        return 'box-%d-%d' % (i, level)

    def _build_class_net_layers(self, norm_activation):
        """Build re-usable layers for class prediction network."""
        if self._use_separable_conv:
            self._class_predict = tf.keras.layers.SeparableConv2D(
                self._num_classes * self._anchors_per_location,
                kernel_size=(3, 3),
                bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
                padding='same',
                name='class-predict')
        else:
            self._class_predict = tf.keras.layers.Conv2D(
                self._num_classes * self._anchors_per_location,
                kernel_size=(3, 3),
                bias_initializer=tf.constant_initializer(-np.log((1 - 0.01) / 0.01)),
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
                padding='same',
                name='class-predict')

        self._class_conv = []
        self._class_norm_activation = {}
        for i in range(self._num_convs):
            if self._use_separable_conv:
                self._class_conv.append(
                    tf.keras.layers.SeparableConv2D(
                        self._num_filters,
                        kernel_size=(3, 3),
                        bias_initializer=tf.zeros_initializer(),
                        activation=None,
                        padding='same',
                        name='class-' + str(i)))
            else:
                self._class_conv.append(
                    tf.keras.layers.Conv2D(
                        self._num_filters,
                        kernel_size=(3, 3),
                        bias_initializer=tf.zeros_initializer(),
                        kernel_initializer=tf.keras.initializers.RandomNormal(
                            stddev=0.01),
                        activation=None,
                        padding='same',
                        name='class-' + str(i)))

            for level in range(self._min_level, self._max_level + 1):
                name = self._class_net_batch_norm_name(i, level)
                self._class_norm_activation[name] = norm_activation(name=name)

    def _build_box_net_layers(self, norm_activation):
        """Build re-usable layers for box prediction network."""
        if self._use_separable_conv:
            self._box_predict = tf.keras.layers.SeparableConv2D(
                4 * self._anchors_per_location,
                kernel_size=(3, 3),
                bias_initializer=tf.zeros_initializer(),
                padding='same',
                name='box-predict')
        else:
            self._box_predict = tf.keras.layers.Conv2D(
                4 * self._anchors_per_location,
                kernel_size=(3, 3),
                bias_initializer=tf.zeros_initializer(),
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5),
                padding='same',
                name='box-predict')

        self._box_conv = []
        self._box_norm_activation = {}
        for i in range(self._num_convs):
            if self._use_separable_conv:
                self._box_conv.append(
                    tf.keras.layers.SeparableConv2D(
                        self._num_filters,
                        kernel_size=(3, 3),
                        activation=None,
                        bias_initializer=tf.zeros_initializer(),
                        padding='same',
                        name='box-' + str(i)))
            else:
                self._box_conv.append(
                    tf.keras.layers.Conv2D(
                        self._num_filters,
                        kernel_size=(3, 3),
                        activation=None,
                        bias_initializer=tf.zeros_initializer(),
                        kernel_initializer=tf.keras.initializers.RandomNormal(
                            stddev=0.01),
                        padding='same',
                        name='box-' + str(i)))

            for level in range(self._min_level, self._max_level + 1):
                name = self._box_net_batch_norm_name(i, level)
                self._box_norm_activation[name] = norm_activation(name=name)

    def __call__(self, fpn_features, is_training=None):
        """Returns outputs of RetinaNet head."""
        class_outputs = {}
        box_outputs = {}
        with keras_utils.maybe_enter_backend_graph(), tf.name_scope('retinanet_head'):
            for level in range(self._min_level, self._max_level + 1):
                features = fpn_features[level]
                class_outputs[str(level)] = self.class_net(features, level, is_training=is_training)
                box_outputs[str(level)] = self.box_net(features, level, is_training=is_training)

        return class_outputs, box_outputs

    def class_net(self, features, level, is_training):
        """Class prediction network for RetinaNet."""
        with self._class_name_scope:
            for i in range(self._num_convs):
                features = self._class_conv[i](features)
                # The convolution layers in the class net are shared among all levels,
                # but each level has its batch normlization to capture the statistical
                # difference among different levels.
                name = self._class_net_batch_norm_name(i, level)
                features = self._class_norm_activation[name](features, is_training=is_training)

            classes = self._class_predict(features)

        return classes

    def box_net(self, features, level, is_training=None):
        """Box regression network for RetinaNet."""
        with self._box_name_scope:
            for i in range(self._num_convs):
                features = self._box_conv[i](features)
                # The convolution layers in the box net are shared among all levels, but
                # each level has its batch normlization to capture the statistical
                # difference among different levels.
                name = self._box_net_batch_norm_name(i, level)
                features = self._box_norm_activation[name](features, is_training=is_training)

            boxes = self._box_predict(features)
        return boxes


def norm_activation_generator(params):
    return nn_ops.norm_activation_builder(momentum=params.batch_norm_momentum,
                                          epsilon=params.batch_norm_epsilon,
                                          trainable=True,
                                          activation=params.activation)


def backbone_generator(params):
    """Generator function for various backbone models."""
    assert params.model_params.architecture.backbone.name == 'resnet'
    resnet_params = params.model_params.architecture.backbone.params
    backbone_fn = resnet.Resnet(resnet_depth=resnet_params.depth,
                                activation=params.model_params.norm_activation.activation,
                                norm_activation=norm_activation_generator(
                                  params.model_params.norm_activation))

    return backbone_fn


def multilevel_features_generator(params):
    """Generator function for various FPN models."""
    assert params.model_params.architecture.multilevel_features == 'fpn'
    fpn_params = params.model_params.architecture.fpn_params
    fpn_fn = Fpn(min_level=params.model_params.architecture.min_level,
                 max_level=params.model_params.architecture.max_level,
                 fpn_feat_dims=fpn_params.fpn_feat_dims,
                 use_separable_conv=fpn_params.use_separable_conv,
                 activation=params.model_params.norm_activation.activation,
                 use_batch_norm=fpn_params.use_batch_norm,
                 norm_activation=norm_activation_generator(
                   params.model_params.norm_activation))

    return fpn_fn


def retinanet_head_generator(params):
    """Generator function for RetinaNet head architecture."""
    head_params = params.model_params.architecture.head_params
    anchors_per_location = params.model_params.anchor.num_scales * len(params.model_params.anchor.aspect_ratios)
    return RetinanetHead(params.model_params.architecture.min_level,
                         params.model_params.architecture.max_level,
                         params.model_params.architecture.num_classes,
                         anchors_per_location,
                         head_params.num_convs,
                         head_params.num_filters,
                         head_params.use_separable_conv,
                         norm_activation=norm_activation_generator(
                           params.model_params.norm_activation))
