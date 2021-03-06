"""Model defination for the RetinaNet Model."""

import tensorflow as tf

from examples.object_detection.modeling import base_model
from examples.object_detection.modeling.architecture import factory
from examples.object_detection.modeling.architecture import keras_utils
from examples.object_detection.ops import postprocess_ops
from examples.object_detection.evaluation import coco_evaluator
from examples.common.logger import logger


def focal_loss(logits, targets, alpha, gamma, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.

    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.

    Args:
      logits: A float32 tensor of size
        [batch, height_in, width_in, num_predictions].
      targets: A float32 tensor of size
        [batch, height_in, width_in, num_predictions].
      alpha: A float32 scalar multiplying alpha to the loss from positive examples
        and (1-alpha) to the loss from negative examples.
      gamma: A float32 scalar modulating loss from hard and easy examples.
      normalizer: A float32 scalar normalizes the total loss from all examples.

    Returns:
      loss: A float32 Tensor of size [batch, height_in, width_in, num_predictions]
        representing normalized loss on the prediction map.
    """

    with tf.name_scope('focal_loss'):
        positive_label_mask = tf.math.equal(targets, 1.0)
        cross_entropy = (tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))

        # Below are comments/derivations for computing modulator.
        # For brevity, let x = logits,  z = targets, r = gamma, and p_t = sigmod(x)
        # for positive samples and 1 - sigmoid(x) for negative examples.
        #
        # The modulator, defined as (1 - P_t)^r, is a critical part in focal loss
        # computation. For r > 0, it puts more weights on hard examples, and less
        # weights on easier ones. However if it is directly computed as (1 - P_t)^r,
        # its back-propagation is not stable when r < 1. The implementation here
        # resolves the issue.
        #
        # For positive samples (labels being 1),
        #    (1 - p_t)^r
        #  = (1 - sigmoid(x))^r
        #  = (1 - (1 / (1 + exp(-x))))^r
        #  = (exp(-x) / (1 + exp(-x)))^r
        #  = exp(log((exp(-x) / (1 + exp(-x)))^r))
        #  = exp(r * log(exp(-x)) - r * log(1 + exp(-x)))
        #  = exp(- r * x - r * log(1 + exp(-x)))
        #
        # For negative samples (labels being 0),
        #    (1 - p_t)^r
        #  = (sigmoid(x))^r
        #  = (1 / (1 + exp(-x)))^r
        #  = exp(log((1 / (1 + exp(-x)))^r))
        #  = exp(-r * log(1 + exp(-x)))
        #
        # Therefore one unified form for positive (z = 1) and negative (z = 0)
        # samples is:
        #      (1 - p_t)^r = exp(-r * z * x - r * log(1 + exp(-x))).

        neg_logits = -1.0 * logits
        modulator = tf.math.exp(gamma * targets * neg_logits - gamma * tf.math.log1p(tf.math.exp(neg_logits)))
        loss = modulator * cross_entropy
        weighted_loss = tf.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
        weighted_loss /= normalizer

    return weighted_loss


class RetinanetClassLoss:
    """RetinaNet class loss."""

    def __init__(self, params, num_classes):
        self._num_classes = num_classes
        self._focal_loss_alpha = params.focal_loss_alpha
        self._focal_loss_gamma = params.focal_loss_gamma

    def __call__(self, cls_outputs, labels, num_positives):
        """Computes total detection loss.

        Computes total detection loss including box and class loss from all levels.

        Args:
          cls_outputs: an OrderDict with keys representing levels and values
            representing logits in [batch_size, height, width,
            num_anchors * num_classes].
          labels: the dictionary that returned from dataloader that includes
            class groundturth targets.
          num_positives: number of positive examples in the minibatch.

        Returns:
          an integar tensor representing total class loss.
        """
        # Sums all positives in a batch for normalization and avoids zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = tf.reduce_sum(input_tensor=num_positives) + 1.0

        cls_losses = []
        for level in cls_outputs.keys():
            cls_losses.append(self.class_loss(cls_outputs[level], labels[int(level)], num_positives_sum))

        # Sums per level losses to total loss.
        return tf.add_n(cls_losses)

    def class_loss(self, cls_outputs, cls_targets, num_positives, ignore_label=-2):
        """Computes RetinaNet classification loss."""
        # Onehot encoding for classification labels.
        cls_targets_one_hot = tf.one_hot(cls_targets, self._num_classes, on_value=None, off_value=None)
        bs, height, width, _, _ = cls_targets_one_hot.get_shape().as_list()
        cls_targets_one_hot = tf.reshape(cls_targets_one_hot, [bs, height, width, -1])

        loss = focal_loss(tf.cast(cls_outputs, tf.float32),
                          tf.cast(cls_targets_one_hot, tf.float32),
                          self._focal_loss_alpha,
                          self._focal_loss_gamma,
                          num_positives)

        ignore_loss = tf.where(tf.equal(cls_targets, ignore_label),
                               tf.zeros_like(cls_targets, dtype=tf.float32),
                               tf.ones_like(cls_targets, dtype=tf.float32))

        ignore_loss = tf.expand_dims(ignore_loss, -1)
        ignore_loss = tf.tile(ignore_loss, [1, 1, 1, 1, self._num_classes])
        ignore_loss = tf.reshape(ignore_loss, tf.shape(input=loss))

        return tf.reduce_sum(input_tensor=ignore_loss * loss)


class RetinanetBoxLoss:
    """RetinaNet box loss."""

    def __init__(self, params):
        self._huber_loss = tf.keras.losses.Huber(delta=params.huber_loss_delta,
                                                 reduction=tf.keras.losses.Reduction.SUM)

    def __call__(self, box_outputs, labels, num_positives):
        """Computes box detection loss.

        Computes total detection loss including box and class loss from all levels.

        Args:
          box_outputs: an OrderDict with keys representing levels and values
            representing box regression targets in [batch_size, height, width,
            num_anchors * 4].
          labels: the dictionary that returned from dataloader that includes
            box groundturth targets.
          num_positives: number of positive examples in the minibatch.

        Returns:
          an integer tensor representing total box regression loss.
        """
        # Sums all positives in a batch for normalization and avoids zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = tf.reduce_sum(input_tensor=num_positives) + 1.0

        box_losses = []
        for level in box_outputs.keys():
            box_targets_l = labels[int(level)]
            box_losses.append(self.box_loss(box_outputs[level], box_targets_l, num_positives_sum))

        # Sums per level losses to total loss.
        return tf.add_n(box_losses)

    def box_loss(self, box_outputs, box_targets, num_positives):
        """Computes RetinaNet box regression loss."""
        # The delta is typically around the mean value of regression target.
        # for instances, the regression targets of 512x512 input with 6 anchors on
        # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
        normalizer = num_positives * 4.0
        mask = tf.cast(tf.not_equal(box_targets, 0.0), tf.float32)
        box_targets = tf.expand_dims(box_targets, axis=-1)
        box_outputs = tf.expand_dims(box_outputs, axis=-1)
        box_loss = self._huber_loss(box_targets, box_outputs, sample_weight=mask)
        box_loss /= normalizer

        return box_loss


class RetinanetModel(base_model.Model):
    """RetinaNet model function."""

    def __init__(self, params):
        super().__init__(params)

        # For eval metrics.
        self._params = params

        # Architecture generators.
        self._backbone_fn = factory.backbone_generator(params)
        self._fpn_fn = factory.multilevel_features_generator(params)
        self._head_fn = factory.retinanet_head_generator(params)

        # Loss function.
        self._cls_loss_fn = RetinanetClassLoss(params.model_params.loss_params,
                                               params.model_params.architecture.num_classes)
        self._box_loss_fn = RetinanetBoxLoss(params.model_params.loss_params)
        self._box_loss_weight = params.model_params.loss_params.box_loss_weight
        self._keras_model = None

        # Predict function.
        self._generate_detections_fn = postprocess_ops.MultilevelDetectionGenerator(
            params.model_params.architecture.min_level,
            params.model_params.architecture.max_level,
            params.model_params.postprocessing)

        # Input layer.
        self._input_layer = tf.keras.layers.Input(
            shape=(None, None, params.preprocessing.num_channels),
            name='',
            dtype=tf.bfloat16 if self._use_bfloat16 else tf.float32)

    def build_outputs(self, inputs, mode):
        backbone_features = self._backbone_fn(inputs)
        fpn_features = self._fpn_fn(backbone_features)
        cls_outputs, box_outputs = self._head_fn(fpn_features)

        if self._use_bfloat16:
            levels = cls_outputs.keys()
            for level in levels:
                cls_outputs[level] = tf.cast(cls_outputs[level], tf.float32)
                box_outputs[level] = tf.cast(box_outputs[level], tf.float32)

        model_outputs = {
            'cls_outputs': cls_outputs,
            'box_outputs': box_outputs,
        }

        return model_outputs

    def build_loss_fn(self):
        if self._keras_model is None:
            raise ValueError('build_loss_fn() must be called after build_model().')

        filter_fn = self.make_filter_trainable_variables_fn()
        trainable_variables = filter_fn(self._keras_model.trainable_variables)

        def _total_loss_fn(labels, outputs):
            cls_loss = self._cls_loss_fn(outputs['cls_outputs'],
                                         labels['cls_targets'],
                                         labels['num_positives'])
            box_loss = self._box_loss_fn(outputs['box_outputs'],
                                         labels['box_targets'],
                                         labels['num_positives'])

            model_loss = cls_loss + self._box_loss_weight * box_loss
            l2_regularization_loss = self.weight_decay_loss(trainable_variables)
            total_loss = model_loss + l2_regularization_loss

            return {
                'total_loss': total_loss,
                'cls_loss': cls_loss,
                'box_loss': box_loss,
                'model_loss': model_loss,
                'l2_regularization_loss': l2_regularization_loss,
            }

        return _total_loss_fn

    def build_model(self, pretrained=True, weights=None, mode=None):
        if self._keras_model is None:
            with keras_utils.maybe_enter_backend_graph():
                outputs = self.model_outputs(self._input_layer, mode)

                model = tf.keras.models.Model(inputs=self._input_layer,
                                              outputs=outputs, name='retinanet')
                assert model is not None, 'Fail to build tf.keras.Model.'
                self._keras_model = model

            if pretrained:
                logger.info('Init backbone')
                init_checkpoint_fn = self.make_restore_checkpoint_fn()
                init_checkpoint_fn(self._keras_model)

            if weights:
                logger.info('Loaded pretrained weights from {}'.format(weights))
                self._keras_model.load_weights(weights)

        return self._keras_model

    def post_processing(self, labels, outputs):
        required_output_fields = ['cls_outputs', 'box_outputs']

        for field in required_output_fields:
            if field not in outputs:
                raise ValueError('"{}" is missing in outputs, requried {} found {}'.format(
                                 field, required_output_fields, outputs.keys()))
        required_label_fields = ['image_info', 'groundtruths']

        for field in required_label_fields:
            if field not in labels:
                raise ValueError('"{}" is missing in outputs, requried {} found {}'.format(
                                 field, required_label_fields, labels.keys()))

        boxes, scores, classes, valid_detections = self._generate_detections_fn(
            outputs['box_outputs'], outputs['cls_outputs'], labels['anchor_boxes'],
            labels['image_info'][:, 1:2, :])
        # Discards the old output tensors to save memory. The `cls_outputs` and
        # `box_outputs` are pretty big and could potentiall lead to memory issue.
        outputs = {
            'source_id': labels['groundtruths']['source_id'],
            'image_info': labels['image_info'],
            'num_detections': valid_detections,
            'detection_boxes': boxes,
            'detection_classes': classes,
            'detection_scores': scores,
        }

        if 'groundtruths' in labels:
            labels['source_id'] = labels['groundtruths']['source_id']
            labels['boxes'] = labels['groundtruths']['boxes']
            labels['classes'] = labels['groundtruths']['classes']
            labels['areas'] = labels['groundtruths']['areas']
            labels['is_crowds'] = labels['groundtruths']['is_crowds']

        return labels, outputs

    def eval_metrics(self):
        # Create evaluator
        evaluator = coco_evaluator.COCOEvaluator(annotation_file=self._params.val_json_file)
        return coco_evaluator.MetricWrapper(evaluator)
