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

from .functions import apply_mask
from ...layers.operation import NNCFOperation, InputType
from ...layers.custom_objects import NNCF_CUSTOM_OBJECTS


@NNCF_CUSTOM_OBJECTS.register()
class BinaryMask(NNCFOperation):
    def build(self, input_shape, input_type, name, layer):
        if input_type is not InputType.WEIGHTS:
            raise ValueError(
                'Binary Mask operation could not be applied to input of the layer: {}'.
                    format(layer.name))

        mask = layer.add_weight(
            name + '_mask',
            shape=input_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN)
        return mask

    def call(self, inputs, weights, _):
        return apply_mask(inputs, weights)
