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
    def __init__(self):
        super().__init__()
        self.bkup_var = None

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

        if not hasattr(layer.layer, name):
            self.bkup_var = self.create_bkup_weights(layer, name)
        return mask

    def call(self, inputs, weights, _):
        if self.bkup_var:
            self.bkup_var.assign(tf.where(weights > 0.5, inputs, self.bkup_var))
            inputs = self.bkup_var
        return apply_mask(inputs, weights)

    def create_bkup_weights(self, layer, name):
        var = None
        for w in layer.layer.weights:
            if w.name.split(":")[0] == name:
                var = w
        if not var:
            return var

        bkup_var = layer.add_weight(
            name + '_bkup',
            shape=var.shape,
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN)

        bkup_var.assign(var.read_value())
        return bkup_var