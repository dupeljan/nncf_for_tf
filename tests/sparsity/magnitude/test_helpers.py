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
from addict import Dict

from nncf.configs.config import Config
from tests.test_helpers import create_conv, check_equal


def get_magnitude_test_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = create_conv(1, 2, 2, 9., -2.)(inputs)
    outputs = create_conv(2, 1, 3, -10., 0.)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_magnitude_model_has_expected_params():
    model = get_magnitude_test_model((4, 4, 1))
    act_weights_1 = model.layers[1].kernel.numpy()
    act_weights_2 = model.layers[2].kernel.numpy()
    act_bias_1 = model.layers[1].bias.numpy()
    act_bias_2 = model.layers[2].bias.numpy()

    sub_tensor = tf.constant([[[[10., 9.],
                                [9., 10.]]]])
    sub_tensor = tf.transpose(sub_tensor, (2, 3, 0, 1))
    ref_weights_1 = tf.concat((sub_tensor, sub_tensor), 3)
    sub_tensor = tf.constant([[[[-9., -10., -10.],
                                [-10., -9., -10.],
                                [-10., -10., -9.]]]])
    sub_tensor = tf.transpose(sub_tensor, (2, 3, 0, 1))
    ref_weights_2 = tf.concat((sub_tensor, sub_tensor), 2)

    check_equal(act_weights_1, ref_weights_1)
    check_equal(act_weights_2, ref_weights_2)

    check_equal(act_bias_1, tf.constant([-2., -2]))
    check_equal(act_bias_2, tf.constant([0]))


def get_basic_magnitude_sparsity_config(input_sample_size=None):
    if input_sample_size is None:
        input_sample_size = [1, 4, 4, 1]
    config = Config()
    config.update(Dict({
        "model": "basic_sparse_conv",
        "input_info":
            {
                "sample_size": input_sample_size,
            },
        "compression":
            {
                "algorithm": "magnitude_sparsity",
                "params": {}
            }
    }))
    return config
