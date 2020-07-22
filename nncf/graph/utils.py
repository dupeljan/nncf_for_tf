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


def is_sequential_or_functional_model(model):
    return is_sequential_model(model) or is_functional_model(model)


def is_sequential_model(model):
    return isinstance(model, tf.keras.Sequential)


def is_functional_model(model):
    return isinstance(model, tf.keras.Model) \
           and not isinstance(model, tf.keras.Sequential) \
           and getattr(model, '_is_graph_network', False)


def get_custom_objects(_):
    return {}


def get_weight_name(name):
    return name.split('/')[-1]