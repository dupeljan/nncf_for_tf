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

import sys
import inspect

import tensorflow as tf

from nncf.layers.wrapper import NNCFWrapper


def is_sequential_or_functional_model(model):
    return is_sequential_model(model) or is_functional_model(model)


def is_sequential_model(model):
    return isinstance(model, tf.keras.Sequential)


def is_functional_model(model):
    return isinstance(model, tf.keras.Model) \
           and not isinstance(model, tf.keras.Sequential) \
           and getattr(model, '_is_graph_network', False)


def get_custom_objects(model):
    # TODO: doesn't work when layer is wrapped by NNCFWrapper
    keras_layers = [class_name for class_name, _ in
                    inspect.getmembers(sys.modules[tf.keras.layers.__name__], inspect.isclass)]
    custom_objects = {}
    for layer in model.layers:
        if layer.__class__.__name__ not in keras_layers:
            custom_objects[layer.__class__.__name__] = layer.__class__
    custom_objects[model.get_layer('keras_layer').__class__.__name__] = model.get_layer('keras_layer').__class__
    return custom_objects


def get_weight_name(name, layer_name=None):
    if layer_name and layer_name in name:
        return name.split(layer_name + '/')[-1]
    return name


def collect_wrapped_layers(model):
    wrapped_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            wrapped_layers += collect_wrapped_layers(layer)
        if isinstance(layer, NNCFWrapper):
            wrapped_layers.append(layer)
    return wrapped_layers
