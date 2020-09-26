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

from collections import OrderedDict
import tensorflow as tf

from .utils import get_custom_objects, is_sequential_or_functional_model, \
    is_functional_model, get_weight_name
from .transformations.commands import TransformationType, TargetType
from ..layers.custom_objects import get_nncf_custom_objects
from ..layers.wrapper import NNCFWrapper


class ModelTransformer:
    """
    Applies transformations to a Keras model graph.
    """
    def __init__(self, model, transformation_layout):
        """
        Constructor

        :param model: Keras model to be transformed
        :param transformation_layout: list of transformations
        """
        if not is_sequential_or_functional_model(model):
            raise ValueError(
                'Only tf.keras sequential or functional models can be transformed.')

        self._model = model
        self._model_config = model.get_config()
        self._custom_objects = dict(
            list(get_custom_objects(model).items()) + list(get_nncf_custom_objects().items())
        )
        self._transformations = transformation_layout.transformations
        self._name_mapping = {}

    def transform(self):
        """ Applies transformations to the Keras model.

        :return: transformed Keras model
        """
        layer_weights_map = {}
        for layer in self._model.layers:
            layer_weights_map[layer.name] = self._get_layer_weights(layer)

        for transform in self._transformations:
            self._apply_transforamtion(transform)

        if is_functional_model(self._model):
            transformed_model = tf.keras.Model.from_config(self._model_config, self._custom_objects)
        else:
            transformed_model = tf.keras.Sequential.from_config(self._model_config, self._custom_objects)

        for layer in transformed_model.layers:
            original_layer = layer.layer if isinstance(layer, NNCFWrapper) else layer
            weights = layer_weights_map.get(original_layer.name)
            if weights:
                self._set_layer_weights(layer, weights)

        return transformed_model

    def _get_layer_weights(self, layer):
        weights_map = OrderedDict()
        for weight_tensor, weight_numpy in \
                zip(layer.weights, layer.get_weights()):
            weights_map[get_weight_name(weight_tensor.name)] = weight_numpy

        return weights_map

    def _set_layer_weights(self, layer, weights_map):
        weight_value_tuples = []
        for weight_tensor in layer.weights:
            weight_name = get_weight_name(weight_tensor.name)
            if weight_name in weights_map:
                weight_value_tuples.append(
                    (weight_tensor, weights_map[weight_name]))

        tf.keras.backend.batch_set_value(weight_value_tuples)

    def _get_layer(self, layer_name):
        for layer in self._model.layers:
            if layer.name == layer_name:
                return layer
        return None

    def _find_layer_config(self, layer_name):
        for idx, layer in enumerate(self._model_config['layers']):
            if layer['name'] == layer_name:
                return idx, layer
        return None, None

    def _apply_transforamtion(self, transformation):
        if transformation.type == TransformationType.INSERT:
            self._insert(transformation.target_point, transformation.insertion_objects)
        elif transformation.type == TransformationType.REMOVE:
            raise NotImplementedError
        else:
            raise TypeError('Type {} of operation does not support'.format(transformation.type))

    def _insert(self, target_point, insertion_objects):
        target_layer_name = target_point.layer_name
        if target_point.layer_name in self._name_mapping:
            target_layer_name = self._name_mapping[target_point.layer_name]

        if target_point.type == TargetType.WEIGHT_OPERATION:
            self._insert_weight_operations(target_layer_name, target_point.weights_attr_name, insertion_objects)
        elif target_point.type == TargetType.AFTER_LAYER:
            self._insert_layers_after(target_layer_name, insertion_objects)
        else:
            raise TypeError('Type {} of target point does not support'.format(target_point.type))

    def _insert_weight_operations(self, layer_name, weights_attr_name, operations):
        layer = self._get_layer(layer_name)
        wrapper = NNCFWrapper(layer)

        for op in operations:
            wrapper.registry_weight_operation(weights_attr_name, op)

        self._replace(layer_name, wrapper)

    def _replace(self, layer_name, raplace_layer):
        raplace_layer_config = tf.keras.utils.serialize_keras_object(raplace_layer)
        raplace_layer_config['name'] = raplace_layer_config['config']['name']

        if is_functional_model(self._model):
            self._replace_functional(layer_name, raplace_layer_config)
        else:
            self._replace_sequential(layer_name, raplace_layer_config)

        self._name_mapping[layer_name] = raplace_layer_config['name']

    def _replace_functional(self, layer_name, replace_layer_config):
        replace_layer_name = replace_layer_config['name']
        for layer in self._model_config['layers']:
            for inbound_node in layer['inbound_nodes']:
                for connection_info in inbound_node:
                    if connection_info[0] == layer_name:
                        connection_info[0] = replace_layer_config['name']

        self._replace_in_model_outputs(layer_name, replace_layer_name)

        idx, layer_config = self._find_layer_config(layer_name)
        replace_layer_config['inbound_nodes'] = layer_config['inbound_nodes']
        self._model_config['layers'][idx] = replace_layer_config

    def _replace_sequential(self, layer_name, raplace_layer_config):
        idx, _ = self._find_layer_config(layer_name)
        self._model_config['layers'][idx] = raplace_layer_config

    def _insert_layers_after(self, layer_name, layers):
        layer_configs = []
        for layer in layers:
            config = tf.keras.utils.serialize_keras_object(layer)
            config['name'] = config['config']['name']
            config['inbound_nodes'] = [[[layer_name, 0, 0, {}]]]
            layer_configs.append(config)

        functional_model = is_functional_model(self._model)

        for config in layer_configs:
            if functional_model:
                self._insert_layer_after_functional(layer_name, config)
            else:
                self._insert_layer_after_sequential(layer_name, config)

    def _insert_layer_after_functional(self, layer_name, layer_config):
        replace_layer_name = layer_config['name']
        for layer in self._model_config['layers']:
            for inbound_node in layer['inbound_nodes']:
                for connection_info in inbound_node:
                    if connection_info[0] == layer_name:
                        connection_info[0] = replace_layer_name

        self._replace_in_model_outputs(layer_name, replace_layer_name)
        self._insert_layer_after_sequential(layer_name, layer_config)

    def _insert_layer_after_sequential(self, layer_name, layer_configs):
        idx, _ = self._find_layer_config(layer_name)
        self._model_config['layers'].insert(idx + 1, layer_configs)

    @staticmethod
    def _replace_output_layer_name(output_layers, layer_name, replace_layer_name):
        for output_layer in output_layers:
            if output_layer[0] == layer_name:
                output_layer[0] = replace_layer_name

    def _replace_in_model_outputs(self, layer_name, replace_layer_name):
        output_layers = self._model_config['output_layers']
        if isinstance(output_layers, list):
            self._replace_output_layer_name(output_layers, layer_name, replace_layer_name)
        elif isinstance(output_layers, dict):
            for out_layers in output_layers.values():
                self._replace_output_layer_name(out_layers.values(), layer_name, replace_layer_name)
