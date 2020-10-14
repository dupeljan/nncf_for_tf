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

from .functions import WEIGHT_IMPORTANCE_FUNCTIONS, calc_magnitude_binary_mask
from .operation import BinaryMask
from ..schedulers import SPARSITY_SCHEDULERS
from ...algorithm_selector import COMPRESSION_ALGORITHMS
from ...api.compression import CompressionAlgorithmController, CompressionAlgorithmBuilder
from ...graph.converter import convert_keras_model_graph_to_nxmodel
from ...graph.transformations.commands import InsertionCommand, RemovalCommand, LayerWeight,\
    LayerWeightOperation, TransformationPriority
from ...graph.transformations.layout import TransformationLayout
from ...graph.model_transformer import ModelTransformer
from ...graph.utils import collect_wrapped_layers
from ...layers.wrapper import NNCFWrapper


PRUNING_LAYERS = {
    'Conv1D': {'weight_attr_name': 'kernel'},
    'Conv2D': {'weight_attr_name': 'kernel'},
    'DepthwiseConv2D': {'weight_attr_name': 'depthwise_kernel'},
    'Conv3D': {'weight_attr_name': 'kernel'},
    'Conv2DTranspose': {'weight_attr_name': 'kernel'},
    'Conv3DTranspose': {'weight_attr_name': 'kernel'},
    'Dense': {'weight_attr_name': 'kernel'},
    'SeparableConv1D': {'weight_attr_name': 'pointwise_kernel'},
    'SeparableConv2D': {'weight_attr_name': 'pointwise_kernel'},
    'Embedding': {'weight_attr_name': 'embeddings'},
    'LocallyConnected1D': {'weight_attr_name': 'kernel'},
    'LocallyConnected2D': {'weight_attr_name': 'kernel'}
}


@COMPRESSION_ALGORITHMS.register('magnitude_sparsity')
class MagnitudeSparsityBuilder(CompressionAlgorithmBuilder):

    def get_transformation_layout(self, model):
        nxmodel = convert_keras_model_graph_to_nxmodel(model)
        transformations = TransformationLayout()
        nodes_to_affect = [node_name for node_name, node in nxmodel.nodes.items()
                           if node['type'] in PRUNING_LAYERS]

        def get_layer_to_nodes_map(model_, nodes):
            map = {layer.name: {'type': layer.__class__.__name__,
                                'nodes': []}
                   for layer in model_.layers}
            for node in nodes:
                layer_name_of_node = node.split('/')[1]  # model_name/layer_name/layer_op_name/...
                if layer_name_of_node not in map:
                    raise RuntimeError('Could not find {} layer in Model'.format(layer_name_of_node))
                map[layer_name_of_node]['nodes'].append(node)
            return map

        def traverse(graph_, node_name):
            while list(graph_.predecessors(node_name)):
                node_name = list(graph_.predecessors(node_name))[-1]
            return node_name

        layer_to_nodes_map = get_layer_to_nodes_map(model, nodes_to_affect)
        for layer, layer_conf in layer_to_nodes_map.items():
            for node in layer_conf['nodes']:
                weight_attr_name = \
                    PRUNING_LAYERS[layer_conf['type']]['weight_attr_name']\
                        if layer_conf['type'] in PRUNING_LAYERS \
                        else traverse(nxmodel, node)
                operation = BinaryMask()
                transformations.register(
                    InsertionCommand(
                        target_point=LayerWeight(layer, weight_attr_name),
                        callable_object=operation,
                        priority=TransformationPriority.SPARSIFICATION_PRIORITY))

        return transformations

    def build_controller(self, model) -> CompressionAlgorithmController:
        """
        Should be called once the compressed model target_model is fully constructed
        """
        return MagnitudeSparsityController(model, self.config.params)


class MagnitudeSparsityController(CompressionAlgorithmController):
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss.
    """
    def __init__(self, target_model, params):
        super().__init__(target_model)
        self.sparsity_level = self.threshold = 0
        self.weight_importance = WEIGHT_IMPORTANCE_FUNCTIONS[params.get('weight_importance', 'normed_abs')]
        scheduler_cls = SPARSITY_SCHEDULERS.get(params.get("schedule", "polynomial"))
        self._scheduler = scheduler_cls(self, params)

    def strip_model(self, model):
        if not isinstance(model, tf.keras.Model):
            raise ValueError(
                'Expected model to be a `tf.keras.Model` instance but got: ', model)

        transformations = TransformationLayout()

        for layer in model.layers:
            if isinstance(layer, NNCFWrapper):
                for weight_attr, ops in layer.weights_attr_ops.items():
                    # BinaryMask operation must be the first operation
                    op_name, op = next(iter(ops.items()))
                    if isinstance(op, BinaryMask):
                        self._apply_mask(layer, weight_attr, op_name)

                        transformations.register(
                            RemovalCommand(
                                target_point=LayerWeightOperation(
                                    layer.name,
                                    weights_attr_name=weight_attr,
                                    operation_name=op_name)
                            ))

        return ModelTransformer(model, transformations).transform()

    def freeze(self):
        pass

    @staticmethod
    def _apply_mask(wrapped_layer, weight_attr, op_name):
        layer_weight = wrapped_layer.layer_weights[weight_attr]
        op = wrapped_layer.weights_attr_ops[weight_attr][op_name]
        layer_weight.assign(
            op(layer_weight,
               wrapped_layer.ops_weights[op_name],
               False)
        )
        wrapped_layer.set_weight(weight_attr, layer_weight)

    def set_sparsity_level(self, sparsity_level):
        if sparsity_level >= 1 or sparsity_level < 0:
            raise AttributeError(
                'Sparsity level should be within interval [0,1), actual value to set is: {}'.format(sparsity_level))
        self.sparsity_level = sparsity_level

        self.threshold = self._select_threshold()
        self._set_masks_for_threshold(self.threshold)

    def _select_threshold(self):
        all_weights = self._collect_all_weights()
        if not all_weights:
            return 0.0
        all_weights_tensor = tf.sort(tf.concat(all_weights, 0))
        index = int(tf.cast(tf.size(all_weights_tensor), all_weights_tensor.dtype) * self.sparsity_level)
        threshold = all_weights_tensor[index].numpy()
        return threshold

    def _set_masks_for_threshold(self, threshold_val):
        for wrapped_layer in collect_wrapped_layers(self._model):
            for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
                weight = wrapped_layer.layer_weights[weight_attr]

                for op_name, op in ops.items():
                    if isinstance(op, BinaryMask):
                        wrapped_layer.ops_weights[op_name].assign(
                            calc_magnitude_binary_mask(weight,
                                                       self.weight_importance,
                                                       threshold_val)
                        )

    def _collect_all_weights(self):
        all_weights = []
        for wrapped_layer in collect_wrapped_layers(self._model):
            for weight_attr, ops in wrapped_layer.weights_attr_ops.items():
                for op in ops.values():
                    if isinstance(op, BinaryMask):
                        all_weights.append(tf.reshape(
                            self.weight_importance(wrapped_layer.layer_weights[weight_attr]),
                            [-1]))
        return all_weights

    def statistics(self):
        sparsity_statistics = {}
        sparsity_levels = []
        mask_names = []
        total_weights_number = tf.constant(0)
        total_sparsified_weights_number = tf.constant(0)
        wrapped_layers = collect_wrapped_layers(self._model)
        for wrapped_layer in wrapped_layers:
            for ops in wrapped_layer.weights_attr_ops.values():
                for op_name, op in ops.items():
                    if isinstance(op, BinaryMask):
                        mask = wrapped_layer.ops_weights[op_name]
                        mask_names.append(mask.name)
                        weights_number = tf.size(mask)
                        sparsified_weights_number = weights_number - tf.reduce_sum(tf.cast(mask, tf.int32))
                        sparsity_levels.append(sparsified_weights_number / weights_number)
                        total_weights_number += weights_number
                        total_sparsified_weights_number += sparsified_weights_number

        sparsity_levels = tf.keras.backend.batch_get_value(sparsity_levels)
        total_sparsity = (total_sparsified_weights_number / total_weights_number).numpy()

        sparsity_statistics.update({
            'total_sparsity': total_sparsity
        })

        mask_sparsity = list(zip(mask_names, sparsity_levels))
        for mask_name, sparsity in mask_sparsity:
            sparsity_statistics.update({
                mask_name + '/sparsity': sparsity
            })

        return sparsity_statistics
