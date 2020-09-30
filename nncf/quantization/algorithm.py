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

import networkx as nx

from .layers import FakeQuantize
from .config import QuantizerConfig, QuantizationMode, QuantizationConstraints
from .initializers.minmax import MinMaxInitializer
from ..algorithm_selector import COMPRESSION_ALGORITHMS
from ..layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from ..layers.common import LAYERS_WITH_WEIGHTS
from ..api.compression import CompressionAlgorithmBuilder, CompressionAlgorithmController
from ..graph.converter import convert_keras_model_to_nxmodel
from ..graph.pattern_matching import search_all
from ..graph import patterns as p
from ..graph.transformations.layout import TransformationLayout
from ..graph.transformations.commands import InsertionCommand, AfterLayer, LayerWeight,\
    TransformationPriority
from ..utils.logger import logger

ACTIVATIONS = "activations"
WEIGHTS = "weights"

QUANTIZER_GROUPS = [
    ACTIVATIONS,
    WEIGHTS
]

QUANTIZATION_LAYERS = LAYERS_WITH_WEIGHTS

NOT_SUPPORT_LAYERS = [
    'Lambda'
]


@COMPRESSION_ALGORITHMS.register('quantization')
class QuantizationBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)

        self.quantize_inputs = self.config.get('quantize_inputs', True)
        self.quantize_outputs = self.config.get('quantize_outputs', False)

        self.global_quantizer_contraints = {}
        self.ignored_scopes_per_group = {}
        self.target_scopes_per_group = {}

        for quantizer_group in QUANTIZER_GROUPS:
            self._parse_group_params(self.config, quantizer_group)

    def build_controller(self, model):
        return QuantizationController(model)

    def _parse_group_params(self, config, quantizer_group):
        params_dict = config.get(quantizer_group, {})
        self.global_quantizer_contraints[quantizer_group] = QuantizationConstraints(
            num_bits=params_dict.get('bits'),
            mode=params_dict.get('mode'),
            signed=params_dict.get('signed'),
            per_channel=params_dict.get('per_channel'),
            narrow_range=(quantizer_group == WEIGHTS)
        )
        self.ignored_scopes_per_group[quantizer_group] = params_dict.get('ignored_scopes')
        self.target_scopes_per_group[quantizer_group] = params_dict.get('target_scopes')

    def _get_default_qconfig(self, constraints: QuantizationConstraints = None):
        qconfig = QuantizerConfig(num_bits=8,
                                  mode=QuantizationMode.SYMMETRIC,
                                  signed=None,
                                  per_channel=False,
                                  narrow_range=False)
        if constraints is not None:
            qconfig = constraints.apply_constraints_to(qconfig)
        return qconfig

    def _create_quantizer(self, qconfig: QuantizerConfig):
        quantizer_cls = NNCF_QUANTIZATION_OPERATONS.get(qconfig.mode)
        return quantizer_cls(qconfig)

    def get_transformation_layout(self, model):
        nxmodel = convert_keras_model_to_nxmodel(model)
        for node_name, node in nxmodel.nodes.items():
            if node['type'] in NOT_SUPPORT_LAYERS:
                logger.warning('The layer {} is not supported by the quantization algorithm'.format(node_name))

        transformations = TransformationLayout()
        qconfig = self._get_default_qconfig(self.global_quantizer_contraints[WEIGHTS])
        for node_name, node in nxmodel.nodes.items():
            if node['type'] not in QUANTIZATION_LAYERS:
                continue

            operation = self._create_quantizer(qconfig)

            weight_attr_name = QUANTIZATION_LAYERS[node['type']]['weight_attr_name']
            transformations.register(
                InsertionCommand(
                    target_point=LayerWeight(node_name, weight_attr_name),
                    callable_object=operation,
                    priority=TransformationPriority.QUANTIZATION_PRIORITY
                ))

        insertion_points = self._find_insertion_points(nxmodel)
        qconfig = self._get_default_qconfig(self.global_quantizer_contraints[ACTIVATIONS])
        for node_name in insertion_points:
            fake_quantize_layer = FakeQuantize(qconfig, name='{}/fake_quantize'.format(node_name))
            transformations.register(
                InsertionCommand(
                    target_point=AfterLayer(node_name),
                    callable_object=fake_quantize_layer,
                    priority=TransformationPriority.QUANTIZATION_PRIORITY
                ))

        return transformations

    def _find_insertion_points(self, nxmodel):
        pattern = p.LINEAR_OPS | p.ARITHMETIC | p.ANY_BN_ACT_COMBO | \
                  p.LINEAR_OPS + p.ANY_AG_BN_ACT_COMBO | p.ARITHMETIC + p.ANY_AG_BN_ACT_COMBO | p.SINGLE_OPS

        matches = search_all(nxmodel, pattern)

        topological_order = {node: k for k, node in enumerate(nx.topological_sort(nxmodel))}
        insertion_points = [max(match, key=topological_order.__getitem__) for match in matches]

        if self.quantize_inputs:
            for node_name, degree in nxmodel.in_degree:
                if degree == 0:
                    insertion_points = [node_name] + insertion_points

        if not self.quantize_outputs:
            outputs = []
            for node_name in insertion_points:
                if nxmodel.out_degree(node_name) == 0:
                    outputs.append(node_name)
            for node_name in outputs:
                insertion_points.remove(node_name)

        return insertion_points


class QuantizationController(CompressionAlgorithmController):
    def __init__(self, target_model):
        super().__init__(target_model)
        self._initializer = MinMaxInitializer()

    def initialize(self, dataset=None, loss=None):
        self._initializer(self._model, dataset, loss)
