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
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from .utils import is_functional_model, is_sequential_model


def convert_keras_model_to_nxmodel(model):
    """
    Convert Keras model graph to the NetworkX directed graph

    :param model: Keras model
    :return: NetworkX directed graph
    """
    func_model = is_functional_model(model)
    seq_model = is_sequential_model(model)

    if not func_model and not seq_model:
        RuntimeError('convert_keras_model_to_nxmodel function supports '
                     'only sequential or functional models')

    nxmodel = nx.DiGraph()
    model_config = model.get_config()
    producer_layer = None
    for layer in model_config['layers']:
        layer_name = layer['name']
        layer_type = _get_layer_type(layer)
        data_format = layer['config'].get('data_format')
        nxmodel.add_node(layer_name, type=layer_type, data_format=data_format)

        if func_model:
            for inbound_nodes in layer['inbound_nodes']:
                for connection_info in inbound_nodes:
                    nxmodel.add_edge(connection_info[0], layer_name)
        elif producer_layer is not None:
            nxmodel.add_edge(producer_layer, layer_name)
            producer_layer = layer_name

    #nx.drawing.nx_pydot.write_dot(nxmodel, str("nxmodel_graph.dot"))

    return nxmodel


def _get_layer_type(layer):
    if layer['class_name'] == 'TensorFlowOpLayer':
        return layer['config']['node_def']['op']
    return layer['class_name']


def convert_keras_model_graph_to_nxmodel(model, use_graph_var_names=False):
    @tf.function
    def g(x):
        return model(x)

    def get_graph_to_layer_var_names_map(concrete_fun):
        names_map = {}
        for layer_var in concrete_fun.variables:
            for value_tensor, graph_name in concrete_fun.graph.captures:
                if layer_var.handle is value_tensor:
                    names_map[graph_name.name.split(':')[0]] = layer_var.name.split(':')[0]
        return names_map

    concr_fn = g.get_concrete_function(tf.TensorSpec(model.input_shape))
    wrapped_function = convert_variables_to_constants_v2(concr_fn, lower_control_flow=False)

    nodes = wrapped_function.graph.as_graph_def().node
    graph_to_layer_names_map = {} if use_graph_var_names else get_graph_to_layer_var_names_map(concr_fn)

    nxmodel = nx.DiGraph()
    for node in nodes:
        nxmodel.add_node(graph_to_layer_names_map.get(node.name, node.name),
                         type=node.op, dtype=node.attr['dtype'])

    for node in nodes:
        for input_node in node.input:
            node_name = graph_to_layer_names_map.get(node.name, node.name)
            input_node_name = graph_to_layer_names_map.get(input_node, input_node)
            if input_node_name in nxmodel.nodes:
                nxmodel.add_edge(input_node_name, node_name)

    return nxmodel
