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
        layer_type = layer['class_name']
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
