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

import os
import pytest
from addict import Dict

import tensorflow as tf

from tests import test_models
from tests.test_helpers import get_empty_config, create_compressed_model_and_algo_for_test


def get_basic_quantization_config(qconfig, input_sample_sizes=None):
    config = get_empty_config(input_sample_sizes=input_sample_sizes)
    config['compression'] = {'algorithm': 'quantization',
                             'activations': {
                                 'mode': qconfig.mode,
                                 'per_channel': qconfig.per_channel
                             },
                             'weights': {
                                 'mode': qconfig.mode,
                                 'per_channel': qconfig.per_channel
                             }}
    return config


def check_graph(graph, ref_graph_dir, ref_graph_file_name):
    data_dir = os.path.join(os.path.dirname(__file__), 'data/reference_graphs')
    graph_dir = os.path.join(data_dir, ref_graph_dir)
    graph_path = os.path.abspath(os.path.join(graph_dir, ref_graph_file_name))

    # validate file with graph manually!
    if not os.path.exists(graph_path):
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)
        tf.io.write_graph(graph, graph_dir, ref_graph_file_name, as_text=False)

    expected_graph = tf.python.GraphDef()
    with open(graph_path, 'rb') as f:
        expected_graph.ParseFromString(f.read())

    tf.test.assert_equal_graph_def(graph, expected_graph)


class QuantizeTestCaseConfiguration:
    def __init__(self, quant_mode, quant_granularity, graph_dir):
        self.qconfig = Dict()
        self.qconfig.mode = quant_mode
        self.qconfig.per_channel = (quant_granularity == 'per_channel')
        self.graph_dir = graph_dir


QUANTIZERS = [('symmetric', 'per_tensor'), ('asymmetric', 'per_channel')]


@pytest.fixture(
    scope='function', params=QUANTIZERS, ids=['{}_{}'.format(mode, granularity) for mode, granularity in QUANTIZERS]
)
def _case_config(request):
    quant_mode, quant_granularity = request.param
    graph_dir = os.path.join('quantized', quant_mode, quant_granularity)
    return QuantizeTestCaseConfiguration(quant_mode, quant_granularity, graph_dir)


class ModelDesc:
    def __init__(self, pb_filename: str, model_builder, input_sample_sizes):
        self.model_name = self._get_model_name(pb_filename)
        self.model_builder = model_builder
        self.pb_filename = pb_filename
        self.input_sample_sizes = input_sample_sizes

    @staticmethod
    def _get_model_name(dot_filename):
        if isinstance(dot_filename, tuple):
            dot_filename = dot_filename[0]
        return dot_filename[:dot_filename.find('.pb')]


TEST_MODELS_DESC = [
    ModelDesc('densenet121.pb', test_models.DenseNet121, [1, 32, 32, 3]),
    pytest.param(
        ModelDesc('inception_resnet_v2.pb', test_models.InceptionResNetV2, [1, 75, 75, 3]),
        marks=pytest.mark.skip(reason='gitlab issue #17')
    ),
    ModelDesc('inception_v3.pb', test_models.InceptionV3, [1, 75, 75, 3]),
    ModelDesc('mobilenet_v1.pb', test_models.MobileNet, [1, 128, 128, 3]),
    ModelDesc('mobilenet_v2.pb', test_models.MobileNetV2, [1, 96, 96, 3]),
    pytest.param(
        ModelDesc('nasnet_mobile.pb', test_models.NASNetMobile, [1, 32, 32, 3]),
        marks=pytest.mark.skip(reason='gitlab issue #18')
    ),
    ModelDesc('resnet50.pb', test_models.ResNet50, [1, 32, 32, 3]),
    ModelDesc('resnet50_v2.pb', test_models.ResNet50V2, [1, 32, 32, 3]),
    ModelDesc('vgg16.pb', test_models.VGG16, [1, 32, 32, 3]),
    ModelDesc('xception.pb', test_models.Xception, [1, 71, 71, 3])
]


def keras_model_to_graph_def(model):
    input_signature = []
    for item in model.inputs:
        input_signature.append(tf.TensorSpec(item.shape, item.dtype))
    concrete_function = tf.function(model).get_concrete_function(input_signature)
    return concrete_function.graph.as_graph_def(add_shapes=True)


def check_model_graph(compressed_model, ref_graph_file_name, ref_graph_dir):
    compressed_graph = keras_model_to_graph_def(compressed_model)
    check_graph(compressed_graph, ref_graph_dir, ref_graph_file_name)


@pytest.mark.parametrize(
    'desc', TEST_MODELS_DESC, ids=[
        m.model_name if isinstance(m, ModelDesc) else m.values[0].model_name for m in TEST_MODELS_DESC
    ]
)
class TestModelsGraph:
    def test_quantize_network(self, desc: ModelDesc, _case_config):
        model = desc.model_builder(input_shape=tuple(desc.input_sample_sizes[1:]))
        config = get_basic_quantization_config(_case_config.qconfig, input_sample_sizes=desc.input_sample_sizes)
        compressed_model, _ = create_compressed_model_and_algo_for_test(model, config)

        check_model_graph(compressed_model, desc.pb_filename, _case_config.graph_dir)
