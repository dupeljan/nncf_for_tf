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

KERAS_LAYERS_WITH_WEIGHTS = {
    'Conv1D': {'weight_attr_name': 'kernel', 'channel_axes': -1},
    'Conv2D': {'weight_attr_name': 'kernel', 'channel_axes': -1},
    'Conv3D': {'weight_attr_name': 'kernel', 'channel_axes': -1},
    'DepthwiseConv2D': {'weight_attr_name': 'depthwise_kernel', 'channel_axes': (2, 3)},
    'Conv1DTranspose': {'weight_attr_name': 'kernel', 'channel_axes': -2},
    'Conv2DTranspose': {'weight_attr_name': 'kernel', 'channel_axes': -2},
    'Conv3DTranspose': {'weight_attr_name': 'kernel', 'channel_axes': -2},
    'Dense': {'weight_attr_name': 'kernel', 'channel_axes': -1}
}

TF_LAYERS_WITH_WEIGHTS = {}

LAYERS_WITH_WEIGHTS = {}
LAYERS_WITH_WEIGHTS.update(KERAS_LAYERS_WITH_WEIGHTS)
LAYERS_WITH_WEIGHTS.update(TF_LAYERS_WITH_WEIGHTS)

KERAS_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT = [
    'Cropping1D',
    'Cropping2D',
    'Cropping3D',
    'Flatten',
    'GlobalMaxPool1D',
    'GlobalMaxPool2D',
    'GlobalMaxPool3D',
    'GlobalMaxPooling1D',
    'GlobalMaxPooling2D',
    'GlobalMaxPooling3D',
    'MaxPool1D',
    'MaxPool2D',
    'MaxPool3D',
    'MaxPooling1D',
    'MaxPooling2D',
    'MaxPooling3D',
    'RepeatVector',
    'Reshape',
    'UpSampling2D'
    'ZeroPadding1D',
    'ZeroPadding2D',
    'ZeroPadding3D'
]

TF_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT = [
    'Identity',
    'Pack',
    'Pad',
    'StridedSlice',
]

LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT = \
    KERAS_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT + TF_LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT
