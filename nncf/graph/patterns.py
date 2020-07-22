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

from .pattern_matching import NodeExpression as N

LINEAR_OPS = N('Dense') | N('Conv2D') | N('DepthwiseConv2D') | N('Conv2DTranspose') | N('Conv3D') | N('Conv3DTranspose')

RELU = N('ReLU') | N('ThresholdedReLU')

BN = N('BatchNormalization')

ANY_BN_RELU_COMBO = BN + RELU | RELU + BN | BN | RELU

POOLING = N('AveragePooling2D') | N('AveragePooling3D') | N('GlobalAveragePooling2D') | N('GlobalAveragePooling3D')

NON_RELU_ACTIVATIONS = N('ELU') | N('PReLU') | N('LeakyReLU')

SINGLE_OPS = NON_RELU_ACTIVATIONS | POOLING | N('Average') | N('LayerNormalization')

ARITHMETIC = N('Add') | N('add') | N('Multiply') | N('multiply')

ELTWISE_UNIFORM_OPS = BN | RELU | NON_RELU_ACTIVATIONS
