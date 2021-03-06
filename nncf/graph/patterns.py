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

import operator
from functools import reduce

from .pattern_matching import NodeExpression as N
from ..layers.common import LAYERS_WITH_WEIGHTS, LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT

LIST_LINEAR_OPS = [N(layer) for layer in LAYERS_WITH_WEIGHTS]
LINEAR_OPS = reduce(operator.or_, LIST_LINEAR_OPS[1:], LIST_LINEAR_OPS[0])

LIST_AGNOSTIC_OPS = [N(layer) for layer in LAYERS_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT]
AG = reduce(operator.or_, LIST_AGNOSTIC_OPS[1:], LIST_AGNOSTIC_OPS[0])

KERAS_ACTIVATIONS = N('ReLU') | N('ThresholdedReLU') | N('ELU') | N('PReLU') | N('LeakyReLU') | N('Activation')
TF_ACTIVATIONS = N('Relu')
ACT = KERAS_ACTIVATIONS | TF_ACTIVATIONS

BN = N('BatchNormalization')

ANY_BN_ACT_COMBO = BN + ACT | ACT + BN | BN | ACT

ANY_AG_BN_ACT_COMBO = AG + ACT | ANY_BN_ACT_COMBO

POOLING = N('AveragePooling2D') | N('AveragePooling3D') | N('GlobalAveragePooling2D') | N('GlobalAveragePooling3D')

SINGLE_OPS = POOLING | N('Average') | N('LayerNormalization')

KERAS_ARITHMETIC = N('Add') | N('Multiply')
TF_ARITHMETIC = N('AddV2') | N('Mul')
ARITHMETIC = KERAS_ARITHMETIC | TF_ARITHMETIC
