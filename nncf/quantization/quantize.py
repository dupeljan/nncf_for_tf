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

import tensorflow_model_optimization as tfmot

from ..algorithm_selector import COMPRESSION_ALGORITHMS

@COMPRESSION_ALGORITHMS.register('quantization')
def quantize(to_quantize, **kwarg):
    quantize_model = tfmot.quantization.keras.quantize_model

    quntaze_model = quantize_model(to_quantize)
    callbacks = []

    return quntaze_model, callbacks
