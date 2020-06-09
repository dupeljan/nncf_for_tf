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

from ...algorithm_selector import COMPRESSION_ALGORITHMS

@COMPRESSION_ALGORITHMS.register('magnitude_sparsity')
def prune(to_prune, config, **kwargs):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    sparsity_target = config.get('params', {}).get('sparsity_target', 0.5)
    prune_model = prune_low_magnitude(to_prune,
                               pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(sparsity_target, 0),
                               block_size=(1, 1),
                               block_pooling_type='AVG',
                               **kwargs)

    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir=config.log_dir, update_freq=100, profile_batch=0)
    ]

    return prune_model, callbacks
