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
from ...api.compression import CompressionAlgorithmController, CompressionAlgorithmBuilder


@COMPRESSION_ALGORITHMS.register('magnitude_sparsity')
class MagnitudeSparsityBuilder(CompressionAlgorithmBuilder):
    def __init__(self, config):
        super().__init__(config)
        config_params = config.get('params', {})
        schedule_params = config_params.get('schedule_params', {})
        schedule_type = config_params.get('schedule_type', 'constant_sparsity')
        begin_step = schedule_params.get('begin_step', 0)
        sparsity_target = schedule_params.get('sparsity_target', 0.5)
        if schedule_type == 'polynomial_decay':
            end_step = schedule_params['end_step']
            initial_sparsity = schedule_params.get('initial_sparsity', 0)
            self.pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(initial_sparsity, sparsity_target, begin_step,
                                                                         end_step)
        elif schedule_type == 'constant_sparsity':
            self.pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(sparsity_target, begin_step)

    def apply_to(self, model):
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        prune_model = prune_low_magnitude(model,
                                          pruning_schedule=self.pruning_schedule,
                                          block_size=(1, 1),
                                          block_pooling_type='AVG')
        return prune_model

    def build_controller(self, model):
        return MagnitudeSparsityController(model, self.config)

    def _get_transformation_layout(self, model):
        pass


class MagnitudeSparsityController(CompressionAlgorithmController):
    def __init__(self, target_model, config):
        super().__init__(target_model)
        self.callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(log_dir=config.log_dir, profile_batch=0)
        ]

    def export_model(self, save_path, model_name=None):
        pass
