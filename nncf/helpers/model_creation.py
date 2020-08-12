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

from ..algorithm_selector import create_compression_algorithm_builder


def create_compressed_model(model, config):
    builder = create_compression_algorithm_builder(config)
    if builder is None:
        return None, model
    compressed_model = builder.apply_to(model)
    compression_ctrl = builder.build_controller(compressed_model)
    return compression_ctrl, compressed_model
