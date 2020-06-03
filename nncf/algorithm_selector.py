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

from functools import partial

from .utils.logger import logger
from .utils.registry import Registry

COMPRESSION_ALGORITHMS = Registry('compression algorithm')


@COMPRESSION_ALGORITHMS.register('NoCompressionAlgorithm')
def no_compression(model, **kwarg):
    return model, list()


def get_compression_algorithm(config):
    algorithm_key = config.get('algorithm', 'NoCompressionAlgorithm')
    logger.info("Creating compression algorithm: {}".format(algorithm_key))
    return COMPRESSION_ALGORITHMS.get(algorithm_key)


def create_compression_algorithm_builder(config):
    compression_config = config.get('compression', {})

    if isinstance(compression_config, dict):
        return partial(get_compression_algorithm(compression_config), config=compression_config)
