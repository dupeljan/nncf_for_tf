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

import datetime
import json
import logging
from os import path as osp
from pathlib import Path

import os
import tarfile

from examples.common.logger import logger as default_logger

GENERAL_LOG_FILE_NAME = "output.log"
NNCF_LOG_FILE_NAME = "nncf_output.log"


def get_name(config):
    dataset = config.get('dataset', 'imagenet')
    if dataset is None:
        dataset = 'imagenet'
    retval = config["model"] + "_" + dataset
    compression_config = config.get('compression', [])
    if not isinstance(compression_config, list):
        compression_config = [compression_config, ]
    for algo_dict in compression_config:
        algo_name = algo_dict["algorithm"]
        if algo_name == "quantization":
            initializer = algo_dict.get("initializer", {})
            precision = initializer.get("precision", {})
            if precision:
                retval += "_mixed_int"
            else:
                activations = algo_dict.get('activations', {})
                a_bits = activations.get('bits', 8)
                weights = algo_dict.get('weights', {})
                w_bits = weights.get('bits', 8)
                if a_bits == w_bits:
                    retval += "_int{}".format(a_bits)
                else:
                    retval += "_a_int{}_w_int{}".format(a_bits, w_bits)
        else:
            retval += "_{}".format(algo_name)
    return retval


def configure_paths(config):
    d = datetime.datetime.now()
    run_id = '{:%Y-%m-%d__%H-%M-%S}'.format(d)
    config.name = get_name(config)
    config.log_dir = osp.join(config.log_dir, "{}/{}".format(config.name, run_id))
    os.makedirs(config.log_dir)

    compression_config = config.get('compression', None)
    if compression_config is not None:
        compression_config.log_dir = config.log_dir

    if config.checkpoint_save_dir is None:
        config.checkpoint_save_dir = config.log_dir

    # create aux dirs
    config.intermediate_checkpoints_path = config.log_dir + '/intermediate_checkpoints'
    os.makedirs(config.intermediate_checkpoints_path)
    os.makedirs(config.checkpoint_save_dir, exist_ok=True)


def configure_logging(sample_logger, config):
    training_pipeline_log_file_handler = logging.FileHandler(osp.join(config.log_dir, GENERAL_LOG_FILE_NAME))
    training_pipeline_log_file_handler.setFormatter(logging.Formatter("%(message)s"))
    sample_logger.addHandler(training_pipeline_log_file_handler)

    nncf_log_file_handler = logging.FileHandler(osp.join(config.log_dir, NNCF_LOG_FILE_NAME))
    nncf_log_file_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    from nncf.utils.logger import logger as nncf_logger
    nncf_logger.addHandler(nncf_log_file_handler)


def create_code_snapshot(root, dst_path, extensions=(".py", ".json", ".cpp", ".cu")):
    """Creates tarball with the source code"""
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


def print_args(config, logger=default_logger):
    for arg in sorted(config):
        logger.info("{: <27s}: {}".format(arg, config.get(arg)))


def serialize_config(config, log_dir):
    with open(osp.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

