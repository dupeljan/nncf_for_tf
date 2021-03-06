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

import logging
import sys

EXAMPLE_LOGGER_NAME = "example"

logger = logging.getLogger(EXAMPLE_LOGGER_NAME)
_LOGGER_INITIALIZED = False

if not _LOGGER_INITIALIZED:
    logger.setLevel(logging.INFO)
    hdl = logging.StreamHandler(stream=sys.stdout)
    hdl.setFormatter(logging.Formatter("%(message)s"))
    hdl.setLevel(logging.INFO)
    logger.addHandler(hdl)
