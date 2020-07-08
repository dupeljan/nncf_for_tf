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

import tensorflow as tf


class ModelTransformer(object):
  """Transforms the Keras model by applying all the specified transforms."""

  def __init__(self, model, transformation_layout):
    """Construct ModelTransformer.

    Arguments:
      model: Keras model to be transformed.
      transformation_layout: Transformation layout to be applied to the model.
    """
    if not self._is_sequential_or_functional_model(model):
      raise ValueError(
          'Only tf.keras sequential or functional models can be transformed.')

    self._model = model
    self._transformation_layout = transformation_layout

  def transform(self):
    raise NotImplementedError

  @staticmethod
  def _is_sequential_or_functional_model(model):
    return ModelTransformer._is_functional_model(model) or isinstance(
        model, tf.keras.Sequential)

  @staticmethod
  def _is_functional_model(model):
    return isinstance(model, tf.keras.Model) \
           and not isinstance(model, tf.keras.Sequential) \
           and getattr(model, '_is_graph_network', False)
