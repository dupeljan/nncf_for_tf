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


class NNCFOperation:
    """
    The abstract class represents main building block for adding compression
    extensions to a model.
    """

    def build(self, input_shape, name, layer):
        """
        This method can be used to create weights that depend on the shape(s)
        of the input(s) and register them in the NNCF Wrapper `layer`. The method
        will be automatically called when NNCF Wrapper `layer` is built.
        :param input_shape: shape of the input
        :param name: operation name
        :param layer: NNCF Wrapper layer, where the operation is registered
        :return: weights dictionary {weight name: weight value}
        """
        pass

    def call(self, inputs, weights, training):
        """
        The method performs the logic of applying the operation to the input tensors
        (which should be passed in as argument).
        :param inputs: input tensors
        :param weights: operation weights
        :param training: identifying that the model is training or evaluating
        :return: output tensors
        """
        raise NotImplementedError
    
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)
    
    
    def __eq__(self, other):
        return self.__class__ is other.__class__
    
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    
    def get_config(self):
        return {}
    
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
