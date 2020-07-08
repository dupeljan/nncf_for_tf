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

from ..graph.model_transformer import ModelTransformer
from ..configs.config import Config


class CompressionLoss:
    """
    Used to calculate additional loss to be added to the base loss during the
    training process. It uses the model graph to measure variables and activations
    values of the layers during the loss construction. For example, the $L_0$-based
    sparsity algorithm calculates the number of non-zero weights in convolutional
    and fully-connected layers to construct the loss function.
    """

    def call(self):
        """
        Returns the compression loss value.
        """
        return 0

    def statistics(self):
        """
        Returns a dictionary of printable statistics.
        """
        return {}

    def __call__(self, *args, **kwargs):
        """
        Invokes the `CompressionLoss` instance.
        Returns:
            the compression loss value.
        """
        return self.call(*args, **kwargs)

    def get_config(self):
        """
        Returns the config dictionary for a `CompressionLoss` instance.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a `CompressionLoss` from its config (output of `get_config()`).
        Arguments:
            config: Output of `get_config()`.
        Returns:
            A `CompressionLoss` instance.
        """
        return cls(**config)


class CompressionScheduler:
    """
    Implements the logic of compression method control during the training process.
    May change the method hyperparameters in regards to the current training step or
    epoch. For example, the sparsity method can smoothly increase the sparsity rate
    over several epochs.
    """

    def call(self, step):
        """
        Implements the logic of compression method control during the training process.
        Arguments:
             step: training step
        """
        pass

    def __call__(self, *args, **kwargs):
        """
        Invokes the `CompressionScheduler` instance.
        """
        return self.call(*args, **kwargs)

    def get_config(self):
        """
        Returns the config dictionary for a `CompressionScheduler` instance.
        """
        return {}

    @classmethod
    def from_config(cls, config):
        """
        Instantiates a `CompressionScheduler` from its config (output of `get_config()`).
        Arguments:
            config: Output of `get_config()`.
        Returns:
            A `CompressionScheduler` instance.
        """
        return cls(**config)


class CompressionAlgorithmInitializer:
    """
    Configures certain parameters of the algorithm that require access to the dataset
    (for example, in order to do range initialization for activation quantizers) or
    to the loss function to be used during fine-tuning (for example, to determine
    quantizer precision bitwidth using HAWQ).
    """

    def call(self, model, dataset=None, loss=None):
        pass

    def __call__(self, *args, **kwargs):
        self.call(*args, **kwargs)


class CompressionAlgorithmController:
    """
    Serves as a handle to the additional modules, parameters and hooks inserted
    into the original uncompressed model in order to enable algorithm-specific compression.
    Hosts entities that are to be used during the training process, such as compression scheduler and
    compression loss.
    """
    def __init__(self, target_model):
        self._model = target_model
        self._loss = CompressionLoss()
        self._scheduler = CompressionScheduler()
        self._initializer = CompressionAlgorithmInitializer()

    @property
    def loss(self):
        return self._loss

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def initializer(self):
        return self._initializer

    def initialize(self, dataset=None, loss=None):
        """
        Configures certain parameters of the algorithm that require access to the dataset
        (for example, in order to do range initialization for activation quantizers) or
        to the loss function to be used during fine-tuning (for example, to determine
        quantizer precision bitwidth using HAWQ).
        """
        self.initializer(self._model, dataset, loss)

    def statistics(self):
        """
        Returns a dictionary of printable statistics.
        """
        return self._loss.statistics()

    def export_model(self, save_path, model_name=None):
        """
        Used to export the compressed model into the IR format.
        Makes method-specific preparations of the model,
        (e.g. removing auxiliary layers that were used for the model compression),
        then exports the model as IR in specified path.
        Arguments:
           `save_path` - a path to export model.
           `model_name` - name under which the model will be saved
        Returns:
            model_path: dictionary { 'model': 'path to xml', 'weights': 'path to bin' }
        """
        raise NotImplementedError


class CompressionAlgorithmBuilder:
    """
    Determines which modifications should be made to the original FP32 model in
    order to enable algorithm-specific compression during fine-tuning.
    """

    def __init__(self, config: Config):
        """
        Arguments:
          `config` - a dictionary that contains parameters of compression method
        """
        self.config = config
        if not isinstance(self.config, list):
            self.ignored_scopes = self.config.get('ignored_scopes')
            self.target_scopes = self.config.get('target_scopes')

    def apply_to(self, model):
        """
        Applies algorithm-specific modifications to the model.
        """
        transformation_layout = self._get_transformation_layout(model)
        return ModelTransformer(model, transformation_layout).transform()

    def build_controller(self, model) -> CompressionAlgorithmController:
        """
        Should be called once the compressed model target_model is fully constructed
        """
        return CompressionAlgorithmController(model)

    def _get_transformation_layout(self, model):
        raise NotImplementedError