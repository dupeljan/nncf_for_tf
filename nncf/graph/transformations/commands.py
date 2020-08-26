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

from ...utils.ordered_enum import OrderedEnum


class TransformationPriority(OrderedEnum):
    DEFAULT_PRIORITY = 0
    SPARSIFICATION_PRIORITY = 2
    QUANTIZATION_PRIORITY = 11
    PRUNING_PRIORITY = 1


class TransformationType(OrderedEnum):
    INSERT = 0
    REMOVE = 1


class TargetType(OrderedEnum):
    BEFORE_LAYER = 0
    AFTER_LAYER = 1
    WEIGHT_OPERATION = 2
    PRE_OPERATION = 3
    POST_OPERATION = 4


class TargetPoint:
    """
    The base class for all TargetPoints

    TargetPoint specifies the object in the model graph to which the
    transformation command will be applied. It can be layer, weight and etc.
    """
    def __init__(self, target_type):
        """
        Constructor

        :param target_type: target point type
        """
        self._target_type = target_type

    @property
    def type(self):
        return self._target_type

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.type == other.type
        return False

    def __str__(self):
        return str(self.type)

    def __hash__(self):
        return hash(str(self))


class TransformationCommand:
    """
    The base class for all transformation commands
    """
    def __init__(self, command_type, target_point):
        """
        Constructor

        :param command_type: transformation command type
        :param target_point: target point, the object in the model
        to which the transformation command will be applied.
        """
        self._command_type = command_type
        self._target_point = target_point

    @property
    def type(self):
        return self._command_type

    @property
    def target_point(self):
        return self._target_point

    def union(self, other):
        pass

    def __add__(self, other):
        return self.union(other)


class InsertionPoint(TargetPoint):
    def __init__(self, target_type, layer_name):
        super().__init__(target_type)
        self._layer_name = layer_name

    @property
    def layer_name(self):
        return self._layer_name

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.type == other.type and self.layer_name == other.layer_name
        return False

    def __str__(self):
        return str(self.type) + " " + self.layer_name


class InsertionWeightsPoint(TargetPoint):
    def __init__(self, layer_name, weights_attr_name):
        super().__init__(TargetType.WEIGHT_OPERATION)
        self._layer_name = layer_name
        self._weights_attr_name = weights_attr_name

    @property
    def layer_name(self):
        return self._layer_name

    @property
    def weights_attr_name(self):
        return self._weights_attr_name

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self.type == other.type and \
                   self.layer_name == other.layer_name and \
                   self.weights_attr_name == other.weights_attr_name
        return False

    def __str__(self):
        return str(self.type) + " " + self.layer_name + " " + self.weights_attr_name


class InsertionCommand(TransformationCommand):
    def __init__(self, target_point, callable_object=None, priority=None):
        super().__init__(TransformationType.INSERT,
                                               target_point)
        self.callable_objects = []

        if callable_object is not None:
            _priority = TransformationPriority.DEFAULT_PRIORITY \
                if priority is None else priority
            self.callable_objects.append((callable_object, _priority))

    @property
    def insertion_objects(self):
        return [x for x, _ in self.callable_objects]

    def union(self, other):
        if self.__class__ != other.__class__ or\
                self.type != other.type or\
                self.target_point != other.target_point:
            raise ValueError('{} and {} commands could not be united'.format(
                type(self).__name__, type(other).__name__))

        com = InsertionCommand(self.target_point)
        com.callable_objects = self.callable_objects + other.callable_objects
        com.callable_objects = sorted(com.callable_objects, key=lambda x: x[1])
        return com
