import tensorflow as tf
import numpy as np


from tensorflow.python.framework import importer
from tensorflow.python.eager import wrap_function
from itertools import zip_longest
from tensorflow.python.pywrap_tfe import TFE_Py_TapeSetShouldRecordBackprop as \
   check_tensor_in_tape
from tensorflow.python.ops.resource_variable_ops import variable_accessed as \
    add_resource_var_in_tape

from tensorflow.python.distribute.values_util import get_current_replica_id_as_int
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.framework.func_graph import _get_defun_inputs_from_args
from tensorflow.python.framework.func_graph import _get_defun_inputs_from_kwargs
from tensorflow.python.framework.func_graph import convert_structure_to_signature
from tensorflow.python.framework.func_graph import flatten
from tensorflow.python.framework.func_graph import check_mutation
import graph_editor as ge

from contrib import input_to_ops
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.ops import init_ops


def name_without_replica_idx(name):
            name = name.split(':')[0]
            if 'replica' in name:
                idx = int(name.split('_')[-1])
                name = '/'.join(name.split('/')[:-1])
            else:
                idx = 0
            return name, idx


class NNCFWrapperCustom(tf.keras.layers.Wrapper):
    def __init__(self, layer, graph_def=None, concrete=None, **kwargs):
        if layer is None:
            raise ValueError('`layer` cannot be None.')

        #if not isinstance(layer, tf.keras.layers.Layer) or \
        #        isinstance(layer, tf.keras.Model):
        #    raise ValueError(
        #        '`layer` can only be a `tf.keras.layers.Layer` instance. '
        #        'You passed an instance of type: {input}.'.format(
        #            input=layer.__class__.__name__))

        if 'name' not in kwargs:
            kwargs['name'] = '{}_{}'.format('nncf_wrapper_custom', layer.name)

        super().__init__(tf.keras.layers.Layer(), **kwargs)
        self.callable = None
        self.graph_def = graph_def
        self.concrete = concrete
        self.real_layer = layer

    def get_custom_graph_fun(self, input_shape):
        layer = tf.keras.layers.Conv1D(1, 10)

        @tf.function
        def f(inputs):
            y = tf.expand_dims(inputs, 2)
            y = layer(y)
            return tf.reshape(y, (-1, y.shape[1]))

        concrete = f.get_concrete_function(*[tf.TensorSpec(input_shape, tf.float32)])
        return concrete, layer.variables

    def build(self, input_shape=None):
        self.layer.build(input_shape[1:])
        if isinstance(self.real_layer, tf.keras.Model):
            tf_f = tf.function(self.real_layer.call)
            concrete = tf_f.get_concrete_function(*[tf.TensorSpec(input_shape, tf.float32)])

            sorted_vars = get_sorted_on_captured_vars(concrete)
            self.mirrored_variables = self.real_layer.variables
        else:
            self.bn_weights_names = []
            gd = self.graph_def
            concrete = make_new_func(gd,
                                     self.concrete.graph.captures,
                                     self.concrete.variables,
                                     self.concrete.inputs,
                                     self.concrete.outputs)

            sorted_vars = get_sorted_on_captured_vars(concrete)
            self.mirrored_variables = create_mirrored_variables(sorted_vars)

        # Save mapping for concrete per replica inputs
        self.bn_weights_names = set(['/'.join(v.name.split('/')[:-1]) for v in concrete.variables if 'replica' in v.name.lower()])
        self.sorted_concrete_vars_names = [v.name for v in sorted_vars]
        mirrored_vars_extended = []
        for v_concrete_name in self.sorted_concrete_vars_names:
            name, _ = name_without_replica_idx(v_concrete_name)
            mirrored_vars_extended.extend([v for v in self.mirrored_variables
                                           if name_without_replica_idx(v.name)[0] == name])

        self.mirrored_variables = mirrored_vars_extended

        # Add new op to layer
        self.op_vars = []
        add_vars = True
        if add_vars:
            with concrete.graph.as_default() as g:
                #target_node_name = [op for op in g.get_operations() if op.type == 'Mul'][0].name
                target_node_name = [op for op in concrete.graph.get_operations() if 'conv2d' in op.type.lower()][0].name
                weights_shape = insert_quant_op(g, target_node_name, create_add_op_with_weights)
                self.output_tensor = g.outputs[0]

            self.add_weight = tf.Variable(tf.fill(weights_shape[1:], 3.))
            self.op_vars.append(self.add_weight)

        else:
            self.output_tensor = concrete.graph.outputs[0]

        self.fn_train = concrete

    def call(self, inputs, training=None):
        replica_context = None
        if tf.distribute.has_strategy():
            replica_context = tf.distribute.get_replica_context()
            if replica_context is not None:
                replica_id = get_current_replica_id_as_int()
                new_variables = []
                new_captured = []
                for concrete_var_name, var, input_tensor in zip_longest(
                                                                self.sorted_concrete_vars_names,
                                                                self.mirrored_variables + self.op_vars,
                                                                self.fn_train.inputs[1:]):
                    if concrete_var_name:
                        name, idx = name_without_replica_idx(concrete_var_name)
                        if name not in self.bn_weights_names:
                            idx = replica_id

                    new_variables.append(var._get_replica(idx))
                    new_captured.append((var._get_replica(idx).handle, input_tensor))

        if not tf.distribute.has_strategy() or not replica_context:
            new_variables = self.fn_train.graph.variables
            new_captured = self.fn_train.graph.captures

        fn_train = make_new_func(self.fn_train.graph.as_graph_def(),
                                 new_captured,
                                 new_variables,
                                 self.fn_train.inputs,
                                 [self.output_tensor])

        return fn_train(inputs)


def insert_softmax_in_graph(fn_train):
    with fn_train.graph.as_default() as g:
        softmax = tf.nn.softmax(g.outputs[0])

        return make_new_func(g.as_graph_def(),
                             g.captures,
                             g.variables,
                             fn_train.inputs,
                             [softmax])


# Copyed from:tensorflow.contrib.quantize.python.common.DropStringPrefix tags/v1.15.0
def RerouteTensor(t0, t1, can_modify=None):
    """Reroute the end of the tensor t0 to the ends of the tensor t1.

    Args:
      t0: a tf.Tensor.
      t1: a tf.Tensor.
      can_modify: iterable of operations which can be modified. Any operation
        outside within_ops will be left untouched by this function.

    Returns:
      The number of individual modifications made by the function.
    """
    nb_update_inputs = 0
    consumers = t1.consumers()
    if can_modify is not None:
        consumers = [c for c in consumers if c in can_modify]
    consumers_indices = {}
    for c in consumers:
        consumers_indices[c] = [i for i, t in enumerate(c.inputs) if t is t1]
    for c in consumers:
        for i in consumers_indices[c]:
            c._update_input(i, t0)  # pylint: disable=protected-access
            nb_update_inputs += 1
    return nb_update_inputs


# Copied from pocketflow:learners.uniform_quantization_tf.utils.insert_quant_op
def insert_quant_op(graph, node_name, node_creation_fn):
    """Insert quantization operations to the specified activation node.

    Args:
    * graph: TensorFlow graph
    * node_name: activation node's name
    :return: shape of weight of the inserted operation.
    """

    # locate the node & activation operation
    #for op in graph.get_operations():
    #    if node_name in [node.name for node in op.outputs]:
    #        tf.logging.info('op: {} / inputs: {} / outputs: {}'.format(
    #            op.name, [node.name for node in op.inputs], [node.name for node in op.outputs]))
    #        conv_pred_act_tensor = op.outputs[0]
    #        conv_pred_act_op = op
    #        break
    conv_pred_act_op = [op for op in graph.get_operations() if node_name == op.name][0]
    conv_pred_act_tensor = conv_pred_act_op.outputs[0]

    # re-route the graph to insert quantization operations
    input_to_ops_map = input_to_ops.InputToOps(graph)
    consumer_ops = input_to_ops_map.ConsumerOperations(conv_pred_act_op)
    insert_op_output_tensor = node_creation_fn(conv_pred_act_tensor, graph)
    #insertion_node_ouput_tensor = None  # Output of the inserting node
    nb_update_inputs = RerouteTensor(insert_op_output_tensor, conv_pred_act_tensor, consumer_ops)
    print(f'nb_update_inputs = {nb_update_inputs}')
    return conv_pred_act_tensor.shape


def create_add_op_with_weights(input_tensor, graph):
    """Should be called in graph context"""
    with variable_scope.variable_scope('new_node'):
        #add_weight = tf.Variable(tf.ones(input_shape[1:]))
        add_weight = variable_scope.get_variable(
            'new_add',
            shape=input_tensor.shape[1:],
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(3),#init_ops.constant_initializer(1),
            trainable=True)
        output_tensor = tf.math.multiply(input_tensor, add_weight)
    return output_tensor


def create_mirrored_variables(vars):
    retval = []
    for var in vars:
        mirrored_var = tf.Variable(var.numpy(),
                                   trainable=var.trainable,
                                   dtype=var.dtype,
                                   name=var.name.split(':')[0] + '_mirrored')
        retval.append(mirrored_var)
    return retval


def get_sorted_on_captured_vars(concrete_fun):
    sorted_vars = []
    for value_tensor, graph_name in concrete_fun.graph.captures:
        for layer_var in concrete_fun.variables:
            if layer_var.handle is value_tensor:
                sorted_vars.append(layer_var)
    return sorted_vars


def make_new_func(output_graph_def, captures, variables, inputs, outputs):
    new_input_names = [tensor.name for tensor in inputs]
    inputs_map = {
        tensor.name: tensor for tensor in inputs
    }
    new_output_names = [tensor.name for tensor in outputs]
    new_func = my_function_from_graph_def(output_graph_def,
                                          new_input_names,
                                          new_output_names,
                                          captures,)
    for input in new_func.inputs:
        input.set_shape(inputs_map[input.name].shape)
        break

    new_func.graph.variables = variables
    return new_func


def my_function_from_graph_def(graph_def, inputs, outputs, ref_captures):
    def _imports_graph_def():
        importer.import_graph_def(graph_def, name="")

    wrapped_import = wrap_function.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    wrapped_import.graph.reset_captures([(tensor, import_graph.get_tensor_by_name(placeholder.name))
                                         for tensor, placeholder in ref_captures])
    return wrapped_import.prune(
        nest.map_structure(import_graph.as_graph_element, inputs),
        nest.map_structure(import_graph.as_graph_element, outputs))


def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))
