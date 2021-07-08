import tensorflow as tf

from tensorflow.python.framework import importer
from tensorflow.python.eager import wrap_function
from tensorflow.python.distribute.values_util import get_current_replica_id_as_int
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from itertools import zip_longest

from contrib import input_to_ops


DUMP_GRAPH = False


class NNCFWrapperCustom(tf.keras.layers.Wrapper):
    def __init__(self, layer, graph_def=None, concrete=None, **kwargs):
        if layer is None:
            raise ValueError('`layer` cannot be None.')

        if 'name' not in kwargs:
            kwargs['name'] = '{}_{}'.format('nncf_wrapper_custom', layer.name)

        super().__init__(tf.keras.layers.Layer(), **kwargs) # For testing
        self.callable = None
        self.graph_def = graph_def
        self.concrete = concrete
        if not concrete:
            self.real_layer = layer

    def build(self, input_shape=None):
        self.layer.build(input_shape[1:])
        if self.graph_def is None:
            tf_f = tf.function(self.real_layer.call)
            concrete = tf_f.get_concrete_function(*[tf.TensorSpec(input_shape, tf.float32)])

            sorted_vars = get_sorted_on_captured_vars(concrete)
            self.mirrored_variables = self.real_layer.variables
        else:
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
        if self.bn_weights_names:
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
                # Trying to find first convolutional layer
                target_node_name = [op for op in concrete.graph.get_operations() if 'conv2d' in op.type.lower()][0].name
                num_fq = insert_quant_op(g, target_node_name, create_add_op_with_weights)
                self.output_tensor = g.outputs[0]

            for _ in range(num_fq):
                self.op_vars.append(tf.Variable(3., dtype=tf.float32))

        else:
            self.output_tensor = concrete.graph.outputs[0]

        self.fn_train = concrete

        if DUMP_GRAPH:
            tf.io.write_graph(concrete.graph, '/tmp', 'mobilenetv2_sub_with_conv.pb')

    def call(self, inputs, training=None):
        replica_context = None
        if tf.distribute.has_strategy():
            replica_context = tf.distribute.get_replica_context()
            if replica_context is not None:
                # Map correspondent replica of MirroredVariable to replica concrete function
                replica_id = get_current_replica_id_as_int()
                new_variables = []
                new_captured = []
                for concrete_var_name, var, input_tensor in zip_longest(
                                                                self.sorted_concrete_vars_names,
                                                                self.mirrored_variables + self.op_vars,
                                                                self.fn_train.inputs[1:]):
                    if concrete_var_name:
                        # Check if some variables from other replicas are needed for
                        # concrete function call
                        name, idx = name_without_replica_idx(concrete_var_name)
                        if name not in self.bn_weights_names:
                            idx = replica_id

                    new_variables.append(var._get_replica(idx))
                    new_captured.append((var._get_replica(idx).handle, input_tensor))

        if not tf.distribute.has_strategy() or not replica_context:
            # If there is no distribute strategy or in compile time
            # don't change vars
            new_variables = self.fn_train.graph.variables
            new_captured = self.fn_train.graph.captures

        fn_train = make_new_func(self.fn_train.graph.as_graph_def(),
                                 new_captured,
                                 new_variables,
                                 self.fn_train.inputs,
                                 [self.output_tensor])

        return fn_train(inputs)


def name_without_replica_idx(name):
    name = name.split(':')[0]
    if 'replica' in name:
        idx = int(name.split('_')[-1])
        name = '/'.join(name.split('/')[:-1])
    else:
        idx = 0
    return name, idx


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
    :return: count of fq inserted into model
    """

    # locate the target node
    target_op = [op for op in graph.get_operations() if node_name == op.name][0]
    pred_target_ops = []
    # Locate all input ops of target op
    for op in graph.get_operations():
        if any([i in [x.name for x in target_op.inputs] for i in [x.name for x in op.outputs]]):
            pred_target_ops.append(op)
            if len(pred_target_ops) == len(target_op.inputs):
                break

    for idx, pred_target_op in enumerate(pred_target_ops):
        # re-route the graph to insert quantization operations
        input_to_ops_map = input_to_ops.InputToOps(graph)
        consumer_ops = input_to_ops_map.ConsumerOperations(pred_target_op)
        insert_op_output_tensor = node_creation_fn(pred_target_op.outputs[0], idx)
        RerouteTensor(insert_op_output_tensor, pred_target_op.outputs[0], consumer_ops)
    return len(pred_target_ops)


def create_add_op_with_weights(input_tensor, idx):
    """Should be called in graph context"""
    with variable_scope.variable_scope('new_node'):
        scale = variable_scope.get_variable(
            f'scale_{idx}',
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(6),#init_ops.constant_initializer(1),
            trainable=True)
        output_tensor = tf.quantization.fake_quant_with_min_max_vars(input_tensor, -scale, scale)
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
