import tensorflow as tf
import numpy as np


from tensorflow.python.framework import importer
from tensorflow.python.eager import wrap_function
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

OUT_GRAPH_PATH = '/tmp/graph_def_test.txt'
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


optimize_by_tf = True
if optimize_by_tf:
    from tensorflow.python.framework.convert_to_constants import _FunctionConverterData
    from tensorflow.python.framework.convert_to_constants import _construct_concrete_function
    from tensorflow.python.framework.convert_to_constants import _GraphDef
else:
    from convert_to_constants import _FunctionConverterData
    from convert_to_constants import _construct_concrete_function
    from convert_to_constants import _GraphDef


def _replace_variables_by_constants(converter_data):
    """Replaces variables by constants on a given graph.

    Given a _ConverterData instance with converted variables in its tensor_data
    field, create a new graph where the respective variables are replaced with the
    converted constants.

    Args:
      converter_data: A pre-populated _ConverterData instance.

    Returns:
      The converted graph.
    """
    input_graph = _GraphDef(converter_data.graph_def)

    for tensor_name, tensor_data in converter_data.tensor_data.items():
        input_graph.nodes[tensor_name].convert_variable_to_constant(
          None, tensor_data)

    converted_graph = input_graph.converted_self().graph_def

    converted_input_indices = {
        t.index
        for t in converter_data.tensor_data.values()
        if t.index is not None
    }

    return converted_graph, converted_input_indices


def optimize_func(func,
                  lower_control_flow=True,
                  aggressive_inlining=False):
    """Replaces all the variables in a graph with constants of the same values.

    TensorFlow 2.0 function for converting all Variable ops into Const ops holding
    the same values. This makes it possible to describe the network fully with a
    single GraphDef file, and allows the removal of a lot of ops related to
    loading and saving the variables. This function runs Grappler's function
    inlining optimization in order to return a single subgraph.

    The current implementation only works for graphs that do not contain any
    control flow or embedding related ops.

    Args:
      func: ConcreteFunction.
      lower_control_flow: Boolean indicating whether or not to lower control flow
        ops such as If and While. (default True)
      aggressive_inlining: Boolean indicating whether or not to to aggressive
        function inlining (might be unsafe if function has stateful ops, not
        properly connected to control outputs). (default False)

    Returns:
      ConcreteFunction containing a simplified version of the original.
    """

    converter_data = _FunctionConverterData(
        func=func,
        lower_control_flow=lower_control_flow,
        aggressive_inlining=aggressive_inlining)

    output_graph_def, converted_input_indices = _replace_variables_by_constants(
        converter_data=converter_data)

    return _construct_concrete_function(func, output_graph_def,
                                        converted_input_indices)


def create_mirrored_variables(vars):
    retval = []
    for var in vars:
        mirrored_var = tf.Variable(var.numpy(),
                                   trainable=var.trainable,
                                   dtype=var.dtype,
                                   name=var.name.split(':')[0] + '_mirrored')
        retval.append(mirrored_var)
    return retval


def get_sorted_on_captured_vals(concrete_fun):
        sorted_vars = []
        for value_tensor, graph_name in concrete_fun.graph.captures:
            for layer_var in concrete_fun.variables:
                if layer_var.handle is value_tensor:
                    sorted_vars.append(layer_var)
        return sorted_vars


class NNCFWrapperCustom(tf.keras.layers.Wrapper):
    def __init__(self, layer, graph_def, concrete, **kwargs):
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
        self.input_shape__ = input_shape
        #self.tf_f = tf.function(self.layer.call)
        #self.tf_f(tf.ones((1,) + input_shape[1:]))
        #from google.protobuf import text_format
        #proto_b = open(OUT_GRAPH_PATH, 'r').read()
        #gd = tf.compat.v1.GraphDef()
        #text_format.Merge(proto_b, gd)
        #self.op_weight_shape = (3, 1, 1)
        #self.op_weigths = tf.Variable(tf.ones(self.op_weight_shape))
        gd = self.graph_def#self.tf_f.get_concrete_function(*[tf.TensorSpec(self.input_shape__, tf.float32)])

        concrete = make_new_func(gd,
                                 self.concrete.graph.captures,        # Wrap frozen graph to ConcreteFunctions
                                 self.concrete.variables,        #concrete = wrap_frozen_graph(graph_def=concrete.graph.as_graph_def(),
                                 self.concrete.inputs,        #                             inputs=[concrete.inputs[0].name],
                                 self.concrete.outputs)        #                             outputs=[concrete.outputs[0].name])

        sorted_vars = get_sorted_on_captured_vals(concrete)
        self.mirrored_variables = create_mirrored_variables(sorted_vars)
        #concrete = optimize_func(concrete)
        # Create graph op which will be added to layer
        #op_concrete, self.op_vars = self.get_custom_graph_fun(input_shape)
        #new_op = \
        #    make_new_func(op_concrete.graph.as_graph_def(),
        #                  op_concrete.graph.captures,
        #                  op_concrete.variables,
        #                  op_concrete.inputs,
        #                  op_concrete.outputs)

        self.op_vars = []
        # Add new op to layer
        #add_var = tf.Variable(tf.ones(input_shape[1:]))
        add_vars = False
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

        #with concrete.graph.as_default() as g:
        #    tf.import_graph_def(new_op.graph.as_graph_def(),
        #                        input_map={new_op.inputs[0].name: g.outputs[0]},
        #                        return_elements=[new_op.outputs[0].name])
        #
        #from tensorflow.python.framework.func_graph import FuncGraph
        #new_func_graph = FuncGraph('')
        #with new_func_graph.as_default():
        #    x = tf.compat.v1.placeholder(tf.float32, concrete.inputs[0].shape, 'inputs')
        #    tf.import_graph_def(concrete.graph.as_graph_def(),
        #                        input_map={concrete.inputs[0].name: x},
        #                        return_elements=[new_op.outputs[0].name])
        #concrete = make_new_func(concrete.graph.as_graph_def(),
        #                         concrete.graph.captures,
        #                         concrete.graph.variables,
        #                         concrete.inputs,
        #                         op_concrete.outputs)

        #fn_train = make_new_func(concrete.graph.as_graph_def(),
        #                         concrete.graph.captures,
        #                         concrete.graph.variables,
        #                         concrete.inputs,
        #                         concrete.outputs)

        #with open(OUT_GRAPH_PATH, 'w') as out:
        #    out.write(str(concrete.graph.as_graph_def()))

        #exit()
        self.fn_train = concrete
        #self.op_concrete = op_concrete
        #self.fn_train_graph = g

    def call(self, inputs, training=None):
        #concrete = self.tf_f.get_concrete_function(*[tf.TensorSpec(self.input_shape__, tf.float32)])
        # Before modifications
        #tf.print('before ', concrete(tf.ones((1,) + self.input_shape__[1:])))
        # Should be [[3 3 3 ... 3 3 3]]
        #from google.protobuf import text_format
        #proto_b = open(OUT_GRAPH_PATH, 'r').read()
        #gd = tf.compat.v1.GraphDef()
        #text_format.Merge(proto_b, gd)

        #@tf.function
        #def fn_train(inputs):
        #    inputs = tf.convert_to_tensor(inputs, tf.float32)
        #    input_map = {
        #      'inputs:0': inputs,
        #      'a:0': self.layer.a,
        #      'b:0': self.layer.b
        #    }
        #    # Import the graph giving x as input and getting the output y
        #    y = tf.graph_util.import_graph_def(
        #        gd, input_map=input_map, return_elements=['Softmax:0'])[0]
        #    return y

        #fn_train = make_new_func(gd,
        #                         concrete.graph.captures,
        #                         concrete.graph.variables,
        #                         concrete.inputs,
        #                         concrete.outputs)
        #func_graph = func_graph_from_func_graph('my_name', self.concrete,
        #                                        3*[tf.TensorSpec(self.input_shape__, tf.float32)],
        #                                        dict())#, func_graph=self.concrete.graph)
        #fn_train = setup_only_concrete_fun(func_graph, self.call)


        #fn_train = setup_concrete_fun(fn_train, self.call)
        # After modifications
        #tf.print('after ', fn_train(tf.ones((1,) + self.input_shape__[1:])))
        # Should be [[[6.64328218e-06 6.64328218e-06 6.64328218e-06 ... 6.64328218e-06 6.64328218e-06 6.64328218e-06]]]
        #fn_train = setup_concrete_fun(fn_train, self.layer.call)concrete = self.tf_f.get_concrete_function(*[tf.TensorSpec(self.input_shape__, tf.float32)])
        #with concrete.graph.as_default() as g:
        #    tf.nn.softmax(g.outputs[0])
        #return self.tf_f(inputs)
        replica_context = None
        if tf.distribute.has_strategy():
            replica_context = tf.distribute.get_replica_context()
            if replica_context is not None:
                replica_id = get_current_replica_id_as_int()
                new_variables = []
                new_captured = []
                for var, input_tensor in zip(self.mirrored_variables + self.op_vars, self.fn_train.inputs[1:]):
                    new_variables.append(var._get_replica(replica_id))
                    new_captured.append((var._get_replica(replica_id).handle, input_tensor))

        if not tf.distribute.has_strategy() or not replica_context:
            new_variables = self.fn_train.graph.variables
            new_captured = self.fn_train.graph.captures

        fn_train = make_new_func(self.graph_def,
                                 new_captured,
                                 new_variables,
                                 self.fn_train.inputs,
                                 [self.output_tensor])

        # Recreate variables
        #func_graph = FuncGraph('')
        #with func_graph.as_default():
        #    outputs = fn_train(inputs)

        #vars_id = sorted(get_concrete_vars_id(fn_train))
        #captures_id = sorted(get_concrete_captured_id(fn_train))
        #if vars_id != captures_id:
        #    # It doesn't work here, but inside concrete function call vars_id changes somehow
        #    print('Gradients will not leak because id\'s id is differs')

        #fn_train.graph.variables = concrete.variables
        return fn_train(inputs)


#######
# To make possible to get gradients out of concrete function
# their vars id and captured id should be equal
#######
def get_concrete_vars_id(concrete):
    res = []
    for var in concrete._func_graph.variables:
        res.append(var.handle._id)
    return res


def get_concrete_captured_id(concrete):
    res = []
    for var in concrete.captured_inputs:
        res.append(var._id)
    return res


def _add_concrete_fun_resource_vars_to_tape(concrete):
    for v in concrete._func_graph.variables:
        add_resource_var_in_tape(v)


def _check_concrete_fun_resource_vars_is_in_tape(concrete):
    return check_tensor_in_tape(concrete.captured_inputs)


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

