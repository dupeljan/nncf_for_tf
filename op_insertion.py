from collections import OrderedDict, defaultdict

import tensorflow as tf
from tensorflow.python.framework import dtypes, importer
# from tensorflow.python.framework.convert_to_constants import _run_inline_graph_optimization
from tensorflow.python.eager import wrap_function
from tensorflow.python.util import nest
from tensorflow.compat.v1 import get_variable

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph

from tensorflow.python.framework.func_graph import FuncGraph

import numpy as np


class NNCFWrapperCustom(tf.keras.layers.Wrapper):
    def __init__(self, layer, **kwargs):
        if layer is None:
            raise ValueError('`layer` cannot be None.')

        if not isinstance(layer, tf.keras.layers.Layer) or \
                isinstance(layer, tf.keras.Model):
            raise ValueError(
                '`layer` can only be a `tf.keras.layers.Layer` instance. '
                'You passed an instance of type: {input}.'.format(
                    input=layer.__class__.__name__))

        if 'name' not in kwargs:
            kwargs['name'] = '{}_{}'.format('nncf_wrapper_custom', layer.name)

        super().__init__(layer, **kwargs)
        self.callable = None
        self.weights_attr_ops = {}
        self._ops_weights = {}
        self.step = 0
        self.pruned_weight_name = None

    def build(self, input_shape=None):
        super().build(input_shape)
        # graph_, concr_fn, func_graph = self._get_inner_graph(input_shape)

        concr_fn = tf.function(self.layer).get_concrete_function(
                        tf.TensorSpec(shape=input_shape, dtype=tf.float32))
        graph_def = _run_inline_graph_optimization(concr_fn, True, False)
        concr_fn = make_new_func(graph_def, concr_fn.graph.captures,
                                 concr_fn.variables, concr_fn.inputs, concr_fn.outputs)
        graph_to_layer_names_map = get_graph_to_layer_var_names_map(concr_fn)

        with concr_fn.graph.as_default() as graph:
            self.prune_ops = [node.inputs[-1].op for node in graph.get_operations() if node.type == 'Conv2D']   # for experiment
        # with graph.as_default():
            # for weight_attr, ops in self.weights_attr_ops.items():
            #     weight = self._get_layer_weight(graph, weight_attr)
            #     for op_name, op in ops.items():
            #         self._ops_weights[op_name] = self._insert_op(op, weight, graph)
            for weight in self.prune_ops:
                self._ops_weights[weight.name + 'mask'] = self._insert_op(None, weight, graph, graph_to_layer_names_map)
                self._non_trainable_weights.append(self._ops_weights[weight.name + 'mask'])  # ?
                self.pruned_weight_name = weight.name  # for experiment
                self.pruned_weight = weight  # for experiment
                self.graph_to_layer_names_map = graph_to_layer_names_map  # for experiment
                break
            softmax = tf.nn.softmax(graph.outputs[0])
        self.callable = make_new_func(graph.as_graph_def(),
                                      captures=graph.captures,
                                      variables=concr_fn.graph.variables + tuple(self._ops_weights.values()),
                                      inputs=concr_fn.inputs, outputs=[softmax])
        self.callable.graph.variables = concr_fn.graph.variables + tuple(self._ops_weights.values())
        # concr_fn = tf.function(self.layer).get_concrete_function(
        #             tf.TensorSpec(shape=input_shape, dtype=tf.float32))
        # graph_def = _run_inline_graph_optimization(concr_fn, False, False)
        # self.callable = make_new_func(concr_fn, graph_def)
        # with self.callable.graph.as_default() as graph:
        #     relus = [op for op in graph.get_operations() if op.type == 'Relu6']
        #     consumers = find_consumers(graph, relus[0])
        #     # with graph.name_scope('scales') as scope:
        #     #     self.scale = tf.Variable(
        #     #         initial_value=1.0, trainable=True,
        #     #         name='scale', dtype=tf.float32)
        #     # fq = symmetric_quantize(
        #     #     relus[0].outputs[0],
        #     #     self.scale,
        #     #     num_bits=8,
        #     #     per_channel=False,
        #     #     narrow_range=False,
        #     #     signed=False,
        #     #     eps=1e-16)
        #     # reroute_tensor(fq, relus[0].outputs[0], consumers)
        #     with graph.name_scope('my_layer') as scope:
        #         self.mask = tf.compat.v1.get_variable(initializer=tf.constant_initializer(1.0), shape=1,
        #                                               trainable=False, name='mask', dtype=tf.float32)
        #     mul = tf.math.multiply(relus[0].outputs[0], self.mask, name='mask_multiple')
        #     reroute_tensor(mul, relus[0].outputs[0], consumers)
        #     softmax = tf.nn.softmax(graph.outputs[0])
        #
        #     self.callable = make_new_func(self.callable, graph.as_graph_def(),
        #                                   outputs=[softmax])
        # self.callable.graph.variables = concr_fn.graph.variables + tuple([self.mask])
        # self.callable.graph.variables = concr_fn.graph.variables + tuple([self.scale])

    def call(self, inputs, **kwargs):
        # if not tf.distribute.in_cross_replica_context():
        #     # id = int(tf.distribute.get_replica_context().replica_id_in_sync_group)
        #     id = int(tf.distribute.get_replica_context().devices[0][0][-1])
        #     if id not in self._concrete_func:
        #         concr_fun = tf.function(self.layer).get_concrete_function(
        #             tf.TensorSpec(shape=self.in_shape, dtype=tf.float32))
        #         with concr_fun.graph.as_default() as graph:
        #             tf.nn.softmax(graph.outputs[0])
        #         self._concrete_func[id] = concr_fun
        #
        #     return self._concrete_func[id](inputs)
        # else:
        #     concr_fun = tf.function(self.layer).get_concrete_function(
        #         tf.TensorSpec(shape=self.in_shape, dtype=tf.float32))
        #     with concr_fun.graph.as_default() as graph:
        #         tf.nn.softmax(graph.outputs[0])
        #     return concr_fun(inputs)
        self.step += 1   # for experiment
        if self.step == 1002:   # for experiment
            self.set_mask(self.calculate_mask_for_threshold())   # for experiment
        return self.callable(inputs)

    def registry_weight_operation(self, weights_attr, op, op_name=None):
        if weights_attr not in self.weights_attr_ops:
            self.weights_attr_ops[weights_attr] = OrderedDict()

        if op_name is None:
            op_name = 'nncf_op_{}:{}'.format(weights_attr, len(self.weights_attr_ops[weights_attr]))

        self.weights_attr_ops[weights_attr][op_name] = op
        return op_name

    def _get_inner_graph(self, input_shape, input_map=None, return_elements=None):
        concr_fn = tf.function(self.layer).get_concrete_function(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32))
        graph_def = _run_inline_graph_optimization(concr_fn, False, False)
        with tf.Graph().as_default() as graph:
            tf.graph_util.import_graph_def(graph_def, input_map, return_elements)  # adds 'import' scope?
        func_graph = FuncGraph('')
        # for tensor, placeholder in concr_fn.graph.captures:
        #     func_graph.add_capture(tensor, placeholder)
        return graph, concr_fn, func_graph

    def _get_layer_weight(self, graph, weight_attr):
        for variable in graph.get_operations():
            if variable.name == weight_attr:
                return variable
        return None

    def _insert_op(self, op, weight_op, graph, graph_to_layer_names_map):
        # graph_var_name = traverse(weight_op).outputs[0].name.split('/', 1)[1]  # import/keras_layer_1/...
        graph_var_name = traverse(weight_op).outputs[0].name
        variable = None
        for var in self.layer.weights:
            if var.name == graph_to_layer_names_map[graph_var_name]:
                variable = var
        with graph.as_default():
            with graph.name_scope('nncf_ops'):
                # mask = get_variable(initializer=tf.constant_initializer(1.0), shape=variable.shape,
                #                     trainable=False, name=weight_op.name + 'mask', dtype=tf.float32)
                consumers = find_consumers(graph, weight_op)
                mask = self.add_weight(
                    variable.name.rstrip(':0') + '_mask',
                    shape=variable.shape,
                    initializer=tf.keras.initializers.Constant(1.0),
                    trainable=False,
                    aggregation=tf.VariableAggregation.MEAN)
                output_tensor = weight_op.outputs[0]
                op = tf.math.multiply(output_tensor, mask)
                reroute_tensor(op, output_tensor, consumers)

        return mask

    def calculate_mask_for_threshold(self):
        graph_var_name = traverse(self.pruned_weight).outputs[0].name
        variable = None
        for var in self.layer.weights:
            if var.name == self.graph_to_layer_names_map[graph_var_name]:
                variable = var
        threshold = calculate_threshold(variable, 0.7)
        mask = tf.cast(tf.math.abs(variable) > threshold, tf.float32)
        a = mask.numpy()
        print("SPARSITY LEVEL {}".format(1-a.sum()/a.size))
        return mask

    @tf.function
    def set_mask(self, mask):
        self._ops_weights[self.pruned_weight_name + 'mask'].assign(mask)


def find_consumers(graph, op):
    inputs_to_ops = defaultdict(set)
    for op_ in graph.get_operations():
        for op_input in op_.inputs:
            inputs_to_ops[op_input.name].add(op_)

    result = set()
    for output in op.outputs:
        result.update(inputs_to_ops[output.name])
    return result


def reroute_tensor(t0, t1, can_modify=None):
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


def make_new_func(output_graph_def, captures, variables, inputs, outputs):
    new_input_names = [tensor.name for tensor in inputs]
    inputs_map = {
        tensor.name: tensor for tensor in inputs
    }
    new_output_names = [tensor.name for tensor in outputs]
    new_func = my_function_from_graph_def(output_graph_def,
                                          new_input_names,
                                          new_output_names,
                                          captures)
    # Manually propagate shape for input tensors where the shape is not correctly
    # propagated. Scalars shapes are lost when wrapping the function.
    for input in new_func.inputs:
        input.set_shape(inputs_map[input.name].shape)
        break

    new_func.graph.variables = variables
    return new_func


def my_function_from_graph_def(graph_def, inputs, outputs, ref_captures):
  """Creates a ConcreteFunction from a GraphDef.

  Args:
    graph_def: A GraphDef to make a function out of.
    inputs: A Tensor name or nested structure of names in `graph_def` which
      should be inputs to the function.
    outputs: A Tensor name or nested structure of names in `graph_def` which
      should be outputs of the function.
    ref_captures: A List of captures of original graph

  Returns:
    A ConcreteFunction.
  """

  def _imports_graph_def():
    importer.import_graph_def(graph_def, name="")

  wrapped_import = wrap_function.wrap_function(_imports_graph_def, [])
  import_graph = wrapped_import.graph
  wrapped_import.graph.reset_captures([(tensor, import_graph.get_tensor_by_name(placeholder.name))
                                       for tensor, placeholder in ref_captures])
  return wrapped_import.prune(
      nest.map_structure(import_graph.as_graph_element, inputs),
      nest.map_structure(import_graph.as_graph_element, outputs))


def traverse(node):
    while node.inputs:
        node = node.inputs[-1].op
    return node


def get_graph_to_layer_var_names_map(concrete_fun):
    names_map = {}
    for layer_var in concrete_fun.variables:
        for value_tensor, graph_name in concrete_fun.graph.captures:
            if layer_var.handle is value_tensor:
                names_map[graph_name.name] = layer_var.name
    return names_map


def calculate_threshold(weight, sparsity_level):
    w = tf.sort(tf.reshape(tf.math.abs(weight), [-1]))
    index = int(tf.cast(tf.size(w), w.dtype) * sparsity_level)
    return w[index].numpy()


def _run_inline_graph_optimization(func, lower_control_flow,
                                   aggressive_inlining):
  """Apply function inline optimization to the graph.

  Returns the GraphDef after Grappler's function inlining optimization is
  applied. This optimization does not work on models with control flow.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to to aggressive
      function inlining (might be unsafe if function has stateful ops not
      properly connected to control outputs).

  Returns:
    GraphDef
  """
  graph_def = func.graph.as_graph_def()
  # if not lower_control_flow:
  #   graph_def = disable_lower_using_switch_merge(graph_def)

  # In some cases, a secondary implementation of the function (e.g. for GPU) is
  # written to the "api_implements" attribute. (e.g. `tf.keras.layers.LSTM` in
  # TF2 produces a CuDNN-based RNN for GPU).
  # This function suppose to inline all functions calls, but "api_implements"
  # prevents this from happening. Removing the attribute solves the problem.
  # To learn more about "api_implements", see:
  #   tensorflow/core/grappler/optimizers/implementation_selector.h
  for function in graph_def.library.function:
    if "api_implements" in function.attr:
      del function.attr["api_implements"]

  meta_graph = export_meta_graph(graph_def=graph_def, graph=func.graph, clear_devices=True,
                                 clear_extraneous_savers=True)

  # Clear the initializer_name for the variables collections, since they are not
  # needed after saved to saved_model.
  for name in [
      "variables", "model_variables", "trainable_variables", "local_variables"
  ]:
    raw_list = []
    for raw in meta_graph.collection_def["variables"].bytes_list.value:
      variable = variable_pb2.VariableDef()
      variable.ParseFromString(raw)
      variable.ClearField("initializer_name")
      raw_list.append(variable.SerializeToString())
    meta_graph.collection_def[name].bytes_list.value[:] = raw_list

  # Add a collection 'train_op' so that Grappler knows the outputs.
  fetch_collection = meta_graph_pb2.CollectionDef()
  for array in func.inputs + func.outputs:
    fetch_collection.node_list.value.append(array.name)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

  # Initialize RewriterConfig with everything disabled except function inlining.
  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  rewrite_options.min_graph_nodes = -1  # do not skip small graphs
  rewrite_options.optimizers.append("function")
  if aggressive_inlining:
    rewrite_options.function_optimization =\
      rewriter_config_pb2.RewriterConfig.AGGRESSIVE
  return tf_optimizer.OptimizeGraph(config, meta_graph)
