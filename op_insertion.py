import tensorflow as tf

from tensorflow.python.framework import importer
from tensorflow.python.eager import wrap_function
from tensorflow.python.distribute.values_util import get_current_replica_id_as_int
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from itertools import zip_longest

from contrib import input_to_ops
from examples.classification.test_models import ModelType


DUMP_GRAPH = False


class NNCFCallableGraph(object):
    pass


class NNCFWrapperCustom(tf.keras.layers.Wrapper):
    def __init__(self, trainable_model, eval_model=None, **kwargs):
        super().__init__(tf.keras.layers.Layer(), **kwargs)
        self.model_type = ModelType.FuncModel
        self.trainable_model = NNCFCallableGraph()
        self.eval_model = NNCFCallableGraph()
        self.mirrored_vars_created = False
        self.ops_vars_created = False
        if isinstance(trainable_model, dict):
            self.model_type = ModelType.KerasLayer

            self.trainable_model.graph_def = trainable_model['graph_def']
            self.trainable_model.concrete = trainable_model['concrete_function']
            self.eval_model.graph_def = eval_model['graph_def']
            self.eval_model.concrete = eval_model['concrete_function']
        else:
            self.trainable_model.orig_model = trainable_model
            self.eval_model.orig_model = trainable_model

    def build(self, input_shape=None):
        for training, model in zip([True, False], [self.trainable_model, self.eval_model]):
            if self.model_type != ModelType.KerasLayer:
                tf_f = tf.function(lambda x: model.orig_model.call(x, training=training))
                concrete = tf_f.get_concrete_function(*[tf.TensorSpec(input_shape, tf.float32)])

                sorted_vars = get_sorted_on_captured_vars(concrete)
                model.mirrored_variables = model.orig_model.variables
            else:
                concrete = make_new_func(model.graph_def,
                                         model.concrete.graph.captures,
                                         model.concrete.variables,
                                         model.concrete.inputs,
                                         model.concrete.outputs)

                sorted_vars = get_sorted_on_captured_vars(concrete)
                model.mirrored_variables = self.create_mirrored_variables(sorted_vars)

            # Save mapping for concrete per replica inputs
            model.bn_weights_names = set(['/'.join(v.name.split('/')[:-1]) for v in concrete.variables if 'replica' in v.name.lower()])
            model.sorted_concrete_vars_names = [v.name for v in sorted_vars]
            if model.bn_weights_names:
                mirrored_vars_extended = []
                for v_concrete_name in model.sorted_concrete_vars_names:
                    name, _ = name_without_replica_idx(v_concrete_name)
                    mirrored_vars_extended.extend([v for v in model.mirrored_variables
                                                   if name_without_replica_idx(v.name)[0] == name])

                model.mirrored_variables = mirrored_vars_extended

            # Add new op to layer
            if not self.ops_vars_created:
                self.op_vars = []
            add_vars = True
            if add_vars:
                num_fq = 0
                with concrete.graph.as_default() as g:
                    # Detect all conv layers
                    conv_ops = [op for op in concrete.graph.get_operations() if op.type == 'Conv2D']
                    for conv in conv_ops:
                        # Find second children of conv
                        relu = [conv]
                        i = 0
                        while i < 2 and len(relu):
                            relu = OperationUtils.get_children_ops(g, relu[0])
                            i += 1
                        relu = relu[0]
                        # If it isn't relu - skip it
                        if relu.type != 'Relu6':
                            continue

                        # Insert fq on conv weights
                        num_fq += insert_op_before(g, conv, 1, create_add_op_with_weights, conv.name)
                        # Insert fq after relu
                        num_fq += insert_op_after(g, relu, 0, create_add_op_with_weights, relu.name)

                    model.output_tensor = g.outputs[0]

                if not self.ops_vars_created:
                    for _ in range(num_fq):
                        self.op_vars.append(tf.Variable(6., dtype=tf.float32))
                    self.ops_vars_created = True

                # Make new concrete to update captured_inputs.
                # This is needed for correct export.
                concrete = make_new_func(concrete.graph.as_graph_def(),
                                         concrete.graph.captures,
                                         concrete.variables,
                                         concrete.inputs,
                                         concrete.outputs)

            else:
                model.output_tensor = concrete.graph.outputs[0]

            model.fn_train = concrete

        if DUMP_GRAPH:
            tf.io.write_graph(concrete.graph, '/tmp', 'mobilenetv2_sub_with_conv.pb')

    def call(self, inputs, training=None):
        model_obj = self.trainable_model if training else self.eval_model
        replica_context = None
        if tf.distribute.has_strategy():
            replica_context = tf.distribute.get_replica_context()
            if replica_context is not None:
                # Map correspondent replica of MirroredVariable to replica concrete function
                replica_id = get_current_replica_id_as_int()
                new_variables = []
                new_captured = []
                for concrete_var_name, var, input_tensor in zip_longest(
                                                                model_obj.sorted_concrete_vars_names,
                                                                model_obj.mirrored_variables + self.op_vars,
                                                                model_obj.fn_train.inputs[1:]):
                    if concrete_var_name:
                        # Check if some variables from other replicas are needed for
                        # concrete function call
                        name, idx = name_without_replica_idx(concrete_var_name)
                        if name not in model_obj.bn_weights_names:
                            idx = replica_id

                    new_variables.append(var._get_replica(idx))
                    new_captured.append((var._get_replica(idx).handle, input_tensor))

        if not tf.distribute.has_strategy() or not replica_context:
            # If there is no distribute strategy or in compile time
            # don't change vars
            new_variables = model_obj.fn_train.graph.variables
            new_captured = model_obj.fn_train.graph.captures

        fn_train = make_new_func(model_obj.fn_train.graph.as_graph_def(),
                                 new_captured,
                                 new_variables,
                                 model_obj.fn_train.inputs,
                                 [model_obj.output_tensor])

        return fn_train(inputs)

    def create_mirrored_variables(self, vars):
        if not self.mirrored_vars_created:
            retval = []
            for var in vars:
                mirrored_var = tf.Variable(var.numpy(),
                                           trainable=var.trainable,
                                           dtype=var.dtype,
                                           name=var.name.split(':')[0] + '_mirrored')
                retval.append(mirrored_var)
            self.mirrored_vars_created = True
            self.mirrored_vars_cache = retval
        else:
            retval = self.mirrored_vars_cache

        return retval

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
def insert_op_before(graph, target_op, input_idx, node_creation_fn, name):
    """Insert quantization operations before node on input_idx.

    Args:
    * graph: TensorFlow graph
    * node_name: activation node's name
    :return: count of fq inserted into model
    """
    target_parent = None
    output_idx = None
    target_op_parents = OperationUtils.get_parent_ops(graph, target_op)
    target_parent_output = target_op.inputs[input_idx]
    for op in target_op_parents:
        for i, outputs in enumerate(op.outputs):
            if outputs.name == target_parent_output.name:
                target_parent = op
                output_idx = i

    if target_parent is None or output_idx is None:
        raise RuntimeError(f'Can\'t find node parent, node name: {target_op.name}')

    # re-route the graph to insert quantization operations
    return insert_op_after(graph, target_parent, output_idx, node_creation_fn, name)


def insert_op_after(graph, target_parent, output_index, node_creation_fn, name):
    input_to_ops_map = input_to_ops.InputToOps(graph)
    consumer_ops = input_to_ops_map.ConsumerOperations(target_parent)
    insert_op_output_tensor = node_creation_fn(target_parent.outputs[output_index], name)
    RerouteTensor(insert_op_output_tensor, target_parent.outputs[output_index], consumer_ops)
    return 1


def create_add_op_with_weights(input_tensor, name):
    """Should be called in graph context"""
    with variable_scope.variable_scope('new_node'):
        scale = variable_scope.get_variable(
            f'scale_{name}',
            shape=(),
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(6),#init_ops.constant_initializer(1),
            trainable=True)
        output_tensor = tf.quantization.fake_quant_with_min_max_vars(input_tensor, -scale, scale)
    return output_tensor


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


class OperationUtils:
    @staticmethod
    def get_parent_ops(graph, target_op):
        retval = []
        target_op_inputs = [x.name for x in target_op.inputs]
        for op in graph.get_operations():
            if any([i in [x.name for x in op.outputs] for i in target_op_inputs]):
                retval.append(op)
                if len(retval) == len(target_op.inputs):
                    break
        return retval

    @staticmethod
    def get_children_ops(graph, target_op):
        retval = []
        target_op_outputs = [x.name for x in target_op.outputs]
        for op in graph.get_operations():
            if any([out in [x.name for x in op.inputs] for out in target_op_outputs]):
                retval.append(op)
                if len(retval) == len(target_op.outputs):
                    break
        return retval
