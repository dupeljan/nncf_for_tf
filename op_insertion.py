from collections import OrderedDict, defaultdict

import tensorflow as tf
from tensorflow.python.framework import dtypes, importer
from tensorflow.python.framework.convert_to_constants import _run_inline_graph_optimization, _construct_concrete_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.util import nest
from tensorflow.compat.v1 import get_variable

from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.framework.func_graph import FuncGraph

import numpy as np

from tensorflow.python.util import compat
from tensorflow.python.eager import execute

def add_gradients_to_op(op):
    # Record the gradient because custom-made ops don't go through the
    # code-gen'd eager call path
    op_type = compat.as_str(op.op_def.name)
    attr_names = [compat.as_str(attr.name) for attr in op.op_def.attr]
    attrs = []
    for attr_name in attr_names:
      attrs.append(attr_name)
      attrs.append(op.get_attr(attr_name))
    attrs = tuple(attrs)
    execute.record_gradient(op_type, op.inputs, attrs, op.outputs)
    return op


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
        # self.weights_attr_ops = {}
        # self._ops_weights = {}
        # self.step = 0
        # self.pruned_weight_name = None

    CREATE_GRAPH_ON_BUILD = True
    def build(self, input_shape=None):
        from tensorflow.python.saved_model.load import _WrapperFunction
        from tensorflow.python.framework import ops
        # with ops.init_scope():
        #     from tensorflow.python.saved_model.load import _WrapperFunction
        #     from tensorflow.python.distribute import distribution_strategy_context
        #     #with distribution_strategy_context._get_default_replica_context():
        #     self.concr_train_fn = tf.function(self.layer.call).get_concrete_function(
        #         tf.TensorSpec(shape=input_shape, dtype=tf.float32), training=True)
        #     self.concr_eval_fn = tf.function(self.layer.call).get_concrete_function(
        #         tf.TensorSpec(shape=input_shape, dtype=tf.float32), training=False)
        #
        if self.CREATE_GRAPH_ON_BUILD:

            #from tensorflow.python.distribute import distribution_strategy_context
            #from tensorflow.python.keras import backend as K
            #from tensorflow.python.keras.saving.saved_model import utils

            self.layer.build(input_shape[1:])

            self.input_shape__ = input_shape


            #return
            tf_f = tf.function(self.layer.call)
            self.train_fn = tf_f
            self.tf_f = tf_f
            return
            #concrete = tf.function(self.layer.call).get_concrete_function(*[tf.TensorSpec(input_shape, tf.float32)]) #,
            #                                                              #tf.TensorSpec(input_shape, tf.float32))
            #gd = concrete.graph.as_graph_def()
            ##@tf.function
            ##def train_fn(inputs):
            ##    inputs = tf.convert_to_tensor(inputs, tf.float32)
            ##    input_map = {
            ##      'inputs:0': inputs,
            ##      'Identity/ReadVariableOp/resource:0': self.layer.a,
            ##      'add/Identity/ReadVariableOp/resource:0': self.layer.b
            ##    }
            ##    # Import the graph giving x as input and getting the output y
            ##    y = tf.graph_util.import_graph_def(
            ##        gd, input_map=input_map, return_elements=['Identity:0'])[0]
            ##    return y
            #fn_train = make_new_func(concrete.graph.as_graph_def(),
            #                         concrete.graph.captures,
            #                         concrete.graph.variables,
            #                         concrete.inputs,
            #                         concrete.outputs)
            #self.train_fn = fn_train
            #return

            l = tf.keras.layers.Dense(10)
            l.build((150528, ))
            concrete_conv1d = tf.function(l.call).get_concrete_function(*[tf.TensorSpec(input_shape, tf.float32)])
            captured_conv1d = make_new_func(concrete_conv1d.graph.as_graph_def(),
                                            concrete_conv1d.graph.captures,
                                            concrete_conv1d.graph.variables,
                                            concrete_conv1d.inputs,
                                            concrete_conv1d.outputs)

            with fn_train.graph.as_default() as g:
                y = g.get_tensor_by_name('Identity_1:0')
                z = tf.graph_util.import_graph_def(
                    captured_conv1d.graph.as_graph_def(), input_map={'inputs:0': y}, return_elements=['Identity:0'])[0]

            gd = g.as_graph_def()
            #for op in fn_train.graph.get_operations():
            #   execute.record_gradient(op.type, op.inputs, op.node_def.attr, op.outputs)

            @tf.function
            def modified_fun(inputs):
                inputs = tf.convert_to_tensor(inputs, tf.float32)
                input_map = {
                  'inputs:0': inputs,
                }
                # Import the graph giving x as input and getting the output y
                y = tf.graph_util.import_graph_def(
                    gd, input_map=input_map, return_elements=['import/Identity:0'])[0]
                return y

            # Create dummy model to save
            class DummyLayer(tf.keras.layers.Layer):
                def call(self, inputs, **kwargs):
                    return modified_fun(inputs)

            model = tf.keras.Sequential([tf.keras.Input(input_shape[1:]),
                                 DummyLayer()])
            self.train_fn = make_new_func(g.as_graph_def(),
                                          g.captures,
                                          g.variables,
                                          fn_train.inputs,
                                          captured_conv1d.outputs)
            #return
            #path = '/tmp/model.pb'
            #model.save(path, save_format='tf')
            #model = tf.keras.models.load_model(path)

            self.vars = concrete.variables
            #self.vars = [
            #    self.layer.a,
            #    self.layer.b
            #]

            #fn_train = _rebuild_func(concrete)

            self.train_fn = fn_train
            #self.train_fn = tf.function(fn_train).python_function#tf_f.python_function


            # Leads to infinite loop
            #@tf.custom_gradient
            #def fn(inputs):
            #    inputs = tf.convert_to_tensor(inputs, tf.float32)
            #    outputs = concrete(inputs)
            #    return outputs, lambda yx, variables: tf.gradients(yx, variables)


            # Leads to deadlock
            #@tf.function
            #def fn(inputs):
            #    inputs = tf.convert_to_tensor(inputs, tf.float32)
            #    outputs = self.train_fn(inputs)
            #    return outputs#, lambda yx, variables: tf.gradients(yx, variables)

            #self.train_fn = fn

            #with K.deprecated_internal_learning_phase_scope(0):
            #    # When saving a model involving batch norm layer within a strategy scope,
            #    # the replica context is not available when calling `add_update()`, and thus
            #    # we use the default replica context here.
            #    #with distribution_strategy_context._get_default_replica_context():  # pylint: disable=protected-access
            #    with utils.keras_option_scope(True):
            #        from tensorflow.python.keras.engine.base_layer import TensorFlowOpLayer
            #        self.concr_train_fn = tf.function(self.layer.call).get_concrete_function(
            #                   tf.TensorSpec(shape=input_shape, dtype=tf.float32), training=True)

            #        gd = self.concr_train_fn.graph.as_graph_def()
            #        #operations = self.concr_train_fn.graph.get_operations()
            #        #for i in range(len(operations)):
            #        #    operations[i] = add_gradients_to_op(operations[i])

            #        #self.fn_train = _WrapperFunction(function_lib.ConcreteFunction(
            #        #                                     func_graph=self.concr_train_fn.graph,
            #        #                                     function_spec=self.concr_train_fn._function_spec))
            #        #self.concr_train_fn.add_gradient_functions_to_graph()
            #        @tf.function
            #        def my_new_train_fn(inputs):
            #            return tf.graph_util.import_graph_def(
            #                       gd, input_map={'inputs:0': inputs}, return_elements=['Identity:0'])[0]

            #        self.fn_train = my_new_train_fn
        #self.evn = _WrapperFunction(function_lib.ConcreteFunction(
        #    self.concr_eval_fn.graph))
        #
        # def inner_call(inputs, training=None):
        #     if training:
        #         return self.train_fn(inputs)
        #     else:
        #         return self.eval_fn(inputs)
        #
        # self._func = tf.function(inner_call)

        #fn_train = make_new_func(self.concr_train_fn.graph.as_graph_def(),
        #                          self.concr_train_fn.graph.captures,
        #                          self.concr_train_fn.graph.variables,
        #                          self.concr_train_fn.inputs,
        #                          self.concr_train_fn.outputs)
        #
        # fn_eval = make_new_func(self.concr_eval_fn.graph.as_graph_def(),
        #                         self.concr_eval_fn.graph.captures,
        #                         self.concr_eval_fn.graph.variables,
        #                         self.concr_eval_fn.inputs,
        #                         self.concr_eval_fn.outputs)

        print('built')
        #concr_train_fn = tf.function(self.layer.call).get_concrete_function(
        #           tf.TensorSpec(shape=input_shape, dtype=tf.float32), training=True)
        #self.fn_train = make_new_func(concr_train_fn.graph.as_graph_def(),
        #                                 concr_train_fn.graph.captures,
        #                                 concr_train_fn.graph.variables,
        #                                 concr_train_fn.inputs,
        #                                 concr_train_fn.outputs)

        #self._concrete_functions[name] = _WrapperFunction(concrete_function)

    REBUILD_CONCREETE = True
    #@tf.function
    def call(self, inputs, training=None):
        #concrete = tf.function(self.layer.call).get_concrete_function(*[tf.TensorSpec(self.input_shape__, tf.float32)]) #,
        concrete = self.tf_f.get_concrete_function(*[tf.TensorSpec(self.input_shape__, tf.float32)])
                                                                      #tf.TensorSpec(input_shape, tf.float32))
        #@tf.function
        #def train_fn(inputs):
        #    inputs = tf.convert_to_tensor(inputs, tf.float32)
        #    input_map = {
        #      'inputs:0': inputs,
        #      'Identity/ReadVariableOp/resource:0': self.layer.a,
        #      'add/Identity/ReadVariableOp/resource:0': self.layer.b
        #    }
        #    # Import the graph giving x as input and getting the output y
        #    y = tf.graph_util.import_graph_def(
        #        gd, input_map=input_map, return_elements=['Identity:0'])[0]
        #    return y
        fn_train = make_new_func(concrete.graph.as_graph_def(),
                                 concrete.graph.captures,
                                 concrete.graph.variables,
                                 concrete.inputs,
                                 concrete.outputs)
        return fn_train(inputs)
        #strategy = tf.distribute.get_strategy()
        #return strategy.run(self.train_fn, (inputs, *self.vars))
        #return self.train_fn(inputs, *self.vars)
        #concrete = tf.function(self.layer.call).get_concrete_function(*[tf.TensorSpec(self.input_shape__, tf.float32)])
        #fn_train = make_new_func(concrete.graph.as_graph_def(),
        #                         concrete.graph.captures,
        #                         concrete.graph.variables,
        #                         concrete.inputs,
        #                         concrete.outputs)
        #print('\ntracing\n')
        #return self.layer.call(inputs)
        #return self.train_fn(inputs)
        #return self.layer.call(inputs)
        '''
        if tf.distribute.has_strategy():
            if not self.REBUILD_CONCREETE:
                fn_train = tf.function(self.layer.call).get_concrete_function(
                    tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=True)
                return fn_train(inputs)
            else:
                if not self.CREATE_GRAPH_ON_BUILD:
                    concr_train_fn = tf.function(self.layer.call).get_concrete_function(
                       tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=True)
                    fn_train = make_new_func(concr_train_fn.graph.as_graph_def(),
                                             concr_train_fn.graph.captures,
                                             concr_train_fn.graph.variables,
                                             concr_train_fn.inputs,
                                             concr_train_fn.outputs)
                    tf.print('\nTrace function\n')
                    return fn_train(inputs)
                return self.fn_train(inputs)

            if tf.distribute.in_cross_replica_context():

                from tensorflow.python.distribute import distribution_strategy_context
                with distribution_strategy_context._get_default_replica_context():
                    fn_train = tf.function(self.layer.call).get_concrete_function(
                       tf.TensorSpec(shape=inputs.shape, dtype=tf.float32))
                    return fn_train(inputs)
                #if training:
                #    from tensorflow.python.distribute import distribution_strategy_context
                #    with distribution_strategy_context._get_default_replica_context():
                #        # fn_train = make_new_func(self.concr_train_fn.graph.as_graph_def(),
                #        #                          self.concr_train_fn.graph.captures,
                #        #                          self.concr_train_fn.graph.variables,
                #        #                          self.concr_train_fn.inputs,
                #        #                          self.concr_train_fn.outputs)
                #        fn_train = tf.function(self.layer.call).get_concrete_function(
                #           tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=True)
                #        return fn_train(inputs)
                #else:
                #    from tensorflow.python.distribute import distribution_strategy_context
                #    with distribution_strategy_context._get_default_replica_context():
                #        fn_eval = tf.function(self.layer.call).get_concrete_function(
                #           tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=False)
                #        # fn_eval = make_new_func(self.concr_eval_fn.graph.as_graph_def(),
                #        #               self.concr_eval_fn.graph.captures,
                #        #               self.concr_eval_fn.graph.variables,
                #        #               self.concr_eval_fn.inputs,
                #        #               self.concr_eval_fn.outputs)
                #        return fn_eval(inputs)
            else:
                from tensorflow.python.distribute import distribution_strategy_context
                with distribution_strategy_context.get_replica_context():
                    fn_train = tf.function(self.layer.call).get_concrete_function(
                       tf.TensorSpec(shape=inputs.shape, dtype=tf.float32))
                    return fn_train(inputs)
                #if training:
                #    from tensorflow.python.distribute import distribution_strategy_context
                #    with distribution_strategy_context.get_replica_context():
                #        fn_train = tf.function(self.layer.call).get_concrete_function(
                #           tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=True)
                #        # fn_train = make_new_func(self.concr_train_fn.graph.as_graph_def(),
                #        #                          self.concr_train_fn.graph.captures,
                #        #                          self.concr_train_fn.graph.variables,
                #        #                          self.concr_train_fn.inputs,
                #        #                          self.concr_train_fn.outputs)
                #        return fn_train(inputs)
                #else:
                #    from tensorflow.python.distribute import distribution_strategy_context
                #    with distribution_strategy_context.get_replica_context():
                #        fn_eval = tf.function(self.layer.call).get_concrete_function(
                #           tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=False)
                #        # fn_eval = make_new_func(self.concr_eval_fn.graph.as_graph_def(),
                #        #                         self.concr_eval_fn.graph.captures,
                #        #                         self.concr_eval_fn.graph.variables,
                #        #                         self.concr_eval_fn.inputs,
                #        #                         self.concr_eval_fn.outputs)
                #        return fn_eval(inputs)
        else:
            if not self.CREATE_GRAPH_ON_BUILD:
                concr_train_fn = tf.function(self.layer.call).get_concrete_function(
                    tf.TensorSpec(shape=inputs.shape, dtype=tf.float32))
                return concr_train_fn(inputs)
            return self.fn_train(inputs)
            if training:
              # return self.concr_train_fn(inputs)
                concr_train_fn = tf.function(self.layer.call).get_concrete_function(
                    tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=True)
                # fn_train = make_new_func(concr_train_fn.graph.as_graph_def(),
                #                          concr_train_fn.graph.captures,
                #                          concr_train_fn.graph.variables,
                #                          concr_train_fn.inputs,
                #                          concr_train_fn.outputs)
                return concr_train_fn(inputs)
            else:
               #return self.concr_eval_fn(inputs)
                concr_eval_fn = tf.function(self.layer.call).get_concrete_function(
                    tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=False)
                # fn_eval = make_new_func(concr_eval_fn.graph.as_graph_def(),
                #                         concr_eval_fn.graph.captures,
                #                         concr_eval_fn.graph.variables,
                #                         concr_eval_fn.inputs,
                #                         concr_eval_fn.outputs)
                return concr_eval_fn(inputs)

        #return self._func(inputs, training)
'''


def make_new_func(output_graph_def, captures, variables, inputs, outputs):
    new_input_names = [tensor.name for tensor in inputs]
    inputs_map = {
        tensor.name: tensor for tensor in inputs
    }
    new_output_names = [tensor.name for tensor in outputs]
   # new_func = my_function_from_graph_def(output_graph_def,
   #                                       new_input_names,
   #                                       new_output_names,
   #                                       captures)
    new_func = mod_my_function_from_graph_def(output_graph_def,
                                              new_input_names,
                                              new_output_names,
                                              captures,)
                                              #inputs_map,
                                              #new_output_names)
    for input in new_func.inputs:
        input.set_shape(inputs_map[input.name].shape)
        break

    new_func.graph.variables = variables
    return new_func

def _rebuild_func(func):
    """Rebuild function from graph_def."""
    gd = func.graph.as_graph_def()
    rebuilt_func = wrap_function.function_from_graph_def(
        gd, [tensor.name for tensor in func.inputs],
        [tensor.name for tensor in func.outputs])
    rebuilt_func.graph.structured_outputs = nest.pack_sequence_as(
        func.graph.structured_outputs, rebuilt_func.graph.structured_outputs)
    # Copy structured input signature from original function (used during
    # serialization)
    rebuilt_func.graph.structured_input_signature = (
        func.structured_input_signature)

    return rebuilt_func


def mod_my_function_from_graph_def(graph_def, inputs, outputs, ref_captures):
    def _imports_graph_def():
        importer.import_graph_def(graph_def, name="")

    wrapped_import = wrap_function.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph
    wrapped_import.graph.reset_captures([(tensor, import_graph.get_tensor_by_name(placeholder.name))
                                         for tensor, placeholder in ref_captures])

    #op_type = compat.as_str(op.op_def.name)
    #attr_names = [compat.as_str(attr.name) for attr in op.op_def.attr]
    #attrs = []
    #for attr_name in attr_names:
    #  attrs.append(attr_name)
    #  attrs.append(op.get_attr(attr_name))
    #attrs = tuple(attrs)
    #execute.record_gradient(op_type, op.inputs, attrs, op.outputs)
    return wrapped_import.prune(
        nest.map_structure(import_graph.as_graph_element, inputs),
        nest.map_structure(import_graph.as_graph_element, outputs))


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

    # def _construct_concrete_function(func, output_graph_def,
    #                                  converted_input_indices):
    #     input_tensors = func.graph.internal_captures
    #     converted_inputs = object_identity.ObjectIdentitySet(
    #         [input_tensors[index] for index in converted_input_indices])
    #     not_converted_inputs = [
    #         tensor for tensor in func.inputs if tensor not in converted_inputs]
    #     not_converted_inputs_map = {
    #         tensor.name: tensor for tensor in not_converted_inputs
    #     }
    #
    #     new_input_names = [tensor.name for tensor in not_converted_inputs]
    #     new_output_names = [tensor.name for tensor in func.outputs]
    #     new_func = wrap_function.function_from_graph_def(output_graph_def,
    #                                                      new_input_names,
    #                                                      new_output_names)
    #
    #     # Manually propagate shape for input tensors where the shape is not correctly
    #     # propagated. Scalars shapes are lost when wrapping the function.
    #     for input_tensor in new_func.inputs:
    #         input_tensor.set_shape(not_converted_inputs_map[input_tensor.name].shape)
    #     return new_func


#-----------------------------------------------------------------------------------------------------

    # def build(self, input_shape=None):
    #     super().build(input_shape)
    #
    #     #from tensorflow.python.framework import func_graph
    #     #graph = func_graph.FuncGraph('build_graph')
    #
    #     #input = tf.random.uniform((1, 224, 224, 3))
    #     #with graph.as_default():
    #     #    output = self.layer.call(input)
    #
    #     # @tf.function
    #     # def wrap_callable(inputs, training=None):
    #     #     if training:
    #     #         fcall = tf.function(self.layer).get_concrete_function(
    #     #             tf.TensorSpec(shape=input_shape, dtype=tf.float32))
    #     #     else:
    #     #         fcall = tf.function(self.layer).get_concrete_function(
    #     #             tf.TensorSpec(shape=input_shape, dtype=tf.float32))
    #     #     return fcall(inputs, training)
    #     #
    #     # self.callable = self.layer
    #
    #     # concr_fn = tf.function(self.layer.call).get_concrete_function(
    #     #     tf.TensorSpec(shape=input_shape, dtype=tf.float32))
    #     #self.concr_fn_training = tf.function(self.layer.call).get_concrete_function(
    #     #    tf.TensorSpec(shape=input_shape, dtype=tf.float32), training=True)
    #     #self.concr_fn_eval = tf.function(self.layer.call).get_concrete_function(
    #     #    tf.TensorSpec(shape=input_shape, dtype=tf.float32), training=False)
    #
    #     # graph_def = concr_fn.graph.as_graph_def()
    #     # graph_def_1 = _run_inline_graph_optimization(concr_fn, True, False)
    #     #
    #     # tf.io.write_graph(graph_def, '/home/alexsu/work/tmp', 'concr_fn.pb',
    #     #                   as_text=False)
    #     # tf.io.write_graph(graph_def_1, '/home/alexsu/work/tmp', 'concr_fn.pb',
    #     #                   as_text=False)
    #     # tf.io.write_graph(concr_fn_eval.graph.as_graph_def(), '/home/alexsu/work/tmp', 'concr_fn_eval.pb',
    #     #                   as_text=False)
    #     # tf.io.write_graph(concr_fn_training.graph.as_graph_def(), '/home/alexsu/work/tmp', 'concr_fn_training.pb',
    #     #                   as_text=False)
    #
    #
    #     #raph_def = _run_inline_graph_optimization(concr_fn, True, False)
    #
    #
    #     #
    #     # self.callable = make_new_func(graph_def,
    #     #                               captures=concr_fn.graph.captures,
    #     #                               variables=concr_fn.graph.variables,
    #     #                               inputs=concr_fn.inputs,
    #     #                               outputs=concr_fn.outputs)
    #
    #
    #     #graph_, concr_fn, func_graph = self._get_inner_graph(input_shape)
    #
    #     # concr_fn = tf.function(self.layer).get_concrete_function(
    #     #                 tf.TensorSpec(shape=input_shape, dtype=tf.float32))
    #     # graph_def = _run_inline_graph_optimization(concr_fn, True, False)
    #     # concr_fn = make_new_func(graph_def, concr_fn.graph.captures,
    #     #                          concr_fn.variables, concr_fn.inputs, concr_fn.outputs)
    #     # graph_to_layer_names_map = get_graph_to_layer_var_names_map(concr_fn)
    #     #
    #     # with concr_fn.graph.as_default() as graph:
    #     #     self.prune_ops = [node.inputs[-1].op for node in graph.get_operations() if node.type == 'Conv2D']   # for experiment
    #     # # with graph.as_default():
    #     #     # for weight_attr, ops in self.weights_attr_ops.items():
    #     #     #     weight = self._get_layer_weight(graph, weight_attr)
    #     #     #     for op_name, op in ops.items():
    #     #     #         self._ops_weights[op_name] = self._insert_op(op, weight, graph)
    #     #     for weight in self.prune_ops:
    #     #         self._ops_weights[weight.name + 'mask'] = self._insert_op(None, weight, graph, graph_to_layer_names_map)
    #     #         self._non_trainable_weights.append(self._ops_weights[weight.name + 'mask'])  # ?
    #     #         self.pruned_weight_name = weight.name  # for experiment
    #     #         self.pruned_weight = weight  # for experiment
    #     #         self.graph_to_layer_names_map = graph_to_layer_names_map  # for experiment
    #     #         break
    #     #     softmax = tf.nn.softmax(graph.outputs[0])
    #     # self.callable = make_new_func(graph.as_graph_def(),
    #     #                               captures=graph.captures,
    #     #                               variables=concr_fn.graph.variables + tuple(self._ops_weights.values()),
    #     #                               inputs=concr_fn.inputs, outputs=[softmax])
    #     # self.callable.graph.variables = concr_fn.graph.variables + tuple(self._ops_weights.values())
    #     # # concr_fn = tf.function(self.layer).get_concrete_function(
    #     # #             tf.TensorSpec(shape=input_shape, dtype=tf.float32))
    #     # # graph_def = _run_inline_graph_optimization(concr_fn, False, False)
    #     # # self.callable = make_new_func(concr_fn, graph_def)
    #     # # with self.callable.graph.as_default() as graph:
    #     # #     relus = [op for op in graph.get_operations() if op.type == 'Relu6']
    #     # #     consumers = find_consumers(graph, relus[0])
    #     # #     # with graph.name_scope('scales') as scope:
    #     # #     #     self.scale = tf.Variable(
    #     # #     #         initial_value=1.0, trainable=True,
    #     # #     #         name='scale', dtype=tf.float32)
    #     # #     # fq = symmetric_quantize(
    #     # #     #     relus[0].outputs[0],
    #     # #     #     self.scale,
    #     # #     #     num_bits=8,
    #     # #     #     per_channel=False,
    #     # #     #     narrow_range=False,
    #     # #     #     signed=False,
    #     # #     #     eps=1e-16)
    #     # #     # reroute_tensor(fq, relus[0].outputs[0], consumers)
    #     # #     with graph.name_scope('my_layer') as scope:
    #     # #         self.mask = tf.compat.v1.get_variable(initializer=tf.constant_initializer(1.0), shape=1,
    #     # #                                               trainable=False, name='mask', dtype=tf.float32)
    #     # #     mul = tf.math.multiply(relus[0].outputs[0], self.mask, name='mask_multiple')
    #     # #     reroute_tensor(mul, relus[0].outputs[0], consumers)
    #     # #     softmax = tf.nn.softmax(graph.outputs[0])
    #     # #
    #     # #     self.callable = make_new_func(self.callable, graph.as_graph_def(),
    #     # #                                   outputs=[softmax])
    #     # # self.callable.graph.variables = concr_fn.graph.variables + tuple([self.mask])
    #     # # self.callable.graph.variables = concr_fn.graph.variables + tuple([self.scale])

#     def call(self, inputs, training=None):
#     #     # if not tf.distribute.in_cross_replica_context():
#     #     #     # id = int(tf.distribute.get_replica_context().replica_id_in_sync_group)
#     #     #     id = int(tf.distribute.get_replica_context().devices[0][0][-1])
#     #     #     if id not in self._concrete_func:
#     #     #         concr_fun = tf.function(self.layer).get_concrete_function(
#     #     #             tf.TensorSpec(shape=self.in_shape, dtype=tf.float32))
#     #     #         with concr_fun.graph.as_default() as graph:
#     #     #             tf.nn.softmax(graph.outputs[0])
#     #     #         self._concrete_func[id] = concr_fun
#     #     #
#     #     #     return self._concrete_func[id](inputs)
#     #     # else:
#     #     #     concr_fun = tf.function(self.layer).get_concrete_function(
#     #     #         tf.TensorSpec(shape=self.in_shape, dtype=tf.float32))
#     #     #     with concr_fun.graph.as_default() as graph:
#     #     #         tf.nn.softmax(graph.outputs[0])
#     #     #     return concr_fun(inputs)
#     #     # self.step += 1   # for experiment
#     #     # if self.step == 1002:   # for experiment
#     #     #     self.set_mask(self.calculate_mask_for_threshold())   # for experiment
#     #     # return self.callable(inputs)
# #        print("I'm here!!!!!!!!!!!!!!!!!!")
# #        tf.print("TF: I'm here!!!!!!!!!!!!!!!!!!")
#
#         from tensorflow.python.eager import wrap_function
#         from tensorflow.python.util import object_identity
#
#         def function_from_graph_def(graph_def, inputs, outputs, ref_captures):
#             """Creates a ConcreteFunction from a GraphDef.
#
#             Args:
#               graph_def: A GraphDef to make a function out of.
#               inputs: A Tensor name or nested structure of names in `graph_def` which
#                 should be inputs to the function.
#               outputs: A Tensor name or nested structure of names in `graph_def` which
#                 should be outputs of the function.
#               ref_captures: A List of captures of original graph
#
#             Returns:
#               A ConcreteFunction.
#             """
#
#             def _imports_graph_def():
#                 importer.import_graph_def(graph_def, name="")
#
#             wrapped_import = wrap_function.wrap_function(_imports_graph_def, [])
#             import_graph = wrapped_import.graph
#             wrapped_import.graph.reset_captures([(tensor, import_graph.get_tensor_by_name(placeholder.name))
#                                                  for tensor, placeholder in ref_captures])
#             return wrapped_import.prune(
#                 nest.map_structure(import_graph.as_graph_element, inputs),
#                 nest.map_structure(import_graph.as_graph_element, outputs))
#
#         def construct_concrete_function(func, output_graph_def,
#                                          converted_input_indices):
#             """Constructs a concrete function from the `output_graph_def`.
#
#             Args:
#               func: ConcreteFunction
#               output_graph_def: GraphDef proto.
#               converted_input_indices: Set of integers of input indices that were
#                 converted to constants.
#
#             Returns:
#               ConcreteFunction.
#             """
#             # Create a ConcreteFunction from the new GraphDef.
#             input_tensors = func.graph.internal_captures
#             converted_inputs = object_identity.ObjectIdentitySet(
#                 [input_tensors[index] for index in converted_input_indices])
#             not_converted_inputs = [
#                 tensor for tensor in func.inputs if tensor not in converted_inputs]
#             not_converted_inputs_map = {
#                 tensor.name: tensor for tensor in not_converted_inputs
#             }
#
#             new_input_names = [tensor.name for tensor in not_converted_inputs]
#             new_output_names = [tensor.name for tensor in func.outputs]
#             new_func = function_from_graph_def(output_graph_def,
#                                                new_input_names,
#                                                new_output_names)
#
#             # Manually propagate shape for input tensors where the shape is not correctly
#             # propagated. Scalars shapes are lost when wrapping the function.
#             for input_tensor in new_func.inputs:
#                 input_tensor.set_shape(not_converted_inputs_map[input_tensor.name].shape)
#             return new_func
#
#
#         # @tf.function
#         def my_call(inputs, training):
#             if training:
#                 fn = tf.function(self.layer.call).get_concrete_function(
#                     tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=True)
#             else:
#                 fn = tf.function(self.layer.call).get_concrete_function(
#                     tf.TensorSpec(shape=inputs.shape, dtype=tf.float32), training=False)
#             return fn(inputs)
#
#         print('Im here')
#         res = my_call(inputs, training)
#
#         #res = self.layer(inputs, training)
#         return res

    # def registry_weight_operation(self, weights_attr, op, op_name=None):
    #     if weights_attr not in self.weights_attr_ops:
    #         self.weights_attr_ops[weights_attr] = OrderedDict()
    #
    #     if op_name is None:
    #         op_name = 'nncf_op_{}:{}'.format(weights_attr, len(self.weights_attr_ops[weights_attr]))
    #
    #     self.weights_attr_ops[weights_attr][op_name] = op
    #     return op_name

    # def _get_inner_graph(self, input_shape, input_map=None, return_elements=None):
    #     concr_fn = tf.function(self.layer).get_concrete_function(
    #         tf.TensorSpec(shape=input_shape, dtype=tf.float32))
    #     graph_def = _run_inline_graph_optimization(concr_fn, True, False)
    #     with tf.Graph().as_default() as graph:
    #         tf.graph_util.import_graph_def(graph_def, input_map, return_elements)  # adds 'import' scope?
    #     func_graph = FuncGraph('')
    #     # for tensor, placeholder in concr_fn.graph.captures:
    #     #     func_graph.add_capture(tensor, placeholder)
    #     return graph, concr_fn, func_graph
    #
    # def _get_layer_weight(self, graph, weight_attr):
    #     for variable in graph.get_operations():
    #         if variable.name == weight_attr:
    #             return variable
    #     return None

    # def _insert_op(self, op, weight_op, graph, graph_to_layer_names_map):
    #     # graph_var_name = traverse(weight_op).outputs[0].name.split('/', 1)[1]  # import/keras_layer_1/...
    #     graph_var_name = traverse(weight_op).outputs[0].name
    #     variable = None
    #     for var in self.layer.weights:
    #         if var.name == graph_to_layer_names_map[graph_var_name]:
    #             variable = var
    #     with graph.as_default():
    #         with graph.name_scope('nncf_ops'):
    #             # mask = get_variable(initializer=tf.constant_initializer(1.0), shape=variable.shape,
    #             #                     trainable=False, name=weight_op.name + 'mask', dtype=tf.float32)
    #             consumers = find_consumers(graph, weight_op)
    #             mask = self.add_weight(
    #                 variable.name.rstrip(':0') + '_mask',
    #                 shape=variable.shape,
    #                 initializer=tf.keras.initializers.Constant(1.0),
    #                 trainable=False,
    #                 aggregation=tf.VariableAggregation.MEAN)
    #             output_tensor = weight_op.outputs[0]
    #             op = tf.math.multiply(output_tensor, mask)
    #             reroute_tensor(op, output_tensor, consumers)
    #
    #     return mask
    #
    # def calculate_mask_for_threshold(self):
    #     graph_var_name = traverse(self.pruned_weight).outputs[0].name
    #     variable = None
    #     for var in self.layer.weights:
    #         if var.name == self.graph_to_layer_names_map[graph_var_name]:
    #             variable = var
    #     threshold = calculate_threshold(variable, 0.7)
    #     mask = tf.cast(tf.math.abs(variable) > threshold, tf.float32)
    #     a = mask.numpy()
    #     print("SPARSITY LEVEL {}".format(1-a.sum()/a.size))
    #     return mask
    #
    # @tf.function
    # def set_mask(self, mask):
    #     self._ops_weights[self.pruned_weight_name + 'mask'].assign(mask)


# def find_consumers(graph, op):
#     inputs_to_ops = defaultdict(set)
#     for op_ in graph.get_operations():
#         for op_input in op_.inputs:
#             inputs_to_ops[op_input.name].add(op_)
#
#     result = set()
#     for output in op.outputs:
#         result.update(inputs_to_ops[output.name])
#     return result
#
#
# def reroute_tensor(t0, t1, can_modify=None):
#     """Reroute the end of the tensor t0 to the ends of the tensor t1.
#   Args:
#     t0: a tf.Tensor.
#     t1: a tf.Tensor.
#     can_modify: iterable of operations which can be modified. Any operation
#       outside within_ops will be left untouched by this function.
#   Returns:
#     The number of individual modifications made by the function.
#   """
#     nb_update_inputs = 0
#     consumers = t1.consumers()
#     if can_modify is not None:
#         consumers = [c for c in consumers if c in can_modify]
#     consumers_indices = {}
#     for c in consumers:
#         consumers_indices[c] = [i for i, t in enumerate(c.inputs) if t is t1]
#     for c in consumers:
#         for i in consumers_indices[c]:
#             c._update_input(i, t0)  # pylint: disable=protected-access
#             nb_update_inputs += 1
#     return nb_update_inputs
#
#
# #
#
# def traverse(node):
#     while node.inputs:
#         node = node.inputs[-1].op
#     return node
#
#
# def get_graph_to_layer_var_names_map(concrete_fun):
#     names_map = {}
#     for layer_var in concrete_fun.variables:
#         for value_tensor, graph_name in concrete_fun.graph.captures:
#             if layer_var.handle is value_tensor:
#                 names_map[graph_name.name] = layer_var.name
#     return names_map
#
#
# def calculate_threshold(weight, sparsity_level):
#     w = tf.sort(tf.reshape(tf.math.abs(weight), [-1]))
#     index = int(tf.cast(tf.size(w), w.dtype) * sparsity_level)
#     return w[index].numpy()
#
