import tensorflow as tf

from tensorflow.python.framework import importer
from tensorflow.python.eager import wrap_function
from tensorflow.python.util import nest


def insert_softmax_in_graph(fn_train):
    with fn_train.graph.as_default() as g:
        softmax = tf.nn.softmax(g.outputs[0])

        return make_new_func(g.as_graph_def(),
                             g.captures,
                             g.variables,
                             fn_train.inputs,
                             [softmax])


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

    def build(self, input_shape=None):
        self.layer.build(input_shape[1:])
        self.input_shape__ = input_shape
        self.tf_f = tf.function(self.layer.call)

    def call(self, inputs, training=None):
        concrete = self.tf_f.get_concrete_function(*[tf.TensorSpec(self.input_shape__, tf.float32)])
        # Before modifications
        #tf.print('before ', concrete(tf.ones((1,) + self.input_shape__[1:])))
        # Should be [[3 3 3 ... 3 3 3]]
        fn_train = insert_softmax_in_graph(concrete)
        # After modifications
        #tf.print('after ', fn_train(tf.ones((1,) + self.input_shape__[1:])))
        # Should be [[[6.64328218e-06 6.64328218e-06 6.64328218e-06 ... 6.64328218e-06 6.64328218e-06 6.64328218e-06]]]
        return fn_train(inputs)


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

