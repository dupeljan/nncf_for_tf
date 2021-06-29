import tensorflow as tf

from tensorflow.python.framework import importer
from tensorflow.python.eager import wrap_function
from tensorflow.python.pywrap_tfe import TFE_Py_TapeSetShouldRecordBackprop as \
   check_tensor_in_tape
from tensorflow.python.ops.resource_variable_ops import variable_accessed as \
    add_resource_var_in_tape

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

OUT_GRAPH_PATH = '/tmp/graph_def_test.txt'
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
        self.tf_f = tf.function(self.layer.call)
        #self.tf_f(tf.ones((1,) + input_shape[1:]))
        #from google.protobuf import text_format
        #proto_b = open(OUT_GRAPH_PATH, 'r').read()
        #gd = tf.compat.v1.GraphDef()
        #text_format.Merge(proto_b, gd)
        #self.op_weight_shape = (3, 1, 1)
        #self.op_weigths = tf.Variable(tf.ones(self.op_weight_shape))
        concrete = self.tf_f.get_concrete_function(*[tf.TensorSpec(self.input_shape__, tf.float32)])
        # Create graph op which will be added to layer
        op_concrete, self.op_vars = self.get_custom_graph_fun(input_shape)
        #new_op = \
        #    make_new_func(op_concrete.graph.as_graph_def(),
        #                  op_concrete.graph.captures,
        #                  op_concrete.variables,
        #                  op_concrete.inputs,
        #                  op_concrete.outputs)

        # Add new op to layer
        with concrete.graph.as_default() as g:
            op_concrete(g.outputs[0])

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
        concrete = make_new_func(concrete.graph.as_graph_def(),
                                 concrete.graph.captures,
                                 concrete.graph.variables,
                                 concrete.inputs,
                                 op_concrete.outputs)

        #fn_train = make_new_func(concrete.graph.as_graph_def(),
        #                         concrete.graph.captures,
        #                         concrete.graph.variables,
        #                         concrete.inputs,
        #                         concrete.outputs)

        #with open(OUT_GRAPH_PATH, 'w') as out:
        #    out.write(str(concrete.graph.as_graph_def()))

        #exit()
        self.fn_train = concrete
        self.op_concrete = op_concrete
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
        replica_context = tf.distribute.get_replica_context()
        if replica_context is not None:
            replica_id = replica_context.replica_id_in_sync_group
            new_variables = []
            new_captured = []
            for var, input_tensor in zip(self.layer.variables + self.op_vars, self.fn_train.inputs[1:]):
                new_variables.append(var._get_replica(replica_id))
                new_captured.append((var._get_replica(replica_id).handle, input_tensor))

        else:
            new_variables = self.fn_train.graph.variables
            new_captured = self.fn_train.graph.captures

        fn_train = make_new_func(self.fn_train.graph.as_graph_def(),
                                 new_captured,
                                 new_variables,
                                 self.fn_train.inputs,
                                 self.op_concrete.outputs)

        # Recreate variables
        #func_graph = FuncGraph('')
        #with func_graph.as_default():
        #    outputs = fn_train(inputs)

        vars_id = sorted(get_concrete_vars_id(fn_train))
        captures_id = sorted(get_concrete_captured_id(fn_train))
        if vars_id != captures_id:
            # It doesn't work here, but inside concrete function call vars_id changes somehow
            print('Gradients will not leak because id\'s id is differs')

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


def setup_only_concrete_fun(g, call):
    from tensorflow.python.eager.function import FunctionSpec
    from tensorflow.python.eager.function import ConcreteFunction
    from tensorflow.python.util import object_identity
    spec = FunctionSpec.from_function_and_signature(
        call,
        None)
    graph_function = ConcreteFunction(g, function_spec=spec)
    seen_names = set()
    captured = object_identity.ObjectIdentitySet(
        graph_function.graph.internal_captures)
    # pylint: disable=protected-access
    graph_function._arg_keywords = []
    prefix_counts = {}
    # pylint: enable=protected-access
    num_positional = 0
    for arg in graph_function.graph.inputs:
        if arg in captured:
            break
        num_positional += 1
        #user_arg_name = compat.as_str(arg.op.get_attr("_user_specified_name"))
        #proposal = user_arg_name
        #while proposal in seen_names:
        #    index = prefix_counts.get(user_arg_name, 1)
        #    proposal = "{}_{}".format(user_arg_name, index)
        #    prefix_counts[user_arg_name] = index + 1
        #seen_names.add(proposal)

    #graph_function._arg_keywords.append(proposal)  # pylint: disable=protected-access
    # Anything can be a positional argument, in the same order as .inputs
    graph_function._num_positional_args = num_positional  # pylint: disable=protected-access
    return graph_function


def setup_concrete_fun(fn_train, call):
    from tensorflow.python.eager.function import FunctionSpec
    from tensorflow.python.eager.function import ConcreteFunction
    from tensorflow.python.util import object_identity
    from tensorflow.python.framework import auto_control_deps
    from tensorflow.python.util import nest

    deps_control_manager = auto_control_deps.AutomaticControlDependencies()
    g = fn_train.graph
    with g.as_default(), deps_control_manager as deps_ctx:
        def convert(x):
            return deps_ctx.mark_as_return(x)

        g.structured_outputs = nest.map_structure(convert, g.outputs,
                                                  expand_composites=True)
        # Returning a closed-over tensor does not trigger convert_to_tensor.
        g.outputs.extend(
            g.capture(x)
            for x in flatten(g.structured_outputs)
            if x is not None)

    g.control_outputs.extend(deps_control_manager.ops_which_must_run)
    g.collective_manager_ids_used = (
        deps_control_manager.collective_manager_ids_used)

    #return fn_train
    spec = FunctionSpec.from_function_and_signature(
        call,
        None)
    graph_function = ConcreteFunction(g, function_spec=spec)
    seen_names = set()
    captured = object_identity.ObjectIdentitySet(
        graph_function.graph.internal_captures)
    # pylint: disable=protected-access
    graph_function._arg_keywords = []
    prefix_counts = {}
    # pylint: enable=protected-access
    num_positional = 0
    for arg in graph_function.graph.inputs:
        if arg in captured:
            break
        num_positional += 1
        #user_arg_name = compat.as_str(arg.op.get_attr("_user_specified_name"))
        #proposal = user_arg_name
        #while proposal in seen_names:
        #    index = prefix_counts.get(user_arg_name, 1)
        #    proposal = "{}_{}".format(user_arg_name, index)
        #    prefix_counts[user_arg_name] = index + 1
        #seen_names.add(proposal)

    #graph_function._arg_keywords.append(proposal)  # pylint: disable=protected-access
    # Anything can be a positional argument, in the same order as .inputs
    graph_function._num_positional_args = num_positional  # pylint: disable=protected-access
    return graph_function


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



def func_graph_from_func_graph(name,
                               python_func,
                               args,
                               kwargs,
                               signature=None,
                               func_graph=None,
                               autograph=False,
                               autograph_options=None,
                               add_control_dependencies=True,
                               arg_names=None,
                               op_return_value=None,
                               collections=None,
                               capture_by_value=None,
                               override_flat_arg_shapes=None):
    """Returns a `FuncGraph` generated from `python_func`.

    Args:
      name: an identifier for the function.
      python_func: the Python function to trace.
      args: the positional args with which the Python function should be called;
        ignored if a signature is provided.
      kwargs: the keyword args with which the Python function should be called;
        ignored if a signature is provided.
      signature: a possibly nested sequence of `TensorSpecs` specifying the shapes
        and dtypes of the arguments. When a signature is provided, `args` and
        `kwargs` are ignored, and `python_func` is traced with Tensors conforming
        to `signature`. If `None`, the shapes and dtypes are inferred from the
        inputs.
      func_graph: Optional. An instance of FuncGraph. If provided, we will use
        this graph else a new one is built and returned.
      autograph: whether to use autograph to compile `python_func`.
        See https://www.tensorflow.org/guide/autograph for more information.
      autograph_options: additional knobs to control when `autograph=True`.
        See https://www.tensorflow.org/guide/autograph for more information.
      add_control_dependencies: If True, automatically adds control dependencies
        to ensure program order matches execution order and stateful ops always
        execute.
      arg_names: Optional list of argument names, used to give input placeholders
        recognizable names.
      op_return_value: Optional. A Tensor. If set and `python_func` returns
        Operations, those return values will be replaced with this value. If not
        set, returning an Operation triggers an error.
      collections: a dictionary of collections this FuncGraph should start
        with. If not specified (None), the FuncGraph will read (but not write to)
        the outer graph's collections that are not allowlisted, and both
        read and write to the outer graph's collections that are allowlisted.
        The current allowlisted collections are the global variables, the
        local variables, and the trainable variables.
        Defaults to None.
      capture_by_value: An optional boolean. If True, the func graph will capture
        Variables by value instead of reference. By default inherit from outer
        graphs, and failing that will default to False.
      override_flat_arg_shapes: An optional list of instances that are either
        `None` or `TensorShape`.  The length must match that of
        `nest.flatten((args, kwargs), expand_composites=True)`.  The entries
        containing value `None` must match entries in flattened arguments
        containing non-tensors, while entries containing a `TensorShape` must
        match entries in the flattened arguments containing tensors.

    Returns:
      A FuncGraph.

    Raises:
      TypeError: If any of `python_func`'s return values is neither `None` nor a
        `Tensor`.
      ValueError: If both `signature` and `override_flat_arg_shapes` are
        passed in.
    """
    if op_return_value is not None:
        assert isinstance(op_return_value, ops.Tensor), op_return_value
    if func_graph is None:
        func_graph = FuncGraph(name, collections=collections,
                               capture_by_value=capture_by_value)
    assert isinstance(func_graph, FuncGraph)
    if add_control_dependencies:
        deps_control_manager = auto_control_deps.AutomaticControlDependencies()
    else:
        deps_control_manager = ops.NullContextmanager()

    with func_graph.as_default(), deps_control_manager as deps_ctx:
        current_scope = variable_scope.get_variable_scope()
        default_use_recource = current_scope.use_resource
        current_scope.set_use_resource(True)

        if signature is not None and override_flat_arg_shapes is not None:
            raise ValueError(
                "Passed both signature and override_flat_arg_shapes: %s and %s."
                % (signature, override_flat_arg_shapes))

        if signature is not None:
            args = signature
            kwargs = {}

        # Creates and names placeholders for all arguments.
        if override_flat_arg_shapes is not None:
            flat_args = nest.flatten(args, expand_composites=True)
            arg_shapes = override_flat_arg_shapes[:len(flat_args)]
            kwarg_shapes = override_flat_arg_shapes[len(flat_args):]
        else:
            arg_shapes = None
            kwarg_shapes = None
        func_args = _get_defun_inputs_from_args(
            args, arg_names, flat_shapes=arg_shapes)
        func_kwargs = _get_defun_inputs_from_kwargs(
            kwargs, flat_shapes=kwarg_shapes)

        # Convert all Tensors into TensorSpecs before saving the structured inputs.
        # If storing pure concrete functions that are not called through polymorphic
        # functions, we don't have access to FunctionSpec, so we need to call the
        # TensorSpecs by their `arg_names` for later binding.
        func_graph.structured_input_signature = (
            convert_structure_to_signature(func_args, arg_names),
            convert_structure_to_signature(func_kwargs))

        flat_func_args = nest.flatten(func_args, expand_composites=True)
        flat_func_kwargs = nest.flatten(func_kwargs, expand_composites=True)
        # Temporarily set inputs to allow graph building code to inspect
        # them. Reassigned below.
        func_graph.inputs = [arg for arg in flat_func_args + flat_func_kwargs
                             if isinstance(arg, ops.Tensor)]

        # Note: `nest.flatten` sorts by keys, as does `_deterministic_dict_values`.
        # Variables to help check whether mutation happens in calling the function
        # Copy the recursive list, tuple and map structure, but not base objects
        func_args_before = nest.pack_sequence_as(func_args, flat_func_args,
                                                 expand_composites=True)
        func_kwargs_before = nest.pack_sequence_as(
            func_kwargs, flat_func_kwargs, expand_composites=True)

        def convert(x):
            """Converts a function output to a Tensor."""
            if x is None:
                return None
            if op_return_value is not None and isinstance(x, ops.Operation):
                # TODO(b/79881896): we currently can't capture external control deps, so
                # this won't work if x needs to be captured (i.e. if python_func returns
                # captured Operations).
                with ops.control_dependencies([x]):
                    x = array_ops.identity(op_return_value)
            elif not isinstance(x, tensor_array_ops.TensorArray):
                try:
                    x = ops.convert_to_tensor_or_composite(x)
                except (ValueError, TypeError):
                    raise TypeError(
                        "To be compatible with tf.eager.defun, Python functions "
                        "must return zero or more Tensors; in compilation of %s, found "
                        "return value of type %s, which is not a Tensor." %
                        (str(python_func), type(x)))
            if add_control_dependencies:
                x = deps_ctx.mark_as_return(x)
            return x

        try:
            if autograph:
                from tensorflow.python import autograph  # pylint: disable=g-import-not-at-top
                _, original_func = tf_decorator.unwrap(python_func)

                def wrapper(*args, **kwargs):
                    """Calls a converted version of original_func."""
                    # TODO(mdan): Push this block higher in tf.function's call stack.
                    try:
                        return autograph.converted_call(
                            original_func,
                            args,
                            kwargs,
                            options=autograph.ConversionOptions(
                                recursive=True,
                                optional_features=autograph_options,
                                user_requested=True,
                            ))
                    except Exception as e:  # pylint:disable=broad-except
                        if hasattr(e, "ag_error_metadata"):
                            raise e.ag_error_metadata.to_exception(e)
                        else:
                            raise

                # Wrapping around a decorator allows checks like tf_inspect.getargspec
                # to be accurate.
                converted_func = tf_decorator.make_decorator(original_func, wrapper)
                python_func = tf_decorator.rewrap(python_func, original_func,
                                                  converted_func)

            else:
                _, original_func = tf_decorator.unwrap(python_func)

            func_outputs = python_func(*func_args, **func_kwargs)

            # invariant: `func_outputs` contains only Tensors, CompositeTensors,
            # TensorArrays and `None`s.
            func_outputs = nest.map_structure(convert, func_outputs,
                                              expand_composites=True)

            check_mutation(func_args_before, func_args, original_func)
            check_mutation(func_kwargs_before, func_kwargs, original_func)
        finally:
            current_scope.set_use_resource(default_use_recource)

        # Variables in `func_args`, `func_kwargs` should be explicit inputs
        # to the function, not captured inputs.
        graph_variables = list(func_graph._watched_variables)  # pylint: disable=protected-access
        arg_variables = object_identity.ObjectIdentitySet()
        inputs = []
        for arg in (nest.flatten(func_args, expand_composites=True) +
                    nest.flatten(func_kwargs, expand_composites=True)):
            if isinstance(arg, resource_variable_ops.BaseResourceVariable):
                # Even if an argument variable was not used in the function, we've
                # already manually captured the resource Tensor when creating argument
                # placeholders.
                resource_placeholder = func_graph.pop_capture(arg.handle)
                if resource_placeholder is None:
                    continue
                arg_variables.add(arg)
                inputs.append(resource_placeholder)
            elif isinstance(arg, ops.Tensor):
                inputs.append(arg)
        variables = [v for v in graph_variables if v not in arg_variables]
        func_graph.inputs = (
                inputs + func_graph.internal_captures + nest.flatten(
            func_graph.deferred_internal_captures, expand_composites=True))
        func_graph.structured_outputs = func_outputs
        # Returning a closed-over tensor does not trigger convert_to_tensor.
        func_graph.outputs.extend(
            func_graph.capture(x)
            for x in flatten(func_graph.structured_outputs)
            if x is not None)

        func_graph.variables = variables

    if add_control_dependencies:
        func_graph.control_outputs.extend(deps_control_manager.ops_which_must_run)
        func_graph.collective_manager_ids_used = (
            deps_control_manager.collective_manager_ids_used)

    return func_graph


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

