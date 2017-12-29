import collections

import tensorflow as tf
import mock

def wrap_variable_creation(func, custom_getter):
    """Provides a custom getter for all variable creations."""
    original_get_variable = tf.get_variable
    def custom_get_variable(*args, **kwargs):
        if hasattr(kwargs, 'custom_getter'):
            raise AttributeError('Custom getters are not supported for '
                'optimizee variables.')
        return original_get_variable(*args, custom_getter=custom_getter, **kwargs)
    # Mock the get_variable method.
    with mock.patch("tensorflow.get_variable", custom_get_variable):
        return func()

def get_variables(func):
    """Calls func, returning any variables created, but ignoring its return value.

    Args:
        func: Function to be called.

    Returns:
        A tuple (variables, constants) where the first element is a list of
        trainable variables and the second is the non-trainable variables.
    """
    variables = []
    constants = []

    def custom_getter(getter, name, **kwargs):
        trainable = kwargs['trainable']
        kwargs['trainable'] = False
        variable = getter(name, **kwargs)
        if trainable:
            variables.append(variable)
        else:
            constants.append(variable)
        return variable

    with tf.name_scope("unused_graph"):
        wrap_variable_creation(func, custom_getter)

    return variables, constants

def run_with_custom_variables(func, variable):
    """Calls func and replaces any trainable variables.

    This returns the output of func, but whenever `get_variable` is called it
    will replace any trainable variables with the tensors in `variables`, in
    the same order. Non-trainable variables will re-use any variables already
    created.

    Args:
        func: Function to be called.
        variables: A list of tensors replacing the trainable variables.

    Returns:
        The return value of func is returned.
    """
    variables = collections.deque(variables)

    def custom_getter(getter, name, **kwargs):
        if kwargs["trainable"]:
            return variables.popleft()
        else:
            kwargs["reuse"] = True
        return getter(name, **kwargs)

    return wrap_variable_creation(func, custom_getter)
