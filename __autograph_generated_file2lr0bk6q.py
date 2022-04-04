# coding=utf-8
def outer_factory():
    model = None

    def inner_factory(ag__):

        def tf___wrapped_model(*args, **kwargs):
            "A concrete tf.function that wraps the model's call function."
            with ag__.FunctionScope('_wrapped_model', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=(), internal_convert_user_code=True)) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                ag__.ld(kwargs)['training'] = False
                with ag__.ld(base_layer_utils).call_context().enter(ag__.ld(model), inputs=None, build_graph=False, training=False, saving=True):
                    outputs = ag__.converted_call(ag__.ld(model), tuple(ag__.ld(args)), dict(**ag__.ld(kwargs)), fscope)
                output_names = ag__.ld(model).output_names

                def get_state():
                    return (output_names,)

                def set_state(vars_):
                    nonlocal output_names
                    (output_names,) = vars_

                def if_body():
                    nonlocal output_names
                    from keras.engine import compile_utils
                    output_names = ag__.converted_call(ag__.ld(compile_utils).create_pseudo_output_names, (ag__.ld(outputs),), None, fscope)

                def else_body():
                    nonlocal output_names
                    pass
                compile_utils = ag__.Undefined('compile_utils')
                ag__.if_stmt((ag__.ld(output_names) is None), if_body, else_body, get_state, set_state, ('output_names',), 1)
                outputs = ag__.converted_call(ag__.ld(tf).nest.flatten, (ag__.ld(outputs),), None, fscope)
                try:
                    do_return = True
                    retval_ = {ag__.ld(name): ag__.ld(output) for (name, output) in ag__.converted_call(ag__.ld(zip), (ag__.ld(output_names), ag__.ld(outputs)), None, fscope)}
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf___wrapped_model
    return inner_factory