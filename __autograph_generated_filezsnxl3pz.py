# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, x):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                dim = ag__.converted_call(ag__.ld(K).int_shape, (ag__.ld(x),), None, fscope)[(- 1)]
                transform_gate = ag__.converted_call(ag__.ld(self).dense_1, (ag__.ld(x),), None, fscope)
                transform_gate = ag__.converted_call(ag__.converted_call(ag__.ld(Activation), ('sigmoid',), None, fscope), (ag__.ld(transform_gate),), None, fscope)
                carry_gate = ag__.converted_call(ag__.converted_call(ag__.ld(Lambda), (ag__.autograph_artifact((lambda x: (1.0 - ag__.ld(x)))),), dict(output_shape=(ag__.ld(dim),)), fscope), (ag__.ld(transform_gate),), None, fscope)
                transformed_data = ag__.converted_call(ag__.ld(self).dense_2, (ag__.ld(x),), None, fscope)
                transformed_data = ag__.converted_call(ag__.converted_call(ag__.ld(Activation), (ag__.ld(self).activation,), None, fscope), (ag__.ld(transformed_data),), None, fscope)
                transformed_gated = ag__.converted_call(ag__.converted_call(ag__.ld(Multiply), (), None, fscope), ([ag__.ld(transform_gate), ag__.ld(transformed_data)],), None, fscope)
                identity_gated = ag__.converted_call(ag__.converted_call(ag__.ld(Multiply), (), None, fscope), ([ag__.ld(carry_gate), ag__.ld(x)],), None, fscope)
                value = ag__.converted_call(ag__.converted_call(ag__.ld(Add), (), None, fscope), ([ag__.ld(transformed_gated), ag__.ld(identity_gated)],), None, fscope)
                try:
                    do_return = True
                    retval_ = ag__.ld(value)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory