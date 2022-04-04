# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, input):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                current_input = ag__.ld(input)

                def get_state():
                    return (out, current_input)

                def set_state(vars_):
                    nonlocal out, current_input
                    (out, current_input) = vars_

                def loop_body(itr):
                    nonlocal out, current_input
                    (dense, dropout) = itr
                    d = ag__.converted_call(ag__.ld(dense), (ag__.ld(current_input),), None, fscope)
                    d = ag__.converted_call(ag__.ld(dropout), (ag__.ld(d),), None, fscope)
                    current_input = ag__.ld(d)
                    out = ag__.ld(current_input)
                out = ag__.Undefined('out')
                dropout = ag__.Undefined('dropout')
                d = ag__.Undefined('d')
                dense = ag__.Undefined('dense')
                ag__.for_stmt(ag__.converted_call(ag__.ld(zip), (ag__.ld(self).dense, ag__.ld(self).dropout), None, fscope), None, loop_body, get_state, set_state, ('out', 'current_input'), {'iterate_names': '(dense, dropout)'})
                out = ag__.converted_call(ag__.ld(self).dense_n, (ag__.ld(out),), None, fscope)
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory