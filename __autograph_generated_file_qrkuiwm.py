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
                    nonlocal current_input, out
                    (out, current_input) = vars_

                def loop_body(itr):
                    nonlocal current_input, out
                    (lstm_f, lstm_b, highway) = itr
                    f = ag__.converted_call(ag__.ld(lstm_f), (ag__.ld(current_input),), None, fscope)
                    b = ag__.converted_call(ag__.ld(lstm_b), (ag__.ld(current_input),), None, fscope)
                    c = ag__.converted_call(ag__.ld(self).concatenate, ([ag__.ld(f), ag__.ld(b)],), None, fscope)
                    c = ag__.converted_call(ag__.ld(self).dropout, (ag__.ld(c),), None, fscope)
                    c = ag__.converted_call(ag__.ld(highway), (ag__.ld(c),), None, fscope)
                    out = ag__.ld(c)
                    current_input = ag__.ld(out)
                c = ag__.Undefined('c')
                lstm_b = ag__.Undefined('lstm_b')
                highway = ag__.Undefined('highway')
                f = ag__.Undefined('f')
                b = ag__.Undefined('b')
                lstm_f = ag__.Undefined('lstm_f')
                out = ag__.Undefined('out')
                ag__.for_stmt(ag__.converted_call(ag__.ld(zip), (ag__.ld(self).forwards, ag__.ld(self).backwards, ag__.ld(self).highway), None, fscope), None, loop_body, get_state, set_state, ('out', 'current_input'), {'iterate_names': '(lstm_f, lstm_b, highway)'})
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory