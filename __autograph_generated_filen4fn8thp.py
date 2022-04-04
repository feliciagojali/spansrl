# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, input):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                x = ag__.converted_call(ag__.ld(self).embedding, (ag__.ld(input),), None, fscope)
                x = ag__.converted_call(ag__.ld(self).dropout_1, (ag__.ld(x),), None, fscope)
                c = [ag__.converted_call(ag__.ld(conv1d_out), (ag__.ld(x),), None, fscope) for conv1d_out in ag__.ld(self).conv1d_out]
                m = [ag__.converted_call(ag__.ld(maxpool_out), (ag__.ld(x),), None, fscope) for (maxpool_out, x) in ag__.converted_call(ag__.ld(zip), (ag__.ld(self).maxpool_out, ag__.ld(c)), None, fscope)]
                f = [ag__.converted_call(ag__.ld(flatten), (ag__.ld(x),), None, fscope) for (flatten, x) in ag__.converted_call(ag__.ld(zip), (ag__.ld(self).flatten, ag__.ld(m)), None, fscope)]
                f = [ag__.converted_call(ag__.ld(dropout_2), (ag__.ld(x),), None, fscope) for (dropout_2, x) in ag__.converted_call(ag__.ld(zip), (ag__.ld(self).dropout_2, ag__.ld(f)), None, fscope)]
                out = ag__.converted_call(ag__.ld(tf).concat, ([ag__.ld(x) for x in ag__.ld(f)], (- 1)), None, fscope)
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory