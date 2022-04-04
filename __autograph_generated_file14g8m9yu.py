# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, input):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                span_start = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(input), ag__.ld(self).start), dict(axis=1, name='span_start'), fscope)
                span_end = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(input), ag__.ld(self).end), dict(axis=1, name='span_end'), fscope)
                batch_size = ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(input),), None, fscope)[0]
                width = ag__.converted_call(ag__.ld(tf).convert_to_tensor, (ag__.ld(self).width,), None, fscope)
                expanded_width = ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(width), 0), None, fscope)
                span_length = ag__.converted_call(ag__.ld(tf).tile, (ag__.ld(expanded_width), [ag__.ld(batch_size), 1]), None, fscope)
                out = [ag__.ld(span_start), ag__.ld(span_end), ag__.ld(span_length)]
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory