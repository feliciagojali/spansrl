# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, inputs):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                (arg_emb, pred_emb) = ag__.ld(inputs)
                num_args = ag__.ld(arg_emb).shape[1]
                num_preds = ag__.ld(pred_emb).shape[1]
                batch_size = ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(arg_emb),), None, fscope)[0]
                out = ag__.converted_call(ag__.ld(tf).zeros, ([ag__.ld(batch_size), ag__.ld(num_preds), ag__.ld(num_args), 1],), None, fscope)
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory