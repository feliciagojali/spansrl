# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, inputs):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                (arg_unary_score, pred_unary_score, pair_score, biaffine_score) = ag__.ld(inputs)
                num_args = ag__.ld(arg_unary_score).shape[1]
                num_preds = ag__.ld(pred_unary_score).shape[1]
                arg_score = ag__.converted_call(ag__.ld(tf).tile, (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(arg_unary_score), 1), None, fscope), [1, ag__.ld(num_preds), 1, ag__.ld(self).num_labels]), None, fscope)
                pred_score = ag__.converted_call(ag__.ld(tf).tile, (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(pred_unary_score), 2), None, fscope), [1, 1, ag__.ld(num_args), ag__.ld(self).num_labels]), None, fscope)
                total_score = ag__.converted_call(ag__.ld(self).add, ([ag__.ld(arg_score), ag__.ld(pred_score), ag__.ld(pair_score), ag__.ld(biaffine_score)],), None, fscope)
                null_score = ag__.converted_call(ag__.ld(self).null_score, ([ag__.ld(arg_unary_score), ag__.ld(pred_unary_score)],), None, fscope)
                final_score = ag__.converted_call(ag__.ld(self).concatenate, ([ag__.ld(total_score), ag__.ld(null_score)],), None, fscope)
                out = ag__.ld(final_score)
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory