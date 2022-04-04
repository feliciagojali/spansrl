# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, inputs):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                (arg_emb, pred_emb) = ag__.ld(inputs)
                arg_emb_expanded = ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(arg_emb), 1), None, fscope)
                pred_emb_expanded = ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(pred_emb), 2), None, fscope)
                num_spans = ag__.ld(arg_emb_expanded).shape[2]
                num_preds = ag__.ld(pred_emb_expanded).shape[1]
                arg_emb_tiled = ag__.converted_call(ag__.ld(tf).tile, (ag__.ld(arg_emb_expanded), [1, ag__.ld(num_preds), 1, 1]), None, fscope)
                pred_emb_tiled = ag__.converted_call(ag__.ld(tf).tile, (ag__.ld(pred_emb_expanded), [1, 1, ag__.ld(num_spans), 1]), None, fscope)
                pair_emb_list = [ag__.ld(pred_emb_tiled), ag__.ld(arg_emb_tiled)]
                pair_emb = ag__.converted_call(ag__.ld(tf).concat, (ag__.ld(pair_emb_list), 3), None, fscope)
                out = ag__.ld(pair_emb)
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory