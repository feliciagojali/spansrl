# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, inputs, training=None):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()

                def get_state():
                    return (char_input, elmo_emb, word_emb)

                def set_state(vars_):
                    nonlocal char_input, elmo_emb, word_emb
                    (char_input, elmo_emb, word_emb) = vars_

                def if_body():
                    nonlocal char_input, elmo_emb, word_emb
                    (word_emb, elmo_emb, char_input) = ag__.ld(inputs)

                def else_body():
                    nonlocal char_input, elmo_emb, word_emb
                    (word_emb, char_input) = ag__.ld(inputs)
                char_input = ag__.Undefined('char_input')
                elmo_emb = ag__.Undefined('elmo_emb')
                word_emb = ag__.Undefined('word_emb')
                ag__.if_stmt(ag__.ld(self).second_emb, if_body, else_body, get_state, set_state, ('char_input', 'elmo_emb', 'word_emb'), 3)
                char_emb = ag__.converted_call(ag__.ld(self).character_block, (ag__.ld(char_input),), None, fscope)

                def get_state_1():
                    return (token_rep,)

                def set_state_1(vars_):
                    nonlocal token_rep
                    (token_rep,) = vars_

                def if_body_1():
                    nonlocal token_rep
                    token_rep = ag__.converted_call(ag__.ld(self).concatenate_1, ([ag__.ld(word_emb), ag__.ld(elmo_emb), ag__.ld(char_emb)],), None, fscope)

                def else_body_1():
                    nonlocal token_rep
                    token_rep = ag__.converted_call(ag__.ld(self).concatenate_1, ([ag__.ld(word_emb), ag__.ld(char_emb)],), None, fscope)
                token_rep = ag__.Undefined('token_rep')
                ag__.if_stmt(ag__.ld(self).second_emb, if_body_1, else_body_1, get_state_1, set_state_1, ('token_rep',), 1)
                token_rep = ag__.converted_call(ag__.ld(self).dropout_1, (ag__.ld(token_rep),), None, fscope)
                token_rep = ag__.converted_call(ag__.ld(self).bihlstm, (ag__.ld(token_rep),), None, fscope)
                mlp_pred_out = ag__.converted_call(ag__.ld(self).mlp_pred, (ag__.ld(token_rep),), None, fscope)
                mlp_arg_out = ag__.converted_call(ag__.ld(self).mlp_arg, (ag__.ld(token_rep),), None, fscope)
                (arg_start_emb, arg_end_emb, arg_length) = ag__.converted_call(ag__.ld(self).arg_endpoints_len, (ag__.ld(mlp_arg_out),), None, fscope)
                arg_head_emb = ag__.converted_call(ag__.ld(self).arg_attention, (ag__.ld(mlp_arg_out),), None, fscope)
                arg_width_emb = ag__.converted_call(ag__.ld(self).arg_length_emb, (ag__.ld(arg_length),), None, fscope)
                arg_rep = ag__.converted_call(ag__.ld(self).concatenate_2, ([ag__.ld(arg_start_emb), ag__.ld(arg_end_emb), ag__.ld(arg_head_emb), ag__.ld(arg_width_emb)],), None, fscope)

                def get_state_2():
                    return (pred_rep,)

                def set_state_2(vars_):
                    nonlocal pred_rep
                    (pred_rep,) = vars_

                def if_body_2():
                    nonlocal pred_rep
                    (pred_start_emb, pred_end_emb, pred_length) = ag__.converted_call(ag__.ld(self).pred_endpoints_len, (ag__.ld(mlp_pred_out),), None, fscope)
                    pred_head_emb = ag__.converted_call(ag__.ld(self).pred_attention, (ag__.ld(mlp_pred_out),), None, fscope)
                    pred_width_emb = ag__.converted_call(ag__.ld(self).pred_length_emb, (ag__.ld(pred_length),), None, fscope)
                    pred_rep = ag__.converted_call(ag__.ld(self).concatenate_3, ([ag__.ld(pred_start_emb), ag__.ld(pred_end_emb), ag__.ld(pred_head_emb), ag__.ld(pred_width_emb)],), None, fscope)

                def else_body_2():
                    nonlocal pred_rep
                    pred_rep = ag__.ld(mlp_pred_out)
                pred_end_emb = ag__.Undefined('pred_end_emb')
                pred_rep = ag__.Undefined('pred_rep')
                pred_width_emb = ag__.Undefined('pred_width_emb')
                pred_start_emb = ag__.Undefined('pred_start_emb')
                pred_length = ag__.Undefined('pred_length')
                pred_head_emb = ag__.Undefined('pred_head_emb')
                ag__.if_stmt((ag__.ld(self).max_pred_span > 1), if_body_2, else_body_2, get_state_2, set_state_2, ('pred_rep',), 1)
                pred_unary_score = ag__.converted_call(ag__.ld(self).pred_unary, (ag__.ld(pred_rep),), None, fscope)
                arg_unary_score = ag__.converted_call(ag__.ld(self).arg_unary, (ag__.ld(arg_rep),), None, fscope)

                def get_state_3():
                    return (arg_rep, arg_unary_score, filtered_arg_idx, filtered_pred_idx, pred_rep, pred_unary_score, ag__.ldu((lambda : self.arg_span_idx_mask), 'self.arg_span_idx_mask'), ag__.ldu((lambda : self.pred_span_idx_mask), 'self.pred_span_idx_mask'))

                def set_state_3(vars_):
                    nonlocal pred_unary_score, arg_unary_score, arg_rep, filtered_pred_idx, pred_rep, filtered_arg_idx
                    (arg_rep, arg_unary_score, filtered_arg_idx, filtered_pred_idx, pred_rep, pred_unary_score, self.arg_span_idx_mask, self.pred_span_idx_mask) = vars_

                def if_body_3():
                    nonlocal pred_unary_score, arg_unary_score, arg_rep, filtered_pred_idx, pred_rep, filtered_arg_idx
                    filtered_arg_idx = ag__.converted_call(ag__.ld(self).arg_prune, (ag__.ld(arg_unary_score),), None, fscope)
                    filtered_pred_idx = ag__.converted_call(ag__.ld(self).pred_prune, (ag__.ld(pred_unary_score),), None, fscope)
                    ag__.ld(self).arg_span_idx_mask = ag__.ld(filtered_arg_idx)
                    ag__.ld(self).pred_span_idx_mask = ag__.ld(filtered_pred_idx)
                    arg_unary_score = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(arg_unary_score), ag__.ld(filtered_arg_idx)), dict(axis=1, batch_dims=1), fscope)
                    pred_unary_score = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(pred_unary_score), ag__.ld(filtered_pred_idx)), dict(axis=1, batch_dims=1), fscope)
                    arg_rep = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(arg_rep), ag__.ld(filtered_arg_idx)), dict(axis=1, batch_dims=1), fscope)
                    pred_rep = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(pred_rep), ag__.ld(filtered_pred_idx)), dict(axis=1, batch_dims=1), fscope)

                def else_body_3():
                    nonlocal pred_unary_score, arg_unary_score, arg_rep, filtered_pred_idx, pred_rep, filtered_arg_idx
                    pass
                filtered_pred_idx = ag__.Undefined('filtered_pred_idx')
                filtered_arg_idx = ag__.Undefined('filtered_arg_idx')
                ag__.if_stmt(ag__.and_((lambda : ag__.not_(ag__.ld(training))), (lambda : ag__.ld(self).use_pruning)), if_body_3, else_body_3, get_state_3, set_state_3, ('arg_rep', 'arg_unary_score', 'filtered_arg_idx', 'filtered_pred_idx', 'pred_rep', 'pred_unary_score', 'self.arg_span_idx_mask', 'self.pred_span_idx_mask'), 8)
                pred_arg_emb = ag__.converted_call(ag__.ld(self).pred_arg_pair, ([ag__.ld(arg_rep), ag__.ld(pred_rep)],), None, fscope)
                pred_arg_score = ag__.converted_call(ag__.ld(self).pred_arg_score, (ag__.ld(pred_arg_emb),), None, fscope)
                relation_score = ag__.converted_call(ag__.ld(self).biaffine_score, ([ag__.ld(pred_rep), ag__.ld(arg_rep)],), None, fscope)
                final_score = ag__.converted_call(ag__.ld(self).compute_score, ([ag__.ld(arg_unary_score), ag__.ld(pred_unary_score), ag__.ld(pred_arg_score), ag__.ld(relation_score)],), None, fscope)
                out = ag__.converted_call(ag__.ld(self).softmax, (ag__.ld(final_score),), None, fscope)

                def get_state_5():
                    return (do_return, retval_)

                def set_state_5(vars_):
                    nonlocal do_return, retval_
                    (do_return, retval_) = vars_

                def if_body_5():
                    nonlocal do_return, retval_
                    try:
                        do_return = True
                        retval_ = ag__.ld(out)
                    except:
                        do_return = False
                        raise

                def else_body_5():
                    nonlocal do_return, retval_

                    def get_state_4():
                        return (do_return, retval_)

                    def set_state_4(vars_):
                        nonlocal do_return, retval_
                        (do_return, retval_) = vars_

                    def if_body_4():
                        nonlocal do_return, retval_
                        try:
                            do_return = True
                            retval_ = (ag__.ld(out), ag__.ld(filtered_pred_idx), ag__.ld(filtered_arg_idx))
                        except:
                            do_return = False
                            raise

                    def else_body_4():
                        nonlocal do_return, retval_
                        try:
                            do_return = True
                            retval_ = ag__.ld(out)
                        except:
                            do_return = False
                            raise
                    ag__.if_stmt(ag__.ld(self).use_pruning, if_body_4, else_body_4, get_state_4, set_state_4, ('do_return', 'retval_'), 2)
                ag__.if_stmt(ag__.ld(training), if_body_5, else_body_5, get_state_5, set_state_5, ('do_return', 'retval_'), 2)
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory