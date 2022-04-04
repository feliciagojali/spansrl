# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, input):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                span_indices = ag__.converted_call(ag__.ld(tf).minimum, ((ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.converted_call(ag__.ld(tf).range, (ag__.ld(self).max_arg_span,), None, fscope), 0), None, fscope) + ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(self).start, 1), None, fscope)), (ag__.ld(self).max_tokens - 1)), None, fscope)
                span_emb = ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(input), ag__.ld(span_indices)), dict(axis=1), fscope)
                head_scores = ag__.converted_call(ag__.ld(self).dense, (ag__.ld(input),), None, fscope)
                span_width = ag__.converted_call(ag__.ld(tf).add, (ag__.converted_call(ag__.ld(tf).subtract, (ag__.ld(self).end, ag__.ld(self).start), None, fscope), 1), None, fscope)
                span_indices_mask = ag__.converted_call(ag__.ld(tf).sequence_mask, (ag__.ld(span_width), ag__.ld(self).max_arg_span), dict(dtype=ag__.ld(tf).float32), fscope)
                span_indices_log_mask = ag__.converted_call(ag__.ld(tf).math.log, (ag__.ld(span_indices_mask),), None, fscope)
                span_head = (ag__.converted_call(ag__.ld(tf).gather, (ag__.ld(head_scores), ag__.ld(span_indices)), dict(axis=1), fscope) + ag__.converted_call(ag__.ld(tf).tile, (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(span_indices_log_mask), (- 1)), None, fscope), [1, 1, ag__.ld(self).num_heads]), None, fscope))
                span_head = ag__.converted_call(ag__.ld(self).softmax, (ag__.ld(span_head),), None, fscope)
                span_emb = ag__.converted_call(ag__.ld(tf).tile, (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(span_emb), (- 1)), None, fscope), [1, 1, 1, 1, ag__.ld(span_head).shape[(- 1)]]), None, fscope)
                span_head = ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(span_head), (- 1)), None, fscope)
                out = ag__.converted_call(ag__.ld(tf).reduce_sum, (ag__.converted_call(ag__.ld(tf).squeeze, (ag__.converted_call(ag__.ld(tf).matmul, (ag__.ld(span_emb), ag__.ld(span_head)), None, fscope), (- 1)), None, fscope), 2), None, fscope)
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory