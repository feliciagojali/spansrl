# coding=utf-8
def outer_factory():

    def inner_factory(ag__):

        def tf__call(self, input):
            with ag__.FunctionScope('call', 'fscope', ag__.STD) as fscope:
                do_return = False
                retval_ = ag__.UndefinedReturnValue()
                (x, y) = ag__.ld(input)

                def get_state():
                    return (x,)

                def set_state(vars_):
                    nonlocal x
                    (x,) = vars_

                def if_body():
                    nonlocal x
                    x = ag__.converted_call(ag__.ld(tf).concat, ([ag__.ld(x), ag__.converted_call(ag__.ld(tf).ones_like, (ag__.ld(x)[:, :, :1],), None, fscope)],), dict(axis=(- 1)), fscope)

                def else_body():
                    nonlocal x
                    pass
                ag__.if_stmt(ag__.ld(self).bias_x, if_body, else_body, get_state, set_state, ('x',), 1)

                def get_state_1():
                    return (y,)

                def set_state_1(vars_):
                    nonlocal y
                    (y,) = vars_

                def if_body_1():
                    nonlocal y
                    y = ag__.converted_call(ag__.ld(tf).concat, ([ag__.ld(y), ag__.converted_call(ag__.ld(tf).ones_like, (ag__.ld(y)[:, :, :1],), None, fscope)],), dict(axis=(- 1)), fscope)

                def else_body_1():
                    nonlocal y
                    pass
                ag__.if_stmt(ag__.ld(self).bias_y, if_body_1, else_body_1, get_state_1, set_state_1, ('y',), 1)
                o = ag__.ld(self).kernel.shape[0]
                x = ag__.converted_call(ag__.ld(tf).tile, (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(x),), dict(axis=1), fscope), [1, ag__.ld(o), 1, 1]), None, fscope)
                y = ag__.converted_call(ag__.ld(tf).tile, (ag__.converted_call(ag__.ld(tf).expand_dims, (ag__.ld(y),), dict(axis=1), fscope), [1, ag__.ld(o), 1, 1]), None, fscope)
                a = ag__.converted_call(ag__.ld(tf).linalg.matmul, (ag__.ld(x), ag__.ld(self).kernel), None, fscope)
                s = ag__.converted_call(ag__.ld(tf).linalg.matmul, (ag__.ld(a), ag__.converted_call(ag__.ld(tf).transpose, (ag__.ld(y),), dict(perm=[0, 1, 3, 2]), fscope)), None, fscope)
                out = ag__.converted_call(ag__.ld(tf).transpose, (ag__.ld(s),), dict(perm=[0, 2, 3, 1]), fscope)
                try:
                    do_return = True
                    retval_ = ag__.ld(out)
                except:
                    do_return = False
                    raise
                return fscope.ret(retval_, do_return)
        return tf__call
    return inner_factory