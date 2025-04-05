def bpp(D, TreeSearch):
    x = D.init_x()
    S = TreeSearch(len(x))
    is_feasible = True
    while True:
        i, b = S.next(is_feasible)
        if i is None:
            return x
        is_feasible = D.update(x, i, b)
        if is_feasible and i == len(x):
            return x


