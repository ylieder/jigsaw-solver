import numpy as np


class Edge(tuple):
    def __init__(self, src, tgt) -> None:
        if src > tgt:
            src, tgt = tgt, src
        super(Edge, self).__init__((src, tgt))

    @property
    def src(self):
        return self[0]

    @property
    def tgt(self):
        return self[1]


def index_iter(arr, axis):
    axis = np.atleast_1d(axis)
    idx_dims = tuple(arr.shape[i] for i in axis)
    transposed_arr = np.moveaxis(arr, axis, range(len(axis)))
    for idx in np.ndindex(idx_dims):
        yield idx, transposed_arr[idx]


# def stuple(iterable):
#     return tuple(sorted(iterable))


def rotate_array(l, n):
    return l[n:] + l[:n]
