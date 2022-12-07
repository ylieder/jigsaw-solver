from collections import defaultdict
from datetime import timedelta
from functools import reduce
from timeit import default_timer

import numpy as np
from ortools.sat.python import cp_model

from .ConstraintMatchingSolver import ConstraintMatchingSolver


class ConstraintMatchingSAT(ConstraintMatchingSolver):
    def __init__(self, edges) -> None:
        model = cp_model.CpModel()

        self.vars = [model.NewBoolVar(str(e)) for e in edges]

        self.edges = edges
        self.model = model
        self.solver = None

        self.index_lookup = {e: i for i, e in enumerate(edges)}

        self.initialize_matching_constraints()

    def initialize_matching_constraints(self):
        adj_edges_lookup = defaultdict(list)

        for i, (src, tgt) in enumerate(self.edges):
            adj_edges_lookup[src].append(i)
            adj_edges_lookup[tgt].append(i)

        for _, adj_edges in adj_edges_lookup.items():
            constraint_vars = [self.vars[e] for e in adj_edges]
            self.model.Add(reduce(lambda x, y: x + y, constraint_vars) == 1)

    def add_neg_conjunction(self, edges):
        constraint_vars = [
            self.vars[self.index_lookup[tuple(sorted(e))]] for e in edges
        ]
        self.model.Add(
            reduce(lambda x, y: x + y, constraint_vars) <= len(constraint_vars) - 1
        )

    def solve(self):
        if self.solver is None:
            self.solver = cp_model.CpSolver()
        status = self.solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            vars = [bool(self.solver.Value(v)) for v in self.vars]
            matching = np.array(self.edges)[vars]
            return matching

        raise Exception("Matching problem cannot be solved.")
