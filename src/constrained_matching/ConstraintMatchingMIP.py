from collections import defaultdict

import numpy as np
import pyomo.environ as pyomo


class ConstraintMatchingMIP:
    def __init__(self, edges, solver="glpk") -> None:
        adj_edges_lookup = defaultdict(list)

        for i, (src, tgt) in enumerate(edges):
            adj_edges_lookup[src].append(i)
            adj_edges_lookup[tgt].append(i)

        model = pyomo.ConcreteModel()
        model.x = pyomo.Var(range(len(edges)), domain=pyomo.Binary)
        model.obj = pyomo.Objective(
            expr=model.x[0] - model.x[0]
        )  # Placeholder to avoid warning of constant objective

        model.matching_constraints = pyomo.ConstraintList()
        for _, adj_edges in adj_edges_lookup.items():
            lhs = 0
            for e in adj_edges:
                lhs += model.x[e]
            model.matching_constraints.add(lhs == 1)

        self.edges = edges
        self.model = model
        self.adj_edges_lookup = adj_edges_lookup

        self.solver = pyomo.SolverFactory(solver)

    def add_neg_conjunction(self, edges):
        if not hasattr(self.model, "xor_constraints"):
            self.model.xor_constraints = pyomo.ConstraintList()

        lhs = 0
        for e in edges:
            lhs += self.model.x[self.edges.index(tuple(sorted(e)))]
        self.model.xor_constraints.add(lhs <= len(edges) - 1)

    def solve(self):
        solution = self.solver.solve(self.model)
        assert solution.solver.termination_condition == "optimal"

        vars = [bool(self.model.x[i].value) for i in range(len(self.model.x))]
        matching = np.array(self.edges)[vars]
        return matching
