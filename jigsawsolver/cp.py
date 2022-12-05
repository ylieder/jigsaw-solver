from collections import defaultdict
from functools import reduce
from typing import List, Tuple, Union

import numpy as np
from ortools.sat.python import cp_model


class JigsawCP:
    def __init__(self, edges: np.ndarray) -> None:
        self.model = cp_model.CpModel()
        self.vars = [self.model.NewBoolVar(str(e)) for e in edges]
        self.slack_vars = []

        self.edges = edges
        self.edge_idx_lookup = {e: i for i, e in enumerate(edges)}

        self.init_matching_constraints()

    def init_matching_constraints(self) -> None:
        """
        Add matching constraints. Each vertex has exactly one incident edge in a valid
        solution.
        """
        incident_edge_idxs = defaultdict(list)
        for i, (src, tgt) in enumerate(self.edges):
            incident_edge_idxs[src].append(i)
            incident_edge_idxs[tgt].append(i)

        for _, edge_idxs in incident_edge_idxs.items():
            self.model.Add(
                reduce(lambda x, y: x + y, [self.vars[idx] for idx in edge_idxs]) == 1,
            )

    def add_grid_layout_constraint(
        self,
        premise: Tuple[Tuple[int, int], Tuple[int, int]],
        consequence: Union[bool, List[Tuple[Tuple[int, int], Tuple[int, int]]]],
    ) -> None:
        if consequence == False:
            e1_var = self.vars[self.edge_idx_lookup[premise[0]]]
            e2_var = self.vars[self.edge_idx_lookup[premise[1]]]
            self.model.Add(e1_var + e2_var <= 1)
        else:
            for e1, e2 in consequence:
                e1_var = self.vars[self.edge_idx_lookup[e1]]
                e2_var = self.vars[self.edge_idx_lookup[e2]]

                # Add a boolean slack variable for each clause of the consequence, which
                # is 1 if and only if both of the edges in the clause are 1.
                self.slack_vars.append(
                    self.model.NewBoolVar(f"s{len(self.slack_vars)}")
                )

                # e1 == 1 AND e2 == 1 => slack_var == 1
                self.model.Add(self.slack_vars[-1] == 1).OnlyEnforceIf(
                    e1_var,
                    e2_var,
                )

                # e1 == 0 => slack_var == 0
                self.model.Add(self.slack_vars[-1] == 0).OnlyEnforceIf(e1_var.Not())

                # e2 == 0 => slack_var == 0
                self.model.Add(self.slack_vars[-1] == 0).OnlyEnforceIf(e2_var.Not())

            # Add implication premise => sum(slack variables for clauses) == 1
            self.model.Add(
                sum(self.slack_vars[-len(consequence) :]) == 1
            ).OnlyEnforceIf(
                self.vars[self.edge_idx_lookup[premise[0]]],
                self.vars[self.edge_idx_lookup[premise[1]]],
            )

    def solve(self) -> Tuple[bool, np.ndarray]:
        solver = cp_model.CpSolver()
        status = solver.Solve(self.model)

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return False, np.array([])

        var_values = [bool(solver.Value(v)) for v in self.vars]
        solution = np.array(self.edges)[var_values]

        return True, solution
