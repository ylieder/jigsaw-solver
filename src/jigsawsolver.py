import itertools
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyomo.environ as pyomo

from constrained_matching import ConstraintMatchingSAT
from Timing import Timing

matplotlib.use("Agg")

ORIENTATIONS = ["N", "E", "S", "W"]
COUNTER_ORIENTATIONS = {
    "N": "S",
    "E": "W",
    "S": "N",
    "W": "E",
}


def index_iter(arr, axis):
    axis = np.atleast_1d(axis)
    idx_dims = tuple(arr.shape[i] for i in axis)
    transposed_arr = np.moveaxis(arr, axis, range(len(axis)))
    for idx in np.ndindex(idx_dims):
        yield idx, transposed_arr[idx]


def stuple(iterable):
    return tuple(sorted(iterable))


def rotate_array(l, n):
    return l[n:] + l[:n]


def create_jigsaw(rows, columns, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    hconnections = np.zeros((rows, columns + 1))
    hconnections[:, 1:-1] = rng.random((rows, columns - 1))

    vconnections = np.zeros((rows + 1, columns))
    vconnections[1:-1, :] = rng.random((rows - 1, columns))

    tiles = np.stack(
        (
            vconnections[:-1, :],
            hconnections[:, 1:],
            vconnections[1:, :],
            hconnections[:, :-1],
        )
    ).transpose((1, 2, 0))

    return tiles


def draw_jigsaw(
    ax,
    tiles,
    tile_ids,
    tv_lookup,
    vertex_offset,
):
    cmap = matplotlib.cm.viridis

    for (r, c), values in index_iter(tiles, axis=(0, 1)):
        tid = tile_ids[r, c]
        ax.add_patch(
            matplotlib.patches.Rectangle(
                (c - vertex_offset, r - vertex_offset),
                0.6,
                0.6,
                color="lightgray",
            )
        )
        ax.text(c, r, tid, ha="center", va="center")

        for i in range(4):
            if i in (0, 2):
                x = [c - vertex_offset, c + vertex_offset]
                y = [r + (i - 1) * vertex_offset, r + (i - 1) * vertex_offset]
                text_pos_x = c
                text_pos_y = r + (i - 1) * vertex_offset
            else:
                x = [c + (2 - i) * vertex_offset, c + (2 - i) * vertex_offset]
                y = [r - vertex_offset, r + vertex_offset]
                text_pos_x = c + (2 - i) * vertex_offset
                text_pos_y = r
            # elif i == 2:
            #     x = [c - vertex_offset, c + vertex_offset]
            #     y =[r + vertex_offset, r + vertex_offset]
            #     text_pos_x = c
            #     text_pos_y = r + vertex_offset
            # elif i == 3:
            #     x = [c - vertex_offset, c - vertex_offset]
            #     y [r - vertex_offset, r + vertex_offset]
            #     text_pos_x = c - vertex_offset
            #     text_pos_y = r

            ax.plot(x, y, c=cmap(values[i]))

            ax.text(
                text_pos_x,
                text_pos_y,
                tv_lookup[tid][i],
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=1),
            )
    ax.set_ylim((-1, tiles.shape[0]))
    ax.set_xlim((-1, tiles.shape[1]))

    ax.set_aspect(1)
    ax.invert_yaxis()


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

    def add_mismatch_constraint(self, mismatch_cycle):
        if not hasattr(self.model, "orientation_constraints"):
            self.model.orientation_constraints = pyomo.ConstraintList()

        lhs = 0
        for e in mismatch_cycle:
            lhs += self.model.x[self.edges.index(stuple(e))]
        self.model.orientation_constraints.add(lhs <= len(mismatch_cycle) - 1)

    def solve(self):
        solution = self.solver.solve(self.model)
        assert solution.solver.termination_condition == "optimal"

        vars = [bool(self.model.x[i].value) for i in range(len(self.model.x))]
        matching = np.array(self.edges)[vars]
        return matching


def find_tile_mismatch(
    vt_lookup,
    tv_lookup,
    matching_lookup,
    stack,
    tile_attributes,
    visited_edges,
):
    while len(stack) > 0:
        tile_id = stack[0]
        tile_vertices = tv_lookup[tile_id]
        (
            current_row,
            current_col,
            current_orientation,
        ) = tile_attributes[tile_id]

        for src in tile_vertices:
            tgt = matching_lookup.get(src, None)

            if tgt is None:
                continue

            tgt_tile = vt_lookup[tgt]

            tgt_rel_loc = np.where(tv_lookup[tgt_tile] == tgt,)[0][
                0
            ]  # target relative position
            src_rel_loc = np.where(tv_lookup[tile_id] == src)[0][
                0
            ]  # source relative position

            src_loc = (
                src_rel_loc - current_orientation
            ) % 4  # source absolute position
            tgt_loc = (src_loc - 2) % 4  # 1 <-> 3, 0 <-> 2, target absolute position

            tgt_orientation = (
                tgt_rel_loc - tgt_loc
            ) % 4  # number of tile rotations required to bring target vertex from its relative position to its target abolute position

            if tgt_loc == 0:
                tgt_row = current_row + 1
                tgt_col = current_col
            elif tgt_loc == 1:
                tgt_row = current_row
                tgt_col = current_col - 1
            elif tgt_loc == 2:
                tgt_row = current_row - 1
                tgt_col = current_col
            else:
                assert tgt_loc == 3
                tgt_row = current_row
                tgt_col = current_col + 1

            if tgt_tile in tile_attributes:
                if tile_attributes[tgt_tile] != (tgt_row, tgt_col, tgt_orientation):
                    return True, stuple((src, tgt))
            else:
                if tgt_tile not in stack:
                    stack.append(tgt_tile)
                tile_attributes[tgt_tile] = (tgt_row, tgt_col, tgt_orientation)
            visited_edges.append(stuple((src, tgt)))
        del stack[0]

        if len(stack) == 0 and len(tile_attributes) != len(tv_lookup):
            # non-connected components
            unvisited_tile_id = next(
                t for t in tv_lookup.keys() if t not in tile_attributes
            )
            stack.append(unvisited_tile_id)

    return False, None


def shortest_mismatch_cycle(
    edges,
    mismatch_edge,
    vt_lookup,
    tv_lookup,
    matching_lookup,
    timer,
):
    mismatch_src, mismatch_tgt = mismatch_edge

    assert stuple((mismatch_src, mismatch_tgt)) not in edges

    timer.start("vertex_lookup")
    edges_ = [(vt_lookup[src], vt_lookup[tgt]) for src, tgt in edges]
    timer.stop("vertex_lookup")

    timer.start("create_graph")
    tile_graph = nx.Graph()
    tile_graph.add_edges_from(edges_)
    # for src, tgt in edges:
    #     if stuple((src, tgt)) != stuple((mismatch_src, mismatch_tgt)):
    #         tile_graph.add_edge(vt_lookup[src], vt_lookup[tgt])
    timer.stop("create_graph")

    timer.start("shortest_path")
    shortest_path_tile = nx.shortest_path(
        tile_graph,
        vt_lookup[mismatch_src],
        vt_lookup[mismatch_tgt],
    )
    timer.stop("shortest_path")

    timer.start("tile_cycle")
    shortest_cycle = [(mismatch_tgt, mismatch_src)]
    for i in range(len(shortest_path_tile) - 1):
        found_ = False
        for candidate in tv_lookup[shortest_path_tile[i]]:
            if vt_lookup.get(
                matching_lookup.get(candidate, None), None
            ) == shortest_path_tile[i + 1] and stuple(
                (candidate, matching_lookup[candidate])
            ) != stuple(  # Edge case: Two tiles have two connecting edges. In this case it is important to select both edges as shortest path and not two times the same edge!
                (mismatch_tgt, mismatch_src)
            ):
                shortest_cycle.append((candidate, matching_lookup[candidate]))
                found_ = True
                break
        assert found_
    timer.stop("tile_cycle")

    return shortest_cycle


def main(
    save_figs=False,
    max_iterations=500,
    remove_strategy="edge",
):
    timer = Timing()
    timer.start("total")

    seed = 0
    rng = np.random.default_rng(seed)

    # rows, cols = 20, 20
    rows, cols = 50, 50

    tiles = create_jigsaw(rows, cols, rng)

    timer.start("preprocessing")

    tile_ids = np.arange(rows * cols).reshape(rows, cols)
    vertex_ids = np.arange(rows * cols * 4).reshape(rows, cols, 4)

    vt_lookup = {v: v // 4 for v in vertex_ids.flatten()}
    tv_lookup = {t: np.arange(t * 4, (t + 1) * 4) for t in tile_ids.flatten()}

    vertex_offset = 0.4
    vertex_pos = np.zeros((rows, cols, 4, 2))
    for r in range(rows):
        for c in range(cols):
            vertex_pos[r, c] = [
                [c, r - vertex_offset],
                [c + vertex_offset, r],
                [c, r + vertex_offset],
                [c - vertex_offset, r],
            ]
    vertex_pos = vertex_pos.reshape((-1, 2))

    edges = []
    for r in range(rows - 1):
        edges.append((vertex_ids[r, 0, 2], vertex_ids[r + 1, 0, 0]))
        edges.append((vertex_ids[r, -1, 2], vertex_ids[r + 1, -1, 0]))

    for c in range(cols - 1):
        edges.append((vertex_ids[0, c, 1], vertex_ids[0, c + 1, 3]))
        edges.append((vertex_ids[-1, c, 1], vertex_ids[-1, c + 1, 3]))

    k = 2

    relevant_vertices = np.zeros(tiles.shape, dtype=np.bool8)
    relevant_vertices[1:-1, 1:-1] = True
    relevant_vertices[0, 1:-1, 2] = True
    relevant_vertices[-1, 1:-1, 0] = True
    relevant_vertices[1:-1, 0, 1] = True
    relevant_vertices[1:-1, -1, 3] = True

    fvids = vertex_ids[relevant_vertices].flatten()
    fvals = tiles[relevant_vertices].flatten()

    for vid, val in zip(fvids, fvals):
        order = np.argsort(np.abs(fvals - val))
        knns = fvids[order][fvids[order] // 4 != vid // 4][:k]
        edges.extend([stuple([vid, neighbor_id]) for neighbor_id in knns])

    edges = sorted(set(edges))

    timer.stop("preprocessing")

    if save_figs:
        _, ax = plt.subplots(figsize=(15, 10))
        draw_jigsaw(
            ax,
            tiles,
            tile_ids,
            tv_lookup,
            0.3,
        )

        for src, tgt in edges:
            ax.plot(
                [vertex_pos[src][0], vertex_pos[tgt][0]],
                [vertex_pos[src][1], vertex_pos[tgt][1]],
                c="black",
            )
        plt.savefig("fig/knns.png", dpi=200)

    # model = ConstraintMatchingMIP(edges)
    model = ConstraintMatchingSAT(edges)
    iteration = 0
    while iteration < max_iterations:
        timer.start("iteration")
        print(iteration)

        timer.start("matching")
        matching = model.solve()
        timer.stop("matching")

        if save_figs:
            _, ax = plt.subplots(figsize=(15, 10))
            draw_jigsaw(
                ax,
                tiles,
                tile_ids,
                tv_lookup,
                0.3,
            )

            for src, tgt in matching:
                ax.plot(
                    [vertex_pos[src][0], vertex_pos[tgt][0]],
                    [vertex_pos[src][1], vertex_pos[tgt][1]],
                    c="black",
                )
            plt.savefig(f"fig/matching_{iteration}.png", dpi=200)

        timer.start("mismatch_check")
        matching_lookup = {src: tgt for src, tgt in matching}
        matching_lookup.update({tgt: src for src, tgt in matching})

        stack = [vt_lookup[matching[0][0]]]
        tile_attributes = {vt_lookup[stack[0]]: (0, 0, 0)}
        visited_edges = []
        # mismatch_edges = []
        removed_edges = []

        found_mismatches = 0
        while True:
            timer.start("find_mismatch")
            has_mismatch, mismatch_edge = find_tile_mismatch(
                vt_lookup,
                tv_lookup,
                matching_lookup,
                stack,
                tile_attributes,
                visited_edges,
            )

            timer.stop("find_mismatch")

            if not has_mismatch:
                if found_mismatches > 0:
                    break

                timer.stop(all=True)
                with open("timing.json", "w") as f:
                    timer.write(f)
                return matching

            timer.start("find_cycle")
            cycle = shortest_mismatch_cycle(
                visited_edges,
                mismatch_edge,
                vt_lookup,
                tv_lookup,
                matching_lookup,
                timer,
            )
            timer.stop("find_cycle")

            if save_figs:
                _, ax = plt.subplots(figsize=(15, 10))
                draw_jigsaw(
                    ax,
                    tiles,
                    tile_ids,
                    tv_lookup,
                    0.3,
                )

                for src, tgt in matching:
                    color = "black"
                    linestyle = "solid"
                    if stuple((src, tgt)) in removed_edges:
                        linestyle = "dashed"
                    elif stuple((src, tgt)) not in visited_edges:
                        color = "gray"

                    ax.plot(
                        [vertex_pos[src][0], vertex_pos[tgt][0]],
                        [vertex_pos[src][1], vertex_pos[tgt][1]],
                        c=color,
                        linestyle=linestyle,
                    )

                for src, tgt in cycle[1:]:
                    ax.plot(
                        [vertex_pos[src][0], vertex_pos[tgt][0]],
                        [vertex_pos[src][1], vertex_pos[tgt][1]],
                        c="blue" if remove_strategy == "edge" else "red",
                    )

                ax.plot(
                    [vertex_pos[mismatch_edge[0]][0], vertex_pos[mismatch_edge[1]][0]],
                    [vertex_pos[mismatch_edge[0]][1], vertex_pos[mismatch_edge[1]][1]],
                    c="red",
                )

                plt.savefig(f"fig/mismatch_{iteration}_{found_mismatches}.png", dpi=200)

            model.add_neg_conjunction(cycle)

            if remove_strategy == "edge":
                del matching_lookup[mismatch_edge[0]], matching_lookup[mismatch_edge[1]]
                removed_edges.append(mismatch_edge)
            else:
                assert remove_strategy == "cycle"
                raise Exception("Not working!")
                for src, tgt in cycle:
                    del matching_lookup[src], matching_lookup[tgt]
                removed_edges.extend(cycle)

            found_mismatches += 1
            plt.close("all")
            # break

        timer.stop("mismatch_check")
        timer.stop("iteration")

        with open("timing.json", "w") as f:
            timer.write(f)
        iteration += 1
        pass

    raise Exception()


if __name__ == "__main__":
    solution = main(
        save_figs=False,
        max_iterations=1000,
        remove_strategy="edge",
    )
