from collections import defaultdict
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from jigsawsolver.plotting import determine_grid_vertex_positions

from .cp import JigsawCP

# from Timing import Timing


def vt(vertex_id: int) -> int:
    """
    Maps a vertex ID to its associated tile ID.
    """
    return vertex_id // 4


def tv(tile_id: int) -> np.ndarray:
    """
    Maps a tile ID to the four associated vertex IDs.
    """
    return np.array([tile_id * 4 + i for i in range(4)])


def stuple(iterable: Iterable) -> Tuple:
    """
    Converts the iterable into a sorted tuple.
    """
    return tuple(sorted(iterable))


def create_jigsaw(
    rows: int,
    cols: int,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Create a simple jigsaw representation. Shapes of tile edges are defined as floats in
    the range (0, 1).

    Returns
    - an array of shape (rows * cols, 4) containing four values for the edges of
    each tile (in clockwise order)
    - the solution of the puzzle (matching vertices)
    - x and y positions of the vertices in the final solution (used only for plotting)
    """
    if rng is None:
        rng = np.random.default_rng()

    hconnections = np.full((rows, cols + 1), np.nan)
    hconnections[:, 1:-1] = rng.random((rows, cols - 1))

    vconnections = np.full((rows + 1, cols), np.nan)
    vconnections[1:-1, :] = rng.random((rows - 1, cols))

    tiles = np.stack(
        (
            vconnections[:-1, :],
            hconnections[:, 1:],
            vconnections[1:, :],
            hconnections[:, :-1],
        )
    ).transpose((1, 2, 0))

    vertex_ids = np.arange(rows * cols * 4).reshape(rows, cols, 4)

    vedges = np.stack(
        (
            vertex_ids[:, : cols - 1, 1].flatten(),
            vertex_ids[:, 1:, 3].flatten(),
        )
    ).T
    hedges = np.stack(
        (
            vertex_ids[: rows - 1, :, 2].flatten(),
            vertex_ids[1:, :, 0].flatten(),
        )
    ).T

    solution = np.concatenate((vedges, hedges))

    # Sort rows
    solution = solution[solution[:, 0].argsort()]

    tiles = tiles.reshape((-1, 4))

    pos = determine_grid_vertex_positions(rows, cols)

    return tiles, solution[solution[:, 0].argsort()], pos

    # TODO

    # Flatten tiles
    tiles = tiles[tile_order, :]

    ## Shuffle tiles
    tile_order = np.random.permutation(rows * cols)

    vertex_ids_new = vertex_ids.reshape(-1, 4)[tile_order, :]
    permutation_lookup = {
        from_id: to_id
        for from_id, to_id in zip(vertex_ids.flatten(), vertex_ids_new.flatten())
    }
    vertex_ids = vertex_ids_new

    # Rotate tiles arbitrarly
    tile_rotation = np.random.randint(0, 4, rows * cols)
    for i in range(rows * cols):
        tiles[i] = np.roll(tiles[i], tile_rotation[i])
        vertex_ids[i] = np.roll(vertex_ids[i], tile_rotation[i])

    # Substitute new vertex order
    solution = np.vectorize(lambda x: permutation_lookup[x])(solution)

    return tiles, solution


def solve_jigsaw(
    jigsaw_tiles: np.ndarray,
    k: int,
    return_candidates: bool = False,
):
    # Size of puzzle
    N = jigsaw_tiles.shape[0]

    # Assign each puzzle tile a unique ID
    vertex_ids = np.arange(N * 4)

    # Create lookup to get CW and CCW neighbors of each vertex
    clockwise_neighbors = {
        v: cn
        for v, cn in zip(
            vertex_ids, np.roll(vertex_ids.reshape(-1, 4), -1, axis=1).flatten()
        )
    }
    counterclockwise_neighbors = {
        v: cn
        for v, cn in zip(
            vertex_ids, np.roll(vertex_ids.reshape(-1, 4), 1, axis=1).flatten()
        )
    }

    # Flatten array of vertex values
    # (rows * cols, 4) -> (rows * cols * 4)
    vertex_values = jigsaw_tiles.flatten()

    # Compute set of matching candidates as k nearest neighbors for each vertex
    candidate_edges = set()
    for v_id, v_val in zip(vertex_ids, vertex_values):
        if not np.isnan(v_val):
            order = np.argsort(np.abs(vertex_values - v_val))

            # Select k nearest neigbors. Don't select neighbors on same tile.
            knns = vertex_ids[order][vertex_ids[order] // 4 != v_id // 4][:k]

            candidate_edges |= {stuple([v_id, neighbor_id]) for neighbor_id in knns}

    candidate_edges = list(candidate_edges)

    # nx.draw_kamada_kawai(drawing, node_size=10, node_color=vertex_values)

    # Create instance of CP
    cpmodel = JigsawCP(candidate_edges)

    # Create adjacency lookup table
    adjacency_lookup = defaultdict(list)
    for src, tgt in candidate_edges:
        adjacency_lookup[src].append(tgt)
        adjacency_lookup[tgt].append(src)

    # Add grid layout constraints.
    # In the following block, v1 - v8 refer always to the following vertex constellation:
    #
    #   +---------+           +---------+
    #   |         |           |         |
    #   |         v4--------- v7        |
    #   |         |           |         |
    #   +--- v3 --+           +--- v8 --+
    #        |                     |
    #        |                     |
    #        |                     |
    #   +--- v1 --+           +--- v6 --+
    #   |         |           |         |
    #   |         v2----------v5        |
    #   |         |           |         |
    #   +---------+           +---------+
    #

    for v1 in vertex_ids:
        v2 = clockwise_neighbors[v1]
        for v3 in adjacency_lookup[v1]:
            v4 = counterclockwise_neighbors[v3]
            for v5 in adjacency_lookup[v2]:
                premise = (stuple((v1, v3)), stuple((v2, v5)))
                consequence = []

                v6 = clockwise_neighbors[v5]
                for v7 in adjacency_lookup[v4]:
                    v8 = counterclockwise_neighbors[v7]
                    if v8 in adjacency_lookup[v6]:
                        # Add clause to consequence disjunction:
                        #   (v1, v3) & (v2, v5) => [... | ((v4, v7) ^ (v6, v8))]
                        #                                 ---------------------
                        #                                       new part
                        consequence.append((stuple((v4, v7)), stuple((v6, v8))))

                if len(consequence) > 0:
                    cpmodel.add_grid_layout_constraint(premise, consequence)
                else:
                    cpmodel.add_grid_layout_constraint(premise, False)

    found_solution, solution = cpmodel.solve()

    if return_candidates:
        return found_solution, solution, candidate_edges
    return found_solution, solution

    # if not (matching[matching[:, 0].argsort()] == solution).all():
    #     print("Wrong solution??")
    # pass

    # fig, ax = plt.subplots()
    # x_pos = x_pos.flatten()
    # y_pos = y_pos.flatten()
    # ax.scatter(
    #     x_pos,
    #     y_pos,
    #     c=vertex_values,
    #     s=10,
    # )

    # ax.invert_yaxis()
    # ax.set_aspect(1)

    # for u, v in matching:
    #     ax.plot([x_pos[u], x_pos[v]], [y_pos[u], y_pos[v]], c="black")

    # pass
