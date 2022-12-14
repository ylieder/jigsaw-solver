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


def find_all_nearest(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the k nearest values of each element in array and returns its indices.

    Returns array of shape (arr.shape[0], k), where row[i] contains the indices of the
    k closest elements to arr[i], sorted ascending by distance to arr[i].
    """
    assert arr.ndim == 1
    assert k <= arr.shape[0]

    idxs = np.argsort(arr)
    arr_sorted = arr[idxs]
    all_nearest = np.empty((arr.shape[0], k), dtype=np.int)

    l = 0
    r = k
    for i in range(arr.shape[0]):
        val = arr_sorted[i]
        while r < arr.shape[0] - 1 and abs(val - arr_sorted[l]) > abs(
            val - arr_sorted[r]
        ):
            r += 1
            l += 1
        order = np.abs(arr_sorted[l:r] - val).argsort()
        all_nearest[i] = idxs[l:r][order]

    return all_nearest[idxs.argsort()]


def create_jigsaw(
    rows: int,
    cols: int,
    shuffle: bool = False,
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
    vertex_ids = vertex_ids.reshape((-1, 4))

    pos = determine_grid_vertex_positions(rows, cols)

    if not shuffle:
        return tiles, solution[solution[:, 0].argsort()], pos

    ## Shuffle tiles
    tile_order = rng.permutation(rows * cols)

    tiles = tiles[tile_order]
    vertex_ids_new = vertex_ids[tile_order]

    pos_1 = np.array(pos[1])[tile_order]

    # Rotate tiles arbitrarly
    tile_rotation = rng.integers(0, 4, rows * cols)
    for i in range(rows * cols):
        tiles[i] = np.roll(tiles[i], tile_rotation[i])
        vertex_ids_new[i] = np.roll(vertex_ids_new[i], tile_rotation[i])

    permutation_lookup = {
        from_id: to_id
        for to_id, from_id in zip(vertex_ids.flatten(), vertex_ids_new.flatten())
    }

    # Substitute new vertex order
    solution = np.vectorize(lambda x: permutation_lookup[x])(solution)
    pos_0 = {permutation_lookup[k]: v for k, v in pos[0].items()}

    pos = (pos_0, pos_1)

    # sort within row
    solution = np.sort(solution)

    # sort rows
    solution = solution[solution[:, 0].argsort()]

    return tiles, solution, pos


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

    all_nearest = find_all_nearest(vertex_values, k + 4)

    # Compute set of matching candidates as k nearest neighbors for each vertex
    candidate_edges = set()
    for v_id, v_val in zip(vertex_ids, vertex_values):
        if not np.isnan(v_val):
            # Select k nearest neigbors. Don't select neighbors on same tile.
            knns = all_nearest[v_id, all_nearest[v_id] // 4 != v_id // 4][:k]

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
