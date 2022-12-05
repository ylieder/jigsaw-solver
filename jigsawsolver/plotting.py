from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Rectangle

TILE_SIZE = 0.3

# if plot:
#     drawing = nx.Graph()

#     for tile_id in range(N):
#         for i in range(4):
#             drawing.add_edge(
#                 tile_id * 4 + i, tile_id * 4 + (i + 1) % 4, weight=2, is_helper=True
#             )

#     for src, tgt in candidate_edges:
#         drawing.add_edge(src, tgt, weight=10, is_helper=False)

#     fig, ax = plt.subplots(1, 2, dpi=200, figsize=(6, 2.5))

#     kamada_kawai = nx.kamada_kawai_layout(drawing)

# tile_centers = []
# tile_rotations = []
# tile_scales = []

# for tile_id in range(N):
#     center = np.mean(
#         np.stack([kamada_kawai[tile_id * 4 + i] for i in range(4)]), axis=0
#     )
#     tile_centers.append(center)
#     x1, y1 = kamada_kawai[tile_id * 4]
#     tile_rotations.append(
#         (np.arctan2(y1 - center[1], x1 - center[0]) * 180 / np.pi - 90) % 360
#     )
#     tile_scales.append(np.sqrt(np.sum(np.square(center - (x1, y1)))))

#     pass

# tile_scale = np.mean(tile_scales)

# for tile_id in range(N):
#     ax[0].add_patch(
#         matplotlib.patches.Rectangle(
#             (
#                 tile_centers[tile_id][0] - tile_scale,
#                 tile_centers[tile_id][1] - tile_scale,
#             ),
#             2 * tile_scale,
#             2 * tile_scale,
#             angle=tile_rotations[tile_id],
#             rotation_point="center",
#             facecolor="gray",
#         )
#     )
#     for i in range(4):
#         rot = (tile_rotations[tile_id] + (4 - i) * 90) % 360
#         x = tile_centers[tile_id][0] + tile_scale * np.cos(np.deg2rad(rot))
#         y = tile_centers[tile_id][1] + tile_scale * np.sin(np.deg2rad(rot))
#         kamada_kawai[tile_id * 4 + i][:] = (x, y)

#     nx.draw_networkx(
#         drawing,
#         pos=kamada_kawai,
#         node_size=5,
#         node_color=vertex_values,
#         with_labels=False,
#         edgelist=candidate_edges,
#         ax=ax[0],
#     )

#     ax[0].set_aspect(1)
#     ax[0].set_ylim((-1, 1))
#     ax[0].set_xlim((-1, 1))
#     ax[1].set_aspect(1)

#     pos_grid = {
#         i: np.array((x, y))
#         for i, x, y in zip(range(4 * N), x_pos.flatten(), y_pos.flatten())
#     }

#     # for tile_id in range(N):
#     #     row = tile_id // cols
#     #     col = tile_id % cols
#     #     ax[1].add_patch(
#     #         matplotlib.patches.Rectangle(
#     #             (col - 0.15, row - 0.15), 0.3, 0.3, facecolor="gray"
#     #         )
#     #     )

#     nx.draw_networkx(
#         drawing,
#         pos=pos_grid,
#         node_size=5,
#         node_color=vertex_values,
#         with_labels=False,
#         edgelist=candidate_edges,
#         ax=ax[1],
#     )
#     ax[1].get_xticks()

#     plt.savefig("fig/candidates.png")
#     pass

#     _, ax = plt.subplots()
#     nx.draw_networkx(
#         drawing,
#         pos=kamada_kawai,
#         node_size=10,
#         node_color=vertex_values,
#         with_labels=False,
#     )
#     pass


def determine_grid_vertex_positions(rows, cols):
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    x = np.tile(x[:, :, None], 4) + np.array([0, TILE_SIZE / 2, 0, -TILE_SIZE / 2])
    y = np.tile(y[:, :, None], 4) + np.array([-TILE_SIZE / 2, 0, TILE_SIZE / 2, 0])

    x = x.reshape(-1)
    y = y.reshape(-1)

    xy = np.stack((x, y)).T

    pos = {i: xy[i] for i in range(4 * rows * cols)}

    tile_squares = [
        {
            "xy": (col - TILE_SIZE / 2, row - TILE_SIZE / 2),
            "width": TILE_SIZE,
            "height": TILE_SIZE,
        }
        for row, col in np.ndindex((rows, cols))
    ]

    return pos, tile_squares


def determine_kk_vertex_positions(tiles, edges, inner_tile_force=0.2):

    # Step 1: Compute good vertex positions using Kamada-Kawai algorithm.
    graph = nx.Graph()

    for tile_id in range(tiles.shape[0]):
        for i in range(4):
            graph.add_edge(
                tile_id * 4 + i,
                tile_id * 4 + (i + 1) % 4,
                weight=inner_tile_force,
                is_helper=True,
            )

    for src, tgt in edges:
        graph.add_edge(src, tgt, weight=1, is_helper=False)

    kamada_kawai = nx.kamada_kawai_layout(graph)

    # Step 2: Align vertices of a tile to form a square (tile).

    tile_centers = []
    tile_rotations = []
    tile_scales = []

    for tile_id in range(tiles.shape[0]):
        center = np.mean(
            np.stack([kamada_kawai[tile_id * 4 + i] for i in range(4)]),
            axis=0,
        )
        tile_centers.append(center)
        x1, y1 = kamada_kawai[tile_id * 4]
        tile_rotations.append(
            (np.arctan2(y1 - center[1], x1 - center[0]) * 180 / np.pi - 90) % 360
        )
        tile_scales.append(
            np.sqrt(np.sum(np.square(center - (x1, y1)))),
        )

    tile_scale = np.mean(tile_scales)
    tile_squares = []

    for tile_id in range(tiles.shape[0]):
        for i in range(4):
            rot = (tile_rotations[tile_id] + (4 - i) * 90 + 90) % 360
            x = tile_centers[tile_id][0] + tile_scale * np.cos(np.deg2rad(rot))
            y = tile_centers[tile_id][1] + tile_scale * np.sin(np.deg2rad(rot))
            kamada_kawai[tile_id * 4 + i][:] = (x, y)

        tile_squares.append(
            {
                "xy": (
                    tile_centers[tile_id][0] - tile_scale,
                    tile_centers[tile_id][1] - tile_scale,
                ),
                "width": 2 * tile_scale,
                "height": 2 * tile_scale,
                "angle": tile_rotations[tile_id],
                "rotation_point": "center",
            }
        )

    return kamada_kawai, tile_squares


def plot_jigsaw(
    tiles,
    edges,
    pos: Optional[Tuple[np.ndarray, List[Dict[str, Any]]]] = None,
    ax=None,
):
    if ax is None:
        _, ax = plt.subplots()

    ax.set_aspect(1)

    if pos is None:
        pos, tile_squares = determine_kk_vertex_positions(tiles, edges)
    else:
        pos, tile_squares = pos

    graph = nx.Graph()
    graph.add_edges_from(edges)

    for square_attrs in tile_squares:
        ax.add_patch(Rectangle(**square_attrs, facecolor="gray"))

    nx.draw_networkx(
        graph,
        pos=pos,
        ax=ax,
        node_size=20,
        node_color=tiles.flatten()[graph.nodes()],
        with_labels=False,
    )

    pass
