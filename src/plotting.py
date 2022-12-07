import matplotlib
# import matplotlib.pyplot as plt


def plot_jigsaw(
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
