import matplotlib.pyplot as plt
import numpy as np

from jigsawsolver.jigsawsolver import create_jigsaw, solve_jigsaw
from jigsawsolver.plotting import plot_jigsaw


def main():
    seed = np.random.randint(2**32)
    print("Seed: ", seed)
    rng = np.random.default_rng(seed)

    showfig = True
    savefig = True

    rows = 4
    cols = 5

    k = 2

    tiles, real_solution, pos = create_jigsaw(rows, cols, rng)

    found_solution, solution, candidates = solve_jigsaw(
        tiles,
        2,
        return_candidates=True,
    )

    assert (solution[solution[:, 0].argsort()] == real_solution).all()

    fig1, ax = plt.subplots()

    plot_jigsaw(
        tiles,
        edges=candidates,
        ax=ax,
    )

    fig2, ax = plt.subplots()

    plot_jigsaw(
        tiles,
        edges=candidates,
        pos=pos,
        ax=ax,
    )

    fig3, ax = plt.subplots()

    plot_jigsaw(
        tiles,
        edges=solution,
        pos=pos,
        ax=ax,
    )

    if savefig:
        fig1.savefig("fig/candidates.png", dpi=200)
        fig2.savefig("fig/candidates_grid.png", dpi=200)
        fig3.savefig("fig/solution.png", dpi=200)

    plt.show()

    pass


if __name__ == "__main__":
    main()
