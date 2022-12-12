import json
from datetime import timedelta
from timeit import default_timer as timer
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from jigsawsolver.jigsawsolver import create_jigsaw, solve_jigsaw
from jigsawsolver.plotting import plot_jigsaw


def verify_solution(solution, real_solution):
    return bool(
        (solution[solution[:, 0].argsort()] == real_solution).all(),
    )


def benchmark(
    sizes: List[Tuple[int, int]],
    k: List[int],
    repetitions: int = 1,
) -> Dict:
    benchmark_results = {}

    for size in sizes:
        benchmark_results[str(size)] = {}
        for k_val in k:
            print(f">>> Run benchmark for size={size} and k={k_val}")
            benchmark_results[str(size)][k_val] = {}
            for repetition in range(repetitions):
                seed = np.random.randint(2**32)
                rng = np.random.default_rng(seed)

                rows, cols = size

                tiles, real_solution, pos = create_jigsaw(rows, cols, rng)

                start = timer()
                found_solution, solution, candidates = solve_jigsaw(
                    tiles,
                    k_val,
                    return_candidates=True,
                )
                runtime = timer() - start

                valid_solution = verify_solution(solution, real_solution)

                benchmark_results[str(size)][k_val][repetition] = {
                    "seed": seed,
                    "found_solution": found_solution,
                    "valid_solution": valid_solution,
                    "runtime": np.round(runtime, 1),
                    "candidate_edges": len(candidates),
                }

    return benchmark_results


def main():
    run_benchmark = False

    if run_benchmark:
        benchmark_results = benchmark(
            sizes=[(10, 10), (30, 30), (40, 60), (61, 82)],
            k=[2, 4, 8, 16],
        )

        with open("benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)

        return

    seed = np.random.randint(0, 2**32)
    print("Seed: ", seed)
    rng = np.random.default_rng(seed)

    savefig = True

    rows = 4
    cols = 5
    shuffle_tiles = True

    k = 2

    tiles, real_solution, pos = create_jigsaw(rows, cols, shuffle_tiles, rng)

    start = timer()
    found_solution, solution, candidates = solve_jigsaw(
        tiles,
        k,
        return_candidates=True,
    )
    end = timer()
    print(f"Finished in {timedelta(seconds=end-start)}")

    if not found_solution:
        print("No solution found! :(")
    elif not verify_solution(solution, real_solution):
        print("Solution is not correct! :(")
    else:
        print("Found correct solution! :)")

    if savefig:
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

        fig1.savefig("fig/candidates.png", dpi=200)
        fig2.savefig("fig/candidates_grid.png", dpi=200)
        fig3.savefig("fig/solution.png", dpi=200)

        plt.show()


if __name__ == "__main__":
    main()
