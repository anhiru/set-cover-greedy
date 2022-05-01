"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from pathlib import Path
import sys
from typing import Callable, Dict

# from openpyxl import NUMPY

from instance import Instance
from point import Point
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper


def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

def solve_greedy(instance: Instance) -> Solution:
    # find index of the max value in service matrix
    def get_max_tower(dim):
        max_tower = Point(0, 0)
        max_val = 0
        for i in range(dim):
            for j in range(dim):
                if service_matrix[i][j] > max_val:
                    max_val = service_matrix[i][j]
                    max_tower = Point(i, j)
        return max_tower

    # get cities covered by a service tower
    def get_cities_covered(tower, cities, r, dim):
        l = []
        x = tower.x
        y = tower.y
        for i in range(max(0, x - r), min(dim, x + r + 1)):
            for j in range(max(0, y - r), min(dim, y + r + 1)):
                c = Point(i, j)
                if c in cities and tower.distance_sq(c) <= r ** 2:
                    l.append(c)
        return l
    
    # update service matrix with val
    def update_service(cities, r, val):
        for city in cities:
            x = city.x
            y = city.y
            for i in range(max(0, x - r), min(dim, x + r + 1)):
                for j in range(max(0, y - r), min(dim, y + r + 1)):
                    c = Point(i, j)
                    if city.distance_sq(c) <= r ** 2:
                        service_matrix[i][j] += val

    # main
    dim = instance.D
    r = instance.R_s
    cities = instance.cities.copy()  # shallow copy to prevent modifying cities directly
    towers = []  # list to return

    # initialize matrix with zeros
    service_matrix = [[0] * instance.D for _ in range(instance.D)]

    # fill matrix with service coverage by incrementing service areas of each city +1
    update_service(cities, r, 1)

    while len(cities) > 0:
        max_tower = get_max_tower(dim)
        towers.append(max_tower)
        covered = get_cities_covered(max_tower, cities, r, dim)
        update_service(covered, r, -1)  # decrement service areas of each covered city -1
        cities = [c for c in cities if c not in covered]  # pop covered cities from cities
        
    return Solution(
        instance=instance,
        towers=towers
    )


SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "greedy": solve_greedy
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")

def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")

def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str, 
                        help="The output file. Use - for stdout.", 
                        default="-")
    main(parser.parse_args())
