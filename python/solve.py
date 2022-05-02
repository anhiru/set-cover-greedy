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
    def get_max_service_towers(dim):
        max_towers = [Point(0, 0)]
        max_val = 0
        for i in range(dim):
            for j in range(dim):
                tower_val = service_matrix[i][j]
                if tower_val > max_val:
                    max_val = tower_val
                    max_towers = [Point(i, j)]
                elif tower_val == max_val:
                    max_towers.append(Point(i, j))
        return max_towers

    # find service tower with the minimum associated penalty 
    def get_min_penalty_tower(towers):
        min_tower = towers[0]
        min_val = float("inf")
        for tower in towers:
            x = tower.x
            y = tower.y
            tower_val = penalty_matrix[x][y]
            if tower_val < min_val:
                min_val = tower_val
                min_tower = tower
        return min_tower
    
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
    def update_service(cities, r_s, val):
        for city in cities:
            x = city.x
            y = city.y
            for i in range(max(0, x - r_s), min(dim, x + r_s + 1)):
                for j in range(max(0, y - r_s), min(dim, y + r_s + 1)):
                    c = Point(i, j)
                    if city.distance_sq(c) <= r_s ** 2:
                        service_matrix[i][j] += val

    # update penalty matrix by incrementing +1
    def update_penalty(tower, r_p):
       x = tower.x
       y = tower.y
       for i in range(max(0, x - r_p), min(dim, x + r_p + 1)):
            for j in range(max(0, y - r_p), min(dim, y + r_p + 1)):
                c = Point(i, j)
                if tower.distance_sq(c) <= r_p ** 2:
                    penalty_matrix[i][j] += 1
    
    # main
    dim = instance.D
    r_s = instance.R_s
    r_p = instance.R_p
    cities = instance.cities.copy()  # shallow copy to prevent modifying cities directly
    towers = []  # list to return

    # initialize matrices with zeros
    service_matrix = [[0] * instance.D for _ in range(instance.D)]
    penalty_matrix = [[0] * instance.D for _ in range(instance.D)]

    # fill matrix with service coverage by incrementing service areas of each city +1
    update_service(cities, r_s, 1)

    while len(cities) > 0:
        max_towers = get_max_service_towers(dim)
        min_tower = get_min_penalty_tower(max_towers)
        towers.append(min_tower)
        update_penalty(min_tower, r_p)
        covered = get_cities_covered(min_tower, cities, r_s, dim)
        update_service(covered, r_s, -1)  # decrement service areas of each covered city -1
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
