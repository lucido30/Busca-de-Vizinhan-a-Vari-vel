from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence


CostFunction = Callable[[Sequence[int]], float]
Neighborhood = Callable[["Solution"], "Solution"]
LocalSearch = Callable[["Solution"], "Solution"]


@dataclass
class Solution:

    representation: List[int]
    _cost_function: CostFunction
    _cached_cost: Optional[float] = None

    def cost(self) -> float:

        if self._cached_cost is None:
            self._cached_cost = self._cost_function(self.representation)
        return self._cached_cost

    def copy(self) -> "Solution":

        return Solution(self.representation.copy(), self._cost_function, self._cached_cost)

    def invalidate(self) -> None:

        self._cached_cost = None

    def with_representation(self, representation: Sequence[int]) -> "Solution":

        return Solution(list(representation), self._cost_function)


def random_swap(solution: Solution) -> Solution:

    if len(solution.representation) < 2:
        return solution.copy()
    new_solution = solution.copy()
    i, j = random.sample(range(len(new_solution.representation)), 2)
    new_solution.representation[i], new_solution.representation[j] = (
        new_solution.representation[j],
        new_solution.representation[i],
    )
    new_solution.invalidate()
    return new_solution


def random_insertion(solution: Solution) -> Solution:

    n = len(solution.representation)
    if n < 3:
        return solution.copy()
    new_solution = solution.copy()
    i, j = random.sample(range(n), 2)
    city = new_solution.representation.pop(i)
    if j > i:
        j -= 1
    new_solution.representation.insert(j, city)
    new_solution.invalidate()
    return new_solution


def random_two_opt(solution: Solution) -> Solution:

    n = len(solution.representation)
    if n < 4:
        return solution.copy()
    new_solution = solution.copy()
    i = random.randrange(0, n - 1)
    j = random.randrange(i + 2, n + 1)
    new_solution.representation[i:j] = reversed(new_solution.representation[i:j])
    new_solution.invalidate()
    return new_solution


def two_opt_local_search(solution: Solution) -> Solution:

    current = solution.copy()
    n = len(current.representation)
    if n < 4:
        return current
    improved = True
    while improved:
        improved = False
        best_cost = current.cost()
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                candidate_representation = current.representation.copy()
                candidate_representation[i:j] = reversed(candidate_representation[i:j])
                candidate = current.with_representation(candidate_representation)
                candidate_cost = candidate.cost()
                if candidate_cost < best_cost - 1e-9:
                    current = candidate
                    best_cost = candidate_cost
                    improved = True
                    break
            if improved:
                break
    return current


def variable_neighborhood_search(
    initial_solution: Solution,
    neighborhoods: Sequence[Neighborhood],
    local_search: LocalSearch,
    k_max: int,
    t_max: float,
) -> Solution:
    if not neighborhoods:
        raise ValueError("At least one neighborhood must be provided.")
    best = initial_solution.copy()
    current = best.copy()
    best_cost = best.cost()
    start_time = time.time()
    while time.time() - start_time < t_max:
        k = 0
        while k < min(k_max, len(neighborhoods)) and time.time() - start_time < t_max:
            shaken = neighborhoods[k](current)
            improved = local_search(shaken)
            improved_cost = improved.cost()
            if improved_cost < best_cost - 1e-9:
                best = improved.copy()
                current = improved
                best_cost = improved_cost
                k = 0
            else:
                k += 1
    return best


def generate_euclidean_distance_matrix(points: Sequence[Sequence[float]]) -> List[List[float]]:

    matrix: List[List[float]] = []
    for i, (x1, y1) in enumerate(points):
        row: List[float] = []
        for j, (x2, y2) in enumerate(points):
            if i == j:
                row.append(0.0)
            else:
                row.append(math.hypot(x1 - x2, y1 - y2))
        matrix.append(row)
    return matrix


def tsp_cost_function(distance_matrix: Sequence[Sequence[float]]) -> CostFunction:

    def evaluate(route: Sequence[int]) -> float:
        total = 0.0
        n = len(route)
        for idx, city in enumerate(route):
            next_city = route[(idx + 1) % n]
            total += distance_matrix[city][next_city]
        return total

    return evaluate


def create_random_tsp_instance(
    dimension: int,
    seed: Optional[int] = None,
) -> tuple[List[List[float]], List[int]]:

    rng = random.Random(seed)
    points = [(rng.random(), rng.random()) for _ in range(dimension)]
    distance_matrix = generate_euclidean_distance_matrix(points)
    tour = list(range(dimension))
    rng.shuffle(tour)
    return distance_matrix, tour


if __name__ == "__main__":
    random.seed(42)
    dimension = 15
    distance_matrix, initial_tour = create_random_tsp_instance(dimension, seed=42)
    cost_function = tsp_cost_function(distance_matrix)
    initial_solution = Solution(initial_tour, cost_function)
    neighborhoods = [random_swap, random_insertion, random_two_opt]
    best_solution = variable_neighborhood_search(
        initial_solution=initial_solution,
        neighborhoods=neighborhoods,
        local_search=two_opt_local_search,
        k_max=len(neighborhoods),
        t_max=1.0,
    )
    print("Initial tour:", initial_solution.representation)
    print("Initial cost:", initial_solution.cost())
    print("Best tour:", best_solution.representation)
    print("Best cost:", best_solution.cost())
