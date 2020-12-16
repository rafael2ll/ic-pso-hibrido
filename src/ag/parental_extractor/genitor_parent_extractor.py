import random

import numpy as np
import tsplib95

from utils.logger import get_logger
from .parent_extractor import ParentExtractor

logger = get_logger(__name__)


class GENITORParentalExtractor(ParentExtractor):
    def extract_parent(self, problem: tsplib95, population: np.array) -> np.array:
        population = np.array([a.position for a in population])
        cyclic_pop = np.hstack((population, np.array([population[:, 0]]).T))
        distances = sorted(
            enumerate([sum(problem.get_weight(a, b) for a, b in zip(chromosome[0:], chromosome[1:])) for chromosome in
                       cyclic_pop]), key=lambda a: a[1])
        weights = [1] * len(distances)
        weights[0] = 100
        weights[1] = 100
        logger.debug(f"Distances: {distances}")
        logger.debug(f"Weights: {weights}")
        parents = random.choices(distances, weights=weights, k=1)
        while True:
            parent2 = random.choices(distances, weights=weights, k=1)[0]
            if parents[0] != parent2:
                parents.append(parent2)
                break
        logger.debug(f'Parents: {parents}')
        return np.array([population[parents[0][0]], population[parents[1][0]]])
