import math
from typing import List

import tsplib95

from utils.logger import get_logger

logger = get_logger(__name__)


def natural_select(problem: tsplib95, population: List, die=0) -> List[int]:
    distances = [sum(problem.get_weight(a, b) for a, b in zip(chromosome.position[0:], chromosome.position[1:])) for
                 chromosome in population]
    logger.debug(distances)
    for d in range(die):
        lowest = max(enumerate(distances), key=lambda d: d[1])[0]
        distances.pop(lowest)
        population.pop(lowest)
    return population
