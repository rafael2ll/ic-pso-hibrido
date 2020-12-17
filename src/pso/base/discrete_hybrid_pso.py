import math
import random
from typing import List

import numpy as np

from ag import natural_select
from ag.crossovers import OX1CrossOver
from ag.parental_extractor import GENITORParentalExtractor
from pso.base.adjuster import fit, discrete_velocity, adjust_discrete_position
from pso.base.particle import DParticle
from utils.logger import get_logger

logger = get_logger(__name__)


class DHybridPSO:
    def __init__(self, problem, size=100):
        self.problem = problem
        self.length = len(list(problem.get_nodes()))

        self.best_path = math.inf
        self.best_path_pos = np.random.permutation(range(len(list(self.problem.get_nodes()))))

        permutations = np.array([
            np.random.permutation(range(len(list(self.problem.get_nodes())))) for _ in list(range(size))])
        self.particles: List[DParticle] = [DParticle(perm) for perm in permutations]
        logger.debug(f"Initial swarm[{self.length}]:\n{self.particles}")

        # Genetic initializations
        self.parent_extractor = GENITORParentalExtractor()
        self.crossover = OX1CrossOver()

    def submit(self, iterations=1000):
        for i in range(iterations):
            for particle in self.particles:
                distance = fit(particle.position, self.problem)
                logger.debug(f"Distance: {distance}\t Path:{particle.position}\tV:{particle.velocity}")

                # Is it the best particle distance so far?
                if distance < particle.best_path_len:
                    particle.best_position = np.copy(particle.position)
                    particle.best_path_len = distance
                    # May be the best global distance as well?
                    if distance < self.best_path:
                        self.best_path = distance
                        self.best_path_pos = np.copy(particle.position)
                        logger.info(f"Best distance: {self.best_path}\tBest Path:{self.best_path_pos}")
                # Adjust velocity
                velocity = discrete_velocity(particle)
                adjust_discrete_position(particle, velocity)

                # Adding genetic vector
                if random.random() <= 0.5:
                    parents = self.parent_extractor.extract_parent(problem=self.problem, population=self.particles)
                    offspring = self.crossover.cross(parents, 2, self.problem)
                    self.particles.extend(DParticle(off) for off in offspring)
                    natural_select(problem=self.problem, population=self.particles, die=len(offspring))

        logger.info(f"{':' * 5} Best distance: {self.best_path}\tBest Path:{self.best_path_pos} {':' * 5}")
        return self.best_path, self.best_path_pos
