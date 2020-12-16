import numpy as np

from .crossover import CrossOver
from utils.logger import get_logger

logger = get_logger(__name__)


def get_pair_pointcuts(high: int):
    while True:
        pc1 = np.random.randint(low=1, high=high, dtype=int)
        pc2 = np.random.randint(low=1, high=high, dtype=int)
        if pc1 != pc2: break

    if pc2 > pc1:
        logger.debug(f"Point cuts pair is {pc1},{pc2}")
        return pc1, pc2
    else:
        logger.debug(f"Point cuts pair is {pc2},{pc1}")
        return pc2, pc1


def ox1(first: np.array, second: np.array):
    size = len(first)
    off1 = np.full(size, -1)
    off2 = np.full(size, -1)
    pc1, pc2 = get_pair_pointcuts(size)
    fillOff1 = {}
    fillOff2 = {}

    i = pc1
    j = 0
    p1 = first[pc1:pc2]
    p2 = second[pc1:pc2]
    while i < pc2:
        a = p1[j]
        b = p2[j]
        off1[i] = a
        fillOff1[a] = True
        off2[i] = b
        fillOff2[b] = True
        logger.debug(f"Copying start values of offspring 1: {a}")
        logger.debug(f"Copying start values of offspring 1: {b}")
        i = i + 1
        j = j + 1

    i = pc2
    j = pc2
    k = pc2
    while i < size:
        logger.debug(f"Copy from second point cut untill end of parent")
        a = second[i]
        b = first[i]

        if a not in fillOff1:
            off1[j] = a
            j = j + 1

        if b not in fillOff2:
            off2[k] = b
            k = k + 1
        i = i + 1

    i = 0
    while i < pc2:
        logger.debug(f"Copy from start of parent untill the second point cut")
        if j == size: j = 0
        if k == size: k = 0
        a = second[i]
        b = first[i]

        if a not in fillOff1:
            off1[j] = a
            j = j + 1
        if b not in fillOff2:
            off2[k] = b
            k = k + 1
        i = i + 1

    return off1, off2


class OX1CrossOver(CrossOver):
    def cross(self, parents: np.array, offspring_count: int = 1, more=None) -> np.array:
        first_parent, second_parent = parents[0], parents[1]
        off1, off2 = ox1(first_parent, second_parent)
        return np.array([off1, off2])
