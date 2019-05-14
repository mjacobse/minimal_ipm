import problem
import numpy


class CentralPath:
    pass


def get_optimal_solution(params):
    return numpy.clip(-params.linear / params.quadratic,
                      0.0, params.upper_bound)
