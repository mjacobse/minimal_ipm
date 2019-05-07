import problem
import numpy


class CentralPath:
    pass


def get_optimal_solution():
    return numpy.clip(-problem.Params.linear / problem.Params.quadratic,
                      0.0, problem.Params.upper_bound)
