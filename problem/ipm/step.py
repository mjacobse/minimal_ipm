import numpy
import problem
from .stepsize import NonNegativityNeighborhood


class ConstantPathFollowing:
    def __init__(self, reduction_factor=0.1):
        self.reduction_factor = reduction_factor

    def calculate_step(self, iterate, kkt_matrix):
        target = self.reduction_factor * iterate.avg_compl()
        rhs = [0, 0, 0,
               -(iterate.x * iterate.mult_x - target),
               -(iterate.s * iterate.mult_s - target)]
        return problem.Step(numpy.linalg.solve(kkt_matrix, rhs))


class MehrotraPredictorCorrector:
    def __init__(self, exponent=3.0):
        self.exponent = exponent

    def calculate_step(self, iterate, kkt_matrix):
        rhs = [0, 0, 0,
               -(iterate.x * iterate.mult_x),
               -(iterate.s * iterate.mult_s)]
        affine_step = problem.Step(numpy.linalg.solve(kkt_matrix, rhs))
        stepsize = iterate.get_max_stepsize(affine_step,
                                            NonNegativityNeighborhood())
        reduction_factor = (iterate.affine_avg_compl(affine_step, stepsize) /
                            iterate.avg_compl())**self.exponent

        target = reduction_factor * iterate.avg_compl()
        rhs[3] -= (affine_step.x * affine_step.mult_x - target)
        rhs[4] -= (affine_step.s * affine_step.mult_s - target)
        return problem.Step(numpy.linalg.solve(kkt_matrix, rhs))
