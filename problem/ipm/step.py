import numpy
import problem
from .stepsize import NonNegativityNeighborhood


class ConstantPathFollowing:
    def calculate_step(self, iterate, kkt_matrix):
        reduction_factor = 0.1
        target = reduction_factor * iterate.avg_compl()
        rhs = [0, 0, 0,
               -(iterate.x * iterate.mult_x - target),
               -(iterate.s * iterate.mult_s - target)]
        return problem.Step(numpy.linalg.solve(kkt_matrix, rhs))


class MehrotraPredictorCorrector:
    def calculate_step(self, iterate, kkt_matrix):
        rhs = [0, 0, 0,
               -(iterate.x * iterate.mult_x),
               -(iterate.s * iterate.mult_s)]
        affine_step = problem.Step(numpy.linalg.solve(kkt_matrix, rhs))
        stepsize = iterate.get_max_stepsize(affine_step,
                                            NonNegativityNeighborhood())
        reduction_factor = (iterate.affine_avg_compl(affine_step, stepsize) /
                            iterate.avg_compl())**3

        target = reduction_factor * iterate.avg_compl()
        rhs[3] -= (affine_step.x * affine_step.mult_x - target)
        rhs[4] -= (affine_step.s * affine_step.mult_s - target)
        return problem.Step(numpy.linalg.solve(kkt_matrix, rhs))
