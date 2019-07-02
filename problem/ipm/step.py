import collections
import numpy
import problem
from . import weighting
from .stepsize import NonNegativityNeighborhood


def solve_with_refinement(matrix, rhs, refine_compl=True):
    solution = numpy.linalg.solve(matrix, rhs)
    old_error = numpy.inf
    max_iterations = 10
    for _ in range(0, max_iterations):
        new_rhs = rhs - matrix.dot(solution)
        if not refine_compl:
            new_rhs[3] = 0.0
            new_rhs[4] = 0.0
        new_error = max(abs(new_rhs))
        if new_error < 1e-15 or new_error >= old_error:
            break
        solution += numpy.linalg.solve(matrix, new_rhs)
        old_error = new_error
    return solution


StepInfo = collections.namedtuple('StepInfo', ['steps', 'target_compl'])


class ConstantPathFollowing:
    def __init__(self, reduction_factor=0.1):
        self.reduction_factor = reduction_factor

    def calculate_step(self, iterate, kkt_matrix):
        target = self.reduction_factor * iterate.avg_compl()
        rhs = -numpy.array(iterate.get_residual() +
                           [(iterate.x * iterate.mult_x - target),
                            (iterate.s * iterate.mult_s - target)])
        step = problem.Step(solve_with_refinement(kkt_matrix, rhs))
        return StepInfo(
            steps=collections.OrderedDict([('combined', step)]),
            target_compl=target)


class MehrotraPredictorCorrector:
    def __init__(self, exponent=3.0, weighting=weighting.WeightingFull()):
        self.exponent = exponent
        self.weighting = weighting

    def calculate_step(self, iterate, kkt_matrix):
        rhs = -numpy.array(iterate.get_residual() +
                           [(iterate.x * iterate.mult_x),
                            (iterate.s * iterate.mult_s)])
        affine_step = problem.Step(solve_with_refinement(kkt_matrix, rhs))
        stepsize = iterate.get_max_stepsize(affine_step,
                                            NonNegativityNeighborhood())
        reduction_factor = (iterate.affine_avg_compl(affine_step, stepsize) /
                            iterate.avg_compl())**self.exponent

        target = reduction_factor * iterate.avg_compl()
        rhs.fill(0.0)
        rhs[3] = -(affine_step.x * affine_step.mult_x - target)
        rhs[4] = -(affine_step.s * affine_step.mult_s - target)
        corrector_step = problem.Step(solve_with_refinement(kkt_matrix, rhs))
        weight = self.weighting.get_weight(iterate, affine_step, corrector_step)
        return StepInfo(
            steps=collections.OrderedDict([
                ('predictor', affine_step),
                ('corrector', corrector_step),
                ('combined', affine_step + weight * corrector_step)]),
            target_compl=target)
