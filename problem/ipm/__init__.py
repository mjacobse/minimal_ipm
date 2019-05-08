from . import step
from . import stepsize
import numpy
import problem


def make_stepsize_valid(stepsize, iterate, step, stepsize_limiter):
    for _ in range(0, 10):
        try:
            new_iterate = problem.FeasibleIterate(
                iterate.x + stepsize * step.x,
                iterate.mult_x + stepsize * step.mult_x)
            if stepsize_limiter.is_fulfilled(new_iterate):
                break
        except ValueError:
            raise
        stepsize = numpy.nextafter(stepsize, 0)
    while True:
        try:
            new_iterate = problem.FeasibleIterate(
                iterate.x + stepsize * step.x,
                iterate.mult_x + stepsize * step.mult_x)
            if stepsize_limiter.is_fulfilled(new_iterate):
                break
        except ValueError:
            raise
        stepsize = 0.999999*stepsize
    return stepsize


def solve(init_x, init_mult_x, max_iterations=500,
          step_calculator=step.MehrotraPredictorCorrector(),
          stepsize_limiter=stepsize.NegativeInfinityNeighborhood()):
    iterate = problem.FeasibleIterate(init_x, init_mult_x)
    yield iterate
    for _ in range(0, max_iterations):
        #assert stepsize_limiter.is_fulfilled(iterate)
        kkt_matrix = numpy.array([[problem.Params.quadratic, 0, -1,  0, 1],
                                  [0, 0, 0, -1, 1],
                                  [1, 1, 0,  0, 0],
                                  [iterate.mult_x, 0, iterate.x, 0, 0],
                                  [0, iterate.mult_s, 0, iterate.s, 0]])
        step = step_calculator.calculate_step(iterate, kkt_matrix)
        stepsize = iterate.get_max_stepsize(step, stepsize_limiter)
        # stepsize = make_stepsize_valid(stepsize, iterate, step,
        #                                stepsize_limiter)
        iterate.update(step, stepsize)
        yield iterate
        if iterate.avg_compl() < 1e-10:
            break
