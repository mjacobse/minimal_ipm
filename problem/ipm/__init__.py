from . import step
from . import stepsize
import numpy
import problem


def is_stepsize_ok(iterate, step, stepsize, stepsize_limiter):
    try:
        new_iterate = problem.FeasibleIterate(
            iterate.x + stepsize * step.x,
            iterate.mult_x + stepsize * step.mult_x)
        if stepsize_limiter.is_fulfilled(new_iterate):
            return True
    except ValueError:
        pass
    return False


def find_exact_stepsize(iterate, step, stepsize, stepsize_limiter):
    fraction = 1e-3

    top = stepsize
    while is_stepsize_ok(iterate, step, top, stepsize_limiter):
        if top >= 1.0:
            return 1.0
        top *= (1.0 + fraction)

    bot = stepsize
    while not is_stepsize_ok(iterate, step, bot, stepsize_limiter):
        bot *= (1.0 - fraction)

    while numpy.nextafter(bot, top) < top:
        mid = (bot + top) / 2.0
        if is_stepsize_ok(iterate, step, mid, stepsize_limiter):
            bot = mid
        else:
            top = mid

    assert bot < top
    assert numpy.nextafter(bot, top) == top
    assert is_stepsize_ok(iterate, step, bot, stepsize_limiter)
    assert not is_stepsize_ok(iterate, step, top, stepsize_limiter)
    return bot


def solve(init_x, init_mult_x, max_iterations=500,
          step_calculator=step.MehrotraPredictorCorrector(),
          stepsize_limiter=stepsize.NegativeInfinityNeighborhood()):
    iterate = problem.FeasibleIterate(init_x, init_mult_x)
    yield iterate
    for _ in range(0, max_iterations):
        assert stepsize_limiter.is_fulfilled(iterate)
        kkt_matrix = numpy.array([[problem.Params.quadratic, 0, -1,  0, 1],
                                  [0, 0, 0, -1, 1],
                                  [1, 1, 0,  0, 0],
                                  [iterate.mult_x, 0, iterate.x, 0, 0],
                                  [0, iterate.mult_s, 0, iterate.s, 0]])
        step = step_calculator.calculate_step(iterate, kkt_matrix)
        stepsize = iterate.get_max_stepsize(step, stepsize_limiter)
        stepsize = find_exact_stepsize(iterate, step, stepsize,
                                       stepsize_limiter)
        iterate.update(step, stepsize)
        yield iterate
        if iterate.avg_compl() < 1e-10:
            break
