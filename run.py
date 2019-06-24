import numpy
import os
import problem
import problem.ipm.stepsize as ipm_stepsize


def main():
    stepsize_limiter = ipm_stepsize.NonNegativityNeighborhood(0.9995)
    stepsize_limiter = ipm_stepsize.NegativeInfinityNeighborhood(0.00005)

    init_x = 0.1
    init_mult_x = 0.1
    iteration_info = problem.ipm.solve(init_x, init_mult_x, problem.Params(),
                                       max_iterations=500,
                                       stepsize_limiter=stepsize_limiter)
    iteration_info = list(iteration_info)
    header_repeat = problem.util.get_period_length(iteration_info)
    for i, (iterate, _) in enumerate(iteration_info):
        if i % header_repeat == 0:
            print('      x        mult_x   s        mult_s')
        print('{0:3d}   {1:6.4f}   {2:6.4f}   {3:6.4f}   {4:6.4f}'.format(
            i, iterate.x, iterate.mult_x, iterate.s, iterate.mult_s))


if __name__ == "__main__":
    main()
