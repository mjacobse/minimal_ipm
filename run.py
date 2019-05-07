import numpy
import os
import problem
import problem.ipm.stepsize as ipm_stepsize


def main():
    stepsize_limiter = ipm_stepsize.NonNegativityNeighborhood(0.9995)
    stepsize_limiter = ipm_stepsize.NegativeInfinityNeighborhood(0.00005)

    init_x = 0.1
    init_mult_x = 0.1
    iterates = problem.ipm.solve(init_x, init_mult_x,
                                 max_iterations=500,
                                 stepsize_limiter=stepsize_limiter)
    for i, iterate in enumerate(iterates):
        print('{0:3d}   {1:10.4e}   {2:10.4e}'.format(
            i, iterate.x, iterate.mult_x))


if __name__ == "__main__":
    main()
