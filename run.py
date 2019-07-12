import argparse
import numpy
import os
import problem
import problem.ipm.stepsize as ipm_stepsize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', default=0.1, type=float,
                        help="Initial value of x")
    parser.add_argument('--mult_x', default=0.1, type=float,
                        help="Initial value of the multiplier for the "
                             "constraint x >= 0")
    parser.add_argument('--fttb', default=0.99, type=float,
                        help="Use fraction to the boundary rule with stepsize "
                             "factor FTTB to determine stepsize ")
    parser.add_argument('--gamma', type=float,
                        help="Use negative infinity neighborhood with "
                             "parameter GAMMA to determine stepsize")
    args = parser.parse_args()

    if args.gamma is not None:
        stepsize_limiter = ipm_stepsize.NegativeInfinityNeighborhood(args.gamma)
    else:
        stepsize_limiter = ipm_stepsize.NonNegativityNeighborhood(args.fttb)

    iteration_info = problem.ipm.solve(args.x, args.mult_x, problem.Params(),
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
