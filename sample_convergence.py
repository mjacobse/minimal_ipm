import argparse
import numpy
import os
import problem
import problem.ipm.stepsize as ipm_stepsize
import random


def get_random_initial_guess(params, compl_space=False):
    if compl_space:
        compl_product_x = 10**random.uniform(-8, 8)
        compl_product_s = 10**random.uniform(-8, 8)
        return problem.info.get_iterates_from_compl_products(
            compl_product_x, compl_product_s, params)
    init_x = random.uniform(0.0, params.upper_bound)
    init_mult_x = 10**(random.uniform(-10, 10))
    return init_x, init_mult_x


def sample_convergence(args):
    if os.path.isfile(args.filepath):
        results = problem.util.ConvergenceResultList.from_file(args.filepath)
    else:
        results = problem.util.ConvergenceResultList(params=problem.Params(),
                                                     max_iterations=100)
    params = results.params
    x_optimal = params.get_optimal_solution()

    max_iterations = results.max_iterations
    if args.gamma is not None:
        stepsize_limiter = ipm_stepsize.NegativeInfinityNeighborhood(args.gamma)
    else:
        stepsize_limiter = ipm_stepsize.NonNegativityNeighborhood(args.fttb)

    for _ in range(0, 10000):
        init_x, init_mult_x = get_random_initial_guess(params,
                                                       args.use_compl_space)
        try:
            initial_iterate = problem.FeasibleIterate(init_x,
                                                      init_mult_x,
                                                      params)
            if not stepsize_limiter.is_fulfilled(initial_iterate):
                raise ValueError
        except ValueError:
            continue
        iteration_info = problem.ipm.solve(initial_iterate.x,
                                           initial_iterate.mult_x,
                                           params,
                                           max_iterations=max_iterations,
                                           stepsize_limiter=stepsize_limiter)
        iteration_info = list(iteration_info)
        num_iterations = len(iteration_info) - 1
        if num_iterations < max_iterations:
            assert abs(iteration_info[-1].iterate.x - x_optimal) < 1e-8
        results.add_result(init_x, init_mult_x, num_iterations)
        #print(num_iterations)

    results.to_file(args.filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath')
    parser.add_argument('--fttb', default=0.99, type=float,
                        help="Use fraction to the boundary rule with stepsize "
                             "factor FTTB to determine stepsize ")
    parser.add_argument('--gamma', type=float,
                        help="Use negative infinity neighborhood with "
                             "parameter GAMMA to determine stepsize")
    parser.add_argument('--compl', dest='use_compl_space',
                        action='store_const', const=True, default=False,
                        help="Sample with random initial complementarity "
                             "products instead of variables")
    args = parser.parse_args()
    while True:
        sample_convergence(args)


if __name__ == "__main__":
    main()
