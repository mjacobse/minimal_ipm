import numpy
import os
import problem
import random


def get_random_initial_guess(params):
    init_x = random.uniform(0.0, params.upper_bound)
    init_mult_x = 10**(random.uniform(-10, 10))
    return init_x, init_mult_x


def main():
    import csv

    params = problem.Params()
    x_optimal = params.get_optimal_solution()

    max_iterations = 100
    stepsize_limiter = problem.ipm.stepsize.NonNegativityNeighborhood(0.9995)
    stepsize_limiter = problem.ipm.stepsize.NegativeInfinityNeighborhood(0.0000605)

    results = []
    for _ in range(0, 10000):
        init_x, init_mult_x = get_random_initial_guess(params)
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
        results.append((init_x, init_mult_x, num_iterations))
        #print(num_iterations)

    is_new = not os.path.isfile('convergence.csv')
    with open('convergence.csv', 'w' if is_new else 'a') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=';')
        if is_new:
            csv_writer.writerow([-numpy.inf, -numpy.inf, max_iterations])
        for result in results:
            csv_writer.writerow([result[0], result[1], result[2]])

if __name__ == "__main__":
    while True:
        main()
