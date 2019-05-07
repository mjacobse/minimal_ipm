import numpy
import os
import problem


def main():
    import csv
    import random

    x_optimal = problem.info.get_optimal_solution()

    max_iterations = 100
    stepsize_limiter = problem.ipm.stepsize.NonNegativityNeighborhood(0.9995)
    stepsize_limiter = problem.ipm.stepsize.NegativeInfinityNeighborhood(0.0000605)

    results = []
    for _ in range(0, 10000):
        init_x = random.uniform(0.0, problem.Params.upper_bound)
        init_mult_x = 10**(random.uniform(-10, 10))
        try:
            initial_iterate = problem.FeasibleIterate(init_x,
                                                      init_mult_x)
            if not stepsize_limiter.is_fulfilled(initial_iterate):
                raise ValueError
        except ValueError:
            continue
        iterates = problem.ipm.solve(initial_iterate.x,
                                     initial_iterate.mult_x,
                                     max_iterations=max_iterations,
                                     stepsize_limiter=stepsize_limiter)
        iterates = list(iterates)
        num_iterations = len(iterates) - 1
        if num_iterations < max_iterations:
            assert abs(iterates[-1].x - x_optimal) < 1e-8
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
