import glob
import os
import problem


def plot_results(results, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.figsize'] = (16, 9)

    plt.figure()
    size = 5000000 / (len(not_converged_x) + len(converged_x))
    plt.scatter(results.not_converged_x, results.not_converged_mult_x,
                edgecolors='none', marker='.', c='black', s=dot_size)
    plt.scatter(results.converged_x, results.converged_mult_x,
                edgecolors='none', marker='.', c=results.converged_iterations,
                cmap='YlGnBu', s=dot_size)
    plt.colorbar(label='Iterations')
    plt.xlim(0, 1)
    plt.yscale('log')
    plt.ylim(1e-10, 1e10)
    plt.title('Convergence depending on initial values')
    plt.xlabel('$x$')
    plt.ylabel('$\\lambda_x$')
    plt.savefig(filename + '.png', format='png')
    plt.close()


def main():
    for filename in glob.glob("convergence*.npz"):
        results = problem.util.ConvergenceResultList.from_file(filename)
        plot_results(results, os.path.splitext(filename)[0])


if __name__ == "__main__":
    main()
