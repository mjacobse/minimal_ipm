import argparse
import glob
import os
import problem


def plot_results(results, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    dot_size = 5000000 / (len(results.not_converged_x) +
                          len(results.converged_x))

    matplotlib.rcParams['figure.figsize'] = (16, 9)
    plt.figure()
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

    matplotlib.rcParams['figure.figsize'] = (12, 9)
    plt.figure()
    plt.scatter(results.not_converged_x * results.not_converged_mult_x,
                results.not_converged_s * results.not_converged_mult_s,
                edgecolors='none', marker='.', c='black', s=dot_size)
    plt.scatter(results.converged_x * results.converged_mult_x,
                results.converged_s * results.converged_mult_s,
                edgecolors='none', marker='.', c=results.converged_iterations,
                cmap='YlGnBu', s=dot_size)
    plt.colorbar(label='Iterations')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-8, 1e8)
    plt.ylim(1e-8, 1e8)
    plt.title('Convergence depending on initial complementarity products')
    plt.xlabel('$x \\lambda_x$')
    plt.ylabel('$s \\lambda_s$')
    plt.savefig(filename + '_compl.png', format='png')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', nargs='+')
    args = parser.parse_args()
    for filepath_pattern in args.filepath:
        for filepath in glob.glob(filepath_pattern):
            results = problem.util.ConvergenceResultList.from_file(filepath)
            plot_results(results, os.path.splitext(filepath)[0])


if __name__ == "__main__":
    main()
