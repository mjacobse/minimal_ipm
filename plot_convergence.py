import glob
import os
import problem


def plot_results(not_converged_x, not_converged_mult_x, converged_x,
                 converged_mult_x, converged_iterations, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['figure.figsize'] = (16, 9)

    fig = plt.figure()
    size = 5000000 / (len(not_converged_x) + len(converged_x))
    plt.scatter(not_converged_x, not_converged_mult_x,
                edgecolors='none', marker='.', c='black', s=size)
    plt.scatter(converged_x, converged_mult_x, edgecolors='none',
                marker='.', c=converged_iterations, cmap='YlGnBu', s=size)
    plt.colorbar(label='Iterations')
    plt.xlim(0, 1)
    plt.yscale('log')
    plt.ylim(1e-10, 1e10)
    plt.title('Convergence depending on initial values')
    plt.xlabel('$x$')
    plt.ylabel('$\\lambda_x$')
    plt.savefig(filename + '.png', format='png')
    plt.close(fig)


def main():
    for filename in glob.glob("convergence*.npz"):
        result = problem.util.ConvergenceResultList.from_file(filename)
        plot_results(result.not_converged_x,
                     result.not_converged_mult_x,
                     result.converged_x,
                     result.converged_mult_x,
                     result.converged_iterations, os.path.splitext(filename)[0])


if __name__ == "__main__":
    main()
