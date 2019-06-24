import numpy
import problem


class ConvergenceResultList:
    def __init__(self, params, max_iterations):
        self.params = params
        self.max_iterations = max_iterations
        self.converged_x = []
        self.converged_mult_x = []
        self.converged_iterations = []
        self.not_converged_x = []
        self.not_converged_mult_x = []

    @classmethod
    def from_file(cls, filepath):
        data = numpy.load(filepath)
        params = problem.Params()
        params.quadratic = data['params_quadratic']
        params.linear = data['params_linear']
        params.upper_bound = data['params_upper_bound']
        max_iterations = data['max_iterations']
        results = ConvergenceResultList(params, max_iterations)
        results.converged_x = data['converged_x'].tolist()
        results.converged_mult_x = data['converged_mult_x'].tolist()
        results.converged_iterations = data['converged_iterations'].tolist()
        results.not_converged_x = data['not_converged_x'].tolist()
        results.not_converged_mult_x = data['not_converged_mult_x'].tolist()
        return results

    def to_file(self, filepath):
        with open(filepath, 'wb') as f:
            numpy.savez_compressed(
                f,
                params_quadratic=self.params.quadratic,
                params_linear=self.params.linear,
                params_upper_bound=self.params.upper_bound,
                max_iterations=self.max_iterations,
                converged_x=self.converged_x,
                converged_mult_x=self.converged_mult_x,
                converged_iterations=self.converged_iterations,
                not_converged_x=self.not_converged_x,
                not_converged_mult_x=self.not_converged_mult_x
            )

    def add_result(self, x, mult_x, iterations):
        if iterations < self.max_iterations:
            self.converged_x.append(x)
            self.converged_mult_x.append(mult_x)
            self.converged_iterations.append(iterations)
        else:
            self.not_converged_x.append(x)
            self.not_converged_mult_x.append(mult_x)
