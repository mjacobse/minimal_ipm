import numpy
import problem
import random


class ConvergenceResultList:
    def __init__(self, params, max_iterations):
        self.params = params
        self.max_iterations = max_iterations
        self.__converged_x = []
        self.__converged_mult_x = []
        self.converged_iterations = []
        self.__not_converged_x = []
        self.__not_converged_mult_x = []

    @property
    def converged_x(self):
        return numpy.array(self.__converged_x)

    @property
    def not_converged_x(self):
        return numpy.array(self.__not_converged_x)

    @property
    def converged_mult_x(self):
        return numpy.array(self.__converged_mult_x)

    @property
    def not_converged_mult_x(self):
        return numpy.array(self.__not_converged_mult_x)

    @property
    def converged_s(self):
        return self.params.get_s(self.converged_x)

    @property
    def not_converged_s(self):
        return self.params.get_s(self.not_converged_x)

    @property
    def converged_mult_s(self):
        return self.params.get_mult_s(self.converged_x, self.converged_mult_x)

    @property
    def not_converged_mult_s(self):
        return self.params.get_mult_s(self.not_converged_x,
                                      self.not_converged_mult_x)

    @classmethod
    def from_file(cls, filepath):
        data = numpy.load(filepath)
        params = problem.Params()
        params.quadratic = data['params_quadratic']
        params.linear = data['params_linear']
        params.upper_bound = data['params_upper_bound']
        max_iterations = data['max_iterations']
        results = ConvergenceResultList(params, max_iterations)
        results.__converged_x = data['converged_x'].tolist()
        results.__converged_mult_x = data['converged_mult_x'].tolist()
        results.converged_iterations = data['converged_iterations'].tolist()
        results.__not_converged_x = data['not_converged_x'].tolist()
        results.__not_converged_mult_x = data['not_converged_mult_x'].tolist()
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
            self.__converged_x.append(x)
            self.__converged_mult_x.append(mult_x)
            self.converged_iterations.append(iterations)
        else:
            self.__not_converged_x.append(x)
            self.__not_converged_mult_x.append(mult_x)


def get_period_length(iteration_info):
    def find_previous_occurence(iterates, current):
        i = current - 1
        while i >= 0 and not iterates[i].isclose(iterates[current]):
            i -= 1
        return i

    iterates = [info.iterate for info in iteration_info]
    num_iterates = len(iterates)
    last = num_iterates - 1
    previous_last = find_previous_occurence(iterates, last)
    period_length = last - previous_last

    period_start = previous_last - period_length + 1
    if period_start < 0:
        return num_iterates  # not enough info to verify period

    for i in range(period_start, previous_last):
        if not iterates[i].isclose(iterates[i + period_length]):
            return num_iterates
    return period_length


def get_random_initial_guess(params, compl_space=False):
    if compl_space:
        compl_product_x = 10**random.uniform(-8, 8)
        compl_product_s = 10**random.uniform(-8, 8)
        return problem.info.get_iterates_from_compl_products(
            compl_product_x, compl_product_s, params)
    init_x = random.uniform(0.0, params.upper_bound)
    init_mult_x = 10**(random.uniform(-10, 10))
    return init_x, init_mult_x
