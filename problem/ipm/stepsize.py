import numpy
import problem


class NonNegativityNeighborhood:
    def __init__(self, fraction_to_boundary=1.0):
        self.fraction_to_boundary = fraction_to_boundary

    def is_fulfilled(self, iterate):
        return (iterate.x > 0.0 and iterate.s > 0.0 and
                iterate.mult_x > 0.0 and iterate.mult_s > 0.0)

    def get_stepsize_limit(self, current, step):
        assert current > 0.0
        limit = numpy.inf
        if step < 0.0:
            allowed_subtract = max(numpy.nextafter(-current, 0.0),
                                   -current * self.fraction_to_boundary)
            limit = allowed_subtract / step
            if limit * step < allowed_subtract:
                limit = numpy.nextafter(limit, 0.0)
            assert current + limit * step > 0.0
        return limit

    def get_stepsize_limit_upper(self, current, step, boundary):
        assert (boundary - current) > 0.0
        limit = numpy.inf
        if step > 0.0:
            allowed_add = numpy.nextafter(((numpy.nextafter(boundary, 0.0) - current) + (boundary - current)) / 2.0, 0.0)
            limit = allowed_add / step
            if limit * step > allowed_add:
                limit = numpy.nextafter(limit, 0.0)
            assert (boundary - (current + limit * step)) > 0.0
        return limit

    def get_max_stepsize(self, iterate, step):
        stepsize_limits = [
            self.get_stepsize_limit(iterate.x, step.x),
            self.get_stepsize_limit_upper(iterate.x, step.x,
                                          problem.Params.upper_bound),
            self.get_stepsize_limit(iterate.mult_x, step.mult_x),
            self.get_stepsize_limit(iterate.mult_s, step.mult_s)
        ]
        limit = min(stepsize_limits)
        assert limit > 0.0
        return min(1.0, limit)


class NegativeInfinityNeighborhood:
    def __init__(self, gamma=0.0003):
        self.gamma = gamma

    def is_fulfilled(self, iterate):
        limit = self.gamma * iterate.avg_compl()
        return (iterate.x * iterate.mult_x >= limit and
                iterate.s * iterate.mult_s >= limit)

    def get_limits_from_parabola(self, a, b, c):
        if a == 0.0:
            if b == 0.0:
                return (-numpy.inf, numpy.inf)
            linear_solution = c / b
            if b > 0.0:
                return (linear_solution, numpy.inf)
            else:
                return (-numpy.inf, linear_solution)

        root_argument = b**2 - 4 * a * c
        if root_argument < 0.0:
            return (-numpy.inf, numpy.inf)
        root = numpy.sqrt(root_argument)
        if a > 0.0:
            return ((-b + root) / (2 * a), numpy.inf)
        return ((-b + root) / (2 * a), (-b - root) / (2 * a))

    def get_max_stepsize(self, iterate, step):
        compl_products = iterate.get_compl_products()
        step_compl_products = step.get_compl_products()
        mixed_compl_products = iterate.get_mixed_products(step)
        assert len(compl_products) == len(step_compl_products)
        assert len(compl_products) == len(mixed_compl_products)
        num_products = len(compl_products)

        lower_limit = -numpy.inf
        upper_limit = numpy.inf
        for i in range(0, num_products):
            quadratic_coefficient = (
                step_compl_products[i] -
                self.gamma * sum(step_compl_products) / num_products
            )
            linear_coefficient = (
                mixed_compl_products[i] -
                self.gamma * sum(mixed_compl_products) / num_products
            )
            constant_coefficient = (
                compl_products[i] -
                self.gamma * sum(compl_products) / num_products
            )

            lower, upper = self.get_limits_from_parabola(quadratic_coefficient,
                                                         linear_coefficient,
                                                         constant_coefficient)
            if lower > lower_limit:
                lower_limit = lower
            if upper < upper_limit:
                upper_limit = upper
        negativity_limit = NonNegativityNeighborhood().get_max_stepsize(
            iterate, step)
        return min(1.0, upper_limit, negativity_limit)
