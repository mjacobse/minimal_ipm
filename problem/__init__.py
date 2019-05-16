from . import ipm
from . import info
from . import util
import numpy
# Solve the problem
#
#  min  1/2 Q x^2 + c x
#   x
#
# s.t.  0 <= x <= XU


class Params:
    def __init__(self, quadratic=1.0, linear=-0.9, upper_bound=1.0):
        self.quadratic = quadratic
        self.linear = linear
        self.upper_bound = upper_bound

    def get_optimal_solution(self):
        return numpy.clip(-self.linear / self.quadratic, 0.0, self.upper_bound)

    def get_s(self, x):
        return self.upper_bound - x

    def get_mult_s(self, x, mult_x):
        return mult_x - (self.quadratic * x + self.linear)


# Primal-Dual feasible iterate
class FeasibleIterate:
    def __init__(self, init_x, init_mult_x, params):
        self.params = params
        if init_x <= 0.0 or init_x >= params.upper_bound:
            raise ValueError("Invalid value for x!")
        if init_mult_x <= 0.0:
            raise ValueError("Invalid value for lower multiplier!")
        self.x = init_x
        self.mult_x = init_mult_x
        if self.s <= 0:
            raise ValueError("Value for x too close to upper bound, causing "
                             "slack to be 0.0!")
        if self.mult_s <= 0:
            raise ValueError("Value for lower multiplier forces upper "
                             "multiplier to be nonpositive!")

    @property
    def s(self):
        return self.params.get_s(self.x)

    @property
    def mult_s(self):
        return self.params.get_mult_s(self.x, self.mult_x)

    @property
    def mult_equality(self):
        return self.mult_s

    def get_compl_products(self):
        return [self.x * self.mult_x, self.s * self.mult_s]

    def avg_compl(self):
        products = self.get_compl_products()
        return sum(products) / len(products)

    def affine_avg_compl(self, step, stepsize):
        return (max(0.0, (self.x + stepsize * step.x) *
                         (self.mult_x + stepsize * step.mult_x)) +
                max(0.0, (self.s + stepsize * step.s) *
                         (self.mult_s + stepsize * step.mult_s))) / 2

    def get_mixed_products(self, step):
        return [self.x * step.mult_x + self.mult_x * step.x,
                self.s * step.mult_s + self.mult_s * step.s]

    def get_residual(self):
        return [0.0, 0.0, 0.0]

    def update(self, step, stepsize):
        self.x += stepsize * step.x
        self.mult_x += stepsize * step.mult_x
        assert self.x > 0.0
        assert self.s > 0.0
        assert self.mult_x > 0.0
        assert self.mult_s > 0.0

    def get_max_stepsize(self, step, stepsize_limiter):
        assert stepsize_limiter.is_fulfilled(self)
        return stepsize_limiter.get_max_stepsize(self, step)


class Step:
    def __init__(self, raw_step):
        dx = raw_step[0]
        ds = raw_step[1]
        dmult_x = raw_step[2]
        dmult_s = raw_step[3]
        dmult_equality = raw_step[4]
        assert numpy.isclose(dx, -ds, rtol=0, atol=1e-14)
        assert numpy.isclose(dmult_s, dmult_equality, rtol=0, atol=1e-14)
        self.x = (dx - ds) / 2.0
        self.mult_x = dmult_x
        self.mult_s = (dmult_s + dmult_equality) / 2.0

    @property
    def s(self):
        return -self.x

    def get_compl_products(self):
        return [self.x * self.mult_x, self.s * self.mult_s]

    def __add__(self, other):
        cls = self.__class__
        result = cls.__new__(cls)
        result.x = self.x + other.x
        result.mult_x = self.mult_x + other.mult_x
        result.mult_s = self.mult_s + other.mult_s
        return result
