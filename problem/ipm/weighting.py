import math
import numpy
from .stepsize import NonNegativityNeighborhood


class WeightingFull:
    def get_weight(self, iterate, step, step_correction):
        return 1.0


class WeightingSettings:
    def __init__(self, num_evals=9, dynamic_min_weight=True,
                 ties_give_larger_weight=True, min_weight=0.0):
        self.num_evals = num_evals
        self.dynamic_min_weight = dynamic_min_weight
        self.ties_give_larger_weight = ties_give_larger_weight
        self.min_weight = min_weight

    def get_test_weights(self, iterate, step, step_correction):
        min_weight = self.min_weight
        if self.dynamic_min_weight:
            nominal_stepsize = iterate.get_max_stepsize(
                step + step_correction, NonNegativityNeighborhood())
            min_weight = min(min_weight, nominal_stepsize**2)
        weights = numpy.linspace(min_weight, 1.0, self.num_evals)
        if self.ties_give_larger_weight:
            weights = reversed(weights)
        return weights


class WeightingMaxStepsize:
    def __init__(self, settings=WeightingSettings()):
        self.settings = settings

    def get_weight(self, iterate, step, step_correction):
        weights = self.settings.get_test_weights(iterate, step, step_correction)
        best_stepsize = 0.0
        for weight in weights:
            stepsize = iterate.get_max_stepsize(step + weight * step_correction,
                                                NonNegativityNeighborhood())
            if stepsize > best_stepsize:
                best_stepsize = stepsize
                best_weight = weight
        return best_weight


class WeightingBestCentrality:
    def __init__(self, settings=WeightingSettings()):
        self.settings = settings

    def get_weight(self, iterate, step, step_correction):
        weights = self.settings.get_test_weights(iterate, step, step_correction)
        best_distance = float('inf')
        for weight in weights:
            step_combined = step + weight * step_correction
            stepsize = iterate.get_max_stepsize(step_combined,
                                                NonNegativityNeighborhood())
            compl_products = iterate.get_affine_compl_products(step_combined,
                                                               stepsize)
            distance = abs(math.log(compl_products[0] / compl_products[1]))
            if distance < best_distance:
                best_distance = distance
                best_weight = weight
        return best_weight
