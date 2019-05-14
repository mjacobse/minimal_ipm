import problem
import numpy


class CentralPath:
    pass


def get_iterates_from_compl_products(compl_product_x, compl_product_s,
                                     params):
    roots = numpy.roots([params.quadratic,
                         params.linear - params.quadratic * params.upper_bound,
                         -(compl_product_x + compl_product_s +
                           params.linear * params.upper_bound),
                         params.upper_bound * compl_product_x])
    good_root = min(roots, key=lambda x: abs(x - params.upper_bound / 2.0))
    init_x = good_root
    init_mult_x = compl_product_x / init_x
    return init_x, init_mult_x
