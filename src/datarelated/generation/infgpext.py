'''A infpy package extension for adding more kernels.

Created on Aug 9, 2016

@author: mraj

'''

import numpy as np
from infpy.gp import gp_1D_X_range
# from infpy.gp import SquaredExponentialKernel as SE
from infpy.gp import GaussianProcess, gp_1D_X_range, gp_plot_samples_from,gp_plot_prediction
from pylab import plot, savefig, title, close, figure, xlabel, ylabel
import infpy
import math






class CustomSquaredExponentialKernel( infpy.gp.RealKernel ):
    """
    A squared exponential kernel

    .. math::
        k(x_1, x_2) = K*k(c*r(x_1, x_2)^-mu) = \\K*textrm{exp}\\big(-c\\frac{r^mu}{2}\\big)

    where :math:`r(x_1, x_2) = |\\frac{x_1 - x_2}{l}|`
    """

    def __init__( self, params = None, priors = None, dimensions = None, K=1,c=1,mu=1 ):
        ( params, priors, dimensions ) = infpy.gp.kernel_init_args_helper(
                params,
                priors,
                dimensions
        )
        infpy.gp.RealKernel.__init__(
                self,
                params,
                priors,
                dimensions
        )


        self.K = K
        self.c = c
        self.mu = mu

    def __str__( self ):
        return """SquaredExpKernel"""

    def __call__( self, x1, x2, identical = False ):
        (x1, x2) = self._check_args( x1, x2 )
        return self.K*math.exp( - self.c*infpy.gp.real_kernel.distance_2( x1, x2, self.params )**self.mu/ 2 )


def gp_get_samples_from(
        gp,
        support,meanf,
        num_samples=1
):
    """
    Plot samples from a Gaussian process.
    """
    samples = []

    mean, sigma, LL = gp.predict(support)

    #gp_plot_prediction(support, mean, sigma)
    for i in xrange(num_samples):
        sample = np.random.multivariate_normal(
            np.asarray(mean).reshape(len(support),),
            sigma
        )
        for i in range(len(sample)):
            sample[i] = sample[i] + meanf(support[i])

        # plot([x[0] for x in support], sample)

        samples.append(sample)

    return samples

# def fn(x):
#     return 4*x
#
# varians = 1
# N = 10
# dims = 30
#
#
# interval = 1.0/dims
# support = gp_1D_X_range(0, 1, interval)
#
# kernel = CustomSquaredExponentialKernel([varians],None,None,1,1,0.2)
# interval = 20.01/dims
# # support = gp_1D_X_range(0, 1, .01)
# gp = GaussianProcess([], [], kernel)
#
# samples = gp_get_samples_from(gp, support, fn, num_samples=5)
# print samples
# # gp_plot_samples_from(gp, support, num_samples=5)
# xlabel('x')
# ylabel('f(x)')
# title('Samples from the prior')
# savefig('samples_from_prior.png')
