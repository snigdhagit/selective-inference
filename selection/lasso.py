"""
This module contains a class `lasso`_ that implements
post selection for the lasso
as described in `post selection LASSO`_.


.. _covTest: http://arxiv.org/abs/1301.7161
.. _Kac Rice: http://arxiv.org/abs/1308.3020
.. _Spacings: http://arxiv.org/abs/1401.3889
.. _post selection LASSO: http://arxiv.org/abs/1311.6238


"""
import numpy as np
from sklearn.linear_model import Lasso
from .affine import (constraints, selection_interval,
                     interval_constraints,
                     stack)

from .variance_estimation import (interpolation_estimate,
                                  truncated_estimate)


from scipy.stats import norm as ndist
import warnings

try:
    import cvxpy as cvx
except ImportError:
    warnings.warn('cvx not available')
    pass

DEBUG = False

class lasso(object):

    r"""
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \frac{1}{2n} \|y-X\beta\|^2_2 + 
            f \lambda_{\max} \|\beta\|_1

    where $f$ is `frac` and 

    .. math::

       \lambda_{\max} = \frac{1}{n} \|X^Ty\|_{\infty}

    """

    # level for coverage is 1-alpha
    alpha = 0.05
    UMAU = False

    def __init__(self, y, X, frac=0.9, sigma=1):


        self.y = y
        self.X = X
        self.frac = frac
        self.sigma = sigma
        n, p = X.shape
        self.lagrange = frac * np.fabs(np.dot(X.T, y)).max() / n
        self._covariance = self.sigma**2 * np.identity(X.shape[0])

    def fit(self, sklearn_alpha=None, **lasso_args):
        """
        Fit the lasso using `Lasso` from `sklearn`.
        This sets the attribute `soln` and
        forms the constraints necessary for post-selection inference
        by caling `form_constraints()`.

        Parameters
        ----------

        sklearn_alpha : float
            Lagrange parameter, in the normalization set by `sklearn`.

        lasso_args : keyword args
             Passed to `sklearn.linear_model.Lasso`_

        Returns
        -------

        soln : np.float
             Solution to lasso with `sklearn_alpha=self.lagrange`.

        """
        if sklearn_alpha is not None:
            self.lagrange = sklearn_alpha
        self._lasso = Lasso(alpha=self.lagrange, **lasso_args)
        self._lasso.fit(self.X, self.y)
        self._soln = self._lasso.coef_
        self.form_constraints()

        return self._soln
      

    def form_constraints(self):
        """
        After having fit lasso, form the constraints
        necessary for inference.

        This sets the attributes: `active_constraints`,
        `inactive_constraints`, `active`.

        Returns
        -------

        None

        """

        X, y, soln, lagrange = self.X, self.y, self.soln, self.lagrange
        n, p = X.shape

        nonzero_coef = soln != 0
        tight_subgrad = np.fabs(np.fabs(np.dot(X.T, y - np.dot(X, soln))) / lagrange - 1) < 1.e-3
        if DEBUG:
            print 'KKT consistency', (nonzero_coef - tight_subgrad).sum()

        A = nonzero_coef
        self.active = np.nonzero(nonzero_coef)[0]
        if A.sum() > 0:
            sA = np.sign(soln[A])
            self.signs = sA
            XA = X[:,A]
            XnotA = X[:,~A]
            self._XAinv = XAinv = np.linalg.pinv(XA)
            self._SigmaA = np.dot(XAinv, XAinv.T)

            self._active_constraints = constraints(  
                -sA[:,None] * XAinv, 
                 -n*lagrange*sA*np.dot(self._SigmaA, 
                                         sA))
            self._active_constraints.covariance *= self.sigma**2
            self._SigmaA *=  self.sigma**2
            self._PA = PA = np.dot(XA, XAinv)
            irrep_subgrad = (n * lagrange * 
                             np.dot(np.dot(XnotA.T, XAinv.T), sA))

        else:
            XnotA = X
            self._PA = PA = 0
            self._XAinv = None
            irrep_subgrad = np.zeros(p)
            self._active_constraints = None

        if A.sum() < X.shape[1]:
            inactiveX = np.dot(np.identity(n) - PA, XnotA)
            scaling = np.ones(inactiveX.shape[1]) # np.sqrt((inactiveX**2).sum(0))
            inactiveX /= scaling[None,:]

            self._inactive_constraints = stack( 
                constraints(-inactiveX.T, 
                              lagrange * n + 
                              irrep_subgrad),
                constraints(inactiveX.T, 
                             lagrange * n -
                             irrep_subgrad))
            self._inactive_constraints.covariance *= self.sigma**2
        else:
            self._inactive_constraints = None

        if (self._active_constraints is not None 
            and self._inactive_constraints is not None):
            self._constraints = stack(self._active_constraints,
                                      self._inactive_constraints)
            self._constraints.covariance *= self.sigma**2
        elif self._active_constraints is not None:
            self._constraints = self._active_constraints
        else:
            self._constraints = self._inactive_constraints

    @property
    def soln(self):
        """
        Solution to the lasso problem, set by `fit` method.
        """
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def active_constraints(self):
        """
        Affine constraints imposed on the
        active variables by the KKT conditions.
        """
        return self._active_constraints

    @property
    def inactive_constraints(self):
        """
        Affine constraints imposed on the
        inactive subgradient by the KKT conditions.
        """
        return self._inactive_constraints

    @property
    def constraints(self):
        """
        Affine constraints for this LASSO problem.
        This is `self.active_constraints` stacked with
        `self.inactive_constraints`.
        """
        return self._constraints

    @property
    def intervals(self):
        """
        Intervals for OLS parameters of active variables
        adjusted for selection.
        """
        if not hasattr(self, "_intervals"):
            self._intervals = []
            C = self.constraints
            XAinv = self._XAinv
            if XAinv is not None:
                for i in range(XAinv.shape[0]):
                    eta = XAinv[i]
                    _interval = C.interval(eta, self.y,
                                           alpha=self.alpha,
                                           UMAU=self.UMAU)
                    self._intervals.append((self.active[i], eta, 
                                            (eta*self.y).sum(), 
                                            _interval))
        return self._intervals

    @property
    def active_pvalues(self, doc="Tests for active variables adjusted " + \
        " for selection."):
        if not hasattr(self, "_pvals"):
            self._pvals = []
            C = self.constraints
            XAinv = self._XAinv
            if XAinv is not None:
                for i in range(XAinv.shape[0]):
                    eta = XAinv[i]
                    _pval = C.pivot(eta, self.y)
                    _pval = 2 * min(_pval, 1 - _pval)
                    self._pvals.append((self.active[i], _pval))
        return self._pvals

    @property
    def nominal_intervals(self):
        """
        Intervals for OLS parameters of active variables
        that have not been adjusted for selection.
        """
        if not hasattr(self, "_intervals_unadjusted"):
            if not hasattr(self, "_constraints"):
                self.form_constraints()
            self._intervals_unadjusted = []
            XAinv = self._XAinv
            for i in range(self.active.shape[0]):
                eta = XAinv[i]
                center = (eta*self.y).sum()
                width = ndist.ppf(1-self.alpha/2.) * np.sqrt(self._SigmaA[i,i])
                _interval = [center-width, center+width]
                self._intervals_unadjusted.append((self.active[i], eta, (eta*self.y).sum(), 
                                        _interval))
        return self._intervals_unadjusted


def estimate_sigma(y, X, frac=0.1, 
                   lower=0.5,
                   upper=2,
                   npts=15,
                   ndraw=5000,
                   burnin=1000):
    r"""
    Estimate the parameter $\sigma$ in $y \sim N(X\beta, \sigma^2 I)$
    after fitting LASSO with Lagrange parameter `frac` times
    $\lambda_{\max}=\|X^Ty\|_{\infty}$.

    ## FUNCTION NEEDS TO BE UPDATED

    Uses `selection.variance_estimation.interpolation_estimate`

    Parameters
    ----------

    y : np.float
        Response to be used for LASSO.

    X : np.float
        Design matrix to be used for LASSO.

    frac : float
        What fraction of $\lambda_{\max}$ should be used to fit
        LASSO.

    lower : float
        Multiple of naive estimate to use as lower endpoint.

    upper : float
        Multiple of naive estimate to use as upper endpoint.

    npts : int
        Number of points in interpolation grid.

    ndraw : int
        Number of Gibbs steps to use for estimating
        each expectation.

    burnin : int
        How many Gibbs steps to use for burning in.

    Returns
    -------

    sigma_hat : float
        The root of the interpolant derived from GCM values.

    interpolant : scipy.interpolate.interp1d
        The interpolant, to be used for plotting or other 
        diagnostics.

    """

    n, p = X.shape
    L = lasso(y, X, frac=frac)
    soln = L.fit(tol=1.e-14, min_its=200)

    # now form the constraint for the inactive variables

    C = L.inactive_constraints
    PR = np.identity(n) - L.PA
    try:
        U, D, V = np.linalg.svd(PR)
    except np.linalg.LinAlgError:
        D, U = np.linalg.eigh(PR)

    keep = D >= 0.5
    U = U[:,keep]
    Z = np.dot(U.T, y)
    Z_inequality = np.dot(C.inequality, U)
    Z_constraint = constraints(Z_inequality, C.inequality_offset)
    if not Z_constraint(Z):
        raise ValueError('Constraint not satisfied. Gibbs algorithm will fail.')
    return interpolation_estimate(Z, Z_constraint,
                                  lower=lower,
                                  upper=upper,
                                  npts=npts,
                                  ndraw=ndraw,
                                  burnin=burnin,
                                  estimator='simulate')


