from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np, statsmodels.api as sm
import regreg.api as rr
from selection.frequentist_eQTL.estimator import M_estimator_exact

from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from selection.frequentist_eQTL.approx_confidence_intervals import neg_log_cube_probability

from selection.randomized.M_estimator import M_estimator
from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues
from selection.tests.instance import gaussian_instance
from selection.api import randomization
from selection.bayesian.initial_soln import selection

def estimate_sigma(X, y, nstep=20, tol=1.e-4):

    old_sigma = 0.8
    for itercount in range(nstep):

        random_Z = np.zeros(p)
        sel = selection(X, y, random_Z, sigma=old_sigma)
        lam, epsilon, active, betaE, cube, initial_soln = sel
        print("active", active.sum())
        ols_fit = sm.OLS(y, X[:, active]).fit()
        new_sigma = np.linalg.norm(ols_fit.resid) / np.sqrt(n - active.sum() - 1)


        print("estimated sigma", new_sigma, old_sigma)
        if np.fabs(new_sigma - old_sigma) < tol :
            sigma = new_sigma
            break
        old_sigma = new_sigma

    return sigma

path = '/Users/snigdhapanigrahi/Results_bayesian/Egene_data/'
X = np.load(os.path.join(path + "X_" + "ENSG00000131697.13") + ".npy")
n, p = X.shape
print("dims", n,p)
X -= X.mean(0)[None, :]
X /= (X.std(0)[None, :] * np.sqrt(n))

y = np.load(os.path.join(path + "y_" + "ENSG00000131697.13") + ".npy")
y = y.reshape((y.shape[0],))

sigma = estimate_sigma(X, y, nstep=20, tol=1.e-5)
y /= sigma

lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * 1.
loss = rr.glm.gaussian(X, y)

epsilon = 1. / np.sqrt(n)

W = np.ones(p) * lam
penalty = rr.group_lasso(np.arange(p),
                         weights=dict(zip(np.arange(p), W)), lagrange=1.)

randomization = randomization.isotropic_gaussian((p,), scale=1.)

M_est = M_estimator_exact(loss, epsilon, penalty, randomization, randomizer='gaussian')

M_est.solve_approx()
active = M_est._overall
active_set = np.asarray([i for i in range(p) if active[i]])
nactive = np.sum(active)
sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
sys.stderr.write("Observed target" + str(M_est.target_observed)+ "\n")