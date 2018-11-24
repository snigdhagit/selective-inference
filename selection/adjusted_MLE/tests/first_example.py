import numpy as np, os, itertools, sys, time
import pandas as pd
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.algorithms.lasso import lasso_full
from scipy.stats import norm as ndist, f as F

from selection.adjusted_MLE.cv_MLE import (sim_xy,
                                           selInf_R,
                                           glmnet_lasso,
                                           BHfilter,
                                           coverage)
from statsmodels.distributions.empirical_distribution import ECDF

def pivot(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20, randomizer_scale=np.sqrt(0.50), full_dispersion=True):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    randomized_lasso = lasso.gaussian(X,
                                      y,
                                      feature_weights=lam_theory * np.ones(p),
                                      randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

    if nonzero.sum() > 0:
        target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero,
                                          dispersion=dispersion)

        toc = time.time()
        MLE_estimate, observed_info_mean, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                                              cov_target,
                                                                                                                              cov_target_score,
                                                                                                                              alternatives)
        tic = time.time()
        cov_MLE, _ = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])

        pivot_MLE = np.true_divide(MLE_estimate - target_randomized, np.sqrt(np.diag(observed_info_mean)))
        time_MLE = tic - toc

        toc = time.time()
        sampler_pivot, sampler_pval, sampler_intervals = randomized_lasso.summary(observed_target,
                                                                                  cov_target,
                                                                                  cov_target_score,
                                                                                  alternatives,
                                                                                  level=0.9,
                                                                                  compute_intervals=True,
                                                                                  ndraw=500000)

        tic = time.time()
        cov_sampler, _ = coverage(sampler_intervals, sampler_pval, target_randomized, beta[nonzero])
        time_sampler = tic - toc

        return pivot_MLE, sampler_pivot, time_MLE, time_sampler, np.mean(cov_MLE), np.mean(cov_sampler)

def plot_pivot(ndraw=500):
    import matplotlib.pyplot as plt

    _pivot_MLE = []
    _pivot_sampler = []
    _cov_MLE = 0.
    _cov_sampler = 0.
    _time_MLE = 0.
    _time_sampler = 0.
    for i in range(ndraw):
        pivot_MLE, pivot_sampler, time_MLE, time_sampler, cov_MLE, cov_sampler = pivot()
        _cov_MLE += cov_MLE
        _cov_sampler += cov_sampler
        _time_MLE += time_MLE
        _time_sampler += time_sampler
        for j in range(pivot_MLE.shape[0]):
            _pivot_MLE.append(pivot_MLE[j])
            _pivot_sampler.append(pivot_sampler[j])

    print("coverage of MLE", _cov_MLE/ndraw, _cov_sampler/ndraw)
    print("times compare", _time_MLE/ndraw,  _time_sampler/ndraw)
    plt.clf()
    ecdf_MLE = ECDF(ndist.cdf(np.asarray(_pivot_MLE)))
    #ecdf_sampler = ECDF(np.asarray(_pivot_sampler))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()
    plt.savefig("/Users/psnigdha/maximum_likelihood_inference/Talk_plots/Motivating_example/Pivot_n500_p100_rho35_snr20.png")

plot_pivot(ndraw=500)

def joint_pivot(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20, randomizer_scale=np.sqrt(0.50), full_dispersion=True):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    randomized_lasso = lasso.gaussian(X,
                                      y,
                                      feature_weights=lam_theory * np.ones(p),
                                      randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

    if nonzero.sum() > 0:
        target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero,
                                          dispersion=dispersion)

        toc = time.time()
        MLE_estimate, observed_info_mean, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                                              cov_target,
                                                                                                                              cov_target_score,
                                                                                                                              alternatives)
        tic = time.time()
        cov_MLE, _ = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])

        pivot_MLE = (MLE_estimate - target_randomized).T.dot(np.linalg.inv(observed_info_mean)).dot(MLE_estimate - target_randomized)
        nactive = nonzero.sum()
        joint_pivot_MLE = F.cdf((n-nactive)*pivot_MLE/((n-1.)*nactive), nactive, n-nactive)
        time_MLE = tic - toc

        return joint_pivot_MLE, time_MLE, np.mean(cov_MLE)

def plot_pivot_joint(ndraw=500):
    import matplotlib.pyplot as plt

    _pivot_MLE = []
    _cov_MLE = 0.
    _time_MLE = 0.
    for i in range(ndraw):
        pivot_MLE, time_MLE, cov_MLE = joint_pivot()
        _cov_MLE += cov_MLE
        _time_MLE += time_MLE
        _pivot_MLE.append(pivot_MLE)

    print("coverage of MLE", _cov_MLE/ndraw)
    print("times compare", _time_MLE/ndraw)
    plt.clf()
    ecdf_MLE = ECDF(np.asarray(_pivot_MLE))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()
    plt.savefig("/Users/psnigdha/maximum_likelihood_inference/Talk_plots/Motivating_example/Joint_Pivot_n500_p100_rho35_snr20.png")

#plot_pivot_joint(ndraw=500)