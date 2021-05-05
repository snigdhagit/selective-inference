import numpy as np
import nose.tools as nt

from selectinf.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selectinf.tests.instance import gaussian_instance

def test_full_targets(n=200, 
                      p=1000, 
                      signal_fac=0.5, 
                      s=5,
                      sigma=3,
                      rho=0.4, 
                      randomizer_scale=0.7,
                      full_dispersion=False):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    while True:
        signal = np.sqrt(signal_fac * 2 * np.log(p))
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=True,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = None

            if n>p:
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = full_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)
            else:
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = debiased_targets(conv.loglike,
                                                  conv._W,
                                                  nonzero,
                                                  penalty=conv.penalty,
                                                  dispersion=dispersion)

            result = conv.selective_MLE(observed_target,
                                        cov_target,
                                        cov_target_score)[0]
            pval = result['pvalue']
            estimate = result['MLE']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
            print("estimate, intervals", estimate, intervals)

            coverage = (beta[nonzero] > intervals[:, 0]) * (beta[nonzero] < intervals[:, 1])
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals


def test_selected_targets(n=2000, 
                          p=200, 
                          signal_fac=1.2,
                          s=5, 
                          sigma=2,
                          rho=0.7,
                          randomizer_scale=1.,
                          full_dispersion=True):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=True,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = 0.8 * np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            dispersion = None
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero, 
                                              dispersion=dispersion)

            result = conv.selective_MLE(observed_target,
                                        cov_target,
                                        cov_target_score)[0]
            estimate = result['MLE']
            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
            
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            # print("check ", np.asarray(result['MLE']), np.asarray(result['unbiased']))

            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def test_instance():

    n, p, s = 500, 100, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:s] = np.sqrt(2 * np.log(p) / n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    scale_ = np.std(Y)
    # uses noise of variance n * scale_ / 4 by default
    L = lasso.gaussian(X, Y, 3 * scale_ * np.sqrt(2 * np.log(p) * np.sqrt(n)))
    signs = L.fit()
    E = (signs != 0)

    M = E.copy()
    M[-3:] = 1
    dispersion = np.linalg.norm(Y - X[:, M].dot(np.linalg.pinv(X[:, M]).dot(Y))) ** 2 / (n - M.sum())
    (observed_target,
     cov_target,
     cov_target_score,
     alternatives) = selected_targets(L.loglike,
                                      L._W,
                                      M,
                                      dispersion=dispersion)

    print("check shapes", observed_target.shape, E.sum())

    result = L.selective_MLE(observed_target,
                             cov_target,
                             cov_target_score)[0]
    estimate = result['MLE']
    pval = result['pvalue']
    intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

    beta_target = np.linalg.pinv(X[:, M]).dot(X.dot(beta))

    coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
    print("observed_opt_state ", L.observed_opt_state)
    #print("check ", np.asarray(result['MLE']), np.asarray(result['unbiased']))

    return coverage

# def main(nsim=500):
#
#     cover = []
#     for i in range(nsim):
#
#         cover_ = test_instance()
#         cover.extend(cover_)
#         print(np.mean(cover), 'coverage so far ')


def test_selected_targets_disperse(n=500,
                                   p=100,
                                   signal_fac=1.,
                                   s=5,
                                   sigma=1.,
                                   rho=0.4,
                                   randomizer_scale=1,
                                   full_dispersion=True):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = 1.

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            dispersion = None
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)

            result = conv.selective_MLE(observed_target,
                                        cov_target,
                                        cov_target_score)[0]

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals


def main(nsim=500, full=False):
    P0, PA, cover, length_int = [], [], [], []
    from statsmodels.distributions import ECDF

    n, p, s = 500, 100, 0

    for i in range(nsim):
        if full:
            if n > p:
                full_dispersion = True
            else:
                full_dispersion = False
            p0, pA, cover_, intervals = test_full_targets(n=n, p=p, s=s, full_dispersion=full_dispersion)
            avg_length = intervals[:, 1] - intervals[:, 0]
        else:
            full_dispersion = True
            p0, pA, cover_, intervals = test_selected_targets(n=n, p=p, s=s, full_dispersion=full_dispersion)
            avg_length = intervals[:, 1] - intervals[:, 0]

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        # print(
        #     np.array(PA) < 0.1, np.mean(P0), np.std(P0), np.mean(np.array(P0) < 0.1), np.mean(np.array(PA) < 0.1), np.mean(cover),
        #     np.mean(avg_length), 'null pvalue + power + length')
        print("coverage and lengths ", np.mean(cover), np.mean(avg_length))

if __name__ == "__main__":
    main(nsim=500, full=True)
