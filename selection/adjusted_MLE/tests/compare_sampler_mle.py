import numpy as np, sys, time

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import highdim
from scipy.stats import norm as ndist
from selection.algorithms.lasso import lasso_full
from selection.tests.instance import gaussian_instance

def BHfilter(pval, q=0.2):
    robjects.r.assign('pval', pval)
    robjects.r.assign('q', q)
    robjects.r('Pval = p.adjust(pval, method="BH")')
    robjects.r('S = which((Pval < q)) - 1')
    S = robjects.r('S')
    ind = np.zeros(pval.shape[0], np.bool)
    ind[np.asarray(S, np.int)] = 1
    return ind

def selInf_R(X, y, beta, lam, sigma, Type, alpha=0.1):
    robjects.r('''
               library("selectiveInference")
               selInf = function(X, y, beta, lam, sigma, Type, alpha= 0.1){
               y = as.matrix(y)
               X = as.matrix(X)
               beta = as.matrix(beta)
               lam = as.matrix(lam)[1,1]
               sigma = as.matrix(sigma)[1,1]
               Type = as.matrix(Type)[1,1]
               if(Type == 1){
                   type = "full"} else{
                   type = "partial"}
               inf = fixedLassoInf(x = X, y = y, beta = beta, lambda=lam, family = "gaussian",
                                   intercept=FALSE, sigma=sigma, alpha=alpha, type=type)
               return(list(ci = inf$ci, pvalue = inf$pv))}
               ''')

    inf_R = robjects.globalenv['selInf']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_beta = robjects.r.matrix(beta, nrow=p, ncol=1)
    r_lam = robjects.r.matrix(lam, nrow=1, ncol=1)
    r_sigma = robjects.r.matrix(sigma, nrow=1, ncol=1)
    r_Type = robjects.r.matrix(Type, nrow=1, ncol=1)
    output = inf_R(r_X, r_y, r_beta, r_lam, r_sigma, r_Type)
    ci = np.array(output.rx2('ci'))
    pvalue = np.array(output.rx2('pvalue'))
    return ci, pvalue

def glmnet_lasso(X, y):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    estimate = np.array(lambda_R(r_X, r_y).rx2('estimate'))
    estimate_min = np.array(lambda_R(r_X, r_y).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.min')))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.1se')))
    return estimate, estimate_min, lam_min, lam_1se

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    #library(bestsubset)
    source('~/best-subset/bestsubset/R/sim.R')
    sim_xy = bestsubset::sim.xy
    ''')

    r_simulate = robjects.globalenv['sim_xy']
    sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma

def coverage(intervals, pval, truth):
    if (truth!=0).sum()!=0:
        avg_power = np.mean(pval[truth != 0])
    else:
        avg_power = 0.
    return np.mean((truth > intervals[:, 0])*(truth < intervals[:, 1])), avg_power

def comparison_risk_inference_cv(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                                 randomizer_scale=np.sqrt(0.25), target = "selected",full_dispersion = True):

    while True:
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        # X, y, beta = gaussian_instance(n=n,
        #                                p=p,
        #                                signal=3.5,
        #                                s=s,
        #                                equicorrelated=False,
        #                                rho=0.20,
        #                                sigma=1,
        #                                random_signs=True)[:3]
        #
        # idx = np.arange(p)
        # Sigma = rho ** np.abs(np.subtract.outer(idx, idx))
        # sigma = 1
        # print("snr", beta.T.dot(Sigma).dot(beta) / ((sigma ** 2.) * n))
        # true_mean = X.dot(beta)

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

        _, _, lam_min, lam_1se = glmnet_lasso(X, y)

        randomized_lasso = highdim.gaussian(X,
                                            y,
                                            n * lam_1se * np.ones(p),
                                            randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

        signs = randomized_lasso.fit()
        nonzero = signs != 0

        sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

        if nonzero.sum()>0:
            #beta_target_rand = np.linalg.pinv(X[:, nonzero]).dot(true_mean)
            beta_target_rand = beta[nonzero]
            toc = time.time()
            mle_estimate, _, _, mle_sel_pval, mle_sel_intervals, ind_unbiased_estimator = randomized_lasso.\
                selective_MLE(target=target,dispersion=dispersion)
            tic = time.time()
            mle_time = tic-toc
            toc = time.time()
            _, sel_pval, sel_intervals = randomized_lasso.summary(target=target, dispersion=dispersion, level=0.9,
                                                                  compute_intervals=True)
            tic = time.time()
            sampler_time = tic-toc

            true_signals = np.zeros(p, np.bool)
            true_signals[beta != 0] = 1
            true_set = np.asarray([u for u in range(p) if true_signals[u]])
            active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])

            active_rand_bool = np.zeros(nonzero.sum(), np.bool)
            for x in range(nonzero.sum()):
                active_rand_bool[x] = (np.in1d(active_set_rand[x], true_set).sum() > 0)

            cov_mle_sel, _ = coverage(mle_sel_intervals, mle_sel_pval, beta_target_rand)
            cov_sel, _ = coverage(sel_intervals, sel_pval, beta_target_rand)

            power_mle_sel = ((active_rand_bool) * (np.logical_or((0. < mle_sel_intervals[:, 0]),
                                                                 (0. > mle_sel_intervals[:, 1])))).sum()
            power_sel = ((active_rand_bool) * (np.logical_or((0. < sel_intervals[:, 0]),
                                                             (0. > sel_intervals[:, 1])))).sum()

            sel_mle_discoveries = BHfilter(mle_sel_pval, q=0.1)
            sel_discoveries = BHfilter(sel_pval, q=0.1)

            power_mle_sel_dis = (sel_mle_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
            power_sel_dis = (sel_discoveries * active_rand_bool).sum() / float((beta != 0).sum())

            fdr_mle_sel_dis = (sel_mle_discoveries * ~active_rand_bool).sum() / float(max(sel_mle_discoveries.sum(), 1.))
            fdr_sel_dis = (sel_discoveries * ~active_rand_bool).sum() / float(max(sel_discoveries.sum(), 1.))
            break

    if True:
        return np.vstack((cov_mle_sel,
                          cov_sel,
                          np.mean(mle_sel_intervals[:, 1] - mle_sel_intervals[:, 0]),
                          np.mean(sel_intervals[:, 1] - sel_intervals[:, 0]),
                          power_mle_sel / float((beta != 0).sum()),
                          power_sel/float((beta != 0).sum()),
                          power_mle_sel_dis,
                          power_sel_dis,
                          fdr_mle_sel_dis,
                          fdr_sel_dis,
                          sel_mle_discoveries.sum(),
                          sel_discoveries.sum(),
                          np.mean(mle_estimate - beta_target_rand),
                          mle_time,
                          sampler_time))


if __name__ == "__main__":

    ndraw = 20

    target = "full"
    n, p, rho, s, beta_type, snr = 200, 1000, 0.35, 10, 1, 1.22

    #n, p, rho, s, beta_type, snr = 3000, 1000, 0.3, 30, 1, 0.15
    output_overall = np.zeros(15)
    for i in range(ndraw):
        if n > p:
            full_dispersion = True
        else:
            full_dispersion = False

        output = comparison_risk_inference_cv(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                              randomizer_scale=np.sqrt(0.50), target=target, full_dispersion=full_dispersion)
        output_overall += np.squeeze(output)

        sys.stderr.write("overall mle coverage " + str(output_overall[0] / float(i + 1)) + "\n")
        sys.stderr.write("overall sampler coverage " + str(output_overall[1] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("overall mle length " + str(output_overall[2] / float(i + 1)) + "\n")
        sys.stderr.write("overall sampler length " + str(output_overall[3] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("overall mle power " + str(output_overall[4] / float(i + 1)) + "\n")
        sys.stderr.write("overall sampler power " + str(output_overall[5] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("overall mle fdr " + str(output_overall[8] / float(i + 1)) + "\n")
        sys.stderr.write("overall sampler fdr " + str(output_overall[9] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("overall mle power post BH " + str(output_overall[6] / float(i + 1)) + "\n")
        sys.stderr.write("overall sampler power post BH " + str(output_overall[7] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("average mle discoveries " + str(output_overall[10] / float(i + 1)) + "\n")
        sys.stderr.write("average sampler discoveries " + str(output_overall[11] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("average bias " + str(output_overall[12] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("mle time " + str(output_overall[13] / float(i + 1)) + "\n")
        sys.stderr.write("sampler time " + str(output_overall[14] / float(i + 1)) + "\n" + "\n")

        sys.stderr.write("iteration completed " + str(i + 1) + "\n")


