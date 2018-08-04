import numpy as np, os, itertools
import pandas as pd

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from scipy.stats import norm as ndist
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.algorithms.lasso import lasso_full
from selection.tests.instance import gaussian_instance

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    #library(bestsubset)
    source('~/best-subset/bestsubset/R/sim.R')
    sim_xy = sim.xy
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

def selInf_R(X, y, beta, lam, sigma, Type, alpha=0.1):
    robjects.r('''
               R.Version()
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
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate.1se = estimate.1se, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    estimate_1se = np.array(lambda_R(r_X, r_y).rx2('estimate.1se'))
    estimate_min = np.array(lambda_R(r_X, r_y).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.min')))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.1se')))
    return estimate_1se, estimate_min, lam_min, lam_1se

def coverage(intervals, pval, truth):
    if (truth!=0).sum()!=0:
        avg_power = np.mean(pval[truth != 0])
    else:
        avg_power = 0.
    return np.mean((truth > intervals[:, 0])*(truth < intervals[:, 1])), avg_power

def BHfilter(pval, q=0.2):
    robjects.r.assign('pval', pval)
    robjects.r.assign('q', q)
    robjects.r('Pval = p.adjust(pval, method="BH")')
    robjects.r('S = which((Pval < q)) - 1')
    S = robjects.r('S')
    ind = np.zeros(pval.shape[0], np.bool)
    ind[np.asarray(S, np.int)] = 1
    return ind

def relative_risk(est, truth, Sigma):
    if (truth!=0).sum>0:
        return (est - truth).T.dot(Sigma).dot(est - truth) / truth.T.dot(Sigma).dot(truth)
    else:
        return (est - truth).T.dot(Sigma).dot(est - truth)

def comparison_cvmetrics_selected(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                                  randomizer_scale = np.sqrt(0.25), full_dispersion = True,
                                  tuning_nonrand = "lambda.min", tuning_rand = "lambda.1se"):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u]!=0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    glm_LASSO_1se, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y)
    if tuning_nonrand == "lambda.min":
        lam_LASSO = lam_min
        glm_LASSO = glm_LASSO_min
    else:
        lam_LASSO = lam_1se
        glm_LASSO = glm_LASSO_1se
    rel_LASSO = np.zeros(p)
    active_LASSO = (glm_LASSO != 0)
    nactive_LASSO = active_LASSO.sum()
    active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])
    active_LASSO_bool = np.asarray([(np.in1d(active_set_LASSO[z], true_set).sum() > 0) for z in range(nactive_LASSO)], np.bool)
    Lee_nreport = 0

    if nactive_LASSO>0:
        rel_LASSO[active_LASSO] = np.linalg.pinv(X[:, active_LASSO]).dot(y)
        Lee_target = np.linalg.pinv(X[:, active_LASSO]).dot(X.dot(beta))
        Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_LASSO, sigma_, Type=0, alpha=0.1)

        if (Lee_pval.shape[0] == Lee_target.shape[0]):

            cov_Lee, _ = coverage(Lee_intervals, Lee_pval, Lee_target)
            inf_entries_bool = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
            inf_entries = np.mean(inf_entries_bool)
            if inf_entries == 1.:
                length_Lee = 0.
            else:
                length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries_bool])
            power_Lee = ((active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]), (0. > Lee_intervals[:, 1])))).sum()/ float((beta != 0).sum())
            Lee_discoveries = BHfilter(Lee_pval, q=0.1)
            power_Lee_BH = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_Lee_BH = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))

            post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
            naive_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
            naive_intervals = np.vstack([post_LASSO_OLS - 1.65 * naive_sd,
                                        post_LASSO_OLS + 1.65 * naive_sd]).T
            naive_pval = ndist.cdf(post_LASSO_OLS / naive_sd)
            cov_naive, _ = coverage(naive_intervals, naive_pval, Lee_target)
            length_naive = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
            power_naive = ((active_LASSO_bool) * (np.logical_or((0. < naive_intervals[:, 0]), (0. > naive_intervals[:, 1])))).sum()/ float((beta != 0).sum())
            naive_discoveries = BHfilter(naive_pval, q=0.1)
            power_naive_BH = (naive_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_naive_BH = (naive_discoveries * ~active_LASSO_bool).sum() / float(max(naive_discoveries.sum(), 1.))
        else:
            Lee_nreport = 1
            cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH = [0., 0., 0., 0., 0., 0.]
            cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH = [0., 0., 0., 0., 0.]
            naive_discoveries = np.zeros(1)
            Lee_discoveries = np.zeros(1)
    elif nactive_LASSO == 0:
        Lee_nreport = 1
        cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH = [0., 0., 0., 0., 0., 0.]
        cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH = [0., 0., 0., 0., 0.]
        naive_discoveries =  np.zeros(1)
        Lee_discoveries = np.zeros(1)

    if tuning_rand == "lambda.min":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= n * lam_min * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    else:
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= n * lam_1se * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())], np.bool)
    sel_MLE = np.zeros(p)
    ind_est = np.zeros(p)
    randomized_lasso_est = np.zeros(p)
    randomized_rel_lasso_est = np.zeros(p)
    MLE_nreport = 0

    if nonzero.sum()>0:
        target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero,
                                          dispersion=dispersion)

        MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                             cov_target,
                                                                                                             cov_target_score,
                                                                                                             alternatives)
        sel_MLE[nonzero] = MLE_estimate
        ind_est[nonzero] = ind_unbiased_estimator
        randomized_lasso_est = randomized_lasso.initial_soln
        randomized_rel_lasso_est = randomized_lasso._beta_full

        cov_MLE, _ = coverage(MLE_intervals, MLE_pval, target_randomized)
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum()/ float((beta != 0).sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)
    else:
        MLE_nreport = 1
        cov_MLE, length_MLE, power_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE= [0., 0., 0., 0., 0., 0.]
        MLE_discoveries = np.zeros(1)

    risks = np.vstack((relative_risk(sel_MLE, beta, Sigma),
                       relative_risk(ind_est, beta, Sigma),
                       relative_risk(randomized_lasso_est, beta, Sigma),
                       relative_risk(randomized_rel_lasso_est, beta, Sigma),
                       relative_risk(rel_LASSO, beta, Sigma),
                       relative_risk(glm_LASSO, beta, Sigma)))

    naive_inf = np.vstack((cov_naive, length_naive, 0., power_naive, power_naive_BH, fdr_naive_BH, naive_discoveries.sum(), nactive_LASSO, 0.))
    Lee_inf = np.vstack((cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH, Lee_discoveries.sum(), nactive_LASSO, 0.))
    Liu_inf = np.zeros((9,1))
    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., power_MLE, power_MLE_BH, fdr_MLE_BH, MLE_discoveries.sum(), nonzero.sum(), bias_MLE))
    nreport = np.vstack((Lee_nreport, 0., MLE_nreport))

    return np.vstack((risks, naive_inf, Lee_inf, Liu_inf, MLE_inf, nreport))

def comparison_cvmetrics_full(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                              randomizer_scale = np.sqrt(0.25), full_dispersion = True,
                              tuning_nonrand = "lambda.min", tuning_rand = "lambda.1se"):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u]!=0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    glm_LASSO_1se, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y)
    if tuning_nonrand == "lambda.min":
        lam_LASSO = lam_min
        glm_LASSO = glm_LASSO_min
    else:
        lam_LASSO = lam_1se
        glm_LASSO = glm_LASSO_1se
    rel_LASSO = np.zeros(p)
    active_LASSO = (glm_LASSO != 0)
    nactive_LASSO = active_LASSO.sum()
    active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])
    active_LASSO_bool = np.asarray([(np.in1d(active_set_LASSO[z], true_set).sum() > 0) for z in range(nactive_LASSO)], np.bool)
    Lee_nreport = 0

    if nactive_LASSO>0:
        rel_LASSO[active_LASSO] = np.linalg.pinv(X[:, active_LASSO]).dot(y)
        Lee_target = beta[active_LASSO]
        Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_LASSO, sigma_, Type=1, alpha=0.1)

        if (Lee_pval.shape[0] == Lee_target.shape[0]):

            cov_Lee, _ = coverage(Lee_intervals, Lee_pval, Lee_target)
            inf_entries_bool = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
            inf_entries = np.mean(inf_entries_bool)
            if inf_entries == 1.:
                length_Lee = 0.
            else:
                length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries_bool])
            power_Lee = ((active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]), (0. > Lee_intervals[:, 1])))).sum()/ float((beta != 0).sum())
            Lee_discoveries = BHfilter(Lee_pval, q=0.1)
            power_Lee_BH = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_Lee_BH = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))

            post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
            naive_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
            naive_intervals = np.vstack([post_LASSO_OLS - 1.65 * naive_sd,
                                        post_LASSO_OLS + 1.65 * naive_sd]).T
            naive_pval = ndist.cdf(post_LASSO_OLS / naive_sd)
            cov_naive, _ = coverage(naive_intervals, naive_pval, Lee_target)
            length_naive = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
            power_naive = ((active_LASSO_bool) * (np.logical_or((0. < naive_intervals[:, 0]), (0. > naive_intervals[:, 1])))).sum()/ float((beta != 0).sum())
            naive_discoveries = BHfilter(naive_pval, q=0.1)
            power_naive_BH = (naive_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
            fdr_naive_BH = (naive_discoveries * ~active_LASSO_bool).sum() / float(max(naive_discoveries.sum(), 1.))
        else:
            Lee_nreport = 1
            cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH = [0., 0., 0., 0., 0., 0.]
            cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH = [0., 0., 0., 0., 0.]
            naive_discoveries = np.zeros(1)
            Lee_discoveries = np.zeros(1)
            
    elif nactive_LASSO == 0:
        Lee_nreport = 1
        cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH = [0., 0., 0., 0., 0., 0.]
        cov_naive, length_naive, power_naive, power_naive_BH, fdr_naive_BH = [0., 0., 0., 0., 0.]
        naive_discoveries =  np.zeros(1)
        Lee_discoveries = np.zeros(1)

    lasso_Liu = lasso_full.gaussian(X, y, n * lam_1se)
    Lasso_soln_Liu = lasso_Liu.fit()
    active_set_Liu = np.nonzero(Lasso_soln_Liu != 0)[0]
    nactive_Liu = active_set_Liu.shape[0]
    active_Liu_bool = np.asarray([(np.in1d(active_set_Liu[a], true_set).sum() > 0) for a in range(nactive_Liu)],np.bool)
    Liu_nreport = 0

    if nactive_Liu > 0:
        Liu_target = beta[Lasso_soln_Liu != 0]
        df = lasso_Liu.summary(level=0.90, compute_intervals=True, dispersion=dispersion)
        Liu_lower, Liu_upper, Liu_pval = np.asarray(df['lower_confidence']), \
                                         np.asarray(df['upper_confidence']), \
                                         np.asarray(df['pval'])
        Liu_intervals = np.vstack((Liu_lower, Liu_upper)).T
        cov_Liu, _ = coverage(Liu_intervals, Liu_pval, Liu_target)
        length_Liu = np.mean(Liu_intervals[:, 1] - Liu_intervals[:, 0])
        power_Liu = ((active_Liu_bool) * (np.logical_or((0. < Liu_intervals[:, 0]),
                                                        (0. > Liu_intervals[:, 1])))).sum()/ float((beta != 0).sum())
        Liu_discoveries = BHfilter(Liu_pval, q=0.1)
        power_Liu_BH = (Liu_discoveries * active_Liu_bool).sum() / float((beta != 0).sum())
        fdr_Liu_BH = (Liu_discoveries * ~active_Liu_bool).sum() / float(max(Liu_discoveries.sum(), 1.))

    else:
        Liu_nreport = 1
        cov_Liu, length_Liu, power_Liu, power_Liu_BH, fdr_Liu_BH= [0., 0., 0., 0., 0.]
        Liu_discoveries = np.zeros(1)

    if tuning_rand == "lambda.min":
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= n * lam_min * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)
    else:
        randomized_lasso = lasso.gaussian(X,
                                          y,
                                          feature_weights= n * lam_1se * np.ones(p),
                                          randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())], np.bool)
    sel_MLE = np.zeros(p)
    ind_est = np.zeros(p)
    randomized_lasso_est = np.zeros(p)
    randomized_rel_lasso_est = np.zeros(p)
    MLE_nreport = 0

    if nonzero.sum()>0:
        target_randomized = beta[nonzero]
        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = full_targets(randomized_lasso.loglike,
                                      randomized_lasso._W,
                                      nonzero,
                                      dispersion=dispersion)
        MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                             cov_target,
                                                                                                             cov_target_score,
                                                                                                             alternatives)
        sel_MLE[nonzero] = MLE_estimate
        ind_est[nonzero] = ind_unbiased_estimator
        randomized_lasso_est = randomized_lasso.initial_soln
        randomized_rel_lasso_est = randomized_lasso._beta_full

        cov_MLE, _ = coverage(MLE_intervals, MLE_pval, target_randomized)
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum()/ float((beta != 0).sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)
    else:
        MLE_nreport = 1
        cov_MLE, length_MLE, power_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE = [0., 0., 0., 0., 0., 0.]
        MLE_discoveries = np.zeros(1)

    risks = np.vstack((relative_risk(sel_MLE, beta, Sigma),
                       relative_risk(ind_est, beta, Sigma),
                       relative_risk(randomized_lasso_est, beta, Sigma),
                       relative_risk(randomized_rel_lasso_est, beta, Sigma),
                       relative_risk(rel_LASSO, beta, Sigma),
                       relative_risk(glm_LASSO, beta, Sigma)))

    naive_inf = np.vstack((cov_naive, length_naive, 0., power_naive, power_naive_BH, fdr_naive_BH, naive_discoveries.sum(), nactive_LASSO, 0.))
    Lee_inf = np.vstack((cov_Lee, length_Lee, inf_entries, power_Lee, power_Lee_BH, fdr_Lee_BH, Lee_discoveries.sum(), nactive_LASSO, 0.))
    Liu_inf = np.vstack((cov_Liu, length_Liu, 0., power_Liu, power_Liu_BH, fdr_Liu_BH, Liu_discoveries.sum(), nactive_Liu, 0.))
    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., power_MLE, power_MLE_BH, fdr_MLE_BH, MLE_discoveries.sum(), nonzero.sum(), bias_MLE))
    nreport = np.vstack((Lee_nreport, Liu_nreport, MLE_nreport))
    return np.vstack((risks, naive_inf, Lee_inf, Liu_inf, MLE_inf, nreport))

def output_file(n=300, p=100, rho=0.35, s=5, beta_type=1, snr_values=np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.42, 0.71, 1.22]),
                target="selected", tuning_nonrand="lambda.min", tuning_rand="lambda.1se",
                randomizing_scale = np.sqrt(0.50), ndraw = 50, outpath = None):

    df_selective_inference = pd.DataFrame()
    df_risk = pd.DataFrame()

    if n > p:
        full_dispersion = True
    else:
        full_dispersion = False

    snr_list = []
    snr_list_0 = []
    for snr in snr_values:
        snr_list.append(snr*np.ones(4))
        snr_list_0.append(snr)
        output_overall = np.zeros(45)
        if target == "selected":
            for i in range(ndraw):
                output_overall += np.squeeze(comparison_cvmetrics_selected(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                                           randomizer_scale=randomizing_scale, full_dispersion=full_dispersion,
                                                                           tuning_nonrand =tuning_nonrand, tuning_rand=tuning_rand))
        elif target == "full":
            for i in range(ndraw):
                output_overall += np.squeeze(comparison_cvmetrics_full(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                                       randomizer_scale=randomizing_scale, full_dispersion=full_dispersion,
                                                                       tuning_nonrand =tuning_nonrand, tuning_rand=tuning_rand))

        nLee = output_overall[42]
        nLiu = output_overall[43]
        nMLE = output_overall[44]

        relative_risk = (output_overall[0:6] / float(ndraw)).reshape((1, 6))
        nonrandomized_naive_inf = (output_overall[6:15] / float(ndraw - nLee)).reshape((1, 9))
        nonrandomized_Lee_inf = (output_overall[15:24] / float(ndraw - nLee)).reshape((1, 9))
        nonrandomized_Liu_inf = (output_overall[24:33] / float(ndraw - nLiu)).reshape((1, 9))
        randomized_MLE_inf = (output_overall[33:42] / float(ndraw - nMLE)).reshape((1, 9))

        df_naive = pd.DataFrame(data=nonrandomized_naive_inf,columns=['coverage', 'length', 'prop-infty',
                                                                      'power', 'power-BH', 'fdr-BH',
                                                                      'tot-discoveries', 'tot-active', 'bias'])
        df_naive['method'] = "Naive"
        df_Lee = pd.DataFrame(data=nonrandomized_Lee_inf, columns=['coverage', 'length', 'prop-infty',
                                                                   'power', 'power-BH', 'fdr-BH',
                                                                   'tot-discoveries', 'tot-active','bias'])
        df_Lee['method'] = "Lee"

        if target=="selected":
            nonrandomized_Liu_inf[nonrandomized_Liu_inf==0] = 'NaN'

        df_Liu = pd.DataFrame(data=nonrandomized_Liu_inf,
                              columns=['coverage', 'length', 'prop-infty',
                                       'power', 'power-BH', 'fdr-BH',
                                       'tot-discoveries', 'tot-active',
                                       'bias'])
        df_Liu['method'] = "Liu"

        df_MLE = pd.DataFrame(data=randomized_MLE_inf, columns=['coverage', 'length', 'prop-infty',
                                                                                'power', 'power-BH', 'fdr-BH',
                                                                                'tot-discoveries', 'tot-active',
                                                                                'bias'])
        df_MLE['method'] = "MLE"
        df_risk_metrics = pd.DataFrame(data=relative_risk, columns=['sel-MLE', 'ind-est', 'rand-LASSO','rel-rand-LASSO', 'rel-LASSO', 'LASSO'])

        df_selective_inference = df_selective_inference.append(df_naive, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_Lee, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_Liu, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_MLE, ignore_index=True)

        df_risk = df_risk.append(df_risk_metrics, ignore_index=True)

    snr_list = list(itertools.chain.from_iterable(snr_list))
    df_selective_inference['n'] = n
    df_selective_inference['p'] = p
    df_selective_inference['s'] = s
    df_selective_inference['rho'] = rho
    df_selective_inference['beta-type'] = beta_type
    df_selective_inference['snr'] = pd.Series(np.asarray(snr_list))
    df_selective_inference['target'] = target

    df_risk['n'] = n
    df_risk['p'] = p
    df_risk['s'] = s
    df_risk['rho'] = rho
    df_risk['beta-type'] = beta_type
    df_risk['snr'] = pd.Series(np.asarray(snr_list_0))
    df_risk['target'] = target

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".csv")
    outfile_risk_csv = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_risk_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".csv")
    outfile_inf_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".html")
    outfile_risk_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_risk_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".html")
    df_selective_inference.to_csv(outfile_inf_csv, index=False)
    df_risk.to_csv(outfile_risk_csv, index=False)
    df_selective_inference.to_html(outfile_inf_html)
    df_risk.to_html(outfile_risk_html)

output_file(outpath='/Users/psnigdha/adjusted_MLE/n_300_p_100/')





















