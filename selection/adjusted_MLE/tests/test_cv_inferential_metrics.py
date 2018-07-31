import numpy as np, sys

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

def relaxed_lasso(X, y):
    robjects.r('''
                library(relaxo)
                relaxed_LASSO = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)
                fit.cv = cvrelaxo(X,y)
                return(list(lam = fit.cv$lambda, alpha = fit.cv$phi))
                }''')

    lambda_R = robjects.globalenv['relaxed_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    lam = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam')))
    alpha = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('alpha')))
    return lam, alpha

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

def tuned_lasso(X, y, X_val, y_val):
    robjects.r('''
        source('~/best-subset/bestsubset/R/lasso.R')
        tuned_lasso_estimator = function(X,Y,X.val,Y.val){
        Y = as.matrix(Y)
        X = as.matrix(X)
        Y.val = as.vector(Y.val)
        X.val = as.matrix(X.val)
        rel.LASSO = lasso(X,Y,intercept=TRUE, nrelax=10, nlam=50, standardize=TRUE)
        LASSO = lasso(X,Y,intercept=TRUE,nlam=50, standardize=TRUE)
        beta.hat.rellasso = as.matrix(coef(rel.LASSO))
        beta.hat.lasso = as.matrix(coef(LASSO))
        min.lam = min(rel.LASSO$lambda)
        max.lam = max(rel.LASSO$lambda)

        lam.seq = exp(seq(log(max.lam),log(min.lam),length=rel.LASSO$nlambda))

        muhat.val.rellasso = as.matrix(predict(rel.LASSO, X.val))
        muhat.val.lasso = as.matrix(predict(LASSO, X.val))
        err.val.rellasso = colMeans((muhat.val.rellasso - Y.val)^2)
        err.val.lasso = colMeans((muhat.val.lasso - Y.val)^2)

        opt_lam = ceiling(which.min(err.val.rellasso)/10)
        lambda.tuned.rellasso = lam.seq[opt_lam]
        lambda.tuned.lasso = lam.seq[which.min(err.val.lasso)]
        fit = glmnet(X, Y, standardize=TRUE, intercept=TRUE)
        estimate.tuned = coef(fit, s=lambda.tuned.lasso, exact=TRUE, x=X, y=Y)[-1]
        beta.hat.lasso = (beta.hat.lasso[,which.min(err.val.lasso)])[-1]
        return(list(beta.hat.rellasso = (beta.hat.rellasso[,which.min(err.val.rellasso)])[-1],
        beta.hat.lasso = beta.hat.lasso,
        lambda.tuned.rellasso = lambda.tuned.rellasso, lambda.tuned.lasso= lambda.tuned.lasso,
        lambda.seq = lam.seq))
        }''')

    r_lasso = robjects.globalenv['tuned_lasso_estimator']

    n, p = X.shape
    nval, _ = X_val.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_X_val = robjects.r.matrix(X_val, nrow=nval, ncol=p)
    r_y_val = robjects.r.matrix(y_val, nrow=nval, ncol=1)

    tuned_est = r_lasso(r_X, r_y, r_X_val, r_y_val)
    estimator_rellasso = np.array(tuned_est.rx2('beta.hat.rellasso'))
    estimator_lasso = np.array(tuned_est.rx2('beta.hat.lasso'))
    lam_tuned_rellasso = np.asscalar(np.array(tuned_est.rx2('lambda.tuned.rellasso')))
    lam_tuned_lasso = np.asscalar(np.array(tuned_est.rx2('lambda.tuned.lasso')))
    lam_seq = np.array(tuned_est.rx2('lambda.seq'))
    return estimator_rellasso, estimator_lasso, lam_tuned_rellasso, lam_tuned_lasso, lam_seq


def relative_risk(est, truth, Sigma):
    return (est - truth).T.dot(Sigma).dot(est - truth) / truth.T.dot(Sigma).dot(truth)

def coverage(intervals, pval, truth):
    if (truth!=0).sum()!=0:
        avg_power = np.mean(pval[truth != 0])
    else:
        avg_power = 0.
    return np.mean((truth > intervals[:, 0])*(truth < intervals[:, 1])), avg_power

def comparison_risk_inference_selected_cv(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                                          randomizer_scale=np.sqrt(0.25), target = "selected",
                                          full_dispersion = True):

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

        true_mean = X.dot(beta)

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

        glm_LASSO, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y)
        active_LASSO = (glm_LASSO_min != 0)
        #active_LASSO = (glm_LASSO != 0)
        nactive_LASSO = active_LASSO.sum()

        randomized_lasso = highdim.gaussian(X,
                                            y,
                                            n * lam_min * np.ones(p),
                                            randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

        signs = randomized_lasso.fit()
        nonzero = signs != 0

        lasso_Liu = lasso_full.gaussian(X, y, n * lam_min)
        Lasso_soln_Liu = lasso_Liu.fit()
        active_set_Liu = np.nonzero(Lasso_soln_Liu != 0)[0]
        nactive_Liu = active_set_Liu.shape[0]

        sys.stderr.write("active variables selected by cv LASSO  " + str(nactive_LASSO) + "\n")
        sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

        if nactive_LASSO>0 and nonzero.sum()>0 and nactive_Liu>0:
            beta_target_rand = np.linalg.pinv(X[:, nonzero]).dot(true_mean)
            beta_target_nonrand = np.linalg.pinv(X[:, active_LASSO]).dot(true_mean)
            beta_target_Liu = beta[active_set_Liu]

            est_LASSO = glm_LASSO_min
            rel_LASSO = np.zeros(p)
            rel_LASSO[active_LASSO] = np.linalg.pinv(X[:, active_LASSO]).dot(y)
            Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO_min, n * lam_min, sigma_, Type=0, alpha=0.1)
            #Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_min, sigma_, Type=0, alpha=0.1)

            df = lasso_Liu.summary(level=0.90, compute_intervals=True, dispersion=dispersion)
            Liu_lower, Liu_upper, Liu_pval = np.asarray(df['lower_confidence']), \
                                             np.asarray(df['upper_confidence']), \
                                             np.asarray(df['pval'])
            Liu_intervals = np.vstack((Liu_lower, Liu_upper)).T

            if (Lee_pval.shape[0] == beta_target_nonrand.shape[0]):
                sel_MLE = np.zeros(p)
                estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(target=target,
                                                                                                                 dispersion=dispersion)
                sel_MLE[nonzero] = estimate
                ind_estimator = np.zeros(p)
                ind_estimator[nonzero] = ind_unbiased_estimator

                post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)
                unad_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
                unad_intervals = np.vstack([post_LASSO_OLS - 1.65 * unad_sd,
                                            post_LASSO_OLS + 1.65 * unad_sd]).T
                unad_pval = ndist.cdf(post_LASSO_OLS / unad_sd)

                true_signals = np.zeros(p, np.bool)
                true_signals[beta != 0] = 1
                true_set = np.asarray([u for u in range(p) if true_signals[u]])
                active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
                active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])

                active_rand_bool = np.zeros(nonzero.sum(), np.bool)
                for x in range(nonzero.sum()):
                    active_rand_bool[x] = (np.in1d(active_set_rand[x], true_set).sum() > 0)
                active_LASSO_bool = np.zeros(nactive_LASSO, np.bool)
                for z in range(nactive_LASSO):
                    active_LASSO_bool[z] = (np.in1d(active_set_LASSO[z], true_set).sum() > 0)
                active_Liu_bool = np.zeros(nactive_Liu, np.bool)
                for a in range(nactive_Liu):
                    active_Liu_bool[a] = (np.in1d(active_set_Liu[a], true_set).sum() > 0)

                cov_sel, _ = coverage(sel_intervals, sel_pval, beta_target_rand)
                cov_Lee, _ = coverage(Lee_intervals, Lee_pval, beta_target_nonrand)
                inf_entries = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
                if inf_entries.sum() == nactive_LASSO:
                    length_Lee = 0.
                else:
                    length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries])
                cov_Liu, _ = coverage(Liu_intervals, Liu_pval, beta_target_Liu)
                cov_unad, _ = coverage(unad_intervals, unad_pval, beta_target_nonrand)

                power_sel = (
                (active_rand_bool) * (np.logical_or((0. < sel_intervals[:, 0]), (0. > sel_intervals[:, 1])))).sum()
                power_Lee = (
                (active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]), (0. > Lee_intervals[:, 1])))).sum()
                power_Liu = ((active_Liu_bool) * (np.logical_or((0. < Liu_intervals[:, 0]),
                                                                (0. > Liu_intervals[:, 1])))).sum()
                power_unad = (
                (active_LASSO_bool) * (np.logical_or((0. < unad_intervals[:, 0]), (0. > unad_intervals[:, 1])))).sum()

                sel_discoveries = BHfilter(sel_pval, q=0.1)
                Lee_discoveries = BHfilter(Lee_pval, q=0.1)
                Liu_discoveries = BHfilter(Liu_pval, q=0.1)
                unad_discoveries = BHfilter(unad_pval, q=0.1)

                power_sel_dis = (sel_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
                power_Lee_dis = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
                power_Liu_dis = (Liu_discoveries * active_Liu_bool).sum() / float((beta != 0).sum())
                power_unad_dis = (unad_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())

                fdr_sel_dis = (sel_discoveries * ~active_rand_bool).sum() / float(max(sel_discoveries.sum(), 1.))
                fdr_Lee_dis = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))
                fdr_Liu_dis = (Liu_discoveries * ~active_Liu_bool).sum() / float(max(Liu_discoveries.sum(), 1.))
                fdr_unad_dis = (unad_discoveries * ~active_LASSO_bool).sum() / float(max(unad_discoveries.sum(), 1.))
                break

    if True:
        return np.vstack((relative_risk(sel_MLE, beta, Sigma),
                          relative_risk(ind_estimator, beta, Sigma),
                          relative_risk(randomized_lasso.initial_soln , beta, Sigma),
                          relative_risk(randomized_lasso._beta_full, beta, Sigma),
                          relative_risk(rel_LASSO, beta, Sigma),
                          relative_risk(est_LASSO, beta, Sigma),
                          cov_sel,
                          cov_Lee,
                          cov_Liu,
                          cov_unad,
                          np.mean(sel_intervals[:, 1] - sel_intervals[:, 0]),
                          length_Lee,
                          np.mean(Liu_intervals[:, 1] - Liu_intervals[:, 0]),
                          np.mean(unad_intervals[:, 1] - unad_intervals[:, 0]),
                          power_sel/float((beta != 0).sum()),
                          power_Lee/float((beta != 0).sum()),
                          power_Liu / float((beta != 0).sum()),
                          power_unad/float((beta != 0).sum()),
                          power_sel_dis,
                          power_Lee_dis,
                          power_Liu_dis,
                          power_unad_dis,
                          fdr_sel_dis,
                          fdr_Lee_dis,
                          fdr_Liu_dis,
                          fdr_unad_dis,
                          nonzero.sum(),
                          nactive_LASSO,
                          nactive_LASSO,
                          sel_discoveries.sum(),
                          Lee_discoveries.sum(),
                          Liu_discoveries.sum(),
                          unad_discoveries.sum(),
                          inf_entries.sum()/float(nactive_LASSO),
                          np.mean(estimate - beta_target_rand)))

def comparison_risk_inference_full_cv(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                                      randomizer_scale=np.sqrt(0.25), target = "full", full_dispersion = True):

    while True:
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        true_mean = X.dot(beta)

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

        glm_LASSO, _, lam_min, lam_1se = glmnet_lasso(X, y)
        active_LASSO = (glm_LASSO != 0)
        nactive_LASSO = active_LASSO.sum()

        randomized_lasso = highdim.gaussian(X,
                                            y,
                                            n * lam_1se * np.ones(p),
                                            randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

        signs = randomized_lasso.fit()
        nonzero = signs != 0

        lasso_Liu = lasso_full.gaussian(X, y, n * lam_1se)
        Lasso_soln_Liu = lasso_Liu.fit()
        active_set_Liu = np.nonzero(Lasso_soln_Liu != 0)[0]
        nactive_Liu = active_set_Liu.shape[0]

        sys.stderr.write("active variables selected by cv LASSO  " + str(nactive_LASSO) + "\n")
        sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

        if nonzero.sum() > 0 and nonzero.sum() < 50 and nactive_LASSO>0 and nactive_Liu>0:
            est_LASSO = glm_LASSO
            rel_LASSO = np.zeros(p)
            rel_LASSO[active_LASSO] = np.linalg.pinv(X[:, active_LASSO]).dot(y)
            active_nonrand = (est_LASSO != 0)
            nactive_nonrand = active_nonrand.sum()

            beta_target_rand = beta[nonzero]
            beta_target_nonrand_py = beta[active_LASSO]
            beta_target_nonrand = beta[active_nonrand]
            beta_target_Liu = beta[active_set_Liu]

            Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_1se, sigma_, Type=1, alpha=0.1)
            df = lasso_Liu.summary(level=0.90, compute_intervals=True, dispersion=dispersion)
            Liu_lower, Liu_upper, Liu_pval = np.asarray(df['lower_confidence']), \
                                             np.asarray(df['upper_confidence']), \
                                             np.asarray(df['pval'])
            Liu_intervals = np.vstack((Liu_lower, Liu_upper)).T

            if (Lee_pval.shape[0] == beta_target_nonrand_py.shape[0]):
                sel_MLE = np.zeros(p)
                estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(
                    target=target,
                    dispersion=dispersion)
                sel_MLE[nonzero] = estimate
                ind_estimator = np.zeros(p)
                ind_estimator[nonzero] = ind_unbiased_estimator

                if Lee_pval.shape[0] != beta_target_nonrand_py.shape[0]:
                    break

                post_LASSO_OLS = np.linalg.pinv(X[:, active_nonrand]).dot(y)
                unad_sd = sigma_ * np.sqrt(
                    np.diag((np.linalg.inv(X[:, active_nonrand].T.dot(X[:, active_nonrand])))))

                unad_intervals = np.vstack([post_LASSO_OLS - 1.65 * unad_sd,
                                            post_LASSO_OLS + 1.65 * unad_sd]).T
                unad_pval = ndist.cdf(post_LASSO_OLS / unad_sd)

                true_signals = np.zeros(p, np.bool)
                true_signals[beta != 0] = 1
                true_set = np.asarray([u for u in range(p) if true_signals[u]])
                active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
                active_set_nonrand = np.asarray([q for q in range(p) if active_nonrand[q]])
                active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])

                active_rand_bool = np.zeros(nonzero.sum(), np.bool)
                for x in range(nonzero.sum()):
                    active_rand_bool[x] = (np.in1d(active_set_rand[x], true_set).sum() > 0)
                active_nonrand_bool = np.zeros(nactive_nonrand, np.bool)
                for w in range(nactive_nonrand):
                    active_nonrand_bool[w] = (np.in1d(active_set_nonrand[w], true_set).sum() > 0)
                active_LASSO_bool = np.zeros(nactive_LASSO, np.bool)
                for z in range(nactive_LASSO):
                    active_LASSO_bool[z] = (np.in1d(active_set_LASSO[z], true_set).sum() > 0)
                active_Liu_bool = np.zeros(nactive_Liu, np.bool)
                for a in range(nactive_Liu):
                    active_Liu_bool[a] = (np.in1d(active_set_Liu[a], true_set).sum() > 0)

                cov_sel, _ = coverage(sel_intervals, sel_pval, beta_target_rand)
                cov_Lee, _ = coverage(Lee_intervals, Lee_pval, beta_target_nonrand_py)
                cov_Liu, _ = coverage(Liu_intervals, Liu_pval, beta_target_Liu)

                inf_entries = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
                if inf_entries.sum() == nactive_LASSO:
                    length_Lee = 0.
                else:
                    length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries])
                cov_unad, _ = coverage(unad_intervals, unad_pval, beta_target_nonrand)

                power_sel = ((active_rand_bool) * (np.logical_or((0. < sel_intervals[:, 0]),
                                                                 (0. > sel_intervals[:, 1])))).sum()
                power_Lee = ((active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]),
                                                                  (0. > Lee_intervals[:, 1])))).sum()
                power_Liu = ((active_Liu_bool) * (np.logical_or((0. < Liu_intervals[:, 0]),
                                                                (0. > Liu_intervals[:, 1])))).sum()
                power_unad = ((active_nonrand_bool) * (np.logical_or((0. < unad_intervals[:, 0]),
                                                                     (0. > unad_intervals[:, 1])))).sum()

                sel_discoveries = BHfilter(sel_pval, q=0.1)
                Lee_discoveries = BHfilter(Lee_pval, q=0.1)
                Liu_discoveries = BHfilter(Liu_pval, q=0.1)
                unad_discoveries = BHfilter(unad_pval, q=0.1)

                power_sel_dis = (sel_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
                power_Lee_dis = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
                power_Liu_dis = (Liu_discoveries * active_Liu_bool).sum() / float((beta != 0).sum())
                power_unad_dis = (unad_discoveries * active_nonrand_bool).sum() / float((beta != 0).sum())

                fdr_sel_dis = (sel_discoveries * ~active_rand_bool).sum() / float(max(sel_discoveries.sum(), 1.))
                fdr_Lee_dis = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))
                fdr_Liu_dis = (Liu_discoveries * ~active_Liu_bool).sum() / float(max(Liu_discoveries.sum(), 1.))
                fdr_unad_dis = (unad_discoveries * ~active_nonrand_bool).sum() / float(
                    max(unad_discoveries.sum(), 1.))

                break

    if True:
        return np.vstack((relative_risk(sel_MLE, beta, Sigma),
                          relative_risk(ind_estimator, beta, Sigma),
                          relative_risk(randomized_lasso.initial_soln, beta, Sigma),
                          relative_risk(randomized_lasso._beta_full, beta, Sigma),
                          relative_risk(rel_LASSO, beta, Sigma),
                          relative_risk(est_LASSO, beta, Sigma),
                          cov_sel,
                          cov_Lee,
                          cov_Liu,
                          cov_unad,
                          np.mean(sel_intervals[:, 1] - sel_intervals[:, 0]),
                          length_Lee,
                          np.mean(Liu_intervals[:, 1] - Liu_intervals[:, 0]),
                          np.mean(unad_intervals[:, 1] - unad_intervals[:, 0]),
                          power_sel / float((beta != 0).sum()),
                          power_Lee / float((beta != 0).sum()),
                          power_Liu / float((beta != 0).sum()),
                          power_unad / float((beta != 0).sum()),
                          power_sel_dis,
                          power_Lee_dis,
                          power_Liu_dis,
                          power_unad_dis,
                          fdr_sel_dis,
                          fdr_Lee_dis,
                          fdr_Liu_dis,
                          fdr_unad_dis,
                          nonzero.sum(),
                          nactive_LASSO,
                          nactive_nonrand,
                          sel_discoveries.sum(),
                          Lee_discoveries.sum(),
                          Liu_discoveries.sum(),
                          unad_discoveries.sum(),
                          inf_entries.sum() / float(nactive_LASSO),
                          np.mean(estimate - beta_target_rand)))

def risk_comparison_cv(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                       randomizer_scale=np.sqrt(0.25), target = "selected", full_dispersion = True):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
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

    glm_LASSO, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y)
    active_LASSO = (glm_LASSO != 0)
    nactive_LASSO = active_LASSO.sum()
    active_LASSO_min = (glm_LASSO_min != 0)
    nactive_LASSO_min = active_LASSO_min.sum()

    est_LASSO = glm_LASSO
    rel_LASSO = np.zeros(p)
    est_LASSO_min = glm_LASSO_min
    rel_LASSO_min = np.zeros(p)

    if nactive_LASSO > 0:
        rel_LASSO[active_LASSO] = np.linalg.pinv(X[:, active_LASSO]).dot(y)
    if nactive_LASSO_min > 0:
        rel_LASSO_min[active_LASSO_min] = np.linalg.pinv(X[:, active_LASSO_min]).dot(y)

    # lam, alpha = relaxed_lasso(X, y)

    # sys.stderr.write("parameters from relaxed lasso: " + str(lam) + str(alpha) + "\n")
    # sys.stderr.write("parameters from cv glmnet: " + str(n * lam_1se) + "\n" + "\n")

    randomized_lasso = highdim.gaussian(X,
                                        y,
                                        n * lam_1se * np.ones(p),
                                        randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    sel_MLE = np.zeros(p)
    ind_estimator = np.zeros(p)

    randomized_lasso_min = highdim.gaussian(X,
                                            y,
                                            n * lam_min * np.ones(p),
                                            randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs_min = randomized_lasso_min.fit()
    nonzero_min = signs_min != 0

    sel_MLE_min = np.zeros(p)
    ind_estimator_min = np.zeros(p)

    sys.stderr.write("active variables selected by cv LASSO lam.1se " + str(nactive_LASSO) + "\n")
    sys.stderr.write("active variables selected by cv LASSO lam.min " + str(nactive_LASSO_min) + "\n")
    sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")

    if nonzero.sum() > 0 and nonzero.sum() < 50:
        estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(
            target=target,
            dispersion=dispersion)
        sel_MLE[nonzero] = estimate
        ind_estimator[nonzero] = ind_unbiased_estimator

    if nonzero_min.sum() > 0 and nonzero_min.sum() < 50:
        estimate_min, _, _, sel_pval_min, sel_intervals_min, ind_unbiased_estimator_min = randomized_lasso_min.selective_MLE(
            target=target,
            dispersion=dispersion)
        sel_MLE_min[nonzero_min] = estimate_min
        ind_estimator_min[nonzero_min] = ind_unbiased_estimator_min

    return np.vstack((relative_risk(sel_MLE, beta, Sigma),
                      relative_risk(ind_estimator, beta, Sigma),
                      relative_risk(randomized_lasso.initial_soln, beta, Sigma),
                      relative_risk(randomized_lasso._beta_full, beta, Sigma),
                      relative_risk(rel_LASSO, beta, Sigma),
                      relative_risk(est_LASSO, beta, Sigma),
                      relative_risk(sel_MLE_min, beta, Sigma),
                      relative_risk(ind_estimator_min, beta, Sigma),
                      relative_risk(randomized_lasso_min.initial_soln, beta, Sigma),
                      relative_risk(randomized_lasso_min._beta_full, beta, Sigma),
                      relative_risk(rel_LASSO_min, beta, Sigma),
                      relative_risk(est_LASSO_min, beta, Sigma)))

def Lee_selected_high(n=200, p=500, nval=200, rho=0.35, s=5, beta_type=2,
                      snr=0.2, randomizer_scale=0.5, target = "full",
                      tuning = "selective_MLE", full_dispersion = True):

    while True:
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
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

        glm_LASSO, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y)
        active_LASSO = (glm_LASSO != 0)
        nactive_LASSO = active_LASSO.sum()
        active_nonrand = active_LASSO
        nactive_nonrand = nactive_LASSO

        if nactive_LASSO > 0:
            beta_target_nonrand_py = np.linalg.pinv(X[:, active_LASSO]).dot(X.dot(beta))
            Lee_intervals, Lee_pval = selInf_R(X, y, glm_LASSO, n * lam_1se, sigma_, Type=0, alpha=0.1)

            if (Lee_pval.shape[0] == beta_target_nonrand_py.shape[0]):

                post_LASSO_OLS = np.linalg.pinv(X[:, active_nonrand]).dot(y)
                unad_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_nonrand].T.dot(X[:, active_nonrand])))))

                unad_intervals = np.vstack([post_LASSO_OLS - 1.65 * unad_sd,
                                            post_LASSO_OLS + 1.65 * unad_sd]).T
                unad_pval = ndist.cdf(post_LASSO_OLS / unad_sd)

                true_signals = np.zeros(p, np.bool)
                true_signals[beta != 0] = 1
                true_set = np.asarray([u for u in range(p) if true_signals[u]])
                active_set_LASSO = np.asarray([r for r in range(p) if active_LASSO[r]])
                active_set_nonrand = np.asarray([q for q in range(p) if active_nonrand[q]])

                active_LASSO_bool = np.zeros(nactive_LASSO, np.bool)
                for z in range(nactive_LASSO):
                    active_LASSO_bool[z] = (np.in1d(active_set_LASSO[z], true_set).sum() > 0)

                active_nonrand_bool = np.zeros(nactive_nonrand, np.bool)
                for w in range(nactive_nonrand):
                    active_nonrand_bool[w] = (np.in1d(active_set_nonrand[w], true_set).sum() > 0)

                cov_Lee, _ = coverage(Lee_intervals, Lee_pval, beta_target_nonrand_py)
                inf_entries = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
                if inf_entries.sum() == nactive_LASSO:
                    length_Lee = 0.
                else:
                    length_Lee = np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries])
                cov_unad, _ = coverage(unad_intervals, unad_pval, beta_target_nonrand_py)

                power_Lee = ((active_LASSO_bool) * (np.logical_or((0. < Lee_intervals[:, 0]),
                                                                  (0. > Lee_intervals[:, 1])))).sum()

                power_unad = ((active_nonrand_bool) * (np.logical_or((0. < unad_intervals[:, 0]),
                                                                     (0. > unad_intervals[:, 1])))).sum()

                Lee_discoveries = BHfilter(Lee_pval, q=0.1)
                unad_discoveries = BHfilter(unad_pval, q=0.1)

                power_Lee_dis = (Lee_discoveries * active_LASSO_bool).sum() / float((beta != 0).sum())
                power_unad_dis = (unad_discoveries * active_nonrand_bool).sum() / float((beta != 0).sum())

                fdr_Lee_dis = (Lee_discoveries * ~active_LASSO_bool).sum() / float(max(Lee_discoveries.sum(), 1.))
                fdr_unad_dis = (unad_discoveries * ~active_nonrand_bool).sum() / float(max(unad_discoveries.sum(), 1.))
                break

    if True:
        return np.vstack((cov_Lee,
                          cov_unad,
                          length_Lee,
                          np.mean(unad_intervals[:, 1] - unad_intervals[:, 0]),
                          power_Lee / float((beta != 0).sum()),
                          power_unad / float((beta != 0).sum()),
                          power_Lee_dis,
                          power_unad_dis,
                          fdr_Lee_dis,
                          fdr_unad_dis,
                          nactive_LASSO,
                          nactive_nonrand,
                          Lee_discoveries.sum(),
                          unad_discoveries.sum(),
                          inf_entries.sum() / float(nactive_LASSO)))

# if __name__ == "__main__":
#
#     ndraw = 50
#
#     target = "selected"
#     n, p, rho, s, beta_type, snr = 3000, 1000, 0.35, 30, 1, 0.10
#
#     if target == "selected":
#         output_overall = np.zeros(35)
#         for i in range(ndraw):
#             if n > p:
#                 full_dispersion = True
#             else:
#                 full_dispersion = False
#
#             output = comparison_risk_inference_selected_cv(n=n, p=p, nval=n, rho=rho, s=s,
#                                                            beta_type=beta_type, snr=snr,
#                                                            randomizer_scale=np.sqrt(0.5), target=target,
#                                                            full_dispersion=full_dispersion)
#             output_overall += np.squeeze(output)
#
#             sys.stderr.write("overall selMLE risk " + str(output_overall[0] / float(i + 1)) + "\n")
#             sys.stderr.write("overall indep est risk " + str(output_overall[1] / float(i + 1)) + "\n")
#             sys.stderr.write("overall randomized LASSO est risk " + str(output_overall[2] / float(i + 1)) + "\n")
#             sys.stderr.write(
#                 "overall relaxed rand LASSO est risk " + str(output_overall[3] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall relLASSO risk " + str(output_overall[4] / float(i + 1)) + "\n")
#             sys.stderr.write("overall LASSO risk " + str(output_overall[5] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective coverage " + str(output_overall[6] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee coverage " + str(output_overall[7] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu coverage " + str(output_overall[8] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad coverage " + str(output_overall[9] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective length " + str(output_overall[10] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee length " + str(output_overall[11] / float(i + 1)) + "\n")
#             sys.stderr.write(
#                 "proportion of Lee intervals that are infty " + str(output_overall[33] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu length " + str(output_overall[12] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad length " + str(output_overall[13] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective power " + str(output_overall[14] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee power " + str(output_overall[15] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu power " + str(output_overall[16] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad power " + str(output_overall[17] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective fdr " + str(output_overall[22] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee fdr " + str(output_overall[23] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu fdr " + str(output_overall[24] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad fdr " + str(output_overall[25] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective power post BH " + str(output_overall[18] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee power post BH  " + str(output_overall[19] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu power post BH  " + str(output_overall[20] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad power post BH " + str(output_overall[21] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("average selective nactive " + str(output_overall[26] / float(i + 1)) + "\n")
#             sys.stderr.write("average Lee nactive  " + str(output_overall[27] / float(i + 1)) + "\n")
#             sys.stderr.write("average tuned LASSO nactive " + str(output_overall[28] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("average selective discoveries " + str(output_overall[29] / float(i + 1)) + "\n")
#             sys.stderr.write("average Lee discoveries " + str(output_overall[30] / float(i + 1)) + "\n")
#             sys.stderr.write("average Liu discoveries " + str(output_overall[31] / float(i + 1)) + "\n")
#             sys.stderr.write("average tuned LASSO discoveries " + str(output_overall[32] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("average bias " + str(output_overall[34] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("iteration completed " + str(i + 1) + "\n")
#
#     elif target == "full":
#         output_overall = np.zeros(35)
#         if n > p:
#             full_dispersion = True
#         else:
#             full_dispersion = False
#         for i in range(ndraw):
#             output = comparison_risk_inference_full_cv(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
#                                                       randomizer_scale=np.sqrt(0.5), target=target,
#                                                       full_dispersion=full_dispersion)
#
#             output_overall += np.squeeze(output)
#
#             sys.stderr.write("overall selMLE risk " + str(output_overall[0] / float(i + 1)) + "\n")
#             sys.stderr.write("overall indep est risk " + str(output_overall[1] / float(i + 1)) + "\n")
#             sys.stderr.write("overall randomized LASSO est risk " + str(output_overall[2] / float(i + 1)) + "\n")
#             sys.stderr.write(
#                 "overall relaxed rand LASSO est risk " + str(output_overall[3] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall relLASSO risk " + str(output_overall[4] / float(i + 1)) + "\n")
#             sys.stderr.write("overall LASSO risk " + str(output_overall[5] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective coverage " + str(output_overall[6] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee coverage " + str(output_overall[7] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu coverage " + str(output_overall[8] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad coverage " + str(output_overall[9] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective length " + str(output_overall[10] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee length " + str(output_overall[11] / float(i + 1)) + "\n")
#             sys.stderr.write(
#                 "proportion of Lee intervals that are infty " + str(output_overall[33] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu length " + str(output_overall[12] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad length " + str(output_overall[13] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective power " + str(output_overall[14] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee power " + str(output_overall[15] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu power " + str(output_overall[16] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad power " + str(output_overall[17] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective fdr " + str(output_overall[22] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee fdr " + str(output_overall[23] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu fdr " + str(output_overall[24] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad fdr " + str(output_overall[25] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("overall selective power post BH " + str(output_overall[18] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Lee power post BH  " + str(output_overall[19] / float(i + 1)) + "\n")
#             sys.stderr.write("overall Liu power post BH  " + str(output_overall[20] / float(i + 1)) + "\n")
#             sys.stderr.write("overall unad power post BH " + str(output_overall[21] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("average selective nactive " + str(output_overall[26] / float(i + 1)) + "\n")
#             sys.stderr.write("average Lee nactive  " + str(output_overall[27] / float(i + 1)) + "\n")
#             sys.stderr.write("average tuned LASSO nactive " + str(output_overall[28] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("average selective discoveries " + str(output_overall[29] / float(i + 1)) + "\n")
#             sys.stderr.write("average Lee discoveries " + str(output_overall[30] / float(i + 1)) + "\n")
#             sys.stderr.write("average Liu discoveries " + str(output_overall[31] / float(i + 1)) + "\n")
#             sys.stderr.write("average tuned LASSO discoveries " + str(output_overall[32] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("average bias " + str(output_overall[34] / float(i + 1)) + "\n" + "\n")
#
#             sys.stderr.write("iteration completed " + str(i + 1) + "\n")

if __name__ == "__main__":

    ndraw = 50

    target = "selected"
    n, p, rho, s, beta_type, snr = 300, 100, 0.35, 5, 1, 0.10

    if target == "selected":
        output_overall = np.zeros(12)
        for i in range(ndraw):
            if n > p:
                full_dispersion = True
            else:
                full_dispersion = False

            output = risk_comparison_cv(n=n, p=p, nval=n, rho=rho, s=s,
                                        beta_type=beta_type, snr=snr,
                                        randomizer_scale=np.sqrt(0.5), target=target,
                                        full_dispersion=full_dispersion)

            output_overall += np.squeeze(output)

            sys.stderr.write("overall selMLE risk at lam.1se " + str(output_overall[0] / float(i + 1)) + "\n")
            sys.stderr.write("overall indep est risk at lam.1se " + str(output_overall[1] / float(i + 1)) + "\n")
            sys.stderr.write("overall randomized LASSO est risk at lam.1se " + str(output_overall[2] / float(i + 1)) + "\n")
            sys.stderr.write(
                "overall relaxed rand LASSO est risk at lam.1se " + str(output_overall[3] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall relLASSO risk at lam.1se " + str(output_overall[4] / float(i + 1)) + "\n")
            sys.stderr.write("overall LASSO risk at lam.1se " + str(output_overall[5] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall selMLE risk at lam.min " + str(output_overall[6] / float(i + 1)) + "\n")
            sys.stderr.write("overall indep est risk at lam.min " + str(output_overall[7] / float(i + 1)) + "\n")
            sys.stderr.write("overall randomized LASSO est risk at lam.min " + str(output_overall[8] / float(i + 1)) + "\n")
            sys.stderr.write(
                "overall relaxed rand LASSO est risk at lam.min " + str(output_overall[9] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("overall relLASSO risk at lam.min " + str(output_overall[10] / float(i + 1)) + "\n")
            sys.stderr.write("overall LASSO risk at lam.min " + str(output_overall[11] / float(i + 1)) + "\n" + "\n")

            sys.stderr.write("iteration completed " + str(i + 1) + "\n")

# if __name__ == "__main__":
#
#     ndraw = 20
#     output_overall = np.zeros(15)
#
#     target = "full"
#     tuning = "selective_MLE"
#     n, p, rho, s, beta_type, snr = 200, 1000, 0.35, 10, 1, 0.10
#
#     if n > p:
#         full_dispersion = True
#     else:
#         full_dispersion = False
#     for i in range(ndraw):
#         output = Lee_selected_high(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
#                                    randomizer_scale=np.sqrt(0.50), target=target, tuning=tuning,
#                                    full_dispersion=full_dispersion)
#
#         output_overall += np.squeeze(output)
#
#         sys.stderr.write("overall Lee coverage " + str(output_overall[0] / float(i + 1)) + "\n")
#         sys.stderr.write("overall unad coverage " + str(output_overall[1] / float(i + 1)) + "\n" + "\n")
#
#         sys.stderr.write("overall Lee length " + str(output_overall[2] / float(i + 1)) + "\n")
#         sys.stderr.write("overall unad length " + str(output_overall[3] / float(i + 1)) + "\n" + "\n")
#
#         sys.stderr.write("proportion of Lee intervals that are infty " + str(output_overall[14] / float(i + 1)) + "\n"+ "\n")
#
#         sys.stderr.write("overall Lee power " + str(output_overall[4] / float(i + 1)) + "\n")
#         sys.stderr.write("overall unad power " + str(output_overall[5] / float(i + 1)) + "\n" + "\n")
#
#         sys.stderr.write("overall Lee fdr " + str(output_overall[6] / float(i + 1)) + "\n")
#         sys.stderr.write("overall unad fdr " + str(output_overall[7] / float(i + 1)) + "\n" + "\n")
#
#         sys.stderr.write("overall Lee power post BH  " + str(output_overall[8] / float(i + 1)) + "\n")
#         sys.stderr.write("overall unad power post BH " + str(output_overall[9] / float(i + 1)) + "\n" + "\n")
#
#         sys.stderr.write("average Lee discoveries  " + str(output_overall[12] / float(i + 1)) + "\n")
#
#         sys.stderr.write("iteration completed " + str(i + 1) + "\n")