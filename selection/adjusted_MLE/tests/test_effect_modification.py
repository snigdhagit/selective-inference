import numpy as np, sys

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import selection.randomized.lasso as L; reload(L)
from selection.randomized.lasso import highdim
from scipy.stats import norm as ndist

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library('glmnet')
                glmnet_LASSO = function(X,y,lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)
                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.1se = estimate.1se, estimate.min = estimate.min,
                            lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)
    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    lam_min = np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.min'))
    lam_1se = np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.1se'))
    estimate_min = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.min'))
    estimate_1se = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.1se'))
    return estimate, lam_min, lam_1se, estimate_min, estimate_1se

def randomized_inference(randomizer_scale = np.sqrt(0.25), target = "full", full_dispersion = True):

    X = np.load("/Users/snigdhapanigrahi/Documents/Research/Effect_modification/predictors.npy")
    y = np.load("/Users/snigdhapanigrahi/Documents/Research/Effect_modification/response.npy")

    n, p = X.shape
    print("size of regression", n, p)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    y= y.reshape((y.shape[0], ))

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2. / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("sigma estimated", sigma_)

    lam_theory = sigma_ * 1.1 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    sys.stderr.write("lam theory " + str(lam_theory) + "\n" + "\n")
    glm_LASSO_est, lam_min, lam_1se, glm_LASSO_est_min, glm_LASSO_est_1se = glmnet_lasso(X, y, lam_theory/float(n))
    sys.stderr.write("lam 1se " + str(n* lam_1se) + "\n" + "\n")
    sys.stderr.write("lam min " + str(n* lam_min) + "\n" + "\n")

    active_LASSO = (glm_LASSO_est != 0)
    active_set = np.asarray([z for z in range(p) if active_LASSO[z]])

    ###running randomized LASSO at theoretical lambda
    randomized_lasso_theory = highdim.gaussian(X,
                                               y,
                                               lam_theory * np.ones(p),
                                               randomizer_scale= np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso_theory.fit(solve_args={'tol': 1.e-5, 'min_its': 100})
    ini_perturb = randomized_lasso_theory._initial_omega
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])

    sys.stderr.write("theoretical lambda: active variables selected by non-randomized LASSO " + str(active_set) + "\n" + "\n")
    sys.stderr.write("theoretical lambda: number of active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")
    sys.stderr.write("theoretical lambda: active variables selected by randomized LASSO " + str(active_set_rand) + "\n" + "\n")

    estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator = randomized_lasso_theory.selective_MLE(target=target,
                                                                                                            dispersion=dispersion)

    sys.stderr.write("theoretical lambda: pvals based on MLE " + str(sel_pval) + "\n" + "\n")
    sys.stderr.write("theoretical lambda: selective MLE " + str(estimate) + "\n" + "\n")
    sys.stderr.write("theoretical lambda: intervals based on MLE " + str(sel_intervals.T) + "\n" + "\n")

    _, pval, intervals = randomized_lasso_theory.summary(target=target, dispersion=sigma_, compute_intervals=True, level=0.90)
    sys.stderr.write("theoretical lambda: pvals based on sampler " + str(pval) + "\n" + "\n")
    sys.stderr.write("theoretical lambda: intervals based on sampler " + str(intervals.T) + "\n" + "\n")

    ###running randomized LASSO at lambda.1se from cv.glmnet
    randomized_lasso_cv1se = highdim.gaussian(X,
                                              y,
                                              n * lam_1se * np.ones(p),
                                              randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso_cv1se.fit(solve_args={'tol': 1.e-5, 'min_its': 100}, perturb=ini_perturb)
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])

    sys.stderr.write("1se lambda: number of active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")
    sys.stderr.write("1se lambda: active variables selected by randomized LASSO " + str(active_set_rand) + "\n" + "\n")

    estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator = randomized_lasso_cv1se.selective_MLE(target=target,
                                                                                                           dispersion=dispersion)

    sys.stderr.write("1se lambda: pvals based on MLE " + str(sel_pval) + "\n" + "\n")
    sys.stderr.write("1se lambda: selective MLE " + str(estimate) + "\n" + "\n")
    sys.stderr.write("1se lambda: intervals based on MLE " + str(sel_intervals.T) + "\n" + "\n")

    _, pval, intervals = randomized_lasso_cv1se.summary(target=target, dispersion=sigma_, compute_intervals=True,
                                                         level=0.90)
    sys.stderr.write("1se lambda: pvals based on sampler " + str(pval) + "\n" + "\n")
    sys.stderr.write("1se lambda: intervals based on sampler " + str(intervals.T) + "\n" + "\n")

    ###running randomized LASSO at lambda.min from cv.glmnet
    randomized_lasso_cvmin = highdim.gaussian(X,
                                              y,
                                              n * lam_min * np.ones(p),
                                              randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso_cvmin.fit(solve_args={'tol': 1.e-5, 'min_its': 100}, perturb=ini_perturb)
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])

    sys.stderr.write(
        "min lambda: number of active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")
    sys.stderr.write("min lambda: active variables selected by randomized LASSO " + str(active_set_rand) + "\n" + "\n")

    estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator = randomized_lasso_cvmin.selective_MLE(
        target=target,
        dispersion=dispersion)

    sys.stderr.write("min lambda: pvals based on MLE " + str(sel_pval) + "\n" + "\n")
    sys.stderr.write("min lambda: selective MLE " + str(estimate) + "\n" + "\n")
    sys.stderr.write("min lambda: intervals based on MLE " + str(sel_intervals.T) + "\n" + "\n")

    _, pval, intervals = randomized_lasso_cvmin.summary(target=target, dispersion=sigma_, compute_intervals=True,
                                                        level=0.90)
    sys.stderr.write("min lambda: pvals based on sampler " + str(pval) + "\n" + "\n")
    sys.stderr.write("min lambda: intervals based on sampler " + str(intervals.T) + "\n" + "\n")

randomized_inference(randomizer_scale = np.sqrt(0.50), target = "full")
