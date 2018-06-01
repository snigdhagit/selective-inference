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
                return(list(estimate = estimate, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)
    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    lam_min = np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.min'))
    lam_1se = np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.1se'))
    return estimate, lam_min, lam_1se

def randomized_inference(randomizer_scale = np.sqrt(0.25), target = "selected", full_dispersion = True):

    X = np.load("/Users/snigdhapanigrahi/Documents/Research/Effect_modification/predictors.npy")
    y = np.load("/Users/snigdhapanigrahi/Documents/Research/Effect_modification/response.npy")

    n, p = X.shape
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

    glm_LASSO_est, lam_min, lam_1se = glmnet_lasso(X, y, lam_theory/float(n))

    sys.stderr.write("lam 1se " + str(lam_1se) + "\n" + "\n")
    sys.stderr.write("lam min " + str(lam_min) + "\n" + "\n")

    active_LASSO = (glm_LASSO_est != 0)
    active_set = np.asarray([z for z in range(p) if active_LASSO[z]])

    randomized_lasso = highdim.gaussian(X,
                                        y,
                                        n* lam_min * np.ones(p),
                                        randomizer_scale= np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso.fit(solve_args={'tol': 1.e-5, 'min_its': 100})
    nonzero = signs != 0
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])

    sys.stderr.write("number of active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")
    sys.stderr.write("active variables selected by non-randomized LASSO " + str(active_set) + "\n" + "\n")
    sys.stderr.write("active variables selected by randomized LASSO " + str(active_set_rand) + "\n" + "\n")


    estimate, _, _, sel_pval, sel_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(target=target,
                                                                                                     dispersion=dispersion)
    sys.stderr.write("intervals based on MLE " + str(sel_intervals) + "\n" + "\n")
    sys.stderr.write("selective MLE " + str(estimate) + "\n" + "\n")
    sys.stderr.write("pvals based on MLE " + str(sel_pval) + "\n" + "\n")

    _, pval, intervals = randomized_lasso.summary(target=target, dispersion=sigma_, compute_intervals=True)
    sys.stderr.write("intervals based on sampler " + str(intervals) + "\n" + "\n")
    sys.stderr.write("pvals based on sampler " + str(pval) + "\n" + "\n")

randomized_inference()
