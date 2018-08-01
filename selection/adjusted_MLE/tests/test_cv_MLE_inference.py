import numpy as np, sys

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from scipy.stats import norm as ndist
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.algorithms.lasso import lasso_full

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    library(bestsubset)
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

