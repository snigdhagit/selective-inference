from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np

def BH_simes(p_simes, level):

    m = p_simes.shape[0]
    p_sorted = np.sort(p_simes)
    indices = np.arange(m)
    indices_order = np.argsort(p_simes)
    nulls = np.argwhere(indices_order< 1000)

    if np.any(p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= np.zeros(m)):

        order_sig = np.max(indices[p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= 0])
        nulls_sig = nulls[nulls<=order_sig]

        return order_sig, nulls_sig.shape[0]

    else:

        return 0.

def simes_selection(X, y, noise_level =1., randomizer= 'gaussian', randomization_scale = 1.):

    n, p = X.shape

    if randomizer == 'gaussian':
        perturb = np.random.standard_normal(p)

    elif randomizer == "None":
        perturb = np.zeros(p)

    sigma = noise_level
    T_stats = X.T.dot(y) / sigma

    randomized_T_stats = T_stats + randomization_scale * perturb

    p_val_randomized = np.sort(2 * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(2.)))))

    simes_p_randomized = np.min((p / (np.arange(p) + 1.)) * p_val_randomized)

    return simes_p_randomized

# class simes_selection_egenes():
#
#     def __init__(self,
#                  X,
#                  y,
#                  randomizer= 'gaussian',
#                  noise_level = 1.,
#                  randomization_scale=1.):
#
#         self.X = X
#         self.y = y
#         self.n, self.p = self.X.shape
#         self.sigma = noise_level
#         self.T_stats = self.X.T.dot(self.y) / self.sigma
#
#         if randomizer == 'gaussian':
#             perturb = np.random.standard_normal(self.p)
#             self.randomized_T_stats = self.T_stats + randomization_scale * perturb
#             self.p_val_randomized = np.sort(
#                 2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(2.)))))
#
#             self.indices_order = np.argsort(
#                 2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(2.)))))
#
#         elif randomizer == 'none':
#             perturb = np.zeros(self.p)
#             self.randomized_T_stats = self.T_stats + randomization_scale * perturb
#
#             self.p_val_randomized = np.sort(
#                 2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(1.)))))
#
#             self.indices_order = np.argsort(
#                 2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(1.)))))
#
#
#     def simes_p_value(self):
#
#         simes_p_randomized = np.min((self.p / (np.arange(self.p) + 1.)) * self.p_val_randomized)
#
#         return simes_p_randomized
#
#     def post_BH_selection(self, level):
#
#         indices = np.arange(self.p)
#
#         significant = indices[self.p_val_randomized - (((indices + 1.)/(self.p))*level)<= 0.]
#
#         i_0 = np.amin(significant)
#
#         t_0 = self.indices_order[i_0]
#
#         T_stats_active = self.T_stats[i_0]
#
#         if i_0 > 0:
#             J = self.indices_order[:i_0]
#
#         else:
#             J = -1 * np.ones(1)
#
#         return i_0, J, t_0, np.sign(T_stats_active)


def BH_selection_egenes(p_simes, level):

    m = p_simes.shape[0]
    p_sorted = np.sort(p_simes)
    indices = np.arange(m)
    indices_order = np.argsort(p_simes)

    #if np.any(p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= np.zeros(m)):

    order_sig = np.max(indices[p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= 0])
    E_sel = indices_order[:(order_sig+1)]

    return order_sig+1, E_sel


class simes_selection_egenes():

    def __init__(self,
                 X,
                 y,
                 randomizer= 'gaussian',
                 noise_level = 1.,
                 randomization_scale=1.):

        self.X = X
        self.y = y
        self.n, self.p = self.X.shape
        self.sigma = noise_level
        self.T_stats = self.X.T.dot(self.y) / self.sigma

        if randomizer == 'gaussian':
            perturb = np.random.standard_normal(self.p)
            self.randomized_T_stats = self.T_stats + randomization_scale * perturb
            self.p_val_randomized = np.sort(
                2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(2.)))))

            self.indices_order = np.argsort(
                2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(2.)))))

        elif randomizer == 'none':
            perturb = np.zeros(self.p)
            self.randomized_T_stats = self.T_stats + randomization_scale * perturb

            self.p_val_randomized = np.sort(
                2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(1.)))))

            self.indices_order = np.argsort(
                2 * (1. - normal.cdf(np.true_divide(np.abs(self.randomized_T_stats), np.sqrt(1.)))))


    def simes_p_value(self):

        simes_p_randomized = np.min((self.p / (np.arange(self.p) + 1.)) * self.p_val_randomized)

        return simes_p_randomized

    def post_BH_selection(self, level):

        i_0 = np.argmin((self.p / (np.arange(self.p) + 1.)) * self.p_val_randomized)

        print("index Simes", i_0)

        t_0 = self.indices_order[i_0]

        T_stats_active = self.T_stats[i_0]

        u_1 = ((i_0+1.)/self.p)* np.min(np.delete((self.p / (np.arange(self.p) + 1.)) * self.p_val_randomized, i_0))

        u_2 = self.p_val_randomized[i_0+1]

        print("u_1, u_2", u_1, u_2)

        lower_threshold = np.sqrt(2.) * normal.ppf(1.-min(u_1, u_2, level*((i_0+1.)/self.p))/2.)

        print("lower threshold", lower_threshold)

        if i_0 >0:
            upper_threshold = np.sqrt(2.) * normal.ppf(1.-self.p_val_randomized[i_0-1]/2.)

        else:
            upper_threshold = 10 ** 10

        print("upper threshold", upper_threshold)

        return t_0, np.sign(T_stats_active), lower_threshold, upper_threshold