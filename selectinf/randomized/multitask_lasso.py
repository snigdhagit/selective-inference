import numpy as np
from scipy.linalg import block_diag
from scipy.stats import norm as ndist

import regreg.api as rr
from .randomization import randomization
from ..base import restricted_estimator

class multi_task_lasso():

     def __init__(self,
                  loglikes,
                  feature_weight,
                  ridge_terms,
                  randomizers,
                  nfeature,
                  ntask,
                  perturbations):

         self.loglikes = loglikes

         self.ntask = ntask
         self.nfeature = nfeature

         self.ridge_terms = ridge_terms
         self.feature_weight = feature_weight

         self._initial_omega = perturbations
         self.randomizers = randomizers

     def fit(self,
             perturbations=None,
             solve_args={'tol': 1.e-12, 'min_its': 50}):

         p = self.nfeature

         ## solve multitask problem

         (self.initial_solns,
          self.initial_subgrads,
          penalty_weight) = self._solve_multitask_problem(perturbations=perturbations)

         ##setting up some initial objects to form our K.K.T map which loops over the K regression tasks

         active_signs = np.zeros((p, self.ntask))
         beta_bar = np.zeros((p, self.ntask))

         _active =  np.zeros((p, self.ntask), np.bool)
         _overall = {}
         inactive = {}

         ordered_variables = {}

         initial_scalings = {}
         _beta_unpenalized = {}

         for i in range(self.ntask):

             active_signs[:, i] = np.sign(self.initial_solns[:, i])
             active_ = _active[:, i] = active_signs[:, i] != 0

             overall_ = _overall[i] = _active[:, i] > 0
             inactive[i] = ~overall_

             ordered_variables[i] = np.nonzero(active_)[0]

             initial_scalings[i] = np.fabs(self.initial_solns[:, i][overall_])

             _beta_unpenalized[i] = restricted_estimator(self.loglikes[i],
                                                         overall_,
                                                         solve_args=solve_args)

             beta_bar[overall_, i] = _beta_unpenalized[i]

         active = self._active = _active
         self._overall = _overall
         self.inactive = inactive

         self.beta_bar = beta_bar

         seltasks = np.array([active[j, :].sum() for j in range(p)])
         self.active_global = np.asarray([j for j in range(p) if seltasks[j] > 0])
         self.seltasks = seltasks[seltasks > 0]

         self.selection_variable = {'sign': active_signs.copy(),
                                    'variables': ordered_variables}

         def signed_basis_vector(p, j, s):
             v = np.zeros(p)
             v[j] = s
             return v

         num_opt_var = np.array([initial_scalings[i].shape[0] for i in range(self.ntask)])
         tot_opt_var = num_opt_var.sum()

         ###defining a transformation tying optimization variables across the K regression tasks

         pi = np.zeros(tot_opt_var)
         indx = 0
         for irow, icol in np.ndindex(active.shape):
             if active[irow, icol] > 0:
                 pi[indx] = active[:, :(icol + 1)].sum() - active[irow:, icol].sum()
                 indx += 1

         pi = pi.astype(int)

         Pi = np.take(np.identity(tot_opt_var), pi, axis=0)

         Psi = np.array([])
         Tau = np.array([])
         A_ = np.array([])

         for j in range(self.seltasks.shape[0]):
             a_ = np.zeros(self.seltasks[j])
             a_[-1] = 1
             A_ = block_diag(A_, a_)

             B_ = np.identity(self.seltasks[j])
             B_[(self.seltasks[j] - 1), :] = np.ones(self.seltasks[j])
             Psi = block_diag(Psi, B_)

             C_ = np.identity(self.seltasks[j])
             C_[(self.seltasks[j] - 1), :] = np.zeros(self.seltasks[j])
             Tau = block_diag(Tau, C_)

         Psi = Psi[1:, :]

         Tau = Tau[1:, :]
         A_ = A_[1:, :]
         Tau = np.delete(Tau, np.array(np.cumsum(self.seltasks) - 1), 0)
         Tau = np.vstack((Tau, A_))

         ## final tranformation is v = Tau.dot(Psi).dot(Pi).dot(o)

         CoV = Tau.dot(Psi).dot(Pi)

         ###setting up K.K.T. map at solution

         _score_linear_term = {}
         observed_score_state = {}
         _opt_offset = {i: self.initial_subgrads[:, i] for i in range(self.ntask)}
         
         omegas = []
         scores = []
         opt_offsets_ = []
         opt_linears_ = np.array([])
         observed_opt_states_ =[]

         for i in range(self.ntask):
             X, y = self.loglikes[i].data
             W = self.loglikes[i].saturated_loss.hessian(X.dot(beta_bar[:, i]))

             _hessian_active = np.dot(X.T, X[:, active[:, i]] * W[:, None])
             _score_linear_term[i] = -_hessian_active

             _observed_score_state = _score_linear_term[i].dot(_beta_unpenalized[i])
             _observed_score_state[self.inactive[i]] += self.loglikes[i].smooth_objective(beta_bar[:,i], 'grad')[self.inactive[i]]
             observed_score_state[i] = _observed_score_state

             active_directions = np.array([signed_basis_vector(p,
                                                               k,
                                                               (active_signs[:,i])[k]) for k in np.nonzero(active[:,i])[0]]).T

             _opt_linear = np.zeros((p, num_opt_var[i]))
             scaling_slice = slice(0, active[:,i].sum())
             if np.sum(active[:,i]) == 0:
                 _opt_hessian = 0
             else:
                 _opt_hessian = (_hessian_active * (active_signs[:,i])[None, active[:,i]]
                                 + self.ridge_terms[i] * active_directions)

             _opt_linear[:, scaling_slice] = _opt_hessian

             omegas.append(self._initial_omega[:, i])

             scores.append(observed_score_state[i])

             opt_offsets_.append(_opt_offset[i])

             observed_opt_states_.extend(initial_scalings[i])

             opt_linears_ = block_diag(opt_linears_, _opt_linear)

         opt_linears_ = opt_linears_[1:, :]
         omegas = np.ravel(np.asarray(omegas))
         scores = np.ravel(np.asarray(scores))
         opt_offsets_ = np.ravel(np.asarray(opt_offsets_))

         observed_opt_states_ = np.asarray(observed_opt_states_)

         opt_linears_ = opt_linears_.dot(np.linalg.inv(CoV))

         observed_opt_states_ = CoV.dot(observed_opt_states_)

         opt_vars = tot_opt_var - self.seltasks.shape[0]

         opt_linears = opt_linears_[:, :opt_vars]

         opt_offsets = opt_offsets_ + opt_linears_[:, opt_vars:].dot(observed_opt_states_[opt_vars:])

         observed_opt_states = observed_opt_states_[:opt_vars]

         ##forming linear constraints on our optimization variables

         ##definining the truncation on sums of our opt variables for each j

         con_vec = self.seltasks[self.seltasks-1 > 0] - 1
         _len_var = np.cumsum(con_vec)

         _con_sum = np.zeros((_len_var.shape[0], opt_vars))
         _con_trunc = (observed_opt_states_[opt_vars:])[self.seltasks-1 > 0]

         for j in range(_len_var.shape[0]):

             if j==0:
                 _con_sum[j, :][:_len_var[j]] = 1
             else:
                 _con_sum[j, :][_len_var[j-1]:_len_var[j]] = 1

         self.linear_con = np.vstack((-np.identity(opt_vars), _con_sum))
         self.offset_con = np.append(np.zeros(opt_vars), _con_trunc)

         self.opt_linears = opt_linears
         self.opt_offsets = opt_offsets
         self.observed_opt_states = observed_opt_states
         self.observed_score_states = scores

         self.omegas = omegas
         self.active_signs = active_signs

         print("check signs of observed opt_states ", ((self.linear_con.dot(observed_opt_states)-self.offset_con) < 0).sum(),
               self.linear_con.shape[0], opt_vars, observed_opt_states.shape[0], self.seltasks.sum())
         print("check  K.K.T. map",
               np.allclose(omegas, self.observed_score_states + self.opt_linears.dot(self.observed_opt_states) + self.opt_offsets, atol=1e-05))

         return active_signs

     def _setup_implied_gaussian(self):

         precs = np.array([])
         for i in range(self.ntask):
             _, prec = self.randomizers[i].cov_prec
             precs = block_diag(precs, prec * np.identity(self.nfeature))

         precs = precs[1:, :]

         cond_precision = self.opt_linears.T.dot(precs.dot(self.opt_linears))
         cond_cov = np.linalg.inv(cond_precision)
         logdens_linear = cond_cov.dot(self.opt_linears.T).dot(precs)

         cond_mean = -logdens_linear.dot(self.observed_score_states + self.opt_offsets)

         self.cond_mean = cond_mean
         self.cond_cov = cond_cov
         self.cond_precision = cond_precision
         self.logdens_linear = logdens_linear
         self.score_offset = self.observed_score_states + self.opt_offsets
         self.precs = precs

         return cond_mean, cond_cov, cond_precision, logdens_linear

     def multitask_inference_hetero(self,
                                    level=0.9,
                                    dispersions=None):

         self._setup_implied_gaussian()
         observed_target, cov_target, cov_target_score = self.multitask_target_hetero(dispersions=dispersions)
         prec_target = np.linalg.inv(cov_target)

         observed_target = np.atleast_1d(observed_target)

         init_soln = self.observed_opt_states
         cond_mean = self.cond_mean
         cond_precision = self.cond_precision
         logdens_linear = self.logdens_linear

         prec_opt = cond_precision

         ntarget = prec_target.shape[0]
         nopt = prec_opt.shape[0]

         target_linear = cov_target_score.T.dot(prec_target)
         target_offset = self.score_offset - cov_target_score.T.dot(prec_target).dot(observed_target)
         target_lin = - logdens_linear.dot(target_linear)

         implied_precision = np.zeros((ntarget + nopt, ntarget + nopt))

         implied_precision[:ntarget][:, :ntarget] = prec_target + (target_linear.T.dot(self.precs).dot(target_linear))

         implied_precision[ntarget:][:, :ntarget] = -prec_opt.dot(target_lin)

         implied_precision[:ntarget][:, ntarget:] = implied_precision[ntarget:][:, :ntarget].T

         implied_precision[ntarget:][:, ntarget:] = prec_opt

         implied_cov = np.linalg.inv(implied_precision)

         _Q = prec_opt.dot(logdens_linear).dot(target_offset)
         _P = (target_linear.T.dot(self.precs).dot(target_offset))
         _prec = np.linalg.inv(implied_cov[:ntarget][:, :ntarget])
         C = cov_target.dot(_P + _prec.dot(implied_cov[:ntarget][:, ntarget:]).dot(_Q))

         conjugate_arg = prec_opt.dot(cond_mean)

         val, soln, hess = solve_barrier_affine_py(conjugate_arg,
                                                   prec_opt,
                                                   init_soln,
                                                   self.linear_con,
                                                   self.offset_con,
                                                   step=1.,
                                                   nstep=10000,
                                                   min_its=5000,
                                                   tol=1.e-12)

         final_estimator = cov_target.dot(_prec).dot(observed_target) \
                           + cov_target.dot(target_lin.T.dot(prec_opt.dot(cond_mean - soln))) + C

         L = target_lin.T.dot(prec_opt)
         observed_info_natural = _prec + L.dot(target_lin) - L.dot(hess.dot(L.T))
         observed_info_mean = cov_target.dot(observed_info_natural.dot(cov_target))

         Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))
         pvalues = ndist.cdf(Z_scores)
         pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

         alpha = 1. - level
         quantile = ndist.ppf(1 - alpha / 2.)
         intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                                final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

         return final_estimator, observed_info_mean, Z_scores, pvalues, intervals

     def multitask_target_hetero(self,
                                dispersions=None):

         observed_targets = []
         cov_targets = np.array([])
         crosscov_target_scores = np.array([])

         for i in range(self.ntask):

             X, y = self.loglikes[i].data
             n, p = X.shape
             features = self._active[:, i]
             W = self.loglikes[i].saturated_loss.hessian(X.dot(self.beta_bar[:, i]))

             Xfeat = X[:, features]
             Qfeat = Xfeat.T.dot(W[:, None] * Xfeat)

             observed_target = np.linalg.pinv(Xfeat).dot(y)

             cov_target = np.linalg.inv(Qfeat)
             _score_linear = -Xfeat.T.dot(W[:, None] * X).T

             crosscov_target_score = _score_linear.dot(cov_target)

             if dispersions is None:  # use Pearson's X^2
                 dispersion = ((y - self.loglikes[i].saturated_loss.mean_function(
                     Xfeat.dot(observed_target))) ** 2 / W).sum() / (n - Xfeat.shape[1])
             else:
                 dispersion = dispersions[i]

             observed_targets.extend(observed_target)
             crosscov_target_scores = block_diag(crosscov_target_scores, crosscov_target_score.T * dispersion)
             cov_targets = block_diag(cov_targets, cov_target * dispersion)

         return np.asarray(observed_targets), cov_targets[1:, :], crosscov_target_scores[1:, :]

     def _solve_randomized_problem(self,
                                   penalty,
                                   solve_args={'tol': 1.e-12, 'min_its': 50}):

        quad_list = [rr.identity_quadratic(self.ridge_terms[i],
                                           0,
                                           -self._initial_omega[:, i],
                                           0)
                     for i in range(self.ntask)]

        problem_list = [rr.simple_problem(self.loglikes[i], penalty) for i in range(self.ntask)]

        initial_solns = np.array([problem_list[i].solve(quad_list[i], **solve_args) for i in range(self.ntask)])
        initial_subgrads = np.array([-(self.loglikes[i].smooth_objective(initial_solns[i, :],
                                                                        'grad') +
                                       quad_list[i].objective(initial_solns[i, :], 'grad'))
                                     for i in range(self.ntask)])

        return initial_solns, initial_subgrads

     def _solve_multitask_problem(self, perturbations=None, num_iter=1000, atol=1.e-5):

        if perturbations is not None:
            self._initial_omega = perturbations

        if self._initial_omega is None:
            self._initial_omega = np.array([self.randomizers[i].sample() for i in range(self.ntask)]).T

        penalty_init = rr.weighted_l1norm(self.feature_weight, lagrange=1.)
        solution_init = self._solve_randomized_problem(penalty=penalty_init)
        beta = solution_init[0].T

        for itercount in range(num_iter):

            beta_prev = beta.copy()

            sum_all_tasks = np.sum(np.absolute(beta), axis=1)

            penalty_weight = 1. / np.maximum(np.sqrt(sum_all_tasks), 10 ** -10)

            feature_weight_current = self.feature_weight * penalty_weight

            penalty_current = rr.weighted_l1norm(feature_weight_current, lagrange=1.)

            solution_current = self._solve_randomized_problem(penalty=penalty_current)

            beta = solution_current[0].T

            if np.sum(np.fabs(beta_prev - beta)) < atol:
                break

        print("check itercount until convergence ", itercount)

        subgrad = solution_current[1].T

        return beta, subgrad, feature_weight_current

     @staticmethod
     def gaussian(predictor_vars,
                  response_vars,
                  feature_weight,
                  noise_levels=None,
                  quadratic=None,
                  ridge_term=None,
                  randomizer_scales=None,
                  perturbations=None):

        ntask = len(response_vars)

        if noise_levels is None:
            noise_levels = np.ones(ntask)

        loglikes = {i: rr.glm.gaussian(predictor_vars[i], response_vars[i], coef=1. / noise_levels[i] ** 2, quadratic=quadratic)
                    for i in range(ntask)}

        sample_sizes = np.asarray([predictor_vars[i].shape[0] for i in range(ntask)])
        nfeatures = [predictor_vars[i].shape[1] for i in range(ntask)]

        if all(x == nfeatures[0] for x in nfeatures) == False:
            raise ValueError("all the predictor matrices must have the same regression dimensions")
        else:
            nfeature = nfeatures[0]

        if ridge_term is None:
            ridge_terms = np.zeros(ntask)
        else:
            ridge_terms = ridge_term

        mean_diag_list = [np.mean((predictor_vars[i] ** 2).sum(0)) for i in range(ntask)]
        if randomizer_scales is None:
            randomizer_scales = np.asarray([np.sqrt(mean_diag_list[i]) * 0.5 * np.std(response_vars[i])
                                            * np.sqrt(sample_sizes[i] / (sample_sizes[i] - 1.)) for i in range(ntask)])

        randomizers = {i: randomization.isotropic_gaussian((nfeature,), randomizer_scales[i]) for i in range(ntask)}

        return multi_task_lasso(loglikes,
                                np.asarray(feature_weight),
                                ridge_terms,
                                randomizers,
                                nfeature,
                                ntask,
                                perturbations)

def solve_barrier_affine_py(conjugate_arg,
                            precision,
                            feasible_point,
                            con_linear,
                            con_offset,
                            step=1,
                            nstep=1000,
                            min_its=200,
                            tol=1.e-10):

    scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. \
                          + np.log(1.+ 1./((con_offset - con_linear.dot(u))/ scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) - con_linear.T.dot(1./(scaling + con_offset - con_linear.dot(u)) -
                                                                       1./(con_offset - con_linear.dot(u)))
    barrier_hessian = lambda u: con_linear.T.dot(np.diag(-1./((scaling + con_offset-con_linear.dot(u))**2.)
                                                 + 1./((con_offset-con_linear.dot(u))**2.))).dot(con_linear)

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(con_offset-con_linear.dot(proposal) > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 1000:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(precision + barrier_hessian(current))

    return current_value, current, hess






