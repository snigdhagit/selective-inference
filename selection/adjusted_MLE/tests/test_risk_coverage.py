import numpy as np, sys, os
import pandas as pd
from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import selection.randomized.lasso as L; reload(L)
from selection.adjusted_MLE.tests.test_inferential_metrics import (BHfilter,
                                                                   selInf_R,
                                                                   glmnet_lasso,
                                                                   sim_xy,
                                                                   tuned_lasso,
                                                                   relative_risk,
                                                                   coverage,
                                                                   comparison_risk_inference_selected,
                                                                   comparison_risk_inference_selected_alt,
                                                                   comparison_risk_inference_full)

from selection.adjusted_MLE.tests.test_cv_inferential_metrics import (comparison_risk_inference_selected_cv,
                                                                      comparison_risk_inference_full_cv,
                                                                      comparison_risk_inference_selected_cv_alt_powerfdr)

def write_ouput(outpath, n=500, p=100, rho=0.35, s=5, beta_type=1, target="selected", tuning = "selective_MLE",
                randomizing_scale= np.sqrt(0.50), CV= True, ndraw = 50):

    df_master = pd.DataFrame()
    df_risk = pd.DataFrame()

    snr_values = np.array([0.30, 0.42, 0.71, 1.22])
#0.10, 0.15, 0.20, 0.25,
    for snr in snr_values:

        if target == "selected":
            if CV is True:
                output_overall = np.zeros(35)
                if n > p:
                    full_dispersion = True
                else:
                    full_dispersion = False
                for i in range(ndraw):
                    output = comparison_risk_inference_selected_cv(n=n, p=p, nval=n, rho=rho, s=s,
                                                                   beta_type=beta_type,snr=snr,
                                                                   randomizer_scale=randomizing_scale,
                                                                   target=target, full_dispersion=full_dispersion)
                    output_overall += np.squeeze(output)

                output_overall /= float(ndraw)
                metrics_selective_MLE = pd.DataFrame({"sample_size": n,
                                                      "regression_dim": p,
                                                      "correlation": rho,
                                                      "SNR": snr,
                                                      "signal_type": beta_type,
                                                      "risk": output_overall[0],
                                                      "coverage": output_overall[6],
                                                      "length": output_overall[10],
                                                      "proportion_infinity": 0.,
                                                      "power": output_overall[14],
                                                      "fdr": output_overall[22],
                                                      "power_post_BH": output_overall[18],
                                                      "nactive": output_overall[26],
                                                      "ndiscoveries": output_overall[29],
                                                      "method": "Selective MLE",
                                                      "tuning": tuning}, index=[0])

                metrics_randomized_LASSO = pd.DataFrame({"sample_size": n,
                                                         "regression_dim": p,
                                                         "correlation": rho,
                                                         "SNR": snr,
                                                         "signal_type": beta_type,
                                                         "risk": output_overall[2],
                                                         "coverage": 0.,
                                                         "length": 0.,
                                                         "proportion_infinity": 0.,
                                                         "power": 0.,
                                                         "fdr": 0.,
                                                         "power_post_BH": 0.,
                                                         "nactive": output_overall[27],
                                                         "ndiscoveries": 0.,
                                                         "method": "Randomized LASSO",
                                                         "tuning": tuning}, index=[0])

                metrics_Lee = pd.DataFrame({"sample_size": n,
                                            "regression_dim": p,
                                            "correlation": rho,
                                            "SNR": snr,
                                            "signal_type": beta_type,
                                            "risk": output_overall[5],
                                            "coverage": output_overall[7],
                                            "length": output_overall[11],
                                            "proportion_infinity": output_overall[33],
                                            "power": output_overall[15],
                                            "fdr": output_overall[23],
                                            "power_post_BH": output_overall[19],
                                            "nactive": output_overall[27],
                                            "ndiscoveries": output_overall[30],
                                            "method": "Lee",
                                            "tuning": tuning}, index=[0])

                metrics_Liu = pd.DataFrame({"sample_size": n,
                                            "regression_dim": p,
                                            "correlation": rho,
                                            "SNR": snr,
                                            "signal_type": beta_type,
                                            "risk": output_overall[5],
                                            "coverage": output_overall[8],
                                            "length": output_overall[12],
                                            "proportion_infinity": 0.,
                                            "power": output_overall[16],
                                            "fdr": output_overall[24],
                                            "power_post_BH": output_overall[20],
                                            "nactive": output_overall[27],
                                            "ndiscoveries": output_overall[31],
                                            "method": "Liu LASSO",
                                            "tuning": tuning}, index=[0])

                metrics_unad = pd.DataFrame({"sample_size": n,
                                             "regression_dim": p,
                                             "correlation": rho,
                                             "SNR": snr,
                                             "signal_type": beta_type,
                                             "risk": output_overall[5],
                                             "coverage": output_overall[9],
                                             "length": output_overall[13],
                                             "proportion_infinity": 0.,
                                             "power": output_overall[17],
                                             "fdr": output_overall[25],
                                             "power_post_BH": output_overall[21],
                                             "nactive": output_overall[28],
                                             "ndiscoveries": output_overall[32],
                                             "method": "Naive",
                                             "tuning": tuning}, index=[0])

                df_master = df_master.append(metrics_selective_MLE, ignore_index=True)
                df_master = df_master.append(metrics_randomized_LASSO, ignore_index=True)
                df_master = df_master.append(metrics_Lee, ignore_index=True)
                df_master = df_master.append(metrics_Liu, ignore_index=True)
                df_master = df_master.append(metrics_unad, ignore_index=True)

            else:
                output_overall = np.zeros(29)
                if n > p:
                    full_dispersion = True
                else:
                    full_dispersion = False
                for i in range(ndraw):
                    output = comparison_risk_inference_selected_cv_alt_powerfdr(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type,
                                                                                snr=snr, randomizer_scale=randomizing_scale,
                                                                                target=target, full_dispersion=full_dispersion)
                    output_overall += np.squeeze(output)

                output_overall /= float(ndraw)
                metrics_selective_MLE = pd.DataFrame({"sample_size": n,
                                                      "regression_dim": p,
                                                      "correlation": rho,
                                                      "SNR": snr,
                                                      "signal_type": beta_type,
                                                      "risk": output_overall[0],
                                                      "coverage": output_overall[6],
                                                      "length": output_overall[9],
                                                      "proportion_infinity": 0.,
                                                      "power": output_overall[12],
                                                      "fdr": output_overall[18],
                                                      "power_post_BH": output_overall[15],
                                                      "nactive": output_overall[21],
                                                      "ndiscoveries": output_overall[24],
                                                      "method": "Selective MLE",
                                                      "tuning": tuning}, index=[0])

                metrics_randomized_LASSO = pd.DataFrame({"sample_size": n,
                                                         "regression_dim": p,
                                                         "correlation": rho,
                                                         "SNR": snr,
                                                         "signal_type": beta_type,
                                                         "risk": output_overall[2],
                                                         "coverage": 0.,
                                                         "length": 0.,
                                                         "proportion_infinity": 0.,
                                                         "power": 0.,
                                                         "fdr": 0.,
                                                         "power_post_BH": 0.,
                                                         "nactive": output_overall[21],
                                                         "ndiscoveries": 0.,
                                                         "method": "Randomized LASSO",
                                                         "tuning": tuning}, index=[0])

                metrics_Lee = pd.DataFrame({"sample_size": n,
                                            "regression_dim": p,
                                            "correlation": rho,
                                            "SNR": snr,
                                            "signal_type": beta_type,
                                            "risk": output_overall[5],
                                            "coverage": output_overall[7],
                                            "length": output_overall[10],
                                            "proportion_infinity": output_overall[27],
                                            "power": output_overall[13],
                                            "fdr": output_overall[19],
                                            "power_post_BH": output_overall[16],
                                            "nactive": output_overall[22],
                                            "ndiscoveries": output_overall[25],
                                            "method": "Lee",
                                            "tuning": tuning}, index=[0])

                metrics_unad = pd.DataFrame({"sample_size": n,
                                             "regression_dim": p,
                                             "correlation": rho,
                                             "SNR": snr,
                                             "signal_type": beta_type,
                                             "risk": output_overall[5],
                                             "coverage": output_overall[8],
                                             "length": output_overall[11],
                                             "proportion_infinity": 0.,
                                             "power": output_overall[14],
                                             "fdr": output_overall[20],
                                             "power_post_BH": output_overall[17],
                                             "nactive": output_overall[23],
                                             "ndiscoveries": output_overall[26],
                                             "method": "Naive",
                                             "tuning": tuning}, index=[0])

                df_master = df_master.append(metrics_selective_MLE, ignore_index=True)
                df_master = df_master.append(metrics_randomized_LASSO, ignore_index=True)
                df_master = df_master.append(metrics_Lee, ignore_index=True)
                df_master = df_master.append(metrics_unad, ignore_index=True)

        elif target == "full":
            output_overall = np.zeros(35)
            if n > p:
                full_dispersion = True
            else:
                full_dispersion = False
            for i in range(ndraw):
                if CV is True:
                    output = comparison_risk_inference_full_cv(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type,
                                                               snr=snr,randomizer_scale=randomizing_scale,target=target,
                                                               full_dispersion=full_dispersion)
                else:
                    output = comparison_risk_inference_full(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type,
                                                            snr=snr,
                                                            randomizer_scale=randomizing_scale,
                                                            target=target, tuning=tuning,
                                                            full_dispersion=full_dispersion)

                output_overall += np.squeeze(output)

            output_overall /= float(ndraw)
            metrics_selective_MLE = pd.DataFrame({"sample_size": n,
                                                  "regression_dim": p,
                                                  "correlation": rho,
                                                  "SNR": snr,
                                                  "signal_type": beta_type,
                                                  "risk": output_overall[0],
                                                  "coverage": output_overall[6],
                                                  "length": output_overall[10],
                                                  "proportion_infinity": 0.,
                                                  "power": output_overall[14],
                                                  "fdr": output_overall[22],
                                                  "power_post_BH": output_overall[18],
                                                  "nactive": output_overall[26],
                                                  "ndiscoveries": output_overall[29],
                                                  "method": "Selective MLE",
                                                  "tuning": tuning}, index=[0])

            metrics_randomized_LASSO = pd.DataFrame({"sample_size": n,
                                                     "regression_dim": p,
                                                     "correlation": rho,
                                                     "SNR": snr,
                                                     "signal_type": beta_type,
                                                     "risk": output_overall[2],
                                                     "coverage": 0.,
                                                     "length": 0.,
                                                     "proportion_infinity": 0.,
                                                     "power": 0.,
                                                     "fdr": 0.,
                                                     "power_post_BH": 0.,
                                                     "nactive": output_overall[27],
                                                     "ndiscoveries": 0.,
                                                     "method": "Randomized LASSO",
                                                     "tuning": tuning}, index=[0])

            metrics_Lee = pd.DataFrame({"sample_size": n,
                                        "regression_dim": p,
                                        "correlation": rho,
                                        "SNR": snr,
                                        "signal_type": beta_type,
                                        "risk": output_overall[5],
                                        "coverage": output_overall[7],
                                        "length": output_overall[11],
                                        "proportion_infinity": output_overall[33],
                                        "power": output_overall[15],
                                        "fdr": output_overall[23],
                                        "power_post_BH": output_overall[19],
                                        "nactive": output_overall[27],
                                        "ndiscoveries": output_overall[30],
                                        "method": "Lee",
                                        "tuning": tuning}, index=[0])

            metrics_Liu = pd.DataFrame({"sample_size": n,
                                        "regression_dim": p,
                                        "correlation": rho,
                                        "SNR": snr,
                                        "signal_type": beta_type,
                                        "risk": output_overall[5],
                                        "coverage": output_overall[8],
                                        "length": output_overall[12],
                                        "proportion_infinity": 0.,
                                        "power": output_overall[16],
                                        "fdr": output_overall[24],
                                        "power_post_BH": output_overall[20],
                                        "nactive": output_overall[27],
                                        "ndiscoveries": output_overall[31],
                                        "method": "Liu LASSO",
                                        "tuning": tuning}, index=[0])

            metrics_unad = pd.DataFrame({"sample_size": n,
                                         "regression_dim": p,
                                         "correlation": rho,
                                         "SNR": snr,
                                         "signal_type": beta_type,
                                         "risk": output_overall[5],
                                         "coverage": output_overall[9],
                                         "length": output_overall[13],
                                         "proportion_infinity": 0.,
                                         "power": output_overall[17],
                                         "fdr": output_overall[25],
                                         "power_post_BH": output_overall[21],
                                         "nactive": output_overall[28],
                                         "ndiscoveries": output_overall[32],
                                         "method": "Naive",
                                         "tuning": tuning}, index=[0])

            df_master = df_master.append(metrics_selective_MLE, ignore_index=True)
            df_master = df_master.append(metrics_randomized_LASSO, ignore_index=True)
            df_master = df_master.append(metrics_Lee, ignore_index=True)
            df_master = df_master.append(metrics_Liu, ignore_index=True)
            df_master = df_master.append(metrics_unad, ignore_index=True)

        metrics = pd.DataFrame({"sample_size": n,
                                "regression_dim": p,
                                "correlation": rho,
                                "SNR": snr,
                                "signal_type": beta_type,
                                "Risk_selMLE": output_overall[0],
                                "Risk_indest": output_overall[1],
                                "Risk_LASSO_rand": output_overall[2],
                                "Risk_relLASSO_rand": output_overall[3],
                                "Risk_relLASSO_nonrand": output_overall[4],
                                "Risk_LASSO_nonrand": output_overall[5],
                                "tuning": tuning}, index=[0])

        df_risk = df_risk.append(metrics, ignore_index=True)

    outfile_metrics = os.path.join(outpath, "metrics_beta_type_"+ str(beta_type) +  "_CV0_" + target + "_rho_"+ str(rho) +".csv")
    outfile_risk = os.path.join(outpath, "risk_beta_type" + str(beta_type) + "_CV0_" + target +"_rho_"+ str(rho) + ".csv")
    df_master.to_csv(outfile_metrics, index=False)
    df_risk.to_csv(outfile_risk, index=False)

write_ouput("/Users/snigdhapanigrahi/adjusted_MLE/results/high_settings/cross_validated/", n=200, p=1000, rho=0.35, s=10, beta_type=1,
            target="selected", tuning = "selective_MLE", randomizing_scale= np.sqrt(0.50), CV=False, ndraw = 50)
