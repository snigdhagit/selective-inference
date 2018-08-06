import numpy as np, os, itertools
import pandas as pd

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr

from selection.adjusted_MLE.tests.cv_MLE import (sim_xy,
                                                 selInf_R,
                                                 glmnet_lasso,
                                                 BHfilter,
                                                 coverage,
                                                 relative_risk,
                                                 comparison_cvmetrics_selected,
                                                 comparison_cvmetrics_full)

def plotRisk(df_risk):
    robjects.r("""
               library("ggplot2")
               library("magrittr")
               library("tidyr")
               library("reshape")
               
               plot_risk <- function(df_risk, outpath="/Users/psnigdha/adjusted_MLE/plots/", resolution=300, height= 7.5, width=15)
                { 
                   date = 1:length(unique(df_risk$snr))
                   df = cbind(df_risk, date)
                   p1= df %>%
                       gather(key, value, sel.MLE, rand.LASSO, LASSO) %>%
                       ggplot(aes(x=date, y=value, colour=key, shape=key, linetype=key)) +
                       geom_point(size=3) +
                       geom_line(aes(linetype=key), size=1) +
                       ylim(0.01,1.2)+
                       labs(y="relative risk", x = "Signal regimes: snr") +
                       scale_x_continuous(breaks=1:length(unique(df_risk$snr)), label = sapply(df_risk$snr, toString)) +
                       theme(legend.position="top", legend.title = element_blank())
                       indices = sort(c(df$sel.MLE[1], df$rand.LASSO[1], df$LASSO[1]), index.return= TRUE)$ix
                       names = c("sel-MLE", "rand-LASSO", "LASSO")
                   p1 = p1 + scale_color_manual(labels = names[indices], values=c("#008B8B", "#104E8B","#B22222")[indices]) +
                        scale_shape_manual(labels = names[indices], values=c(15, 17, 16)[indices]) +
                        scale_linetype_manual(labels = names[indices], values = c(1,1,2)[indices])
                   outfile = paste(outpath, 'risk.png', sep="")                       
                   ggsave(outfile, plot = p1, dpi=resolution, dev='png', height=height, width=width, units="cm")}
                """)

    robjects.pandas2ri.activate()
    r_df_risk = robjects.conversion.py2ri(df_risk)
    R_plot = robjects.globalenv['plot_risk']
    R_plot(r_df_risk)

def output_file(n=300, p=100, rho=0.35, s=5, beta_type=1, snr_values=np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.42, 0.71, 1.22]),
                target="selected", tuning_nonrand="lambda.min", tuning_rand="lambda.1se",
                randomizing_scale = np.sqrt(0.50), ndraw = 10, outpath = None):

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

    plotRisk(df_risk)

output_file(outpath='/Users/psnigdha/adjusted_MLE/n_300_p_100/')





















