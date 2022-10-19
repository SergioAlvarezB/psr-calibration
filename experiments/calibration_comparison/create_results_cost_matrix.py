import os
import joblib
import numpy as np
from IPython import embed
from psrcal.calibration import calibrate, AffineCalLogLoss, AffineCalLogLossPlusECE, AffineCalBrier
from psrcal.losses import LogLoss, LogLossSE, ECE, CostFunction, Brier
import torch
import matplotlib.pyplot as plt

colors = {'raw': 'k', 'Log': 'blue', 'Cal': 'blue', 'Bri': 'red', 'Log+w*ECE': 'green'}

def compute_and_print_results(dir, trndset, tstdset, ece_weight=0.1, cost_family='alpha_in_row'):

    # Load the scores for training the calibration models and for evaluation
    trn_score_path = "data/{}/{}/predictions.npy".format(dir,trndset)
    tst_score_path = "data/{}/{}/predictions.npy".format(dir,tstdset)
    trn_label_path = "data/{}/{}/targets.npy".format(dir,trndset)
    tst_label_path = "data/{}/{}/targets.npy".format(dir,tstdset)

    trn_scores = torch.as_tensor(np.load(trn_score_path), dtype=torch.float32)
    tst_scores = torch.as_tensor(np.load(tst_score_path), dtype=torch.float32)
    trn_labels = torch.as_tensor(np.load(trn_label_path), dtype=torch.int64)
    tst_labels = torch.as_tensor(np.load(tst_label_path), dtype=torch.int64)

    # Normalize the scores in case they were unnormalized logits
    # After this operation, we can treat the scores as (potentially miscalibrated)
    # log posteriors
    tst_scores -= torch.logsumexp(tst_scores, axis=-1, keepdim=True) 
    trn_scores -= torch.logsumexp(trn_scores, axis=-1, keepdim=True) 

    # Define various series of cost functions with varying alpha
    nclasses = tst_scores.shape[1]
    cost0 = 1-np.eye(nclasses)
    cost_matrices = []
    min_alpha, max_alpha = [-1, 0] if cost_family == 'alpha_for_abstention' else [-4, 4]
    delta = 0.2
    alphas = np.arange(min_alpha, max_alpha+delta, delta)
    for alpha in alphas:
        alphae = np.exp(alpha)
        cost1 = cost0.copy()
        if cost_family == 'alpha_in_row':
            cost1[0,:] = alphae
            cost1[0,0] = 0
        elif cost_family == 'alpha_in_col':
            cost1[:,0] = alphae
            cost1[0,0] = 0
        elif cost_family == 'alpha_for_abstention':
            cost1 = np.c_[cost1, alphae*np.ones(cost1.shape[0])].T
        cost_matrices.append(cost1)


    def _get_metrics(scores, raw_scores=None):
        xent = LogLoss(scores, tst_labels, norm=True)
        ece  = ECE(scores, tst_labels)
        bri  = Brier(scores, tst_labels)
        table_metrics = [bri, ece, xent]

        if raw_scores is not None:
            # Assume scores is the calibrated version of raw_scores and compute 
            # both versions of the cross-entropy decomposition (empirical and semi-empirical)
            xent_raw    = LogLoss(raw_scores, tst_labels, norm=True)
            xent_se_raw = LogLossSE(raw_scores, scores, norm=True) 
            xent_se     = LogLossSE(scores, scores, norm=True) 
            table_metrics += [xent_se, xent_raw, xent_se_raw]

        costs = []
        for cost_matrix in cost_matrices:
            costs.append(CostFunction(scores, tst_labels, cost_matrix, norm=True).item())
        return table_metrics, costs


    def _format_results(params, mname, mvals, has_bias=None):
        if print_transform:
            print("%6s  %6s  %-20s  %-76s   %-9s       "%(tstdset, trndset, dir, params, mname), end="")
        else:
            print("%6s  %6s  %-20s  %-8s  %-9s       "%(tstdset, trndset, dir, has_bias, mname), end="")
        print("".join(["%6.3f     "%m for m in mvals]))


    plt.figure(figsize=(10,5))

    print("-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
    if print_transform:
        print("TstSet   TrnSet %-20s   %-75s   "%("System", "Transform"), end='')
    else:
        print("TstSet   TrnSet %-20s  %8s   "%("System", "Has_bias"), end='')
    print("CalLoss          %-10s %-10s %-10s %-10s %-10s %-10s\n"%("Brier", "ECE", "Log", "LogSE", "Log_precal", "LogSE_precal"))
    

    table_metrics, costs = _get_metrics(tst_scores)
    _format_results(" None", "raw", table_metrics)
    plt.plot(alphas, costs, label="raw", color=colors['raw'])

    maxy = np.max(costs)
    miny = np.min(costs)
    for bias in [True, False]:

        cal_out = dict()

        # Different ways of calibrating the scores. 
        cal_out['Log']       = calibrate(trn_scores, trn_labels, tst_scores, AffineCalLogLoss,        bias=bias)
        cal_out['Bri']       = calibrate(trn_scores, trn_labels, tst_scores, AffineCalBrier,          bias=bias)
        #cal_out['Log+w*ECE'] = calibrate(trn_scores, trn_labels, tst_scores, AffineCalLogLossPlusECE, bias=bias, ece_weight=ece_weight)

        for cal_type, (cal_tst_scores, cal_params) in cal_out.items():
            table_metrics, costs = _get_metrics(cal_tst_scores, tst_scores)
            params = "%5.2f "%cal_params[0]
            if len(cal_params) > 1:
                params += " ".join(["%5.2f "%f for f in cal_params[1]])
            _format_results(params, cal_type, table_metrics, bias)
            plt.plot(alphas, costs, label=cal_type+" with bias" if bias else cal_type+" no bias", color=colors[cal_type], linestyle='-' if bias else ':')
            maxy = max(maxy, np.max(costs))
            miny = min(miny, np.min(costs))

    plt.legend()
#    plt.ylim([miny, maxy])
    plt.xlabel("alpha")
    plt.ylabel("normalized cost")
    plt.title("%s %s \n cal trained on %s"%(tstdset, cost_family, trndset))
    outres = "results/%s"%dir
    if not os.path.isdir(outres):
        os.makedirs(outres)
    plt.savefig("%s/costs_%s_%s_%s.png"%(outres,cost_family,trndset,tstdset), dpi=300)
    plt.close()


# Weight given to the ECE term when optimizing LogLoss + w * ECE for calibration
ece_weight = 0.5
print_transform = True

for cost_family in ['alpha_in_col', 'alpha_in_row', 'alpha_for_abstention']:

    for sys in ["emotion_ep12", "emotion_final", "spkr_verif", "resnet-50_cifar10"]:

        compute_and_print_results(sys, "trn", "trn", ece_weight=ece_weight, cost_family=cost_family)
        compute_and_print_results(sys, "tst", "tst", ece_weight=ece_weight, cost_family=cost_family)

        if sys == 'resnet-50_cifar10':
            compute_and_print_results(sys, "tst_sh", "tst_sh", ece_weight=ece_weight, cost_family=cost_family)
            compute_and_print_results(sys, "val", "tst", ece_weight=ece_weight, cost_family=cost_family)
            
