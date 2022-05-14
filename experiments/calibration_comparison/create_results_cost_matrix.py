import os
import joblib
import numpy as np
from IPython import embed
from psrcal.calibration import calibrate, AffineCalLogLoss, AffineCalLogLossPlusECE, AffineCalBrier
from psrcal.losses import LogLoss, ECE, CostFunction, Brier
import torch
import matplotlib.pyplot as plt

def compute_and_print_results(dir, dset, ece_weight=0.1, cost_family='alpha_in_row'):

    # This code assumes that the priors in train and test are the same
    scor_path = "{}/{}/predictions".format(dir,dset)
    lab_path = "{}/{}/targets".format(dir,dset)

    if dir in ["emotion_ep12", "emotion_final"]:
        scores = torch.tensor(np.concatenate(joblib.load(scor_path)))
        labels = torch.tensor(np.concatenate(joblib.load(lab_path)), dtype=torch.int64)

        # There is no validation set for these datasets
        tst_scores = scores
        tst_labels = labels

        

    elif dir in ['resnet-50_cifar10']:
        scores = torch.as_tensor(np.load(scor_path + '.npy'), dtype=torch.float32)
        labels = torch.as_tensor(np.load(lab_path + '.npy'), dtype=torch.int64)
        if dset == 'val':
            tst_scores = torch.as_tensor(np.load("{}/{}/predictions".format(dir,'tst') + '.npy'), dtype=torch.float32)
            tst_labels = torch.as_tensor(np.load("{}/{}/targets".format(dir,'tst') + '.npy'), dtype=torch.int64)
        else:
            tst_scores = scores
            tst_labels = labels

    else:
        raise ValueError('Not available data for {}'.format(dir))

    logp_raw = tst_scores - torch.logsumexp(tst_scores, axis=-1, keepdim=True) 

    # Define a cost matrix with 0s in the diagonal, and 1s everywhere else
    # except for one column or row where the values are exp(alpha), with
    # varying alpha

    # Number of classes
    nclasses = scores.shape[1]

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


    def _get_metrics(scores):
        xent = LogLoss(scores, tst_labels, norm=True)
        ece  = ECE(scores, tst_labels)
        bri  = Brier(scores, tst_labels)
        table_metrics = [xent, bri, ece]
        costs = []
        for cost_matrix in cost_matrices:
            costs.append(CostFunction(scores, tst_labels, cost_matrix, norm=True).item())
        return table_metrics, costs

    def _format_results(dset, dir, params, mname, mvals):
        print("%3s  %-20s  %-35s   %-9s       "%(dset, dir, params, mname), end="")
        print("".join(["%6.3f     "%m for m in mvals]))


    plt.figure(figsize=(10,5))

    print("--------------------------------------------------------------------------------------------------------------------")
    print("Set  %-20s   %-34s   CalLoss          %-8s   %-8s   %-8s "%("System", "Transform", "Log", "Brier", "ECE"))
    print("")

    table_metrics, costs = _get_metrics(logp_raw)
    _format_results(dset, dir, " None", "raw", table_metrics)
    print("")
    plt.plot(alphas, costs, label="raw", color='k')

    color = {'Log': 'blue', 'Bri': 'red', 'Log+w*ECE': 'green'}

    maxy = np.max(costs)
    miny = np.min(costs)
    for bias in [True, False]:

        cal_out = dict()

        # Different ways of calibrating the scores. 
        cal_out['Log']     = calibrate(tst_scores, labels, AffineCalLogLoss, scores,        bias=bias)
        cal_out['Bri']     = calibrate(tst_scores, labels, AffineCalBrier, scores,          bias=bias)
        cal_out['Log+w*ECE'] = calibrate(tst_scores, labels, AffineCalLogLossPlusECE, scores, bias=bias, ece_weight=ece_weight)

        for cal_type, (cal_scores, cal_params) in cal_out.items():
            table_metrics, costs = _get_metrics(cal_scores)
            params = "%5.2f "%cal_params[0]
            if len(cal_params) > 1:
                params += " ".join(["%5.2f "%f for f in cal_params[1]])
            _format_results(dset, dir, params, cal_type, table_metrics)
            plt.plot(alphas, costs, label=cal_type if bias else None, color=color[cal_type], linestyle='-' if bias else ':')
            maxy = max(maxy, np.max(costs))
            miny = min(miny, np.min(costs))
        print("")

    plt.legend()
    plt.ylim([miny*0.9, maxy*1.2])
    plt.xlabel("alpha")
    plt.ylabel("normalized cost")
    plt.title("%s"%cost_family)
    plt.savefig("results/costs_%s_%s_%s.pdf"%(cost_family,dir,dset))
    plt.close()


# Weight given to the ECE term when optimizing LogLoss + w * ECE for calibration
ece_weight = {
    'emotion_final': 0.05,
    'resnet-50_cifar10': 0.05,
}

for cost_family in ['alpha_in_col', 'alpha_in_row', 'alpha_for_abstention']:
#for cost_family in ['alpha_for_abstention']:

    #for sys in ["emotion_ep12", "emotion_final"]:
    for sys in ["emotion_final", "resnet-50_cifar10"]:

        compute_and_print_results(sys, "trn", ece_weight=ece_weight[sys], cost_family=cost_family)
        compute_and_print_results(sys, "tst", ece_weight=ece_weight[sys], cost_family=cost_family)
        if sys == 'resnet-50_cifar10':
            compute_and_print_results(sys, "val", ece_weight=ece_weight[sys], cost_family=cost_family)
            compute_and_print_results(sys, "tst_sh", ece_weight=ece_weight[sys], cost_family=cost_family)
            compute_and_print_results(sys, "tst_sc", ece_weight=ece_weight[sys], cost_family=cost_family)

