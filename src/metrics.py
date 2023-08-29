from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss, matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
from sklearn import metrics
import pickle
from src.constants import *
from src.utils import *
import numpy as np
import pandas as pd
import os
import json


# taken from https://www.kaggle.com/cpmpml/optimizing-probabilities-for-best-mcc
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)
        
def get_best_threshold_mcc(y_true, y_prob):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    prev_proba = -1
    best_proba = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_id = i
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0

    y_pred = (y_prob >= best_proba).astype(int)
    score = matthews_corrcoef(y_true, y_pred)
    # print(score, best_mcc)
    # plt.plot(mccs)
    return best_proba

def get_optimal_threshold(output_df, data_df):
    test_df = data_df.merge(output_df)
    
    predictions = np.stack(test_df["preds"].to_numpy())
    actuals = np.stack(test_df["Target"].to_numpy())
    
    optimal_thresholds = np.zeros((11,))
    for i in range(11):
        fpr, tpr, thresholds = metrics.roc_curve(actuals[:, i], predictions[:, i])
        optimal_idx = np.argmax(tpr - fpr)
        optimal_thresholds[i] = thresholds[optimal_idx]

    return optimal_thresholds

def get_optimal_threshold_pr(output_df, data_df):
    test_df = data_df.merge(output_df)
    
    predictions = np.stack(test_df["preds"].to_numpy())
    actuals = np.stack(test_df["Target"].to_numpy())
    
    optimal_thresholds = np.zeros((11,))
    for i in range(11):
        pr, re, thresholds = metrics.precision_recall_curve(actuals[:, i], predictions[:, i])
        fscores = (2 * pr * re) / (pr + re)
        optimal_idx = np.argmax(fscores)
        optimal_thresholds[i] = thresholds[optimal_idx]

    return optimal_thresholds

def get_optimal_threshold_mcc(output_df, data_df):
    test_df = data_df.merge(output_df)
    
    predictions = np.stack(test_df["preds"].to_numpy())
    actuals = np.stack(test_df["Target"].to_numpy())
    
    optimal_thresholds = np.zeros((11,))
    for i in range(11):
        optimal_thresholds[i] = get_best_threshold_mcc(actuals[:, i], predictions[:, i])

    return optimal_thresholds

def calculate_sl_metrics_fold(test_df, thresholds):
    print("Computing fold")
    predictions = np.stack(test_df["preds"].to_numpy())
    outputs = predictions>thresholds
    actuals = np.stack(test_df["Target"].to_numpy())

    ypred_membrane = outputs[:, 0]
    ypred_subloc = outputs[:,1:]
    y_membrane = actuals[:, 0]
    y_subloc = actuals[:,1:]

    metrics_dict = {}

    metrics_dict["NumLabels"] = y_subloc.sum(1).mean()
    metrics_dict["NumLabelsTest"] = ypred_subloc.sum(1).mean()
    metrics_dict["ACC_membrane"] = (ypred_membrane == y_membrane).mean()
    metrics_dict["MCC_membrane"] = matthews_corrcoef(y_membrane, ypred_membrane)
    metrics_dict["ACC_subloc"] = (np.all((ypred_subloc == y_subloc), axis=1)).mean()
    metrics_dict["HammLoss_subloc"] = 1-hamming_loss(y_subloc, ypred_subloc)
    metrics_dict["Jaccard_subloc"] = jaccard_score(y_subloc, ypred_subloc, average="samples")
    metrics_dict["MicroF1_subloc"] = f1_score(y_subloc, ypred_subloc, average="micro")
    metrics_dict["MacroF1_subloc"] = f1_score(y_subloc, ypred_subloc, average="macro")
    for i in range(10):
      metrics_dict[f"{CATEGORIES[1+i]}"] = matthews_corrcoef(y_subloc[:,i], ypred_subloc[:,i])

    # for i in range(10):
    #    metrics_dict[f"{categories[1+i]}"] = roc_auc_score(y_subloc[:,i], predictions[:,i+1])
    return metrics_dict

def calculate_sl_metrics(model_attrs: ModelAttributes, datahandler: DataloaderHandler, thresh_type="mcc", inner_i="1Layer"):
    with open(os.path.join(model_attrs.outputs_save_path, f"thresholds_sl_{thresh_type}.pkl"), "rb") as f:
        threshold_dict = pickle.load(f)
    print(np.array(list(threshold_dict.values())).mean(0))
    metrics_dict_list = {}
    full_data_df = []
    for outer_i in range(5):
        data_df = datahandler.get_partition(outer_i)
        output_df = pd.read_pickle(os.path.join(model_attrs.outputs_save_path, f"{outer_i}_{inner_i}.pkl"))
        data_df = data_df.merge(output_df)
        full_data_df.append(data_df)
        threshold = threshold_dict[f"{outer_i}_{inner_i}"]
        metrics_dict = calculate_sl_metrics_fold(data_df, threshold)
        for k in metrics_dict:
            metrics_dict_list.setdefault(k, []).append(metrics_dict[k])

    output_dict = {}
    for k in metrics_dict_list:
        output_dict[k] = [f"{round(np.array(metrics_dict_list[k]).mean(), 2):.2f} pm {round(np.array(metrics_dict_list[k]).std(), 2):.2f}"]

    print(pd.DataFrame(output_dict).to_latex())
    for k in metrics_dict_list:
        print("{0:21s} : {1}".format(k, f"{round(np.array(metrics_dict_list[k]).mean(), 2):.2f} + {round(np.array(metrics_dict_list[k]).std(), 2):.2f}"))
    for k in metrics_dict_list:
        print("{0}".format(f"{round(np.array(metrics_dict_list[k]).mean(), 2):.2f} + {round(np.array(metrics_dict_list[k]).std(), 2):.2f}"))


def calculate_ss_metrics_fold(y_test, y_test_preds, thresh):
    y_preds = y_test_preds > thresh

    metrics_dict = {}

    metrics_dict["microF1"] = f1_score(y_test, y_preds, average="micro")
    metrics_dict["macroF1"] = f1_score(y_test, y_preds, average="macro")
    metrics_dict["accuracy"] = (np.all((y_preds == y_test), axis=1)).mean()

    for j in range(len(SS_CATEGORIES)-1):
        metrics_dict[f"{SS_CATEGORIES[j+1]}"]  = matthews_corrcoef(y_preds[:, j],y_test[:, j])

    return metrics_dict

def calculate_ss_metrics(model_attrs: ModelAttributes, datahandler: DataloaderHandler, thresh_type="mcc"):
    with open(os.path.join(model_attrs.outputs_save_path, f"thresholds_ss_{thresh_type}.pkl"), "rb") as f:
        threshold_dict = pickle.load(f)
    # print(np.array(list(threshold_dict.values())).mean(0))
    metrics_dict_list = {}
    thresh = np.array([threshold_dict[k] for k in SS_CATEGORIES[1:]])
    
    for outer_i in range(5):
        _,_,_, y_test = datahandler.get_swissprot_ss_xy(model_attrs.outputs_save_path, outer_i)
        y_test_preds = pickle.load(open(f"{model_attrs.outputs_save_path}/ss_{outer_i}.pkl", "rb"))
        metrics_dict = calculate_ss_metrics_fold(y_test, y_test_preds, thresh)
        for k in metrics_dict:
            metrics_dict_list.setdefault(k, []).append(metrics_dict[k])

    output_dict = {}
    for k in metrics_dict_list:
        output_dict[k] = [f"{round(np.array(metrics_dict_list[k]).mean(), 2):.2f} pm {round(np.array(metrics_dict_list[k]).std(), 2):.2f}"]
    print(pd.DataFrame(output_dict).to_latex())
