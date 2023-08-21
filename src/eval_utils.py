
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import hamming_loss, matthews_corrcoef, confusion_matrix
from sklearn import metrics
import torch
from src.utils import ModelAttributes
from src.data import DataloaderHandler
import os

if torch.cuda.is_available():
    device = "cuda" #torch.device("cuda")
elif torch.backends.mps.is_available():
    device = "cpu" #torch.device("mps")
else:
    device = "cpu" #torch.device("cpu")

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

def predict_values(dataloader, model):
    output_dict = {}
    annot_dict = {}
    pool_dict = {}
    with torch.no_grad():
      for i, (toks, lengths, np_mask, targets, targets_seq, labels) in enumerate(dataloader):
        with torch.autocast(device_type=device,dtype=torch.bfloat16):
            y_pred, y_pool, y_attn = model.predict(toks.to("cuda"), lengths.to("cuda"), np_mask.to("cuda"))
        x = torch.sigmoid(y_pred).cpu().numpy()
        for j in range(len(labels)):
            if len(labels) == 1:
                output_dict[labels[j]] = x
                pool_dict[labels[j]] = y_pool.cpu().numpy()
                annot_dict[labels[j]] = y_attn[:lengths[j]].cpu().numpy()
            else:
                output_dict[labels[j]] = x[j]
                pool_dict[labels[j]] = y_pool[j].cpu().numpy()
                annot_dict[labels[j]] = y_attn[j,:lengths[j]].cpu().numpy()

    output_df = pd.DataFrame(output_dict.items(), columns=['ACC', 'preds'])
    annot_df = pd.DataFrame(annot_dict.items(), columns=['ACC', 'pred_annot'])
    pool_df = pd.DataFrame(pool_dict.items(), columns=['ACC', 'embeds'])
    return output_df.merge(annot_df).merge(pool_df)
    
def generate_outputs(
        model_attrs: ModelAttributes, 
        datahandler: DataloaderHandler, 
        thresh_type="mcc", 
        inner_i="1_layer", 
        reuse=False):
    
    threshold_dict = {}
    if not os.path.exists(f"{model_attrs.outputs_save_path}"):
        os.makedirs(f"{model_attrs.outputs_save_path}")
        
    for outer_i in range(5):
        print("Generating output for ensemble model", outer_i)
        dataloader, data_df = datahandler.get_partition_dataloader_inner(outer_i)
        if not reuse:
            path = f"{model_attrs.save_path}/{outer_i}_{inner_i}.ckpt"
            model = model_attrs.class_type.load_from_checkpoint(path).to(device).eval()
            pred_df = predict_values(dataloader, model)
            pred_df.to_pickle(os.path.join(model_attrs.outputs_save_path, f"inner_{outer_i}_{inner_i}.pkl"))
        else:
            pred_df = pd.read_pickle(os.path.join(model_attrs.outputs_save_path, f"inner_{outer_i}_{inner_i}.pkl"))

        if thresh_type == "roc":
            thresholds = get_optimal_threshold(pred_df, data_df)
        elif thresh_type == "pr":
            thresholds = get_optimal_threshold_pr(pred_df, data_df)
        else:
            thresholds = get_optimal_threshold_mcc(pred_df, data_df)
        threshold_dict[f"{outer_i}_{inner_i}"] = thresholds

        if not reuse:
            dataloader, data_df = datahandler.get_partition_dataloader(outer_i)
            output_df = predict_values(dataloader, model)
            output_df.to_pickle(os.path.join(model_attrs.outputs_save_path, f"{outer_i}_{inner_i}.pkl"))

    with open(os.path.join(model_attrs.outputs_save_path, f"thresholds_sl_{thresh_type}.pkl"), "wb") as f:
        pickle.dump(threshold_dict, f)


