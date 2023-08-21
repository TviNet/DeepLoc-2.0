from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss, matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report
import pickle
from src.constants import *
from src.utils import *
import numpy as np
import pandas as pd
import os

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