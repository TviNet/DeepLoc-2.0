from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
from src.constants import * 

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import ProgressBar
import pytorch_lightning as pl
from src.model import *
from src.data import *

def train_model(outer_i, X, y):
    train_dataloader, val_dataloader = get_train_dataloader(X, y)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=path,
        filename= f"{outer_i}",
        save_top_k=1,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=True
    )

    early_stopping_callback = EarlyStopping(
         monitor='val_loss',
         patience=5, 
         mode='min'
    )

    # Initialize trainer
    trainer = pl.Trainer(max_epochs=500, 
                        default_root_dir=path + f"{outer_i}",
                        check_val_every_n_epoch = 1,
                        callbacks=[checkpoint_callback, early_stopping_callback],
                        progress_bar_refresh_rate=0
                        )
                        # precision=16,
                        # gpus=1)
                        #tpu_cores=8)
    clf = SignalTypeMLP()#.load_from_checkpoint(path + f"{outer_i}_{inner_i}.ckpt")
    print(f"Training clf {outer_i}")
    trainer.fit(clf, train_dataloader, val_dataloader)

def test_model(outer_i, X):
    clf = SignalTypeMLP.load_from_checkpoint(path + f"{outer_i}.ckpt")
    y_preds = torch.sigmoid(clf(torch.tensor(X).float()))
    return y_preds.detach().cpu().numpy()

def convert_to_binary(x):
    types_binary = np.zeros((len(SS_CATEGORIES)-1,))
    for c in x.split("_"):
      types_binary[SS_CATEGORIES.index(c)-1] = 1
    return types_binary

def calculate_metrics_fold(i):
    
    train_annot_pred_df = pd.read_pickle(f"outputs_prott5/inner_{i}_1Layer.pkl")
    test_annot_pred_df = pd.read_pickle(f"outputs_prott5/{i}_1Layer.pkl")
    assert train_annot_pred_df.merge(test_annot_pred_df, on="ACC").empty == True

    
    filt_annot_df = annot_df[annot_df["Types"]!=""].reset_index(drop=True)
    seq_df = filt_annot_df.merge(train_annot_pred_df)
    seq_df["Sequence"] = seq_df["Sequence"].apply(lambda x: clip_middle(x))
    seq_df["Target"] = seq_df[CATEGORIES].values.tolist()
    seq_df["TargetSignal"] = seq_df["Types"].apply(lambda x: convert_to_binary(x))

    annot_true_df = seq_df
    X_true_train, y_true_train = np.concatenate((np.stack(annot_true_df["embeds"].to_numpy()), np.stack(annot_true_df["Target"].to_numpy())), axis=1) , np.stack(annot_true_df["TargetSignal"].to_numpy())
    annot_pred_df = seq_df
    X_pred_target = np.stack(annot_true_df["preds"].to_numpy())# > threshold_dict[f"{i}_multidct"]
    X_pred_train, y_pred_train = np.concatenate((np.stack(annot_pred_df["embeds"].to_numpy()), X_pred_target), axis=1), np.stack(annot_pred_df["TargetSignal"].to_numpy())

    seq_df = filt_annot_df.merge(test_annot_pred_df)
    seq_df["Sequence"] = seq_df["Sequence"].apply(lambda x: clip_middle(x))
    seq_df["Target"] = seq_df[CATEGORIES].values.tolist()
    seq_df["TargetSignal"] = seq_df["Types"].apply(lambda x: convert_to_binary(x))

    annot_test_df = seq_df
    X_test_target = np.stack(annot_test_df["preds"].to_numpy())# > threshold_dict[f"{i}_multidct"]
    X_test, y_test = np.concatenate((np.stack(annot_test_df["embeds"].to_numpy()), X_test_target), axis=1), np.stack(annot_test_df["TargetSignal"].to_numpy())
    
    X_train = np.concatenate((X_true_train, X_pred_train), axis=0)
    y_train = np.concatenate((y_true_train, y_pred_train), axis=0)
    print(X_train.shape, X_test.shape)

    train_model(i, X_train, y_train)
    y_train_preds = test_model(i, X_train)
     
    thresh = np.zeros((9,))
    print("thresholds")
    for type_i in range(9):
        thresh[type_i] = get_best_threshold_mcc(y_train[:, type_i], y_train_preds[:, type_i])
        print(SS_CATEGORIES[i+1], thresh[type_i])

    with open(f"{modelattrs.outputs_save_path}/thresholds_ss_{thresh_type}.pkl", "wb") as f:
        pickle.dump(threshold_dict, f)
    y_test_preds = test_model(i, X_test)
    y_preds = y_test_preds > thresh

    annot_test_df["SignalPred"] = pd.Series(y_preds.tolist())

    metrics_dict = {}

    metrics_dict["microF1"] = f1_score(y_test, y_preds, average="micro")
    metrics_dict["macroF1"] = f1_score(y_test, y_preds, average="macro")
    metrics_dict["accuracy"] = (np.all((y_preds == y_test), axis=1)).mean()

    for j in range(len(SS_CATEGORIES)-1):
        metrics_dict[f"{SS_CATEGORIES[j+1]}"]  = matthews_corrcoef(y_preds[:, j],y_test[:, j])

    return metrics_dict, annot_test_df
        

def train_test_signaltype_prediction():
    metrics_dict_list = {}
    annot_test_dfs = []
    for i in range(5):
        metrics_dict, annot_test_df = calculate_metrics_fold(i)
        annot_test_dfs.append(annot_test_df)
        for k in metrics_dict:
            metrics_dict_list.setdefault(k, []).append(metrics_dict[k])

    output_dict = {}
    for k in metrics_dict_list:
        output_dict[k] = [f"{round(np.array(metrics_dict_list[k]).mean(), 2):.2f} pm {round(np.array(metrics_dict_list[k]).std(), 2):.2f}"]
    print(pd.DataFrame(output_dict).to_latex())
    #pd.concat(annot_test_dfs).to_pickle("outputs_esm12/attnreg/test_signal_preds.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m","--model", 
        default="Fast",
        choices=['Accurate', 'Fast'],
        type=str,
        help="Model to use."
    )
    args = parser.parse_args()
    model_attrs = get_train_model_attributes(model_type=model_type)
    datahandler = DataloaderHandler(alphabet=model_attrs.alphabet, embedding_file=model_attrs.embedding_file)
    generate_outputs(model_attrs, datahandler)

    for i in range(0, 5):
        train_model(args.model, i)
    