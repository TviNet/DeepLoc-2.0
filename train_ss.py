from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
import argparse
from src.constants import * 

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import ProgressBar
import pytorch_lightning as pl
from src.model import *
from src.data import *
from src.utils import *
from src.eval_utils import *


def train_model(model_attrs: ModelAttributes, datahandler:DataloaderHandler, outer_i: int):
    train_dataloader, val_dataloader = datahandler.get_ss_train_val_dataloader(model_attrs.outputs_save_path, outer_i)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=model_attrs.ss_save_path,
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
                        default_root_dir=model_attrs.ss_save_path + f"/{outer_i}",
                        check_val_every_n_epoch = 1,
                        callbacks=[checkpoint_callback, early_stopping_callback],
                        )
                        # precision=16,
                        # gpus=1)
                        #tpu_cores=8)
    clf = SignalTypeMLP()
    print(f"Training clf {outer_i}")
    trainer.fit(clf, train_dataloader, val_dataloader)
        
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

    model_attrs = get_train_model_attributes(model_type=args.model)
    datahandler = DataloaderHandler(
        clip_len=model_attrs.clip_len, 
        alphabet=model_attrs.alphabet, 
        embedding_file=model_attrs.embedding_file,
        embed_len=model_attrs.embed_len
    )

    print("Training sorting signal type prediction models")
    for i in range(0, 5):
        print(f"Training model {i+1} / 5")
        if not os.path.exists(os.path.join(model_attrs.save_path, f"signaltype/{i}.ckpt")):
            train_model(model_attrs, datahandler, i)
    
    print("Finished training sorting signal type prediction models")

    print("Using trained models to generate outputs of signal prediction")
    generate_ss_outputs(model_attrs=model_attrs, datahandler=datahandler)
    print("Generated outputs!")

    print("Computing sorting signal type prediction performance on swissprot CV dataset")
    calculate_ss_metrics(model_attrs=model_attrs, datahandler=datahandler)


    