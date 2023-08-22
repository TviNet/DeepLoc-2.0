from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.data import *
from src.utils import *
from src.eval_utils import *
from src.embedding import *
from src.metrics import *
import argparse
import subprocess
import os
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

def train_model(model_attrs: ModelAttributes, datahandler:DataloaderHandler, outer_i: int):
    train_dataloader, val_dataloader = datahandler.get_train_val_dataloaders(outer_i)

    checkpoint_callback = ModelCheckpoint(
        monitor='bce_loss',
        dirpath=model_attrs.save_path,
        filename= f"{outer_i}_1Layer",
        save_top_k=1,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=True
    )

    early_stopping_callback = EarlyStopping(
         monitor='bce_loss',
         patience=5, 
         mode='min'
    )

    # Initialize trainer
    trainer = pl.Trainer(max_epochs=14, 
                        default_root_dir=model_attrs.save_path + f"/{outer_i}_1Layer",
                        check_val_every_n_epoch = 1,
                        callbacks=[
                            checkpoint_callback, 
                            early_stopping_callback
                        ],
                        precision=16,
                        accelerator="auto")
    clf = model_attrs.class_type()
    trainer.fit(clf, train_dataloader, val_dataloader)
    return trainer

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
    if not os.path.exists(model_attrs.embedding_file):
        print("Embeddings not found, generating......")
        generate_embeddings(model_attrs)
        print("Embeddings created!")
    else:
        print("Using existing embeddings")
    
    if not os.path.exists(model_attrs.embedding_file):
        raise Exception("Embeddings could not be created. Verify that data_files/embeddings/<MODEL_DATASET> is deleted")

    datahandler = DataloaderHandler(
        clip_len=model_attrs.clip_len, 
        alphabet=model_attrs.alphabet, 
        embedding_file=model_attrs.embedding_file,
        embed_len=model_attrs.embed_len
    )
    print("Training subcellular localization models")
    for i in range(0, 5):
        print(f"Training model {i+1} / 5")
        if not os.path.exists(os.path.join(model_attrs.save_path, f"{i}_1Layer.ckpt")):
            train_model(model_attrs, datahandler, i)
    print("Finished training subcellular localization models")

    print("Using trained models to generate outputs for signal prediction training")
    generate_sl_outputs(model_attrs=model_attrs, datahandler=datahandler)
    print("Generated outputs! Can train sorting signal prediction now")


    print("Computing subcellular localization performance on swissprot CV dataset")
    calculate_sl_metrics(model_attrs=model_attrs, datahandler=datahandler)