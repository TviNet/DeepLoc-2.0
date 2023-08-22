# DeepLoc 2.0

Multi-label subcellular localization and sorting signal prediction based on protein foundation models (https://github.com/agemagician/ProtTrans, https://github.com/facebookresearch/esm).

Prediction webserver is available at https://services.healthtech.dtu.dk/services/DeepLoc-2.0/

More details can be found in the paper https://academic.oup.com/nar/article/50/W1/W228/6576357


## Data
The 'data_files' folder contains the data for training
1. multisub_5_partitions_unique.csv: Annotated SwissProt Sequences, labels, and partitions for subcellular localization
2. multisub_ninesignals.pkl, sorting_signals.csv: Annotated SwissProt Sequences and sorting signal annotations
3. Processed FASTA files for generating embeddings

## Models
Two models dubbed Fast (ESM1b) and Accurate (ProtT5) are used. `<MODEL-TYPE>` refers to one of these. 

## Setup

It is recommened to setup a conda environment using

`conda env create -f environment.yml`

## Training Workflow

Training is divided into two stages:

`python train_sl.py --model <MODEL-TYPE>`
1. Generate and store embeddings for faster training. Note: h5 files of ~30-40 GB are stored in "data_files/embeddings".
2. Train subcellular localization and interpretable attention.
3. Generate predictions and intermediate representations for sorting signal prediction.
4. Compute metrics on the SwissProt CV dataset.


`python train_ss.py --model <MODEL-TYPE>`
1. Train sorting signal prediction
2. Predict and compute metrrics on the SwissProt CV dataset.

## Citation

If you found this useful please consider citing

```
@article{deeploc22022,
    author = {\textbf{Vineet Thumuluri}* and Almagro Armenteros*, José Juan and Johansen, Alexander Rosenberg and Nielsen, Henrik and Winther, Ole},
    title = "{DeepLoc 2.0: multi-label subcellular localization prediction using protein language models}",
    journal = {Nucleic Acids Research},
    year = {2022},
    month = {04},
    issn = {0305-1048},
    doi = {10.1093/nar/gkac278},
    url = {https://doi.org/10.1093/nar/gkac278},
    eprint = {https://academic.oup.com/nar/advance-article-pdf/doi/10.1093/nar/gkac278/43515314/gkac278.pdf},
}
```
