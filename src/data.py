import pickle
import torch
from Bio import SeqIO
import re
import pandas as pd

class FastaBatchedDatasetTorch(torch.utils.data.Dataset):
    def __init__(self, data_df):
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)
    
    def shuffle(self):
        self.data_df = self.data_df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, idx):
        return self.data_df["Sequence"][idx], self.data_df["ACC"][idx]

    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.data_df["Sequence"])]
        sizes.sort(reverse=True)
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        start = 0
        #start = random.randint(0, len(sizes))
        for j in range(len(sizes)):
            i = (start + j) % len(sizes)
            sz = sizes[i][0]
            idx = sizes[i][1]    
            sz += extra_toks_per_seq
            if (max(sz, max_len) * (len(buf) + 1) > toks_per_batch):
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(idx)

        _flush_current_buf()
        return batches

class BatchConverterProtT5(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        #print(len(raw_batch[0]), raw_batch[1], raw_batch[2])
        max_len = max(len(seq_str) for seq_str, _ in raw_batch)
        labels = []
        lengths = []
        strs = []
        for i, (seq_str, label) in enumerate(raw_batch):
            #seq_str = seq_str[1:]
            labels.append(label)
            lengths.append(len(seq_str))
            strs.append(seq_str)
        
        proteins = [" ".join(list(item)) for item in strs]
        proteins = [re.sub(r"[UZOB]", "X", sequence) for sequence in proteins]
        ids = self.alphabet.batch_encode_plus(proteins, add_special_tokens=True, padding=True)
        non_pad_mask = torch.tensor(ids['input_ids']) > -100 # B, T

        return ids, torch.tensor(lengths), non_pad_mask, labels


class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        #print(len(raw_batch[0]), raw_batch[1], raw_batch[2])
        max_len = max(len(seq_str) for seq_str, _ in raw_batch)
        tokens = torch.empty((batch_size, max_len + int(self.alphabet.prepend_bos) + \
            int(self.alphabet.append_eos)), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)
        labels = []
        lengths = []
        strs = []
        for i, (seq_str, label) in enumerate(raw_batch):
            #seq_str = seq_str[1:]
            labels.append(label)
            lengths.append(len(seq_str))
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = torch.tensor([self.alphabet.get_idx(s) for s in seq_str], dtype=torch.int64)
            tokens[i, int(self.alphabet.prepend_bos) : len(seq_str) + int(self.alphabet.prepend_bos)] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_str) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
        
        non_pad_mask = ~tokens.eq(self.alphabet.padding_idx) &\
         ~tokens.eq(self.alphabet.cls_idx) &\
         ~tokens.eq(self.alphabet.eos_idx)# B, T

        return tokens, torch.tensor(lengths), non_pad_mask, labels

def read_fasta(fastafile):
    """Parse a file with sequences in FASTA format and store in a dict"""
    proteins = list(SeqIO.parse(fastafile, "fasta"))
    res = {}
    for prot in proteins:
        res[str(prot.id)] = str(prot.seq)
    return res

# with open("/tools/src/deeploc-2.0/models/ESM1b_alphabet.pkl", "rb") as f:
#     alphabet = pickle.load(f)

###################################
#######   TRAINING STUFF  #########
###################################

import h5py
import numpy as np
import pickle5
from sklearn.model_selection import ShuffleSplit
from src.constants import *

def get_swissprot_df(clip_len):  
    with open(SIGNAL_DATA, "rb") as f:
        annot_df = pickle5.load(f)
    nes_exclude_list = ['Q7TPV4','P47973','P38398','P38861','Q16665','O15392','Q9Y8G3','O14746','P13350','Q06142']
    swissprot_exclusion_list = ['Q04656-5   ','O43157','Q9UPN3-2']
    def clip_middle_np(x):
        if len(x)>clip_len:
            x = np.concatenate((x[:clip_len//2],x[-clip_len//2:]), axis=0)
        return x
    def clip_middle(x):
      if len(x)>clip_len:
          x = x[:clip_len//2] + x[-clip_len//2:]
      return x
 
    annot_df["TargetAnnot"] = annot_df["ANNOT"].apply(lambda x: clip_middle_np(x))
    data_df = pd.read_csv(LOCALIZATION_DATA)
    data_df["Sequence"] = data_df["Sequence"].apply(lambda x: clip_middle(x))
    data_df["Target"] = data_df[CATEGORIES].values.tolist()    

    annot_df = annot_df[~annot_df.ACC.isin(nes_exclude_list)].reset_index(drop=True)
    data_df = data_df[~data_df.ACC.isin(swissprot_exclusion_list)].reset_index(drop=True)
    data_df = data_df.merge(annot_df[["ACC", "ANNOT", "Types", "TargetAnnot"]], on="ACC", how="left")
    data_df['TargetAnnot'] = data_df['TargetAnnot'].fillna(0)

    # embedding_fasta = read_fasta(f"{embedding_path}/remapped_sequences_file.fasta")
    # embedding_df = pd.DataFrame(embedding_fasta.items(), columns=["details", "RawSeq"])
    # embedding_df["Hash"] = embedding_df.details.apply(lambda x: x.split()[0])
    # embedding_df["ACC"] = embedding_df.details.apply(lambda x: x.split()[1])
    # data_df = data_df.merge(embedding_df[["ACC", "Hash"]]).reset_index(drop=True)

    return data_df

class EmbeddingsLocalizationDataset(torch.utils.data.Dataset):
    """
    Dataset of protein embeddings and the corresponding subcellular localization label.
    """

    def __init__(self, embedding_file, data_df) -> None:
        super().__init__()
        self.data_df = data_df
        self.embeddings_file = h5py.File(embedding_file, "r")
    
    def __getitem__(self, index: int):
        embedding = self.embeddings_file[self.data_df["ACC"][index]]
        print(self.data_df["ACC"][index], embedding.shape, len(self.data_df["Sequence"][index]))
        return self.data_df["Sequence"][index], embedding, self.data_df["Target"][index], self.data_df["TargetAnnot"][index], self.data_df["ACC"][index]
    
    def get_batch_indices(self, toks_per_batch, max_batch_size, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.data_df["Sequence"])]
        sizes.sort(reverse=True)
        batches = []
        buf = []
        max_len = 0

        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        start = 0
        #start = random.randint(0, len(sizes))
        for j in range(len(sizes)):
            i = (start + j) % len(sizes)
            sz = sizes[i][0]
            idx = sizes[i][1]    
            sz += extra_toks_per_seq
            if (max(sz, max_len) * (len(buf) + 1) > toks_per_batch) or len(buf) >= max_batch_size:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(idx)

        _flush_current_buf()
        return batches

    def __len__(self) -> int:
        return len(self.data_df)

class TrainBatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch):
        batch_size = len(raw_batch)
        max_len = max(len(seq_str) for seq_str, _, _, _, _ in raw_batch)
        embedding_tensor = torch.zeros((batch_size, max_len, 1280), dtype=torch.float32)
        np_mask = torch.zeros((batch_size, max_len))
        target_annots = torch.zeros((batch_size, max_len), dtype=torch.int64)
        labels = []
        lengths = []
        strs = []
        targets = torch.zeros((batch_size, 11), dtype=torch.float32)
        for i, (seq_str, embedding, target, target_annot, label) in enumerate(raw_batch):
            #seq_str = seq_str[1:]
            labels.append(label)
            lengths.append(len(seq_str))
            strs.append(seq_str)
            targets[i] = torch.tensor(target)
            print(len(seq_str), embedding.shape, targets[i].shape)
            embedding_tensor[i, :len(seq_str)] = torch.tensor(np.array(embedding))
            target_annots[i, :len(seq_str)] = torch.tensor(target_annot)
            np_mask[i, :len(seq_str)] = 1
        np_mask = np_mask == 1
        return embedding_tensor, torch.tensor(lengths), np_mask, targets, target_annots, labels


    
class SignalTypeDataset(torch.utils.data.Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = X
        self.y = y
    
    def __getitem__(self, index: int):
        return torch.tensor(self.X[index]).float(), torch.tensor(self.y[index]).float()

    def __len__(self):
        return self.X.shape[0]


class DataloaderHandler:
    def __init__(self, clip_len, alphabet, embedding_file) -> None:
        self.clip_len = clip_len
        self.alphabet = alphabet
        self.embedding_file = embedding_file

    def get_train_val_dataloaders(self, outer_i):
        data_df = get_swissprot_df(self.clip_len)
        
        train_df = data_df[data_df.Partition != outer_i].reset_index(drop=True)

        X = np.stack(train_df["ACC"].to_numpy())
        sss_tt = ShuffleSplit(n_splits=1, test_size=2048, random_state=0)
        
        (split_train_idx, split_val_idx) = next(sss_tt.split(X))
        split_train_df =  train_df.iloc[split_train_idx].reset_index(drop=True)
        split_val_df = train_df.iloc[split_val_idx].reset_index(drop=True)

        # print(split_train_df[CATEGORIES].mean())
        # print(split_val_df[CATEGORIES].mean())
        
        train_dataset = EmbeddingsLocalizationDataset(self.embedding_file, split_train_df)
        train_batches = train_dataset.get_batch_indices(4096*4, BATCH_SIZE, extra_toks_per_seq=0)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=TrainBatchConverter(self.alphabet), batch_sampler=train_batches)

        val_dataset = EmbeddingsLocalizationDataset(self.embedding_file, split_val_df)
        val_batches = val_dataset.get_batch_indices(4096*4, BATCH_SIZE, extra_toks_per_seq=0)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, collate_fn=TrainBatchConverter(self.alphabet), batch_sampler=val_batches)
        return train_dataloader, val_dataloader

    def get_partition(self, outer_i):
        data_df = get_swissprot_df(self.clip_len )
        test_df = data_df[data_df.Partition == outer_i].reset_index(drop=True)
        return test_df

    def get_partition_dataloader(self, outer_i):
        data_df = get_swissprot_df(self.clip_len)
        test_df = data_df[data_df.Partition == outer_i].reset_index(drop=True)
        
        test_dataset = EmbeddingsLocalizationDataset(self.embedding_file, test_df)
        test_batches = test_dataset.get_batch_indices(4096*4, BATCH_SIZE, extra_toks_per_seq=0)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=TrainBatchConverter(self.alphabet), batch_sampler=test_batches)
        return test_dataloader, test_df

    def get_partition_dataloader_inner(self, partition_i):
        data_df = get_swissprot_df(self.clip_len)
        test_df = data_df[data_df.Partition != partition_i].reset_index(drop=True)
        test_dataset = EmbeddingsLocalizationDataset(self.embedding_file, test_df)
        test_batches = test_dataset.get_batch_indices(4096*4, BATCH_SIZE, extra_toks_per_seq=0)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=TrainBatchConverter(self.alphabet), batch_sampler=test_batches)

        return test_dataloader, test_df
    
    def get_sl_train_val_dataloader(self, X, y):
        sss_tt = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
        
        (split_train_idx, split_val_idx) = next(sss_tt.split(y))
        split_train_X, split_train_y =  X[split_train_idx], y[split_train_idx]
        split_val_X, split_val_y = X[split_val_idx], y[split_val_idx]

        print(split_train_X.shape, split_train_y.shape, split_val_X.shape, split_val_y.shape)
        
        train_dataset = SignalTypeDataset(split_train_X, split_train_y)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=BATCH_SIZE,
            drop_last=True)

        val_dataset = SignalTypeDataset(split_val_X, split_val_y)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=BATCH_SIZE)
        
        return train_dataloader, val_dataloader











