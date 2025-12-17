from dataset import * 
import torch 
from torch.utils.data import TensorDataset, DataLoader

n_train = 9*len(FPATHS) // 10 
train_ds = RCNNDataset(FPATHS[:n_train], ROIS[:n_train], CLSS[:n_train], DELTAS[:n_train], GTBBS[:n_train])

test_ds = RCNNDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:])

# data loaders 
train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True)
test_ds = DataLoader(test_ds, batch_size=2 ,collate_fn=test_ds.collate_fn, drop_last=True)
