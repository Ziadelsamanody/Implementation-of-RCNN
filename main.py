from dataset import * 
import torch 
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim 
from model import * 
n_train = 9*len(FPATHS) // 10 
train_ds = RCNNDataset(FPATHS[:n_train], ROIS[:n_train], CLSS[:n_train], DELTAS[:n_train], GTBBS[:n_train])

test_ds = RCNNDataset(FPATHS[n_train:], ROIS[n_train:], CLSS[n_train:], DELTAS[n_train:], GTBBS[n_train:])

# data loaders 
train_loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn, drop_last=True)
test_loader  = DataLoader(test_ds, batch_size=2 ,collate_fn=test_ds.collate_fn, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

rcnn = RCNN(num_classes=len(label2target)).to(device)
criterion = rcnn.calc_loss
optimizer = optim.SGD(rcnn.parameters(), lr=1e-3)

epochs = 5 
log = Report(n_epochs=epochs)

for epoch in range(epochs):
    _n = len(train_loader)
    for idx , inputs in enumerate(train_loader):
        loss, loc_loss, regr_loss, accs = train_batch(inputs, rcnn, optimizer, criterion)
        pos = (epoch + (idx + 1) / _n)
        log.record(pos, trn_loss =loss.item(), trn_loc_loss=loc_loss, trn_regr_loss=regr_loss, trn_acc=accs.mean(), end='\r')
    _n = len(test_loader)
    for idx , inputs in enumerate(test_loader):
        _clss, _deltas, loss, loc_loss, regr_loss , accs = validate_batch(inputs, rcnn, criterion)
        pos = (epoch + (idx + 1) / _n)
        log.record(
            pos,
            val_loss = loss.item(),
            val_loc_loss = loc_loss,
            val_regr_loss = regr_loss,
            val_acc = accs.mean(), end='\r'
        )
# plot training and validation metrics 
log.plot_epochs('trn_loss, val_loss'.split(','))
