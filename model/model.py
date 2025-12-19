import torch
import  torch.nn as nn 
from torchvision import models
import sys 
sys.path.append('../')
from dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg_backbone = models.vgg16(weights=True)
vgg_backbone.classifier = nn.Sequential()

# freeze paramter 
for param in vgg_backbone.parameters():
    param.requires_grad = False
vgg_backbone.eval().to(device)

class RCNN(nn.Module):
    def __init__(self, num_classes):
        super(RCNN, self).__init__()
        feature_dim = 25088
        self.backbone = vgg_backbone
        self.cls_score = nn.Linear(feature_dim, num_classes)
        self.bbox = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Tanh()
        )
        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.L1Loss()
    def forward(self,input):
        features = self.backbone(input)
        cls_scores = self.cls_score(features)
        bbox_reg = self.bbox(features)
        return cls_scores, bbox_reg
    
    def calc_loss(self, probs, _deltas, labels, deltas):
        detection_loss = self.cel(probs, labels)
        # calcute loss for detection classes exclude the background 
        idx, = torch.where(labels !=0)
        _deltas = _deltas[idx]
        deltas = deltas[idx]
        self.lmb = 10.0
        if len(idx) > 0 :
            regression_loss = self.sl1(_deltas, deltas)
            return detection_loss + self.lmb *\
            regression_loss , detection_loss.detach(), regression_loss.detach()
        else : 
            regression_loss = 0
            return detection_loss + self.lmb * regression_loss, \
            detection_loss.detach(), regression_loss.detach()
        
def train_batch(inputs, model, optimizer,  criterion):
    input, clss, deltas = inputs
    model.train()
    optimizer.zero_grad()
    _clss, _deltas = model(input)
    loss, loc_loss, regr_loss = criterion(_clss,_deltas, clss, deltas)
    accs = clss = decode(_clss)
    loss.backward()
    optimizer.step()
    return loss.detach(), loc_loss, regr_loss, accs.cpu().numpy()

@torch.no_grad()
def validate_batch(inputs, model, citerion):
    input, clss, deltas = inputs 
    with torch.no_grad():
        model.eval()
        _clss, _deltas = model(input)
        loss, loc_loss, regr_loss = citerion(_clss, _deltas, clss, deltas)
        _, _clss = _clss.max(-1)
        accs = clss == _clss
        return _clss, _deltas, loss.detach() , \
            loc_loss, regr_loss, accs.cpu().numpy()
    