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
        