import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.models as models

class Resnet(nn.Module):
    def __init__(self, mode='linear',pretrained=True):
        super().__init__()
        """
        use the resnet18 model from torchvision models. Remember to set pretrained as true
        
        mode has three options:
        1) features: to extract features only, we do not want the last fully connected layer of 
            resnet18. Use nn.Identity() to replace this layer.
        2) linear: For this model, we want to freeze resnet18 features, then train a linear 
            classifier which takes the features before FC (again we do not want 
            resnet18 FC). And then write our own FC layer: which takes in the features and 
            output scores of size 100 (because we have 100 categories).
            Because we want to freeze resnet18 features, we have to iterate through parameters()
            of our model, and manually set some parameters to requires_grad = False
            Or use other methods to freeze the features
        3) finetune: Same as 2), except that we we do not need to freeze the features and
           can finetune on the pretrained resnet model.
        """
        self.resnet = None
        self.resnet = models.resnet18(pretrained = pretrained)

        if mode == 'feature':
          self.resnet.fc = nn.Identity()
        
        if mode == 'linear':
          for param in self.resnet.parameters():
            param.requires_grad = False
          self.resnet.fc = nn.Linear(512, 2)

        if mode == 'finetune':
          for param in self.resnet.parameters():
            param.requires_grad = True
          self.resnet.fc = nn.Linear(512, 2)
    #####################################################################################

    def forward(self, x):
        x = self.resnet(x)
        return x
    
    def to(self,device):
        return self.resnet.to(device=device)
    

class Resnet_Categorize(nn.Module):
    def __init__(self, mode='linear',pretrained=True):
        super().__init__()
        self.resnet = None
        self.resnet = models.resnet18(pretrained = pretrained)

        if mode == 'feature':
          self.resnet.fc = nn.Identity()
        
        if mode == 'linear':
          for param in self.resnet.parameters():
            param.requires_grad = False
          self.resnet.fc = nn.Linear(512, 3)

        if mode == 'finetune':
          for param in self.resnet.parameters():
            param.requires_grad = True
          self.resnet.fc = nn.Linear(512, 3)
    #####################################################################################

    def forward(self, x):

        return self.resnet(x)
    
    def to(self,device):
        return self.resnet.to(device=device)