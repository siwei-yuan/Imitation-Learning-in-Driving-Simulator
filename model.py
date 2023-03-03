import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
import torchvision.models as models

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

#####################################################
# Naive ResNet
#####################################################
class Resnet(nn.Module):
    def __init__(self, mode='linear',pretrained=True):
        super().__init__()
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

    def forward(self, x):
        x = self.resnet(x)
        return x
    
    def to(self,device):
        return self.resnet.to(device=device)
    
#####################################################
# ResNet for classification task
#####################################################
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

    def forward(self, x):

        return self.resnet(x)
    
    def to(self,device):
        return self.resnet.to(device=device)
    

#####################################################
# ViT
#####################################################
def tokenize_image(img, patch_size=16, stride=16):
    """
    Tokenize an image into non-overlapping image patches.
    Args:
        img (torch.Tensor): The input image with shape (C, H, W).
        patch_size (int): The size of each patch.
        stride (int): The stride of the sliding window.
    Returns:
        patches (torch.Tensor): The tokenized patches with shape (N, patch_size*patch_size*C).
    """
    
    C, H, W = img.shape
    patches = []
    for i in range(0, H, stride):
      for j in range(0, W, stride):
        patch = torch.permute(img[:, i:i+patch_size, j:j+patch_size], (1, 2, 0))
        patches.append(torch.flatten(patch).tolist())
    patches = torch.Tensor(patches)
    return patches


class Tokenization_layer(nn.Module):
  def __init__(self, dim, patch_dim,patch_height, patch_width):
    super().__init__()
    self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
    self.norm1 = torch.nn.LayerNorm(patch_dim)
    self.fc1 = torch.nn.Linear(patch_dim, dim)
    self.norm2 = torch.nn.LayerNorm(dim)


  def forward(self, x):
    out = self.to_patch(x)
    out = self.norm1(out)
    out = self.fc1(out)
    out = self.norm2(out)
    return out

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.heads = heads
        
        self.inner_dim = dim_head *  heads    
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias = False)

        self.dim_head = dim_head
        self.final_linear = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        out = None
        
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #                                     2, 196, (8 * 64 =512). -> 2, 8, 196, 64
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        qk = torch.matmul(q, k.transpose(-1, -2))
        qk = qk/(self.dim_head ** 0.5)
        qk = torch.nn.functional.softmax(qk, dim=-1)
        qk = self.dropout(qk)
        # (2, 8, 196, 196)
        attention = torch.matmul(qk, v)
        # (2, 8, 196, 64)
        attention = rearrange(attention, 'b h n d -> b n (h d)')
        attention = self.final_linear(attention)
        out = self.dropout(attention)
        
        return out
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        # keey the resiual connection here
        return self.fn(self.norm(x), **kwargs)+x
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.activation = nn.GELU()
        self.fc1 = nn.Linear(dim, mlp_dim) 
        self.fc2 = nn.Linear(mlp_dim, dim) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        
        return out

class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads, dim_head, dropout))
        self.ff = PreNorm(dim, PositionwiseFeedForward(dim, mlp_dim, dropout))
        
    def forward(self, x):
        res1 = self.attn(x)
        res2 = self.ff(res1)
        return res2

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    def __init__(self, *, 
                 image_size,
                 patch_size, 
                 num_classes, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 pool = 'cls', 
                 channels = 3, 
                 dim_head = 64, 
                 dropout = 0., 
                 emb_dropout = 0.,
                ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = 0
        patch_dim = 0
        
        num_patches = int((image_height/patch_height) * (image_width/patch_width))
        patch_dim = patch_height * patch_width * channels
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.transformers = nn.ModuleList([])

        self.to_path_embedding = Tokenization_layer(dim, patch_dim, patch_height, patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(num_patches+1, dim), True)
        self.cls_token = nn.Parameter(torch.randn(dim), True)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformers = nn.ModuleList([Transformer(dim, heads, dim_head, mlp_dim, dropout) for i in range(depth)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )


        
    def forward(self, img):
        # (batch_size,C,H,W)
        out = self.to_path_embedding(img)
        # (batch_size, num_of_patches, dim_of_each_patch)
        cls_token_repeated = self.cls_token.repeat((out.shape[0], 1, 1))
        out = torch.cat((out, cls_token_repeated), 1)
        out += self.pos_embedding
        out = self.dropout(out)
        # (batch_size,N+1,dim)
        for each_transformer_block in self.transformers:
          out = each_transformer_block(out)

        # (batch_size,N+1,dim)
        if self.pool == 'cls':
          out = torch.squeeze(out[:, -1, :])
        else:
          out = torch.mean(out, 1, True)
          out = torch.squeeze(out)
        # (batch_size,dim)
        out = self.mlp_head(out)

        return out

class ViT_with_speed(nn.Module):
    "Implements Vision Transfromer"
    def __init__(self, *, 
                 image_size,
                 patch_size, 
                 num_classes, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 pool = 'cls', 
                 channels = 3, 
                 dim_head = 64, 
                 dropout = 0., 
                 emb_dropout = 0.,
                ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = 0
        patch_dim = 0
        
        
        num_patches = int((image_height/patch_height) * (image_width/patch_width))
        patch_dim = patch_height * patch_width * channels
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.transformers = nn.ModuleList([])

        self.to_path_embedding = Tokenization_layer(dim, patch_dim, patch_height, patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(num_patches+3, dim), True)
        self.cls_token = nn.Parameter(torch.randn((3, dim)), True)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformers = nn.ModuleList([Transformer(dim, heads, dim_head, mlp_dim, dropout) for i in range(depth)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(3*dim),
            nn.Linear(3*dim, 128),
            nn.Linear(128, num_classes)
        )


        
    def forward(self, img):
        # (batch_size,C,H,W)
        out = self.to_path_embedding(img)
        # (batch_size, num_of_patches, dim_of_each_patch)
        cls_token_repeated = self.cls_token.repeat((out.shape[0], 1, 1))

        # IMPORTANT: REPEATED CLS TOKEN 3 TIMES TO PREDICT STEERING, ACCEL, SPEED
        out = torch.cat((out, cls_token_repeated), 1)
        
        out += self.pos_embedding
        out = self.dropout(out)
        # (batch_size,N+1,dim)
        for each_transformer_block in self.transformers:
          out = each_transformer_block(out)

        # (batch_size,N+1,dim)
        speed = torch.squeeze(out[:, -1, :])
        accel = torch.squeeze(out[:, -2, :])
        steering = torch.squeeze(out[:, -3, :])

        res = torch.cat((steering, accel, speed), dim=-1)

        # (batch_size,dim)
        action = self.mlp_head(res)
    
        return action