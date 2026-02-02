import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch.optim as optim
import torchvision

class CBDetector(nn.Module):

    def __init__(self, 
                 num_classes,
                 feature_dim=64,
                 dropout=0.0,
                 num_transformer_layers=1,
                 num_transformer_heads=4, 
                 device='cuda'):
        super(CBDetector, self).__init__()

        self.dropout = dropout
        self.num_transformer_layers = num_transformer_layers
        self.num_transformer_heads = num_transformer_heads
        self.num_classes = num_classes
        self.device = device
        self.feature_extractor = ResNet(feature_dim=feature_dim)
        self.feature_dim = feature_dim

        self.sp_dropout = nn.Dropout(dropout)
        self.tm_dropout = nn.Dropout(dropout)

        self.sp_transformer = nn.Sequential(*[AttentionBlock(embed_dim=self.feature_dim, 
                                                             hidden_dim=2*self.feature_dim, 
                                                             num_heads=self.num_transformer_heads, 
                                                             dropout=self.dropout) for _ in range(self.num_transformer_layers)])
        
        self.tm_transformer = nn.Sequential(*[AttentionBlock(embed_dim=self.feature_dim*3, 
                                                             hidden_dim=2*self.feature_dim*3, 
                                                             num_heads=self.num_transformer_heads, 
                                                             dropout=self.dropout) for _ in range(self.num_transformer_layers)])

        self.sp_cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        self.sp_pos_embedding = nn.Parameter(torch.randn(1, 1+3, self.feature_dim))

        self.tm_cls_token = nn.Parameter(torch.randn(1, 1, self.feature_dim*3))
        self.tm_pos_embedding = nn.Parameter(torch.randn(1, 1+2, self.feature_dim*3))

        self.tm_compress = nn.Linear(self.feature_dim*3, self.feature_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim*2),
            nn.Linear(self.feature_dim*2, num_classes)
        )
        

    def forward(self, x):
        t_1_images = x[0]
        t_images = x[1]
        t_1_features = (self.feature_extractor(t_1_images[0]), self.feature_extractor(t_1_images[1]), self.feature_extractor(t_1_images[2]))
        t_features = (self.feature_extractor(t_images[0]), self.feature_extractor(t_images[1]), self.feature_extractor(t_images[2]))

        spatial_input = torch.stack(t_features, dim=1)
        temporal_input = torch.stack([torch.cat(t_1_features, dim=1),
                                     torch.cat(t_features, dim=1)], dim=1)
        
        batch_size, sp_seq_ln, feature_dim = spatial_input.shape

        # Add CLS token and positional encoding
        sp_cls_token = self.sp_cls_token.repeat(batch_size, 1, 1)
        spatial_x = torch.cat([sp_cls_token, spatial_input], dim=1)
        spatial_x = spatial_x + self.sp_pos_embedding[:,:3+1]

        tm_cls_token = self.tm_cls_token.repeat(batch_size, 1, 1)
        temporal_x = torch.cat([tm_cls_token, temporal_input], dim=1)
        temporal_x = temporal_x + self.tm_pos_embedding[:,:2+1]
        
        # Apply Transforrmer
        spatial_x = self.sp_dropout(spatial_x)
        spatial_x = self.sp_transformer(spatial_x)

        temporal_x = self.tm_dropout(temporal_x)
        temporal_x = self.tm_transformer(temporal_x)

        # Perform classification prediction
        sp_cls = spatial_x[:, 0]
        tm_cls = self.tm_compress(temporal_x[:, 0])
        out = self.mlp_head(torch.cat([sp_cls, tm_cls], dim=1))
        
        return out

class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout,
                                          batch_first=True) # NOTE: batch_first is True
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x

class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module):
    
    def __init__(self, image_channels=3, feature_dim=64):
        
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, feature_dim)

    def __make_layer(self, in_channels, out_channels, stride):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride), 
            Block(out_channels, out_channels)
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
       
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(out_channels)
        )
