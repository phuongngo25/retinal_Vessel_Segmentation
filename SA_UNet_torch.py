import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from SA_torch import SpatialAttention
from Dropblock_torch import * 


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block_size=7, keep_prob=0.9):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dropblock1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropblock2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropblock1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.dropblock2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class Backbone(nn.Module):
    def __init__(self, input_channels=3, block_size=7, keep_prob=0.9, start_neurons=16):
        super(Backbone, self).__init__()
        
        # Encoder
        self.conv1 = ConvBlock(input_channels, start_neurons, block_size, keep_prob)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = ConvBlock(start_neurons, start_neurons*2, block_size, keep_prob)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = ConvBlock(start_neurons*2, start_neurons*4, block_size, keep_prob)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Middle
        self.convm = ConvBlock(start_neurons*4, start_neurons*8, block_size, keep_prob)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(start_neurons*8, start_neurons*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv3 = ConvBlock(start_neurons*8, start_neurons*4, block_size, keep_prob)  # *8 because of concat
        
        self.upconv2 = nn.ConvTranspose2d(start_neurons*4, start_neurons*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv2 = ConvBlock(start_neurons*4, start_neurons*2, block_size, keep_prob)  # *4 because of concat
        
        self.upconv1 = nn.ConvTranspose2d(start_neurons*2, start_neurons, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv1 = ConvBlock(start_neurons*2, start_neurons, block_size, keep_prob)  # *2 because of concat
        
        # Output
        self.outconv = nn.Conv2d(start_neurons, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        # Middle
        convm = self.convm(pool3)
        
        # Decoder
        deconv3 = self.upconv3(convm)
        concat3 = torch.cat([deconv3, conv3], dim=1)
        uconv3 = self.uconv3(concat3)
        
        deconv2 = self.upconv2(uconv3)
        concat2 = torch.cat([deconv2, conv2], dim=1)
        uconv2 = self.uconv2(concat2)
        
        deconv1 = self.upconv1(uconv2)
        concat1 = torch.cat([deconv1, conv1], dim=1)
        uconv1 = self.uconv1(concat1)
        
        # Output
        output = self.outconv(uconv1)
        output = self.sigmoid(output)
        
        return output


class SA_UNet(nn.Module):
    def __init__(self, input_channels=3, block_size=7, keep_prob=0.9, start_neurons=16):
        super(SA_UNet, self).__init__()
        
        # Encoder
        self.conv1 = ConvBlock(input_channels, start_neurons, block_size, keep_prob)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = ConvBlock(start_neurons, start_neurons*2, block_size, keep_prob)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = ConvBlock(start_neurons*2, start_neurons*4, block_size, keep_prob)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Middle with Spatial Attention
        self.conv_m1 = nn.Conv2d(start_neurons*4, start_neurons*8, kernel_size=3, padding=1)
        self.dropblock_m1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn_m1 = nn.BatchNorm2d(start_neurons*8)
        self.relu_m1 = nn.ReLU(inplace=True)
        
        # Spatial Attention module
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        self.conv_m2 = nn.Conv2d(start_neurons*8, start_neurons*8, kernel_size=3, padding=1)
        self.dropblock_m2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn_m2 = nn.BatchNorm2d(start_neurons*8)
        self.relu_m2 = nn.ReLU(inplace=True)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(start_neurons*8, start_neurons*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv3 = ConvBlock(start_neurons*8, start_neurons*4, block_size, keep_prob)  # *8 because of concat
        
        self.upconv2 = nn.ConvTranspose2d(start_neurons*4, start_neurons*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv2 = ConvBlock(start_neurons*4, start_neurons*2, block_size, keep_prob)  # *4 because of concat
        
        self.upconv1 = nn.ConvTranspose2d(start_neurons*2, start_neurons, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv1 = ConvBlock(start_neurons*2, start_neurons, block_size, keep_prob)  # *2 because of concat
        
        # Output
        self.outconv = nn.Conv2d(start_neurons, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        # Middle with Spatial Attention
        convm = self.conv_m1(pool3)
        convm = self.dropblock_m1(convm)
        convm = self.bn_m1(convm)
        convm = self.relu_m1(convm)
        
        # Apply spatial attention
        convm = self.spatial_attention(convm)
        
        convm = self.conv_m2(convm)
        convm = self.dropblock_m2(convm)
        convm = self.bn_m2(convm)
        convm = self.relu_m2(convm)
        
        # Decoder
        deconv3 = self.upconv3(convm)
        concat3 = torch.cat([deconv3, conv3], dim=1)
        uconv3 = self.uconv3(concat3)
        
        deconv2 = self.upconv2(uconv3)
        concat2 = torch.cat([deconv2, conv2], dim=1)
        uconv2 = self.uconv2(concat2)
        
        deconv1 = self.upconv1(uconv2)
        concat1 = torch.cat([deconv1, conv1], dim=1)
        uconv1 = self.uconv1(concat1)
        
        # Output
        output = self.outconv(uconv1)
        output = self.sigmoid(output)
        
        return output


def create_backbone(input_size=(3, 512, 512), block_size=7, keep_prob=0.9, start_neurons=16, lr=1e-3):
    """
    Factory function to create the Backbone model
    
    Args:
        input_size: Input size as (channels, height, width)
        block_size: Size of the blocks for DropBlock
        keep_prob: Keep probability for DropBlock
        start_neurons: Number of filters to start with
        lr: Learning rate for the optimizer
        
    Returns:
        model: The Backbone model
    """
    model = Backbone(
        input_channels=input_size[0],
        block_size=block_size,
        keep_prob=keep_prob,
        start_neurons=start_neurons
    )
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    return model, criterion, optimizer


def create_sa_unet(input_size=(3, 512, 512), block_size=7, keep_prob=0.9, start_neurons=16, lr=1e-3):
    """
    Factory function to create the SA-UNet model
    
    Args:
        input_size: Input size as (channels, height, width)
        block_size: Size of the blocks for DropBlock
        keep_prob: Keep probability for DropBlock
        start_neurons: Number of filters to start with
        lr: Learning rate for the optimizer
        
    Returns:
        model: The SA-UNet model
    """
    model = SA_UNet(
        input_channels=input_size[0],
        block_size=block_size,
        keep_prob=keep_prob,
        start_neurons=start_neurons
    )
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    return model, criterion, optimizer
