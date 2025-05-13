import torch
import torch.nn as nn
import torch.nn.functional as F
from Dropblock import DropBlock1D, DropBlock2D 

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels=2, 
                             out_channels=1, 
                             kernel_size=kernel_size,
                             padding=(kernel_size-1)//2,  
                             bias=False)
        # Initialize with he_normal (kaiming_normal in PyTorch)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        # Get input shape
        batch_size, channels, height, width = x.size()
        
        # Channel first is default in PyTorch, so we don't need to permute here
        
        # Average pool across channel dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        # Max pool across channel dimension
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along the channel dimension
        concat = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply convolution with sigmoid activation
        attention = self.conv(concat)
        attention = torch.sigmoid(attention)
        
        # Apply attention to input feature
        return x * attention


def spatial_attention(input_feature, kernel_size=7):
    """
    Functional interface for the spatial attention module.
    
    Args:
        input_feature: Input tensor of shape [batch_size, channels, height, width]
        kernel_size: Size of the convolutional kernel
        
    Returns:
        Tensor with spatial attention applied
    """
    # Get input shape
    batch_size, channels, height, width = input_feature.size()
    
    # Average pool across channel dimension
    avg_pool = torch.mean(input_feature, dim=1, keepdim=True)
    # Max pool across channel dimension
    max_pool, _ = torch.max(input_feature, dim=1, keepdim=True)
    
    # Concatenate along the channel dimension
    concat = torch.cat([avg_pool, max_pool], dim=1)
    
    # Apply convolution with sigmoid activation
    conv = nn.Conv2d(in_channels=2, 
                    out_channels=1, 
                    kernel_size=kernel_size,
                    padding=(kernel_size-1)//2,  # 'same' padding
                    bias=False)
    
    # Initialize with he_normal (kaiming_normal in PyTorch)
    nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
    
    # Move conv to the same device as input
    conv = conv.to(input_feature.device)
    
    attention = torch.sigmoid(conv(concat))
    
    # Apply attention to input feature
    return input_feature * attention
