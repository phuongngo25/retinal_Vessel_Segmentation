import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock1D(nn.Module):


    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        """
        super(DropBlock1D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = 'channels_last' if data_format is None else data_format
        if self.data_format not in ['channels_first', 'channels_last']:
            raise ValueError('data_format must be either "channels_first" or "channels_last"')

    def forward(self, x, training=None):
        if training is None:
            training = self.training
        
        if not training or self.keep_prob == 1:
            return x
        
        # Handle data format
        if self.data_format == 'channels_first':
            # Convert from [B, C, L] to [B, L, C]
            x = x.permute(0, 2, 1)
        
        # Get shape
        shape = x.shape
        batch_size, seq_length, channels = shape
        
        # Compute gamma (dropout rate)
        gamma = self._get_gamma(seq_length)
        
        # Create mask
        if self.sync_channels:
            mask_shape = [batch_size, seq_length, 1]
        else:
            mask_shape = [batch_size, seq_length, channels]
            
        # Sample mask
        mask = torch.bernoulli(torch.ones(mask_shape, device=x.device) * gamma)
        
        # Apply valid seed region constraint
        valid_seed_region = self._compute_valid_seed_region(seq_length).to(x.device)
        mask = mask * valid_seed_region
        
        # Extend dropped area by max pooling
        mask = mask.unsqueeze(1)  # Add channel dim for F.max_pool2d: [B, 1, L, C]
        mask = F.max_pool2d(
            mask,
            kernel_size=(self.block_size, 1),
            stride=(1, 1),
            padding=(self.block_size // 2, 0)
        )
        mask = mask.squeeze(1)  # Remove channel dim: [B, L, C]
        mask = 1.0 - mask  # Invert mask
        
        # Apply mask and scale
        x = x * mask * (mask.numel() / mask.sum().clamp(min=1.0))
        
        # Restore original format if needed
        if self.data_format == 'channels_first':
            x = x.permute(0, 2, 1)  # Convert back to [B, C, L]
            
        return x

    def _get_gamma(self, feature_dim):
        """Get the number of activation units to drop"""
        feature_dim = float(feature_dim)
        block_size = float(self.block_size)
        return ((1.0 - self.keep_prob) / block_size) * (feature_dim / (feature_dim - block_size + 1.0))

    def _compute_valid_seed_region(self, seq_length):
        """Compute valid seed region to ensure blocks don't go out of bounds"""
        positions = torch.arange(seq_length).float()
        half_block_size = self.block_size // 2
        
        valid_seed_bool = (positions >= half_block_size) & (positions < seq_length - half_block_size)
        valid_seed_region = valid_seed_bool.float()
        
        return valid_seed_region.unsqueeze(0).unsqueeze(-1)  # Shape: [1, L, 1]


class DropBlock2D(nn.Module):

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        """
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = 'channels_last' if data_format is None else data_format
        if self.data_format not in ['channels_first', 'channels_last']:
            raise ValueError('data_format must be either "channels_first" or "channels_last"')

    def forward(self, x, training=None):
        if training is None:
            training = self.training
        
        if not training or self.keep_prob == 1:
            return x
        
        # Handle data format
        if self.data_format == 'channels_first':
            # Convert from [B, C, H, W] to [B, H, W, C] - Batch, Channel, Height, Width
            x = x.permute(0, 2, 3, 1)
        
        # Get shape
        shape = x.shape
        batch_size, height, width, channels = shape
        
        # Compute gamma (dropout rate)
        gamma = self._get_gamma(height, width)
        
        # Create mask
        if self.sync_channels:
            mask_shape = [batch_size, height, width, 1]
        else:
            mask_shape = [batch_size, height, width, channels]
            
        # Sample mask
        mask = torch.bernoulli(torch.ones(mask_shape, device=x.device) * gamma)
        
        # Apply valid seed region constraint
        valid_seed_region = self._compute_valid_seed_region(height, width).to(x.device)
        mask = mask * valid_seed_region
        
        # Extend dropped area by max pooling
        mask = mask.permute(0, 3, 1, 2)  # [B, C, H, W] for pooling
        mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        )
        mask = mask.permute(0, 2, 3, 1)  # [B, H, W, C]
        mask = 1.0 - mask  # Invert mask
        
        # Apply mask and scale
        x = x * mask * (mask.numel() / mask.sum().clamp(min=1.0))
        
        # Restore original format if needed
        if self.data_format == 'channels_first':
            x = x.permute(0, 3, 1, 2)  # Convert back to [B, C, H, W]
            
        return x

    def _get_gamma(self, height, width):
        """Get the number of activation units to drop"""
        height, width = float(height), float(width)
        block_size = float(self.block_size)
        return ((1.0 - self.keep_prob) / (block_size ** 2)) * \
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self, height, width):
        """Compute valid seed region to ensure blocks don't go out of bounds"""
        half_block_size = self.block_size // 2
        
        # Create position tensors
        y_positions = torch.arange(height).view(height, 1).expand(height, width)
        x_positions = torch.arange(width).view(1, width).expand(height, width)
        
        # Create valid seed region mask
        valid_seed_bool = (
            (y_positions >= half_block_size) & 
            (y_positions < height - half_block_size) &
            (x_positions >= half_block_size) & 
            (x_positions < width - half_block_size)
        )
        valid_seed_region = valid_seed_bool.float()
        
        return valid_seed_region.unsqueeze(0).unsqueeze(-1)  # Shape: [1, H, W, 1]
