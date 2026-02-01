"""Neural network models for comparing standard vs. alternating training approaches.

This module implements feedforward neural networks that can be trained
using either standard backpropagation or an alternating approach where different
halves of the network are frozen during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    """Simple feedforward neural network with two hidden layers.
    
    This network is designed to be split into two halves for alternating training.
    The first half consists of the first hidden layer, and the second half consists
    of the second hidden layer and output layer.
    
    Args:
        input_size (int): Size of input features (default: 3072 for CIFAR-10: 32x32x3)
        hidden_size (int): Size of hidden layers (default: 512)
        num_classes (int): Number of output classes (default: 10)
        dropout_rate (float): Dropout probability (default: 0.3)
    """
    
    def __init__(self, input_size=3072, hidden_size=512, num_classes=10, dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        
        # First half of the network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Second half of the network
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
                or (batch_size, channels, height, width) for images
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Flatten input if needed
        x = x.view(x.size(0), -1)
        
        # First half
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second half
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
    def get_first_half_params(self):
        """Get parameters for the first half of the network.
        
        Returns:
            list: List of parameters in the first half (fc1, bn1)
        """
        params = []
        params.extend(list(self.fc1.parameters()))
        params.extend(list(self.bn1.parameters()))
        return params
    
    def get_second_half_params(self):
        """Get parameters for the second half of the network.
        
        Returns:
            list: List of parameters in the second half (fc2, bn2, fc3)
        """
        params = []
        params.extend(list(self.fc2.parameters()))
        params.extend(list(self.bn2.parameters()))
        params.extend(list(self.fc3.parameters()))
        return params
    
    def freeze_first_half(self):
        """Freeze parameters in the first half of the network."""
        for param in self.fc1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
    
    def freeze_second_half(self):
        """Freeze parameters in the second half of the network."""
        for param in self.fc2.parameters():
            param.requires_grad = False
        for param in self.bn2.parameters():
            param.requires_grad = False
        for param in self.fc3.parameters():
            param.requires_grad = False
    
    def unfreeze_first_half(self):
        """Unfreeze parameters in the first half of the network."""
        for param in self.fc1.parameters():
            param.requires_grad = True
        for param in self.bn1.parameters():
            param.requires_grad = True
    
    def unfreeze_second_half(self):
        """Unfreeze parameters in the second half of the network."""
        for param in self.fc2.parameters():
            param.requires_grad = True
        for param in self.bn2.parameters():
            param.requires_grad = True
        for param in self.fc3.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters in the network."""
        self.unfreeze_first_half()
        self.unfreeze_second_half()
    
    def count_parameters(self, trainable_only=False):
        """Count the number of parameters in the model.
        
        Args:
            trainable_only (bool): If True, count only trainable parameters
            
        Returns:
            int: Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


class DeepNN(nn.Module):
    """Deeper feedforward neural network for more complex experiments.
    
    This network has more layers and can be split into arbitrary halves
    for alternating training experiments.
    
    Args:
        input_size (int): Size of input features (default: 784 for MNIST)
        hidden_sizes (list): List of hidden layer sizes (default: [512, 256, 128, 64])
        num_classes (int): Number of output classes (default: 10)
        dropout_rate (float): Dropout probability (default: 0.3)
    """
    
    def __init__(self, input_size=784, hidden_sizes=None, num_classes=10, dropout_rate=0.3):
        super(DeepNN, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128, 64]
        
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        # Build layers
        layers = []
        in_features = input_size
        
        for i, out_features in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_sizes[-1], num_classes)
        
        # Calculate split point (half of the layers)
        self.split_idx = (len(hidden_sizes) // 2) * 4  # Each layer has 4 components
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_first_half_params(self):
        """Get parameters for the first half of the network."""
        params = []
        for i, layer in enumerate(self.features):
            if i < self.split_idx:
                params.extend(list(layer.parameters()))
        return params
    
    def get_second_half_params(self):
        """Get parameters for the second half of the network."""
        params = []
        for i, layer in enumerate(self.features):
            if i >= self.split_idx:
                params.extend(list(layer.parameters()))
        params.extend(list(self.classifier.parameters()))
        return params
    
    def freeze_first_half(self):
        """Freeze parameters in the first half of the network."""
        for i, layer in enumerate(self.features):
            if i < self.split_idx:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def freeze_second_half(self):
        """Freeze parameters in the second half of the network."""
        for i, layer in enumerate(self.features):
            if i >= self.split_idx:
                for param in layer.parameters():
                    param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
    
    def unfreeze_first_half(self):
        """Unfreeze parameters in the first half of the network."""
        for i, layer in enumerate(self.features):
            if i < self.split_idx:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def unfreeze_second_half(self):
        """Unfreeze parameters in the second half of the network."""
        for i, layer in enumerate(self.features):
            if i >= self.split_idx:
                for param in layer.parameters():
                    param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters in the network."""
        for param in self.parameters():
            param.requires_grad = True
    
    def count_parameters(self, trainable_only=False):
        """Count the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())