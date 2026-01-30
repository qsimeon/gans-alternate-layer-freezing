"""Neural network models for comparing standard vs. alternating training approaches.

This module implements simple feedforward neural networks that can be trained
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
        input_size (int): Size of input features (default: 784 for MNIST)
        hidden_size (int): Size of hidden layers (default: 256)
        num_classes (int): Number of output classes (default: 10)
    """
    
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleNN, self).__init__()
        
        # First half of the network
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Second half of the network
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            
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
