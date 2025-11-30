"""
Physics-Informed Neural Network (PINN) Model for Lid-Driven Cavity Flow
"""

import torch
import torch.nn as nn
import numpy as np


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving Navier-Stokes equations.
    Takes (x, y) coordinates as input and outputs (u, v, p) velocity and pressure fields.
    """
    
    def __init__(self, layers=[2, 50, 50, 50, 50, 3], activation=nn.Tanh()):
        """
        Initialize PINN model
        
        Args:
            layers: List of layer sizes [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function (default: Tanh)
        """
        super(PINN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.activation = activation
        
        # Build fully connected layers
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
    
    def forward(self, x, y):
        """
        Forward pass of the neural network
        
        Args:
            x: x-coordinates (tensor of shape [N, 1] or [N])
            y: y-coordinates (tensor of shape [N, 1] or [N])
            
        Returns:
            u, v, p: velocity components and pressure (each of shape [N])
        """
        # Ensure inputs are 2D tensors
        if x.dim() == 1:
            x = x.unsqueeze(1)
        if y.dim() == 1:
            y = y.unsqueeze(1)
        
        # Concatenate x and y coordinates
        xy = torch.cat([x, y], dim=1)
        
        # Forward through layers
        for i, layer in enumerate(self.layers[:-1]):
            xy = layer(xy)
            xy = self.activation(xy)
        
        # Final layer (no activation)
        output = self.layers[-1](xy)
        
        # Split output into u, v, p
        u = output[:, 0]
        v = output[:, 1]
        p = output[:, 2]
        
        return u, v, p

