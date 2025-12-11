import torch
import torch.nn as nn

class VolumeNet(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=4):
        super(VolumeNet, self).__init__()
        
        layers = []
        # Input is (x, y, z) -> 3
        layers.append(nn.Linear(3, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        # Output is intensity -> 1
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid()) # Ensure output is between 0 and 1
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
