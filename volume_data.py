import torch
import numpy as np
from torch.utils.data import Dataset

def generate_synthetic_volume(size=32):
    """
    Generates a synthetic 3D volume (size x size x size) with a sphere in the center.
    Values are normalized between 0 and 1.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    z = np.linspace(-1, 1, size)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Distance from center
    dist = np.sqrt(xx**2 + yy**2 + zz**2)
    
    # Create a soft sphere: 1 at center, fading to 0 at radius 0.8
    volume = np.clip(1.0 - dist / 0.8, 0, 1)
    
    return volume

class VolumeDataset(Dataset):
    def __init__(self, volume_data):
        """
        volume_data: numpy array of shape (D, H, W)
        """
        self.shape = volume_data.shape
        
        # Create coordinate grid
        D, H, W = self.shape
        x = np.linspace(-1, 1, D)
        y = np.linspace(-1, 1, H)
        z = np.linspace(-1, 1, W)
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten coordinates and values
        self.coords = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1).astype(np.float32)
        self.values = volume_data.flatten().astype(np.float32)
        
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx):
        return self.coords[idx], self.values[idx]
