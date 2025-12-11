import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from volume_data import generate_synthetic_volume, VolumeDataset
from model import VolumeNet
import os

def train_model(epochs=10, batch_size=1024, learning_rate=0.001, save_path='volume_net.pth'):
    print("Generating synthetic volume data...")
    volume = generate_synthetic_volume(size=32)
    dataset = VolumeDataset(volume)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = VolumeNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    for epoch in range(epochs):
        total_loss = 0
        for coords, values in dataloader:
            coords = coords.to(device)
            values = values.to(device).unsqueeze(1) # Match output shape
            
            optimizer.zero_grad()
            outputs = model(coords)
            loss = criterion(outputs, values)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
        
    print("Training complete.")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

if __name__ == "__main__":
    train_model()
