import torch
import os
from train import train_model
from renderer import raycast_render, save_image
from model import VolumeNet

def main():
    # 1. Train the model (or load if exists, but for this demo we train fresh)
    print("Step 1: Training Neural Network on Volume Data")
    # Train for a few epochs to get a result quickly
    model = train_model(epochs=20, batch_size=4096)
    
    # 2. Render the volume
    print("\nStep 2: Rendering Volume with Neural Network and VTK Transfer Functions")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Render
    image = raycast_render(model, width=256, height=256, device=device)
    
    # 3. Save the result
    save_image(image, 'neural_volume_render.png')
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
