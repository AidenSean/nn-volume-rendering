import sys
import os
import cv2
import numpy as np
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_gen import generate_planet_volume
from volume_data import VolumeDataset
from traditional_renderer import render_volume_direct
from renderer import raycast_render
from model import VolumeNet

def test_fpp():
    print("Testing FPP Game Pipeline...")
    
    # 1. Generate Data
    volume_size = 16
    volume_data = generate_planet_volume(size=volume_size)
    
    # 2. Setup Model
    device = torch.device("cpu")
    model = VolumeNet().to(device)
    
    # 3. Test Render with Camera
    camera_pos = np.array([0.0, 0.0, -2.5])
    camera_dir = np.array([0.0, 0.0, 1.0])
    camera_up = np.array([0.0, 1.0, 0.0])
    
    print("Rendering FPP Frame (Traditional)...")
    trad_img = render_volume_direct(volume_data, width=64, height=64,
                                  camera_pos=camera_pos, camera_dir=camera_dir, camera_up=camera_up)
    
    print("Rendering FPP Frame (Neural)...")
    neur_img = raycast_render(model, width=64, height=64, device=device,
                            camera_pos=camera_pos, camera_dir=camera_dir, camera_up=camera_up)
    
    # Save output
    trad_bgr = cv2.cvtColor((trad_img * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    neur_bgr = cv2.cvtColor((neur_img * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    combined = np.hstack([trad_bgr, neur_bgr])
    
    cv2.imwrite("test_fpp_output.png", combined)
    print("Test Complete.")

if __name__ == "__main__":
    test_fpp()
