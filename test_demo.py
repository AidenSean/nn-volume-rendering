import cv2
import numpy as np
import torch
from world_gen import generate_world_data
from volume_data import VolumeDataset
from traditional_renderer import render_volume_direct
from renderer import raycast_render
from model import VolumeNet

def test_demo():
    print("Testing Demo Pipeline (Realistic Planet)...")
    volume_size = 16 # Small for speed
    from world_gen import generate_planet_volume
    volume_data = generate_planet_volume(size=volume_size)
    
    device = torch.device("cpu") # Force CPU for test stability
    model = VolumeNet().to(device)
    
    # Render one frame
    print("Rendering Traditional...")
    trad_img = render_volume_direct(volume_data, width=64, height=64, azimuth=0, elevation=0)
    
    print("Rendering Neural...")
    neur_img = raycast_render(model, width=64, height=64, device=device, azimuth=0, elevation=0)
    
    print("Saving test output...")
    trad_bgr = cv2.cvtColor((trad_img * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    neur_bgr = cv2.cvtColor((neur_img * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    combined = np.hstack([trad_bgr, neur_bgr])
    cv2.imwrite("test_demo_output.png", combined)
    print("Test Complete.")

if __name__ == "__main__":
    test_demo()
