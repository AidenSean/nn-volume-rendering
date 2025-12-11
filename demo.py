import cv2
import numpy as np
import torch
import time
from world_gen import generate_world_data
from volume_data import VolumeDataset
from traditional_renderer import render_volume_direct
from renderer import raycast_render
from train import train_model
from model import VolumeNet

def main():
    print("Generating World Data (Realistic Planet)...")
    volume_size = 32
    # Use the new planet generation
    from world_gen import generate_planet_volume
    volume_data = generate_planet_volume(size=volume_size)
    
    # Train Neural Network on this data
    print("Training Neural Network on World Data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Quick training
    dataset = VolumeDataset(volume_data)
    # Re-use train_model logic but we need to pass data or save/load
    # Let's just instantiate and train here quickly to avoid file I/O complexity
    model = VolumeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Flatten data for training
    coords = torch.from_numpy(dataset.coords).to(device)
    values = torch.from_numpy(dataset.values).to(device).unsqueeze(1)
    
    epochs = 50
    batch_size = 4096
    num_samples = len(coords)
    
    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0
        for i in range(0, num_samples, batch_size):
            idx = perm[i:i+batch_size]
            batch_coords = coords[idx]
            batch_values = values[idx]
            
            optimizer.zero_grad()
            outputs = model(batch_coords)
            loss = criterion(outputs, batch_values)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.5f}")
            
    print("Training Complete.")
    
    # Interactive Loop
    print("\nStarting Interactive Demo...")
    print("Controls:")
    print("  Mouse Drag: Rotate View (Left Click)")
    print("  Arrow Keys: Rotate View")
    print("  ESC: Exit")
    
    width, height = 256, 256
    azimuth = 0
    elevation = 0
    
    # Mouse handling
    mouse_down = False
    last_x, last_y = 0, 0
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_down, last_x, last_y, azimuth, elevation
        # Debug print to see what events are firing
        # print(f"Event: {event}, Flags: {flags}") 
        
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
            last_x, last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_down = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if mouse_down:
                dx = x - last_x
                dy = y - last_y
                azimuth -= dx * 0.5
                elevation += dy * 0.5
                last_x, last_y = x, y
                
    cv2.namedWindow("Neural vs Traditional Volume Rendering")
    cv2.setMouseCallback("Neural vs Traditional Volume Rendering", mouse_callback)
    
    while True:
        # Render Traditional
        # Downsample for speed during interaction if needed, but 32^3 is small enough
        t0 = time.time()
        trad_img = render_volume_direct(volume_data, width=width, height=height, 
                                      azimuth=azimuth, elevation=elevation)
        
        # Render Neural
        neur_img = raycast_render(model, width=width, height=height, device=device,
                                azimuth=azimuth, elevation=elevation)
        
        # Combine
        # Convert RGBA to BGR for OpenCV
        # trad_img is (H, W, 4)
        trad_bgr = cv2.cvtColor((trad_img * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
        neur_bgr = cv2.cvtColor((neur_img * 255).astype(np.uint8), cv2.COLOR_RGBA2BGR)
        
        # Add labels
        cv2.putText(trad_bgr, "Traditional (Raycast)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(neur_bgr, "Neural (Implicit Rep)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        combined = np.hstack([trad_bgr, neur_bgr])
        
        cv2.imshow("Neural vs Traditional Volume Rendering", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == 82: # Up Arrow (might vary by platform, usually 82 or 0)
            elevation -= 5
        elif key == 84: # Down Arrow
            elevation += 5
        elif key == 81: # Left Arrow
            azimuth -= 5
        elif key == 83: # Right Arrow
            azimuth += 5
        # Fallback for some systems where arrow keys are different
        elif key == ord('w'): elevation -= 5
        elif key == ord('s'): elevation += 5
        elif key == ord('a'): azimuth -= 5
        elif key == ord('d'): azimuth += 5
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
