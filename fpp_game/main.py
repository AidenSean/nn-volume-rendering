import sys
import os
import pygame
import numpy as np
import torch
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world_gen import generate_planet_volume
from volume_data import VolumeDataset
from traditional_renderer import render_volume_direct
from renderer import raycast_render
from model import VolumeNet

class FPPGame:
    def __init__(self):
        pygame.init()
        self.width = 1024 # Wider for side-by-side
        self.height = 512
        self.render_width = 256
        self.render_height = 256
        
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Neural Volume Rendering - FPP Game")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        
        # Game State
        self.running = True
        self.camera_pos = np.array([0.0, 0.0, -2.5])
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        
        # Mouse State
        self.mouse_locked = False
        self.mouse_sensitivity = 0.2
        self.move_speed = 0.1
        
        # Initialize Data & Model
        self.init_volume()
        
    def init_volume(self):
        print("Generating World...")
        self.volume_data = generate_planet_volume(size=32)
        
        print("Training Neural Network...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VolumeNet().to(self.device)
        
        dataset = VolumeDataset(self.volume_data)
        coords = torch.from_numpy(dataset.coords).to(self.device)
        values = torch.from_numpy(dataset.values).to(self.device).unsqueeze(1)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.MSELoss()
        
        # Quick training
        epochs = 20
        batch_size = 4096
        num_samples = len(coords)
        
        for epoch in range(epochs):
            perm = torch.randperm(num_samples)
            for i in range(0, num_samples, batch_size):
                idx = perm[i:i+batch_size]
                optimizer.zero_grad()
                outputs = self.model(coords[idx])
                loss = criterion(outputs, values[idx])
                loss.backward()
                optimizer.step()
        print("Ready!")

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_TAB:
                    self.mouse_locked = not self.mouse_locked
                    pygame.mouse.set_visible(not self.mouse_locked)
                    pygame.event.set_grab(self.mouse_locked)
            elif event.type == pygame.MOUSEMOTION and self.mouse_locked:
                dx, dy = event.rel
                self.camera_yaw += dx * self.mouse_sensitivity
                self.camera_pitch -= dy * self.mouse_sensitivity
                self.camera_pitch = np.clip(self.camera_pitch, -89, 89)

        # Keyboard Movement
        keys = pygame.key.get_pressed()
        
        # Calculate Vectors
        yaw_rad = np.radians(self.camera_yaw)
        pitch_rad = np.radians(self.camera_pitch)
        
        dx = np.sin(yaw_rad) * np.cos(pitch_rad)
        dy = np.sin(pitch_rad)
        dz = np.cos(yaw_rad) * np.cos(pitch_rad)
        
        fwd = np.array([dx, dy, dz])
        fwd = fwd / np.linalg.norm(fwd)
        
        up = np.array([0, 1, 0])
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        
        # Re-orthogonalize up
        self.camera_up = np.cross(right, fwd)
        self.camera_dir = fwd
        
        if keys[pygame.K_w]: self.camera_pos += fwd * self.move_speed
        if keys[pygame.K_s]: self.camera_pos -= fwd * self.move_speed
        if keys[pygame.K_a]: self.camera_pos -= right * self.move_speed
        if keys[pygame.K_d]: self.camera_pos += right * self.move_speed
        if keys[pygame.K_q]: self.camera_pos += self.camera_up * self.move_speed
        if keys[pygame.K_e]: self.camera_pos -= self.camera_up * self.move_speed

    def render(self):
        self.screen.fill((30, 30, 30))
        
        # Render Traditional
        trad_img = render_volume_direct(self.volume_data, width=self.render_width, height=self.render_height,
                                      camera_pos=self.camera_pos, camera_dir=self.camera_dir, camera_up=self.camera_up)
        
        # Render Neural
        neur_img = raycast_render(self.model, width=self.render_width, height=self.render_height, device=self.device,
                                camera_pos=self.camera_pos, camera_dir=self.camera_dir, camera_up=self.camera_up)
        
        # Convert to Pygame Surfaces
        # Numpy is (H, W, 4) RGBA -> Pygame expects (W, H) or we can use make_surface with swapaxes
        
        # Traditional
        trad_surf = pygame.image.frombuffer((trad_img * 255).astype(np.uint8).tobytes(), 
                                          (self.render_width, self.render_height), "RGBA")
        trad_surf = pygame.transform.scale(trad_surf, (self.width // 2, self.height))
        
        # Neural
        neur_surf = pygame.image.frombuffer((neur_img * 255).astype(np.uint8).tobytes(), 
                                          (self.render_width, self.render_height), "RGBA")
        neur_surf = pygame.transform.scale(neur_surf, (self.width // 2, self.height))
        
        # Blit
        self.screen.blit(trad_surf, (0, 0))
        self.screen.blit(neur_surf, (self.width // 2, 0))
        
        # UI
        fps = int(self.clock.get_fps())
        fps_text = self.font.render(f"FPS: {fps}", True, (255, 255, 255))
        self.screen.blit(fps_text, (10, 10))
        
        pos_text = self.font.render(f"Pos: {self.camera_pos}", True, (255, 255, 255))
        self.screen.blit(pos_text, (10, 30))
        
        label_trad = self.font.render("Traditional (Raycast)", True, (255, 255, 255))
        self.screen.blit(label_trad, (self.width // 4 - label_trad.get_width() // 2, self.height - 30))
        
        label_neur = self.font.render("Neural (Implicit Rep)", True, (255, 255, 255))
        self.screen.blit(label_neur, (3 * self.width // 4 - label_neur.get_width() // 2, self.height - 30))
        
        if not self.mouse_locked:
            msg = self.font.render("Press TAB to Lock Mouse", True, (255, 255, 0))
            self.screen.blit(msg, (self.width // 2 - msg.get_width() // 2, self.height // 2))
            
        # Crosshair
        cx, cy = self.width // 2, self.height // 2
        pygame.draw.line(self.screen, (255, 255, 255), (cx - 10, cy), (cx + 10, cy), 1)
        pygame.draw.line(self.screen, (255, 255, 255), (cx, cy - 10), (cx, cy + 10), 1)
        
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_input()
            self.render()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    game = FPPGame()
    game.run()
