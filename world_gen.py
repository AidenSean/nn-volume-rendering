import numpy as np
import noise

def generate_planet_volume(size=64, scale=0.1):
    """
    Generates a 3D volume representing a planet with:
    - Core/Mantle (Solid)
    - Surface Terrain (Mountains/Valleys)
    - Oceans (Water level)
    - Atmosphere/Clouds
    
    Values:
    0.0 - 0.3: Air
    0.3 - 0.4: Clouds
    0.4 - 0.5: Water
    0.5 - 0.7: Land
    0.7 - 0.9: Mountain
    0.9 - 1.0: Snow
    """
    volume = np.zeros((size, size, size), dtype=np.float32)
    
    cx, cy, cz = size/2, size/2, size/2
    radius = size * 0.4
    
    x = np.linspace(0, size-1, size)
    y = np.linspace(0, size-1, size)
    z = np.linspace(0, size-1, size)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    
    # Distance from center
    dist = np.sqrt((xx-cx)**2 + (yy-cy)**2 + (zz-cz)**2)
    
    # 1. Base Planet Shape (Sphere)
    # Normalized distance: 0 at center, 1 at surface
    norm_dist = dist / radius
    
    # 2. Terrain Noise (3D Perlin)
    # We only care about noise near the surface
    terrain_noise = np.zeros_like(dist)
    
    # Optimization: Only compute noise near surface
    surface_mask = (norm_dist > 0.8) & (norm_dist < 1.2)
    
    # We iterate to compute noise (vectorized noise is not available in this lib easily)
    # But for 64^3 it's fast enough to loop or we can use a trick.
    # Let's loop over the whole volume for simplicity of code, or just surface.
    # To keep it fast, we'll just loop.
    
    print("Generating Terrain...")
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if surface_mask[i, j, k]:
                    # Terrain noise
                    val = noise.pnoise3(i*scale, j*scale, k*scale, octaves=4, persistence=0.5, lacunarity=2.0)
                    terrain_noise[i, j, k] = val
                    
    # 3. Combine to form density
    # Surface is at norm_dist = 1.0
    # We want:
    # If dist < radius + terrain_height -> Solid
    # If dist > radius + terrain_height -> Air
    
    # Terrain height perturbation (-0.1 to 0.1 radius)
    height_map = terrain_noise * 0.15 
    
    # Effective radius at this point
    r_eff = 1.0 + height_map
    
    # Solid Mask
    solid_mask = norm_dist < r_eff
    
    # Water Level (at exactly radius 1.0)
    water_mask = (norm_dist < 1.0) & (~solid_mask)
    
    # Assign Values
    # Land/Mountain: Map height to 0.5 - 1.0
    # height_map ranges approx -0.15 to 0.15
    # We want land (0.5) to snow (1.0)
    # Normalize height_map to 0-1 range relative to min/max terrain
    
    # Base land value
    volume[solid_mask] = 0.5 + (height_map[solid_mask] + 0.05) * 2.5 
    # Clip land to valid range
    volume[solid_mask] = np.clip(volume[solid_mask], 0.5, 1.0)
    
    # Water value
    volume[water_mask] = 0.45 # Constant water density
    
    # 4. Clouds
    # Cloud layer at 1.1 radius
    print("Generating Clouds...")
    cloud_layer_mask = (norm_dist > 1.05) & (norm_dist < 1.2)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if cloud_layer_mask[i, j, k]:
                    # High freq noise for clouds
                    c_val = noise.pnoise3(i*scale*2, j*scale*2, k*scale*2 + 100, octaves=2)
                    if c_val > 0.2: # Threshold for cloud formation
                        volume[i, j, k] = 0.35 # Cloud density
                        
    # Air is 0 (already initialized)
    
    return volume

# Wrapper for compatibility
def generate_world_data(size=64, **kwargs):
    return generate_planet_volume(size=size)
