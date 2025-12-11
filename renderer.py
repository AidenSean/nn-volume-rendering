import torch
import numpy as np
import vtk
from vtk.util import numpy_support

def get_vtk_transfer_functions():
    """
    Creates and returns VTK color and opacity transfer functions.
    Defaulting to realistic earth-like mapping.
    """
    return get_realistic_transfer_functions()

def get_realistic_transfer_functions():
    """
    Maps density values to Earth-like materials.
    """
    # Color Transfer Function
    color_tf = vtk.vtkColorTransferFunction()
    color_tf.AddRGBPoint(0.0, 0.0, 0.0, 0.0) # Air
    color_tf.AddRGBPoint(0.30, 1.0, 1.0, 1.0) # Clouds
    color_tf.AddRGBPoint(0.40, 1.0, 1.0, 1.0)
    color_tf.AddRGBPoint(0.41, 0.0, 0.0, 0.5) # Water
    color_tf.AddRGBPoint(0.49, 0.0, 0.4, 0.8)
    color_tf.AddRGBPoint(0.50, 0.0, 0.6, 0.0) # Land
    color_tf.AddRGBPoint(0.65, 0.4, 0.5, 0.2)
    color_tf.AddRGBPoint(0.70, 0.5, 0.4, 0.3) # Mountain
    color_tf.AddRGBPoint(0.85, 0.3, 0.3, 0.3)
    color_tf.AddRGBPoint(0.90, 0.9, 0.9, 0.9) # Snow
    color_tf.AddRGBPoint(1.00, 1.0, 1.0, 1.0)
    
    # Opacity Transfer Function
    opacity_tf = vtk.vtkPiecewiseFunction()
    opacity_tf.AddPoint(0.0, 0.0)
    opacity_tf.AddPoint(0.29, 0.0)
    opacity_tf.AddPoint(0.30, 0.1)
    opacity_tf.AddPoint(0.40, 0.2)
    opacity_tf.AddPoint(0.401, 0.0)
    opacity_tf.AddPoint(0.41, 0.8)
    opacity_tf.AddPoint(0.49, 0.8)
    opacity_tf.AddPoint(0.50, 1.0)
    opacity_tf.AddPoint(1.00, 1.0)
    
    return color_tf, opacity_tf

def ray_box_intersection(ray_origins, ray_dirs, box_min, box_max):
    """
    Vectorized Ray-Box Intersection (AABB).
    Returns t_min and t_max for each ray.
    """
    inv_dirs = 1.0 / (ray_dirs + 1e-6)
    
    t1 = (box_min - ray_origins) * inv_dirs
    t2 = (box_max - ray_origins) * inv_dirs
    
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)
    
    t_enter = np.max(t_min, axis=-1)
    t_exit = np.min(t_max, axis=-1)
    
    # Check if intersection exists
    mask = t_exit >= t_enter
    
    # Also check if box is behind camera (t_exit < 0)
    mask = mask & (t_exit > 0)
    
    # If inside box, t_enter is negative, clamp to 0
    t_enter = np.maximum(t_enter, 0.0)
    
    return t_enter, t_exit, mask

def raycast_render(model, width=256, height=256, device='cpu', 
                   camera_pos=None, camera_dir=None, camera_up=None,
                   azimuth=None, elevation=None):
    """
    Performs raycasting using the trained model.
    Supports arbitrary camera via camera_pos/dir/up OR azimuth/elevation.
    """
    model.eval()
    
    # Handle Legacy Azimuth/Elevation if provided
    if azimuth is not None and elevation is not None:
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        cam_dist = 2.0
        cx = cam_dist * np.cos(el_rad) * np.sin(az_rad)
        cy = cam_dist * np.sin(el_rad)
        cz = cam_dist * np.cos(el_rad) * np.cos(az_rad)
        camera_pos = np.array([cx, cy, cz])
        camera_dir = -camera_pos / np.linalg.norm(camera_pos)
        camera_up = np.array([0, 1, 0])
    
    # Default if nothing provided
    if camera_pos is None:
        camera_pos = np.array([0, 0, -2.0])
        camera_dir = np.array([0, 0, 1.0])
        camera_up = np.array([0, 1, 0])
        
    # Normalize vectors
    fwd = camera_dir / np.linalg.norm(camera_dir)
    right = np.cross(fwd, camera_up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, fwd)
    
    # Ray generation (Perspective Projection)
    fov = 60.0
    aspect = width / height
    scale = np.tan(np.radians(fov * 0.5))
    
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    # Ray directions in camera space
    # P_cam = [x * aspect * scale, y * scale, 1]
    # dir = normalize(P_cam)
    # But we need world space
    
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    
    # Directions in world space
    # dir = (xx * aspect * scale) * right + (yy * scale) * up + fwd
    ray_dirs = (xx_flat[:, None] * aspect * scale * right[None, :] + 
                yy_flat[:, None] * scale * new_up[None, :] + 
                fwd[None, :])
    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    
    # Ray origins are all camera_pos
    ray_origins = np.broadcast_to(camera_pos, ray_dirs.shape)
    
    # Ray-Box Intersection with Unit Cube [-1, 1]
    t_enter, t_exit, mask = ray_box_intersection(ray_origins, ray_dirs, -1.0, 1.0)
    
    image = np.zeros((height * width, 4))
    
    # Only process rays that hit the box
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) > 0:
        # Ray marching parameters
        num_steps = 64
        
        # We march from t_enter to t_exit
        # Each ray has different start/end
        
        # To vectorize efficiently, we can step in normalized [0, 1] range of the segment
        # pos = origin + dir * (t_enter + step_idx * step_size)
        # where step_size = (t_exit - t_enter) / num_steps
        
        valid_origins = ray_origins[valid_indices]
        valid_dirs = ray_dirs[valid_indices]
        valid_t_enter = t_enter[valid_indices]
        valid_t_exit = t_exit[valid_indices]
        
        step_sizes = (valid_t_exit - valid_t_enter) / num_steps
        
        # Accumulators for valid rays
        acc_color = np.zeros((len(valid_indices), 3))
        acc_alpha = np.zeros(len(valid_indices))
        
        color_tf, opacity_tf = get_vtk_transfer_functions()
        
        # Pre-compute LUT
        lut_size = 256
        lut_colors = np.zeros((lut_size, 3))
        lut_opacity = np.zeros(lut_size)
        for i in range(lut_size):
            val = i / (lut_size - 1)
            c = color_tf.GetColor(val)
            o = opacity_tf.GetValue(val)
            lut_colors[i] = c
            lut_opacity[i] = o
            
        with torch.no_grad():
            for i in range(num_steps):
                # Current t for each ray
                t = valid_t_enter + i * step_sizes
                
                # Sample points
                sample_points = valid_origins + valid_dirs * t[:, None]
                
                # Query model
                tensor_points = torch.from_numpy(sample_points.astype(np.float32)).to(device)
                intensities = model(tensor_points).cpu().numpy().flatten()
                
                # Lookup TF
                lut_idx = (np.clip(intensities, 0, 1) * 255).astype(int)
                colors = lut_colors[lut_idx]
                opacities = lut_opacity[lut_idx]
                
                # Composite
                # src_alpha = opacities * step_sizes * density_factor
                # Density factor ensures opacity is consistent regardless of step size
                # But step_sizes vary per ray here!
                src_alpha = opacities * step_sizes * 5.0 # Tuned factor
                src_alpha = np.clip(src_alpha, 0, 1)
                
                rem_alpha = 1.0 - acc_alpha
                
                # Early termination check could go here (if rem_alpha < epsilon)
                
                acc_color += rem_alpha[:, None] * colors * src_alpha[:, None]
                acc_alpha += rem_alpha * src_alpha
        
        # Write back to image
        image[valid_indices, :3] = acc_color
        image[valid_indices, 3] = acc_alpha
            
    return image.reshape((height, width, 4))

def save_image(image, filename='output_render.png'):
    from PIL import Image
    img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8, 'RGBA')
    img_pil.save(filename)
    print(f"Image saved to {filename}")
