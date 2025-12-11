import numpy as np
import vtk
from renderer import get_vtk_transfer_functions, ray_box_intersection

def render_volume_direct(volume, width=256, height=256, 
                         camera_pos=None, camera_dir=None, camera_up=None,
                         azimuth=None, elevation=None):
    """
    Renders the volume using standard raycasting.
    Supports FPP camera.
    """
    D, H, W = volume.shape
    
    # Handle Legacy
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
        
    if camera_pos is None:
        camera_pos = np.array([0, 0, -2.0])
        camera_dir = np.array([0, 0, 1.0])
        camera_up = np.array([0, 1, 0])
        
    fwd = camera_dir / np.linalg.norm(camera_dir)
    right = np.cross(fwd, camera_up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, fwd)
    
    # Perspective Projection
    fov = 60.0
    aspect = width / height
    scale = np.tan(np.radians(fov * 0.5))
    
    x = np.linspace(-1, 1, width)
    y = np.linspace(-1, 1, height)
    xx, yy = np.meshgrid(x, y)
    
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    
    ray_dirs = (xx_flat[:, None] * aspect * scale * right[None, :] + 
                yy_flat[:, None] * scale * new_up[None, :] + 
                fwd[None, :])
    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    
    ray_origins = np.broadcast_to(camera_pos, ray_dirs.shape)
    
    # Ray-Box Intersection
    t_enter, t_exit, mask = ray_box_intersection(ray_origins, ray_dirs, -1.0, 1.0)
    
    image = np.zeros((height * width, 4))
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) > 0:
        valid_origins = ray_origins[valid_indices]
        valid_dirs = ray_dirs[valid_indices]
        valid_t_enter = t_enter[valid_indices]
        valid_t_exit = t_exit[valid_indices]
        
        num_steps = 128
        step_sizes = (valid_t_exit - valid_t_enter) / num_steps
        
        acc_color = np.zeros((len(valid_indices), 3))
        acc_alpha = np.zeros(len(valid_indices))
        
        color_tf, opacity_tf = get_vtk_transfer_functions()
        lut_size = 256
        lut_colors = np.zeros((lut_size, 3))
        lut_opacity = np.zeros(lut_size)
        for i in range(lut_size):
            val = i / (lut_size - 1)
            c = color_tf.GetColor(val)
            o = opacity_tf.GetValue(val)
            lut_colors[i] = c
            lut_opacity[i] = o
            
        for i in range(num_steps):
            t = valid_t_enter + i * step_sizes
            pos = valid_origins + valid_dirs * t[:, None]
            
            # Map pos from [-1, 1] to [0, D-1]
            norm_pos = (pos + 1) / 2
            
            # Nearest Neighbor
            idx = (norm_pos * np.array([D-1, H-1, W-1])).astype(int)
            idx[:, 0] = np.clip(idx[:, 0], 0, D-1)
            idx[:, 1] = np.clip(idx[:, 1], 0, H-1)
            idx[:, 2] = np.clip(idx[:, 2], 0, W-1)
            
            vals = volume[idx[:, 0], idx[:, 1], idx[:, 2]]
            
            lut_idx = (vals * 255).astype(int)
            c = lut_colors[lut_idx]
            o = lut_opacity[lut_idx]
            
            src_alpha = o * step_sizes * 5.0
            src_alpha = np.clip(src_alpha, 0, 1)
            
            rem_alpha = 1.0 - acc_alpha
            
            acc_color += rem_alpha[:, None] * c * src_alpha[:, None]
            acc_alpha += rem_alpha * src_alpha
            
        image[valid_indices, :3] = acc_color
        image[valid_indices, 3] = acc_alpha
            
    return image.reshape((height, width, 4))
