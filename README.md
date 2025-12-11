# Project Documentation

This project implements a Volume Rendering system that compares a Traditional Raycasting approach with a Neural Network-based approach (Implicit Neural Representation). It also includes realistic procedural world generation.

## File Descriptions

### 1. `model.py`
**Purpose**: Defines the Neural Network architecture.
- **Class `VolumeNet`**: A Multi-Layer Perceptron (MLP) that takes 3D coordinates `(x, y, z)` as input and outputs a scalar intensity value.
    - **`__init__`**: Defines the layers (Linear -> ReLU -> ... -> Linear -> Sigmoid).
    - **`forward`**: Performs the forward pass.
    - **Why**: We use a neural network to "memorize" the volume data. This allows us to represent infinite resolution (continuous function) and potentially compress the data.

### 2. `renderer.py`
**Purpose**: Implements the Neural Renderer and Transfer Functions.
- **Function `get_realistic_transfer_functions`**: Returns VTK objects (`vtkColorTransferFunction`, `vtkPiecewiseFunction`) that map density values to colors (Water, Land, Clouds) and opacity.
- **Function `raycast_render`**: The core rendering loop for the neural network.
    - Generates rays for each pixel.
    - Marches along rays, querying the `VolumeNet` at each step.
    - Applies transfer functions to the predicted intensities.
    - Composites the colors using alpha blending to form the final image.
    - **Why**: This simulates the physics of light passing through a volume, but uses the neural network as the data source.

### 3. `traditional_renderer.py`
**Purpose**: Implements the Traditional Renderer for comparison.
- **Function `render_volume_direct`**: Similar to `raycast_render` but samples directly from a 3D NumPy array.
    - Uses Nearest Neighbor or Trilinear interpolation to fetch values from the grid.
    - **Why**: To provide a ground truth comparison. It shows the discrete nature of grid-based data vs the continuous neural representation.

### 4. `world_gen.py`
**Purpose**: Generates procedural 3D volume data.
- **Function `generate_planet_volume`**: Uses Perlin Noise (`noise` library) to create a realistic planet.
    - Layers: Core, Terrain (Mountains), Oceans, Atmosphere (Clouds).
    - **Why**: To provide interesting and complex data for the renderer, rather than a simple sphere.

### 5. `train.py`
**Purpose**: Handles the training of the Neural Network.
- **Function `train_model`**:
    - Generates the volume data.
    - Creates a `VolumeDataset`.
    - Optimizes the `VolumeNet` to minimize the Mean Squared Error (MSE) between its predictions and the actual volume data.
    - **Why**: The network must be "fitted" to the data before it can render it.

### 6. `volume_data.py`
**Purpose**: Data handling for PyTorch.
- **Class `VolumeDataset`**: A PyTorch `Dataset` that wraps the 3D volume array and provides `(coordinate, value)` pairs for training.

### 7. `demo.py`
**Purpose**: An interactive demonstration application.
- **Function `main`**:
    - Generates data and trains the model on the fly.
    - Opens an OpenCV window showing Traditional vs Neural rendering side-by-side.
    - Handles User Input (Mouse/Keyboard) to rotate the view.

### 8. `main.py`
**Purpose**: A simple batch script to run the pipeline once and save an image.
- Useful for headless execution or quick verification without interaction.

## Dependencies
- **PyTorch**: For the neural network.
- **VTK**: For professional-grade Color and Opacity transfer functions.
- **NumPy**: For numerical operations and array handling.
- **OpenCV (`cv2`)**: For the interactive display window.
- **Noise**: For Perlin noise generation.
