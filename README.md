# Structure from Motion (SFM) Pipeline

A Python implementation of Structure from Motion reconstruction from image sequences.

**Authors:** Rayan Atif, Asad Ayub  
**Student IDs:** 26100166, 27100413  
**Course:** CS436 - Computer Vision

## Project Overview

This project implements a complete Structure from Motion pipeline organized by weekly deliverables:

- **Week 1**: Feature Matching and Two-View Reconstruction
- **Week 3**: Multi-View SfM with Incremental Image Addition and Bundle Adjustment

## Project Structure

```
CV Proj/
├── src/                    # Source code package
│   ├── __init__.py        # Package initialization
│   ├── models.py          # Data classes (CameraPose, MapPoint)
│   ├── io.py              # Image loading and PLY file I/O
│   ├── features.py        # Feature detection and matching
│   ├── geometry.py        # Camera geometry and triangulation
│   ├── reconstruction.py  # SFM reconstruction algorithms
│   └── visualization.py   # Plotting and visualization
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── dataset/               # Image dataset
└── outputs/               # Generated outputs (created automatically)
```

## Installation

1. **Clone or download the repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Week 1 Deliverable: Feature Matching & Two-View Reconstruction

### Features

1. **Image Loading and Preprocessing**
   - Load images from dataset
   - Resize to consistent width (800px) while maintaining aspect ratio
   - Display image grid

2. **Feature Matching**
   - SIFT feature detection
   - Lowe's ratio test for match filtering
   - Visualization of feature matches

3. **Two-View Reconstruction**
   - Essential matrix estimation
   - Camera pose recovery
   - 3D point triangulation with cheirality check
   - Best image pair selection based on reconstruction quality

### Functions (Week 1)

**`src/io.py`:**
- `load_and_resize()` - Load and preprocess images
- `load_image_dataset()` - Load entire dataset
- `save_ply()` - Save 3D point clouds to PLY format

**`src/features.py`:**
- `find_and_filter_matches()` - SIFT + Lowe's ratio test
- `get_matches()` - Get matched points as numpy arrays
- `draw_matches()` - Visualize matches

**`src/geometry.py`:**
- `build_intrinsic_matrix()` - Create K matrix from image dimensions
- `triangulate_and_check()` - Triangulate with cheirality check
- `triangulate_between_views()` - Multi-view triangulation

**`src/reconstruction.py`:**
- `evaluate_pair()` - Evaluate image pair for reconstruction quality
- `find_best_pair()` - Search for optimal image pair
- `two_view_reconstruction()` - Perform two-view reconstruction

**`src/visualization.py`:**
- `display_image_grid()` - Display grid of images
- `visualize_matches()` - Visualize feature matches
- `plot_3d_pointcloud()` - 3D point cloud visualization
- `plot_2d_projections()` - 2D projections of point cloud

### Usage (Week 1)

```bash
# Run feature matching
python main.py --mode features

# Run two-view reconstruction
python main.py --mode two-view
```

## Week 3 Deliverable: Multi-View SfM & Refinement

### Features

1. **Incremental Multi-View SfM**
   - Initialize from two-view reconstruction
   - Incrementally add images using PnP pose estimation
   - Match features to existing 3D points using descriptor matching
   - Triangulate new points between views

2. **Bundle Adjustment**
   - Refine camera poses and 3D points
   - Minimize reprojection error
   - Coordinate system normalization for numerical stability

3. **Advanced Functions**
   - Descriptor-based 3D point matching
   - Observation tracking for bundle adjustment
   - Reprojection error computation

### Functions (Week 3)

**`src/reconstruction.py` (Week 3 additions):**
- `match_features_to_3d()` - Match 2D features to existing 3D points using descriptors
- `estimate_pose_pnp()` - Estimate camera pose using PnP (returns success flag)
- `incremental_sfm()` - Incremental Structure from Motion pipeline
- `bundle_adjust()` - Bundle adjustment optimization
- `run_bundle_adjustment()` - Full bundle adjustment with normalization (Week 3 specific)

**Helper Functions:**
- `rodrigues_to_rotation()` - Convert rotation vector to matrix
- `rotation_to_rodrigues()` - Convert rotation matrix to vector
- `bundle_adjustment_residuals()` - Compute residuals for bundle adjustment
- `compute_reprojection_error()` - Compute reprojection error

### Usage (Week 3)

```bash
# Run incremental SFM
python main.py --mode incremental

# Run complete pipeline (Week 1 + Week 3)
python main.py --mode all
```

## Command-Line Interface

### Options

- `--dataset PATH`: Path to image dataset folder (default: `dataset`)
- `--pattern PATTERN`: Glob pattern for image files (default: `img_*.jpeg`)
- `--width WIDTH`: Target width for image resizing (default: 800)
- `--mode MODE`: Pipeline mode to run:
  - `features`: Week 1 - Feature matching only
  - `two-view`: Week 1 - Two-view reconstruction
  - `incremental`: Week 3 - Incremental SFM
  - `all`: Run complete pipeline (default)
- `--output DIR`: Output directory for results (default: `outputs`)
- `--display-grid`: Display image grid visualization

### Examples

**Run Week 1 feature matching:**
```bash
python main.py --mode features --display-grid
```

**Run Week 1 two-view reconstruction:**
```bash
python main.py --mode two-view
```

**Run Week 3 incremental SFM:**
```bash
python main.py --mode incremental
```

**Run complete pipeline:**
```bash
python main.py --mode all --display-grid
```

## Python API

You can also use the package as a Python library:

### Week 1 Example

```python
from src import (
    load_image_dataset, save_ply,
    find_best_pair, two_view_reconstruction,
    plot_3d_pointcloud
)

# Load images
images, image_files = load_image_dataset('dataset')

# Find best pair for reconstruction
idx1, idx2, data = find_best_pair(images)

# Visualize matches
from src.visualization import visualize_matches
visualize_matches(images[idx1], data['kp1'], images[idx2], data['kp2'], 
                 data['good_matches'])
```

### Week 3 Example

```python
from src import incremental_sfm, match_features_to_3d, estimate_pose_pnp
from src.geometry import build_intrinsic_matrix

# Build intrinsic matrix
K = build_intrinsic_matrix(images[0].shape)

# Run incremental SFM
poses, map_points = incremental_sfm(images, K, idx1, idx2)

# Match features to 3D points (for adding new images)
point_to_descriptor_map = {...}  # Map from point index to descriptor
points_3d_matched, points_2d_matched, indices, kp, des = match_features_to_3d(
    new_image, points_3d, point_to_descriptor_map, K
)

# Estimate pose for new image
R, t, success = estimate_pose_pnp(points_3d_matched, points_2d_matched, K)
```

## Output Files

The pipeline generates the following output files in the `outputs/` directory:

- `pointcloud.ply`: Week 1 - Two-view reconstruction point cloud
- `incremental_pointcloud.ply`: Week 3 - Incremental SFM point cloud
- `matches_*.png`: Feature match visualizations

## Dependencies

- **opencv-python**: Computer vision operations
- **numpy**: Numerical computations
- **matplotlib**: Visualization
- **scipy**: Optimization (bundle adjustment)

## Module Documentation

### `src.models`
- `CameraPose`: Represents camera pose with rotation and translation
- `MapPoint`: Represents 3D point with optional color

### `src.io`
- `load_and_resize()`: Load and resize single image
- `load_image_dataset()`: Load entire image dataset
- `save_ply()`: Save 3D point cloud to PLY format

### `src.features`
- `find_and_filter_matches()`: SIFT feature matching with Lowe's ratio test
- `get_matches()`: Get matched points as numpy arrays
- `draw_matches()`: Visualize feature matches

### `src.geometry`
- `build_intrinsic_matrix()`: Create camera intrinsic matrix
- `triangulate_and_check()`: Triangulate points with cheirality check
- `triangulate_between_views()`: Triangulate between two camera views
- `project_points()`: Project 3D points to 2D
- `rvec_tvec_from_pose()`: Convert R,t to rvec,tvec
- `pose_from_rvec_tvec()`: Convert rvec,tvec to R,t

### `src.reconstruction`
**Week 1 Functions:**
- `evaluate_pair()`: Evaluate image pair for reconstruction quality
- `find_best_pair()`: Search for optimal image pair
- `two_view_reconstruction()`: Perform two-view reconstruction

**Week 3 Functions:**
- `match_features_to_3d()`: Match 2D features to existing 3D points using descriptors
- `estimate_pose_pnp()`: Estimate camera pose using PnP
- `incremental_sfm()`: Incremental Structure from Motion
- `solve_pnp()`: Solve Perspective-n-Point problem
- `bundle_adjust()`: Bundle adjustment optimization
- `match_2d_3d()`: Match 2D features to 3D map points (projection-based)

### `src.visualization`
- `display_image_grid()`: Display grid of images
- `visualize_matches()`: Visualize feature matches
- `plot_3d_pointcloud()`: 3D point cloud visualization
- `plot_2d_projections()`: 2D projections of point cloud

## Notes

- Images are automatically resized to 800px width while maintaining aspect ratio
- The pipeline uses SIFT features with Lowe's ratio test (threshold: 0.7)
- Essential matrix estimation uses RANSAC with 99.9% confidence
- Point clouds are saved in PLY format with RGB colors
- Week 3 bundle adjustment includes coordinate normalization for numerical stability

## License

This project is part of a course assignment. Please refer to your institution's academic integrity policies.

## Acknowledgments

- OpenCV for computer vision algorithms
- SciPy for optimization routines
