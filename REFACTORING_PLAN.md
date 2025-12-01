# Refactoring Plan: Jupyter Notebook to Python Package

## Project Structure
```
CV Proj/
├── src/
│   ├── __init__.py
│   ├── models.py          # CameraPose, MapPoint classes
│   ├── io.py              # Image loading, PLY file I/O
│   ├── features.py        # SIFT feature detection and matching
│   ├── geometry.py        # Camera geometry, pose conversions, triangulation
│   ├── reconstruction.py  # Two-view and incremental SFM
│   └── visualization.py   # Plotting and visualization functions
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .gitignore             # Git ignore file
├── dataset/               # Image dataset (existing)
└── outputs/               # Generated outputs (PLY files, visualizations)
```

## File Organization

### 1. `src/models.py`
**Purpose:** Data classes for SFM pipeline
- `CameraPose` class (R, t)
- `MapPoint` class (xyz, color)

### 2. `src/io.py`
**Purpose:** File I/O operations
- `load_and_resize()` - Load and preprocess images
- `load_image_dataset()` - Load entire dataset
- `save_ply()` - Save 3D point clouds to PLY format

### 3. `src/features.py`
**Purpose:** Feature detection and matching
- `find_and_filter_matches()` - SIFT + Lowe's ratio test
- `get_matches()` - Alternative matching function (returns points)
- `draw_matches()` - Visualize matches

### 4. `src/geometry.py`
**Purpose:** Camera geometry and 3D operations
- `build_intrinsic_matrix()` - Create K matrix from image dimensions
- `rvec_tvec_from_pose()` - Convert R,t to rvec,tvec
- `pose_from_rvec_tvec()` - Convert rvec,tvec to R,t
- `project_points()` - Project 3D points to 2D
- `triangulate_and_check()` - Triangulate with cheirality check
- `triangulate_between_views()` - Triangulate between two views

### 5. `src/reconstruction.py`
**Purpose:** Structure from Motion algorithms
- `evaluate_pair()` - Evaluate image pair for reconstruction quality
- `find_best_pair()` - Search for optimal image pair
- `two_view_reconstruction()` - Perform two-view reconstruction
- `match_2d_3d()` - Match 2D features to 3D map points
- `solve_pnp()` - Solve PnP for camera pose estimation
- `bundle_adjust()` - Bundle adjustment optimization
- `incremental_sfm()` - Incremental structure from motion

### 6. `src/visualization.py`
**Purpose:** Visualization utilities
- `display_image_grid()` - Display grid of images
- `visualize_matches()` - Visualize feature matches
- `plot_3d_pointcloud()` - 3D point cloud visualization
- `plot_2d_projections()` - 2D projections of point cloud

### 7. `main.py`
**Purpose:** Main execution script
- Command-line interface
- Run feature matching
- Run two-view reconstruction
- Run incremental SFM
- Generate outputs

## Implementation Steps

1. ✅ Create project directory structure
2. ✅ Create `src/models.py` with data classes
3. ✅ Create `src/io.py` with I/O functions
4. ✅ Create `src/features.py` with feature matching
5. ✅ Create `src/geometry.py` with geometry functions
6. ✅ Create `src/reconstruction.py` with SFM algorithms
7. ✅ Create `src/visualization.py` with plotting functions
8. ✅ Create `main.py` with CLI interface
9. ✅ Create `requirements.txt`
10. ✅ Create `README.md`
11. ✅ Create `.gitignore`
12. ✅ Test all functionality

## Dependencies
- opencv-python (cv2)
- numpy
- matplotlib
- scipy (for bundle adjustment)

## Notes
- Keep the notebook for reference/documentation
- All functions should have docstrings
- Use type hints where appropriate
- Follow PEP 8 style guidelines

