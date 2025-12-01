"""
Structure from Motion (SFM) Package

A Python package for performing Structure from Motion reconstruction
from image sequences.
"""

__version__ = "1.0.0"

from .models import CameraPose, MapPoint
from .io import load_and_resize, load_image_dataset, save_ply
from .features import find_and_filter_matches, get_matches, draw_matches
from .geometry import (
    build_intrinsic_matrix,
    rvec_tvec_from_pose,
    pose_from_rvec_tvec,
    project_points,
    triangulate_and_check,
    triangulate_between_views
)
from .reconstruction import (
    evaluate_pair,
    find_best_pair,
    two_view_reconstruction,
    match_2d_3d,
    solve_pnp,
    bundle_adjust,
    incremental_sfm,
    match_features_to_3d,  # Week 3
    estimate_pose_pnp  # Week 3
)
from .visualization import (
    display_image_grid,
    visualize_matches,
    plot_3d_pointcloud,
    plot_2d_projections
)

__all__ = [
    # Models
    'CameraPose',
    'MapPoint',
    # I/O
    'load_and_resize',
    'load_image_dataset',
    'save_ply',
    # Features
    'find_and_filter_matches',
    'get_matches',
    'draw_matches',
    # Geometry
    'build_intrinsic_matrix',
    'rvec_tvec_from_pose',
    'pose_from_rvec_tvec',
    'project_points',
    'triangulate_and_check',
    'triangulate_between_views',
    # Reconstruction
    'evaluate_pair',
    'find_best_pair',
    'two_view_reconstruction',
    'match_2d_3d',
    'solve_pnp',
    'bundle_adjust',
    'incremental_sfm',
    # Week 3 functions
    'match_features_to_3d',
    'estimate_pose_pnp',
    # Visualization
    'display_image_grid',
    'visualize_matches',
    'plot_3d_pointcloud',
    'plot_2d_projections',
]

