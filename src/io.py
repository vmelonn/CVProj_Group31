"""
Input/Output utilities for image loading and point cloud saving.
"""

import cv2
import numpy as np
import glob
import os
from typing import List, Optional, Tuple


def load_and_resize(image_path: str, width: int = 800) -> Optional[np.ndarray]:
    """
    Load an image from a file path and resize it to a fixed width
    while maintaining the aspect ratio.
    
    Args:
        image_path (str): Path to the image file
        width (int): Target width in pixels. Defaults to 800.
    
    Returns:
        np.ndarray: Resized RGB image, or None if loading failed
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return None

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize maintaining aspect ratio
    (h, w) = image.shape[:2]
    r = width / float(w)
    dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized


def load_image_dataset(folder: str = 'dataset', pattern: str = 'img_*.jpeg', 
                      width: int = 800) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all images from a folder matching a pattern.
    
    Args:
        folder (str): Folder path containing images
        pattern (str): Glob pattern for image files
        width (int): Target width for resizing
    
    Returns:
        Tuple[List[np.ndarray], List[str]]: 
            - List of loaded images (RGB format)
            - List of corresponding file paths
    """
    image_files = sorted(glob.glob(os.path.join(folder, pattern)))
    images = []
    valid_files = []
    
    for f in image_files:
        img = load_and_resize(f, width)
        if img is not None:
            images.append(img)
            valid_files.append(f)
    
    print(f"Loaded {len(images)} images from {folder}")
    return images, valid_files


def save_ply(filename: str, points_3d: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    """
    Save 3D point cloud to PLY format.
    
    Args:
        filename (str): Output PLY file path
        points_3d (np.ndarray): 3D points, shape (N, 3)
        colors (np.ndarray, optional): RGB colors, shape (N, 3). 
            Defaults to zeros (black).
    
    Note:
        Colors are saved in BGR order as per PLY convention.
    """
    points_3d = np.array(points_3d)

    if colors is not None:
        colors = np.array(colors)
    else:
        colors = np.zeros_like(points_3d)

    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points_3d, colors):
            # Note: BGR order (c[2], c[1], c[0]) as in provided code
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n")
    
    print(f"Saved {len(points_3d)} points to {filename}")

