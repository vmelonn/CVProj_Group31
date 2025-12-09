"""
Camera Utilities Module
Camera matrix estimation and pose utilities
"""
import numpy as np
import cv2
from typing import Tuple


def get_camera_matrix(img: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Estimate camera intrinsic matrix K based on image dimensions
    
    Args:
        img: Input image (BGR)
        
    Returns:
        Tuple of (K, width, height) where K is 3x3 camera matrix
    """
    h, w = img.shape[:2]
    # Focal length approx: 1.2 * max dimension
    f = 1.2 * max(w, h)
    K = np.array([
        [f, 0, w / 2],
        [0, f, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    return K, w, h


def triangulate_points(P1: np.ndarray, P2: np.ndarray, 
                       pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    """
    Triangulate 2D points to 3D using projection matrices
    
    Args:
        P1: First camera projection matrix (3x4)
        P2: Second camera projection matrix (3x4)
        pts1: 2D points in first image (Nx2)
        pts2: 2D points in second image (Nx2)
        
    Returns:
        3D points (Nx3 array)
    """
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D = pts4D[:3] / pts4D[3]
    return pts3D.T

