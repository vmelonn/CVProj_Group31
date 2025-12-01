"""
Camera geometry and 3D operations.

This module provides functions for camera pose conversions, point projections,
and triangulation operations.
"""

import cv2
import numpy as np
from typing import Tuple


def build_intrinsic_matrix(img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Build camera intrinsic matrix K from image dimensions.
    
    Args:
        img_shape (Tuple[int, int]): Image shape (height, width)
    
    Returns:
        np.ndarray: 3x3 intrinsic matrix K
    """
    h, w = img_shape[:2]
    fx = fy = w
    cx = w / 2
    cy = h / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return K


def rvec_tvec_from_pose(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert rotation matrix and translation vector to rotation vector and translation.
    
    Args:
        R (np.ndarray): 3x3 rotation matrix
        t (np.ndarray): 3x1 translation vector
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation vector (3x1) and translation vector (3x1)
    """
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t.reshape(3, 1)


def pose_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert rotation vector and translation to rotation matrix and translation vector.
    
    Args:
        rvec (np.ndarray): Rotation vector (3x1)
        tvec (np.ndarray): Translation vector (3x1)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation matrix (3x3) and translation vector (3x1)
    """
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3, 1)


def project_points(points_3d: np.ndarray, R: np.ndarray, t: np.ndarray, 
                   K: np.ndarray) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d (np.ndarray): 3D points, shape (N, 3)
        R (np.ndarray): 3x3 rotation matrix
        t (np.ndarray): 3x1 translation vector
        K (np.ndarray): 3x3 intrinsic matrix
    
    Returns:
        np.ndarray: Projected 2D points, shape (N, 2)
    """
    rvec, tvec = rvec_tvec_from_pose(R, t)
    pts = cv2.projectPoints(points_3d, rvec, tvec, K, None)[0].reshape(-1, 2)
    return pts


def triangulate_and_check(R: np.ndarray, t: np.ndarray, pts1: np.ndarray, 
                         pts2: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Triangulate 3D points from two views and check cheirality (points in front of cameras).
    
    Args:
        R (np.ndarray): 3x3 rotation matrix (relative pose)
        t (np.ndarray): 3x1 translation vector (relative pose)
        pts1 (np.ndarray): Points in first image, shape (N, 2)
        pts2 (np.ndarray): Points in second image, shape (N, 2)
        K (np.ndarray): 3x3 intrinsic matrix
    
    Returns:
        Tuple[np.ndarray, int]: 
            - Triangulated 3D points, shape (N, 3)
            - Number of points in front of both cameras
    """
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))

    pts4d_h = cv2.triangulatePoints(P0, P1, pts1.T, pts2.T)
    pts4d = pts4d_h[:3] / pts4d_h[3]

    z1 = pts4d[2]
    z2 = (R @ pts4d + t.reshape(3, 1))[2]
    count_front = np.sum((z1 > 0) & (z2 > 0))

    return pts4d.T, count_front


def triangulate_between_views(R0: np.ndarray, t0: np.ndarray, R1: np.ndarray, 
                              t1: np.ndarray, pts0: np.ndarray, pts1: np.ndarray, 
                              K: np.ndarray) -> np.ndarray:
    """
    Triangulate 3D points between two camera views.
    
    Args:
        R0 (np.ndarray): 3x3 rotation matrix of first camera
        t0 (np.ndarray): 3x1 translation vector of first camera
        R1 (np.ndarray): 3x3 rotation matrix of second camera
        t1 (np.ndarray): 3x1 translation vector of second camera
        pts0 (np.ndarray): Points in first image, shape (N, 2)
        pts1 (np.ndarray): Points in second image, shape (N, 2)
        K (np.ndarray): 3x3 intrinsic matrix
    
    Returns:
        np.ndarray: Triangulated 3D points, shape (N, 3)
    """
    P0 = K @ np.hstack((R0, t0))
    P1 = K @ np.hstack((R1, t1))
    pts4d = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    return pts3d

