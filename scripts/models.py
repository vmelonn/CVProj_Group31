"""
Data models for Structure from Motion.

This module contains the core data structures used throughout the SFM pipeline.
"""

import numpy as np


class CameraPose:
    """
    Represents a camera pose with rotation and translation.
    
    Attributes:
        R (np.ndarray): 3x3 rotation matrix
        t (np.ndarray): 3x1 translation vector
    """
    
    def __init__(self, R=None, t=None):
        """
        Initialize camera pose.
        
        Args:
            R (np.ndarray, optional): 3x3 rotation matrix. Defaults to identity.
            t (np.ndarray, optional): 3x1 translation vector. Defaults to zeros.
        """
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros((3, 1))
        
        self.R = R
        self.t = t
    
    def __repr__(self):
        return f"CameraPose(R={self.R.shape}, t={self.t.shape})"


class MapPoint:
    """
    Represents a 3D point in the map with optional color information.
    
    Attributes:
        xyz (np.ndarray): 3D coordinates (x, y, z)
        color (np.ndarray): RGB color values [R, G, B]
    """
    
    def __init__(self, xyz, color=None):
        """
        Initialize map point.
        
        Args:
            xyz (array-like): 3D coordinates, shape (3,) or (3,1)
            color (array-like, optional): RGB color [R, G, B]. 
                Defaults to gray [128, 128, 128].
        """
        self.xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        
        if color is not None:
            self.color = np.asarray(color, dtype=np.uint8)
        else:
            self.color = np.array([128, 128, 128], dtype=np.uint8)
    
    def __repr__(self):
        return f"MapPoint(xyz={self.xyz}, color={self.color})"

