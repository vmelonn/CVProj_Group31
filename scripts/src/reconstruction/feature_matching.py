"""
Feature Detection and Matching Module
Extracted from reconstruct_wall.py for reusability
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional


def detect_and_match(img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List, List]:
    """
    Feature matching using SIFT and Ratio Test
    
    Args:
        img1: First image (BGR)
        img2: Second image (BGR)
        
    Returns:
        Tuple of (pts1, pts2, kp1, kp2) where:
        - pts1, pts2: Matched 2D points (Nx2 float32 arrays)
        - kp1, kp2: Keypoint lists
    """
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return np.array([]), np.array([]), [], []

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []
    
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.float32(pts1), np.float32(pts2), kp1, kp2


def count_matches(img1_path: str, img2_path: str) -> int:
    """
    Count feature matches between two images (for pair selection)
    
    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        
    Returns:
        Number of good matches found
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 0

    # Resize huge images for faster checking
    h, w = img1.shape
    if w > 1000:
        scale = 1000 / w
        img1 = cv2.resize(img1, (0, 0), fx=scale, fy=scale)
        img2 = cv2.resize(img2, (0, 0), fx=scale, fy=scale)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except:
        return 0

    good_count = 0
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_count += 1
            
    return good_count

