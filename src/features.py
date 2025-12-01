"""
Feature detection and matching utilities.

This module provides functions for detecting and matching SIFT features
between images using Lowe's ratio test.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def find_and_filter_matches(img1_rgb: np.ndarray, img2_rgb: np.ndarray, 
                            ratio_threshold: float = 0.7) -> Tuple[List, List, List]:
    """
    Find and filter SIFT feature matches between two images using Lowe's Ratio Test.
    
    Args:
        img1_rgb (np.ndarray): First image in RGB format
        img2_rgb (np.ndarray): Second image in RGB format
        ratio_threshold (float): Lowe's ratio test threshold. Defaults to 0.7.
    
    Returns:
        Tuple[List, List, List]: 
            - Keypoints from image 1
            - Keypoints from image 2
            - List of good matches
    """
    gray1 = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2_rgb, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    good_matches = []
    
    if des1 is None or des2 is None:
        return kp1, kp2, []

    try:
        matches = bf.knnMatch(des1, des2, k=2)
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    except cv2.error:
        pass

    return kp1, kp2, good_matches


def get_matches(img1: np.ndarray, img2: np.ndarray, 
                ratio_threshold: float = 0.7) -> Tuple[Optional[np.ndarray], 
                                                       Optional[np.ndarray], 
                                                       List, Optional[List], 
                                                       Optional[List]]:
    """
    Get SIFT feature matches between two images using Lowe's ratio test.
    Returns matched points as numpy arrays.
    
    Args:
        img1 (np.ndarray): First image
        img2 (np.ndarray): Second image
        ratio_threshold (float): Lowe's ratio test threshold. Defaults to 0.7.
    
    Returns:
        Tuple containing:
            - pts1 (np.ndarray or None): Matched points from image 1, shape (N, 2)
            - pts2 (np.ndarray or None): Matched points from image 2, shape (N, 2)
            - good_matches (List): List of good match objects
            - kp1 (List or None): Keypoints from image 1
            - kp2 (List or None): Keypoints from image 2
    """
    # Convert RGB to grayscale if needed
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.dtype != np.uint8 or img1.shape[2] == 3 else img1
    else:
        gray1 = img1
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.dtype != np.uint8 or img2.shape[2] == 3 else img2
    else:
        gray2 = img2

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return None, None, [], None, None

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2, good_matches, kp1, kp2


def draw_matches(img1: np.ndarray, kp1: List, img2: np.ndarray, kp2: List, 
                 matches: List) -> np.ndarray:
    """
    Draw the filtered matches between two images.
    
    Args:
        img1 (np.ndarray): First image in RGB format
        kp1 (List): Keypoints from image 1
        img2 (np.ndarray): Second image in RGB format
        kp2 (List): Keypoints from image 2
        matches (List): List of match objects to draw
    
    Returns:
        np.ndarray: Image with matches drawn, in RGB format
    """
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    img_matches = cv2.drawMatches(
        img1_bgr, kp1,
        img2_bgr, kp2,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    return img_matches

