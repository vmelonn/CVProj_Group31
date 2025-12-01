"""
Structure from Motion reconstruction algorithms.

This module provides functions for two-view and incremental SFM reconstruction.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Optional, Dict
import copy

from .models import CameraPose, MapPoint
from .features import get_matches
from .geometry import build_intrinsic_matrix, triangulate_and_check, triangulate_between_views
from .geometry import rvec_tvec_from_pose, pose_from_rvec_tvec, project_points


def evaluate_pair(img1: np.ndarray, img2: np.ndarray, idx1: int, idx2: int) -> Tuple[int, int, int, bool, Optional[Dict]]:
    """
    Evaluate a pair of images for reconstruction quality.
    
    Args:
        img1 (np.ndarray): First image
        img2 (np.ndarray): Second image
        idx1 (int): Index of first image
        idx2 (int): Index of second image
    
    Returns:
        Tuple containing:
            - num_matches (int): Number of feature matches
            - num_inliers (int): Number of inliers after essential matrix estimation
            - num_valid_3d (int): Number of valid 3D points (in front of cameras)
            - success (bool): Whether reconstruction was successful
            - data (Dict or None): Reconstruction data if successful
    """
    # Get matches
    pts1, pts2, good_matches, kp1, kp2 = get_matches(img1, img2)

    if pts1 is None or len(good_matches) < 8:
        return 0, 0, 0, False, None

    # Build intrinsic matrix K
    K = build_intrinsic_matrix(img1.shape)

    # Estimate Essential Matrix
    try:
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        if E is None:
            return len(good_matches), 0, 0, False, None
    except:
        return len(good_matches), 0, 0, False, None

    # Get inliers
    pts1_inliers = pts1[mask.ravel() == 1]
    pts2_inliers = pts2[mask.ravel() == 1]
    num_inliers = pts1_inliers.shape[0]

    if num_inliers < 8:
        return len(good_matches), num_inliers, 0, False, None

    # Decompose Essential Matrix
    try:
        R1, R2, t = cv2.decomposeEssentialMat(E)
    except:
        return len(good_matches), num_inliers, 0, False, None

    # Check all 4 possible poses
    poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    best_pts = None
    best_count = -1
    best_pose = None

    for R, t_vec in poses:
        pts3d, count = triangulate_and_check(R, t_vec, pts1_inliers, pts2_inliers, K)
        if count > best_count:
            best_count = count
            best_pts = pts3d
            best_pose = (R, t_vec)

    if best_count > 0:
        return len(good_matches), num_inliers, best_count, True, {
            'R': best_pose[0], 't': best_pose[1],
            'points_3d': best_pts,
            'pts1_inliers': pts1_inliers, 'pts2_inliers': pts2_inliers,
            'kp1': kp1, 'kp2': kp2,
            'good_matches': good_matches, 'E': E, 'K': K
        }
    else:
        return len(good_matches), num_inliers, 0, False, None


def find_best_pair(images: List[np.ndarray], gaps_to_try: List[int] = None) -> Tuple[int, int, Dict]:
    """
    Search through multiple image pairs to find the best one for reconstruction.
    
    Args:
        images (List[np.ndarray]): List of images
        gaps_to_try (List[int], optional): List of gaps to try. Defaults to [5, 10, 15, 20, 25, 30, 35, 40].
    
    Returns:
        Tuple containing:
            - idx1 (int): Index of first image
            - idx2 (int): Index of second image
            - data (Dict): Reconstruction data
    """
    if gaps_to_try is None:
        gaps_to_try = [5, 10, 15, 20, 25, 30, 35, 40]
    
    results = []

    print("Searching for the best image pair...")
    print("=" * 60)

    for gap in gaps_to_try:
        for start_idx in range(len(images) - gap):
            idx1 = start_idx
            idx2 = start_idx + gap

            img1 = images[idx1]
            img2 = images[idx2]

            num_matches, num_inliers, num_valid_3d, success, data = evaluate_pair(img1, img2, idx1, idx2)

            if success:
                # Score: prioritize valid 3D points, then inliers, then matches
                score = num_valid_3d * 1000 + num_inliers * 10 + num_matches
                results.append({
                    'idx1': idx1, 'idx2': idx2, 'gap': gap,
                    'num_matches': num_matches, 'num_inliers': num_inliers,
                    'num_valid_3d': num_valid_3d, 'score': score, 'data': data
                })
                print(f"Pair ({idx1:2d}, {idx2:2d}) gap={gap:2d}: "
                      f"{num_matches:3d} matches, {num_inliers:3d} inliers, "
                      f"{num_valid_3d:3d} valid 3D points ✓")

    if not results:
        print("\nNo successful pairs found! Trying all pairs with smaller gaps...")
        # Fallback: try smaller gaps
        for gap in range(3, 10):
            for start_idx in range(len(images) - gap):
                idx1 = start_idx
                idx2 = start_idx + gap
                img1 = images[idx1]
                img2 = images[idx2]
                num_matches, num_inliers, num_valid_3d, success, data = evaluate_pair(img1, img2, idx1, idx2)
                if success:
                    score = num_valid_3d * 1000 + num_inliers * 10 + num_matches
                    results.append({
                        'idx1': idx1, 'idx2': idx2, 'gap': gap,
                        'num_matches': num_matches, 'num_inliers': num_inliers,
                        'num_valid_3d': num_valid_3d, 'score': score, 'data': data
                    })

    if results:
        # Sort by score (best first)
        results.sort(key=lambda x: x['score'], reverse=True)
        best_pair = results[0]

        print("\n" + "=" * 60)
        print("BEST PAIR SELECTED:")
        print(f"  Images: {best_pair['idx1']} and {best_pair['idx2']} (gap={best_pair['gap']})")
        print(f"  Matches: {best_pair['num_matches']}")
        print(f"  Inliers: {best_pair['num_inliers']}")
        print(f"  Valid 3D Points: {best_pair['num_valid_3d']}")
        print("=" * 60)

        return best_pair['idx1'], best_pair['idx2'], best_pair['data']
    else:
        print("\nERROR: Could not find any suitable pair. Using fallback pair (0, 20).")
        idx1 = 0
        idx2 = min(20, len(images) - 1)
        _, _, _, _, data = evaluate_pair(images[idx1], images[idx2], idx1, idx2)
        return idx1, idx2, data


def match_2d_3d(img: np.ndarray, kp: List, des: np.ndarray, map_points: List[MapPoint], 
                K: np.ndarray, dist_thresh: float = 4.0) -> Tuple[Optional[np.ndarray], 
                                                                   Optional[np.ndarray], 
                                                                   List, List, Optional[np.ndarray], 
                                                                   Optional[np.ndarray]]:
    """
    Match 2D image features to 3D map points.
    
    Args:
        img (np.ndarray): Input image
        kp (List): Keypoints from image
        des (np.ndarray): Descriptors from image
        map_points (List[MapPoint]): List of 3D map points
        K (np.ndarray): Intrinsic matrix
        dist_thresh (float): Distance threshold for matching
    
    Returns:
        Tuple containing matched 2D points, 3D points, and other data
    """
    pts3d = np.array([p.xyz for p in map_points])
    colors = np.array([p.color for p in map_points])
    proj = project_points(pts3d, np.eye(3), np.zeros((3,1)), K)

    pts2d = []
    pts3d_matched = []
    for p2, p3 in zip(proj, pts3d):
        dists = np.linalg.norm(np.array([k.pt for k in kp]) - p2, axis=1)
        j = np.argmin(dists)
        if dists[j] < dist_thresh:
            pts2d.append(kp[j].pt)
            pts3d_matched.append(p3)
    
    if len(pts2d) < 6:
        return None, None, [], [], None, None
    
    return np.float32(pts2d), np.float32(pts3d_matched), pts2d, pts3d_matched, proj, colors


def solve_pnp(K: np.ndarray, pts2d: np.ndarray, pts3d: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Solve Perspective-n-Point problem to estimate camera pose.
    
    Args:
        K (np.ndarray): Intrinsic matrix
        pts2d (np.ndarray): 2D image points, shape (N, 2)
        pts3d (np.ndarray): 3D world points, shape (N, 3)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Rotation matrix R and translation vector t, or (None, None) if failed
    """
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, None,
        iterationsCount=200, reprojectionError=3.0, confidence=0.999
    )
    if not retval or inliers is None or len(inliers) < 6:
        return None, None
    R, t = pose_from_rvec_tvec(rvec, tvec)
    return R, t


def estimate_pose_pnp(points_3d: np.ndarray, points_2d: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Estimate camera pose using PnP (Week 3 function).
    
    Args:
        points_3d (np.ndarray): 3D world points, shape (N, 3)
        points_2d (np.ndarray): 2D image points, shape (N, 2)
        K (np.ndarray): Intrinsic matrix
    
    Returns:
        Tuple[np.ndarray, np.ndarray, bool]: R, t, success flag
    """
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d, points_2d, K, None,
        iterationsCount=200, reprojectionError=3.0, confidence=0.999
    )
    if not retval or inliers is None or len(inliers) < 4:
        return None, None, False
    R, t = pose_from_rvec_tvec(rvec, tvec)
    return R, t, True


def match_features_to_3d(image: np.ndarray, points_3d: np.ndarray, 
                         point_to_descriptor_map: Dict[int, np.ndarray], 
                         K: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, np.ndarray]:
    """
    Match 2D features in a new image to existing 3D points using SIFT descriptors (Week 3 function).
    
    Args:
        image (np.ndarray): New image
        points_3d (np.ndarray): Existing 3D points, shape (N, 3)
        point_to_descriptor_map (Dict): Map from point index to descriptor
        K (np.ndarray): Intrinsic matrix
    
    Returns:
        Tuple containing:
            - points_3d_matched: Matched 3D points
            - points_2d_matched: Matched 2D points
            - point_indices_matched: Indices into original points_3d array
            - kp_new: Keypoints in new image
            - des_new: Descriptors in new image
    """
    if len(point_to_descriptor_map) == 0:
        return np.array([]), np.array([]), np.array([]), [], np.array([])
    
    # Extract features from new image
    sift = cv2.SIFT_create()
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    kp_new, des_new = sift.detectAndCompute(gray, None)
    
    if des_new is None or len(kp_new) < 4:
        return np.array([]), np.array([]), np.array([]), kp_new, des_new if des_new is not None else np.array([])
    
    # Prepare existing descriptors
    existing_indices = []
    existing_descriptors = []
    for idx, desc in point_to_descriptor_map.items():
        existing_indices.append(idx)
        existing_descriptors.append(desc)
    
    existing_descriptors = np.array(existing_descriptors)
    
    # Match using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des_new, existing_descriptors, k=2)
    
    # Filter matches with Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        return np.array([]), np.array([]), np.array([]), kp_new, des_new
    
    # Extract matched points
    points_3d_matched = []
    points_2d_matched = []
    point_indices_matched = []
    
    for match in good_matches:
        pt_3d_idx = existing_indices[match.trainIdx]
        points_3d_matched.append(points_3d[pt_3d_idx])
        points_2d_matched.append(kp_new[match.queryIdx].pt)
        point_indices_matched.append(pt_3d_idx)
    
    return (np.array(points_3d_matched), np.array(points_2d_matched), 
            np.array(point_indices_matched), kp_new, des_new)


def bundle_adjust(K: np.ndarray, poses: List[Tuple], points_3d: np.ndarray, obs: List[Tuple]) -> Tuple[List, np.ndarray]:
    """
    Perform bundle adjustment optimization.
    
    Args:
        K (np.ndarray): Intrinsic matrix
        poses (List[Tuple]): List of (rvec, tvec) tuples for each camera
        points_3d (np.ndarray): 3D points, shape (N, 3)
        obs (List[Tuple]): Observations as (pose_idx, point_idx, u, v) tuples
    
    Returns:
        Tuple[List, np.ndarray]: Optimized poses and points
    """
    def pack(poses, pts):
        x = []
        for r, t in poses:
            x.extend(r.ravel())
            x.extend(t.ravel())
        x.extend(pts.ravel())
        return np.array(x)

    def unpack(x, n_poses, n_pts):
        poses = []
        idx = 0
        for _ in range(n_poses):
            r = x[idx:idx+3]
            t = x[idx+3:idx+6]
            idx += 6
            poses.append((r.reshape(3,1), t.reshape(3,1)))
        pts = x[idx:].reshape(n_pts, 3)
        return poses, pts

    def residuals(x, n_poses, n_pts):
        poses, pts = unpack(x, n_poses, n_pts)
        res = []
        for (pi, qi, u, v) in obs:
            rvec, tvec = poses[pi]
            pt = pts[qi].reshape(1, 3)
            proj, _ = cv2.projectPoints(pt, rvec, tvec, K, None)
            proj = proj.ravel()
            res.extend(proj - np.array([u, v]))
        return np.array(res)

    x0 = pack(poses, points_3d)
    res = least_squares(residuals, x0, verbose=0, x_scale='jac', ftol=1e-4, method='trf',
                        args=(len(poses), len(points_3d)))
    poses_opt, pts_opt = unpack(res.x, len(poses), len(points_3d))
    return poses_opt, pts_opt


def incremental_sfm(images: List[np.ndarray], K: Optional[np.ndarray], 
                   init_idx0: int, init_idx1: int) -> Tuple[List[Optional[CameraPose]], List[MapPoint]]:
    """
    Perform incremental Structure from Motion reconstruction.
    
    Args:
        images (List[np.ndarray]): List of images
        K (np.ndarray, optional): Intrinsic matrix. If None, will be extracted from first pair.
        init_idx0 (int): Index of first image for initialization
        init_idx1 (int): Index of second image for initialization
    
    Returns:
        Tuple containing:
            - poses (List[CameraPose]): Camera poses for each image (None if not estimated)
            - map_points (List[MapPoint]): Reconstructed 3D map points
    """
    poses = [None] * len(images)
    map_points = []
    keypoints = []
    descriptors = []

    sift = cv2.SIFT_create()
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)

    R0 = np.eye(3)
    t0 = np.zeros((3, 1))
    poses[init_idx0] = CameraPose(R0, t0)
    
    _, _, _, success, data = evaluate_pair(images[init_idx0], images[init_idx1], init_idx0, init_idx1)
    if not success or data is None:
        raise ValueError(f"Failed to initialize with pair ({init_idx0}, {init_idx1})")
    
    # Use K from data if K parameter is None, otherwise use the provided K
    K_used = K if K is not None else data['K']
    
    R1, t1 = data['R'], data['t']
    poses[init_idx1] = CameraPose(R1, t1)

    pts3d = data['points_3d']
    colors = []
    for pt1 in data['pts1_inliers']:
        x, y = map(int, pt1)
        colors.append(images[init_idx0][y, x] if y < images[init_idx0].shape[0] and x < images[init_idx0].shape[1] else [128, 128, 128])
    
    for p, c in zip(pts3d, colors):
        map_points.append(MapPoint(p, c))

    ordered_idxs = list(range(len(images)))
    for idx in ordered_idxs:
        if poses[idx] is not None:
            continue
        
        kp = keypoints[idx]
        des = descriptors[idx]
        pts2d, pts3d_matched, _, _, _, _ = match_2d_3d(images[idx], kp, des, map_points, K_used)
        
        if pts2d is None or len(pts2d) < 8:
            continue
        
        R, t = solve_pnp(K_used, pts2d, pts3d_matched)
        if R is None:
            continue
        
        poses[idx] = CameraPose(R, t)

        prev = max([i for i, p in enumerate(poses) if p is not None and i < idx], default=None)
        if prev is not None:
            pts_prev, pts_cur, good_matches, kp_prev, kp_cur = get_matches(images[prev], images[idx])
            if pts_prev is not None and len(good_matches) >= 8:
                new_pts3d = triangulate_between_views(poses[prev].R, poses[prev].t, R, t, pts_prev, pts_cur, K_used)
                for p in new_pts3d:
                    map_points.append(MapPoint(p))
    
    return poses, map_points

