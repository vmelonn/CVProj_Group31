"""
Incremental Structure-from-Motion Reconstruction
Processes multiple images per wall to build 3D point clouds
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import open3d as o3d
import json
from reconstruction.feature_matching import detect_and_match
from reconstruction.camera_utils import get_camera_matrix, triangulate_points
from utils.file_utils import get_image_files

# --- CONFIGURATION ---
MIN_MATCH_COUNT = 10  # Minimum matches to accept a new frame
REPROJ_THRESH = 4.0   # PnP RANSAC Threshold
MAX_IMAGES = 15       # Limit images per wall for performance

# Output directories
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def process_folder(folder_path: str, wall_id: int):
    """Process a folder of images to create 3D reconstruction"""
    print(f"--- Starting Incremental SfM for Wall {wall_id} ---")
    
    # 1. Load Images
    image_files = get_image_files(folder_path)
    
    if len(image_files) < 2:
        print("Not enough images.")
        return

    # Use first 2 images for Initialization (Phase 1) 
    img0 = cv2.imread(image_files[0])
    img1 = cv2.imread(image_files[1])
    K, w, h = get_camera_matrix(img0)

    # Match 0 and 1
    pts0, pts1, kp0, kp1 = detect_and_match(img0, img1)
    
    if len(pts0) < 100:
        print("Initialization failed: Not enough matches.")
        return

    # Phase 1: Essential Matrix & Pose
    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]
    _, R, t, mask = cv2.recoverPose(E, pts0, pts1, K)

    # Initial Projection Matrices
    P0 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = K @ np.hstack((R, t))

    # Initial 3D Cloud
    cloud_points = triangulate_points(P0, P1, pts0, pts1)
    
    # Store Camera Poses (relative paths for web compatibility)
    cameras = []
    rel_path0 = os.path.relpath(image_files[0], Path(__file__).parent.parent)
    rel_path1 = os.path.relpath(image_files[1], Path(__file__).parent.parent)
    cameras.append({"rotation": np.eye(3).tolist(), "translation": [0,0,0], "filename": rel_path0.replace("\\", "/")})
    cameras.append({"rotation": R.tolist(), "translation": t.flatten().tolist(), "filename": rel_path1.replace("\\", "/")})

    # Keep track of last image data for chaining
    prev_img = img1
    prev_des = cv2.SIFT_create().detectAndCompute(prev_img, None)[1]
    prev_kp = kp1
    prev_P = P1
    
    # Store all 3D points and colors
    all_3d_points = [cloud_points]
    all_colors = []
    
    # Get colors for initial points
    colors0 = []
    for pt in pts0:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < w and 0 <= y < h:
            colors0.append(img0[y, x, ::-1] / 255.0)
        else:
            colors0.append([0.5,0.5,0.5])
    all_colors.append(np.array(colors0))

    # --- PHASE 2: INCREMENTAL LOOP ---
    for i in range(2, min(len(image_files), MAX_IMAGES)):
        curr_img_path = image_files[i]
        curr_img = cv2.imread(curr_img_path)
        print(f"Adding View {i}: {os.path.basename(curr_img_path)}")

        # Match Current vs Previous
        sift = cv2.SIFT_create()
        curr_kp, curr_des = sift.detectAndCompute(curr_img, None)
        
        # FLANN Match
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(prev_des, curr_des, k=2)
        
        good = []
        q_idx = []
        t_idx = []
        
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
                q_idx.append(m.queryIdx)
                t_idx.append(m.trainIdx)

        if len(good) < MIN_MATCH_COUNT:
            print("  -> Lost track. Skipping.")
            continue

        # Get matched 2D points
        pts_prev_2d = np.float32([prev_kp[idx].pt for idx in q_idx])
        pts_curr_2d = np.float32([curr_kp[idx].pt for idx in t_idx])

        E_new, mask_new = cv2.findEssentialMat(pts_prev_2d, pts_curr_2d, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E_new is None: 
            continue
        
        _, R_rel, t_rel, _ = cv2.recoverPose(E_new, pts_prev_2d, pts_curr_2d, K)

        # Accumulate Pose
        last_cam = cameras[-1]
        R_prev_stored = np.array(last_cam['rotation'])
        t_prev_stored = np.array(last_cam['translation']).reshape(3,1)

        # Update Pose
        R_curr = R_rel @ R_prev_stored
        t_curr = R_rel @ t_prev_stored + t_rel
        
        # Save Camera with relative path
        rel_path = os.path.relpath(curr_img_path, Path(__file__).parent.parent)
        cameras.append({
            "rotation": R_curr.tolist(),
            "translation": t_curr.flatten().tolist(),
            "filename": rel_path.replace("\\", "/")
        })

        # Triangulate New Points
        P_curr = K @ np.hstack((R_curr, t_curr))
        new_cloud = triangulate_points(prev_P, P_curr, pts_prev_2d, pts_curr_2d)
        
        # Filter noise
        mask_z = (new_cloud[:, 2] > -50) & (new_cloud[:, 2] < 50)
        new_cloud = new_cloud[mask_z]
        pts_curr_2d_filtered = pts_curr_2d[mask_z]

        # Get colors
        new_colors = []
        for pt in pts_curr_2d_filtered:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                new_colors.append(curr_img[y, x, ::-1] / 255.0)
            else:
                new_colors.append([0.5,0.5,0.5])

        all_3d_points.append(new_cloud)
        all_colors.append(np.array(new_colors))

        # Update "Previous" for next loop
        prev_P = P_curr
        prev_kp = curr_kp
        prev_des = curr_des

    # --- SAVE OUTPUT ---
    final_points = np.vstack(all_3d_points)
    final_colors = np.vstack(all_colors)
    
    # Save PLY
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_points)
    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    
    # Statistical Outlier Removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    output_ply = OUTPUT_DIR / f"wall_{wall_id}.ply"
    o3d.io.write_point_cloud(str(output_ply), pcd)
    print(f"Saved {output_ply} ({len(final_points)} points)")
    
    # Save Cameras JSON
    output_json = OUTPUT_DIR / f"cameras_wall_{wall_id}.json"
    with open(output_json, 'w') as f:
        json.dump({"cameras": cameras}, f, indent=2)
    print(f"Saved {output_json}")


def main():
    """Main entry point"""
    base_dir = Path(__file__).parent.parent / "data" / "Dataset"
    
    for i in range(1, 5):
        wall_path = base_dir / f"Wall {i}"
        if wall_path.exists():
            process_folder(str(wall_path), i)
        else:
            print(f"Skipping Wall {i} (Folder not found)")


if __name__ == "__main__":
    main()

