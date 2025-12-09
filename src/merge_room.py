"""
Merge Multiple Wall Reconstructions into Unified Room
Applies transformations and scaling to align walls
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import open3d as o3d
import numpy as np
import json

INPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def transform_camera_pose(R_old: np.ndarray, t_old: np.ndarray, T_total: np.ndarray):
    """
    Transforms a single camera pose (R, t) using a combined matrix T_total.
    T_total includes Centering -> Scaling -> Positioning.
    """
    R_transform = T_total[:3, :3]
    t_transform = T_total[:3, 3]

    # 1. Convert t_old to actual Camera Center (C_old)
    C_old = -R_old.T @ t_old

    # 2. Transform Camera Orientation
    R_new = R_transform @ R_old

    # 3. Transform Camera Center (Apply the full T_total)
    C_new = R_transform @ C_old + t_transform

    # 4. Recompute the new translation vector t_new
    t_new = -R_new @ C_new

    return R_new, t_new


def process_wall(wall_id: int, T_position: np.ndarray, scale_factor: float, all_cameras_list: list):
    """Process a single wall: load, transform, and merge"""
    ply_file = INPUT_DIR / f"wall_{wall_id}.ply"
    json_file = INPUT_DIR / f"cameras_wall_{wall_id}.json"
    
    print(f"Processing wall_{wall_id}...")
    
    try:
        pcd = o3d.io.read_point_cloud(str(ply_file))
    except:
        print(f"Warning: Could not find {ply_file}")
        return None

    # Visual Colors
    colors = [[1, 0.7, 0], [0, 0.65, 0.93], [0, 0.8, 0], [0.8, 0, 0]]
    pcd.paint_uniform_color(colors[wall_id-1])

    # --- STEP 1: CALCULATE CENTERING MATRIX ---
    center = pcd.get_center()
    T_center = np.eye(4)
    T_center[:3, 3] = -center

    # --- STEP 2: CALCULATE SCALE MATRIX ---
    S_scale = np.eye(4) * scale_factor
    S_scale[3, 3] = 1.0

    # --- COMBINE ALL MATRICES ---
    T_total = T_position @ S_scale @ T_center

    # --- APPLY TO POINTS ---
    pcd.transform(T_total)

    # --- APPLY TO CAMERAS ---
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        for cam in data['cameras']:
            R_old = np.array(cam['rotation'])
            t_old = np.array(cam['translation'])
            
            # Apply the transformation to the camera
            R_new, t_new = transform_camera_pose(R_old, t_old, T_total)
            
            cam['rotation'] = R_new.tolist()
            cam['translation'] = t_new.tolist()
            cam['wall_id'] = wall_id
            all_cameras_list.append(cam)
            
    except FileNotFoundError:
        print(f"Warning: Could not find {json_file}")

    return pcd


def main():
    """Main entry point"""
    all_cameras = []
    combined_pcd = o3d.geometry.PointCloud()

    # --- SCALES (From diagnostic output) ---
    scale1 = 1.0
    scale2 = 2.2225 / 0.3956
    scale3 = 2.2225 / 1.0353
    scale4 = 2.2225 / 0.4801
    
    # Target size for the room box
    ROOM_SIZE = 2.15

    # --- WALL 1 (Base - Back Wall) ---
    T1 = np.eye(4)
    pcd1 = process_wall(1, T1, scale1, all_cameras)
    if pcd1: 
        combined_pcd += pcd1

    # --- WALL 2 (Right Wall) ---
    theta2 = np.radians(90)
    R2 = np.array([[np.cos(theta2), 0, np.sin(theta2)], [0, 1, 0], [-np.sin(theta2), 0, np.cos(theta2)]])
    t2 = np.array([ROOM_SIZE/2, 0.0, ROOM_SIZE/2]) 
    T2 = np.eye(4)
    T2[:3, :3] = R2
    T2[:3, 3] = t2
    pcd2 = process_wall(2, T2, scale2, all_cameras)
    if pcd2: 
        combined_pcd += pcd2

    # --- WALL 3 (Front/Opposite Wall) ---
    theta3 = np.radians(180)
    R3 = np.array([[np.cos(theta3), 0, np.sin(theta3)], [0, 1, 0], [-np.sin(theta3), 0, np.cos(theta3)]])
    t3 = np.array([0.0, 0.0, ROOM_SIZE]) 
    T3 = np.eye(4)
    T3[:3, :3] = R3
    T3[:3, 3] = t3
    pcd3 = process_wall(3, T3, scale3, all_cameras)
    if pcd3: 
        combined_pcd += pcd3

    # --- WALL 4 (Left Wall) ---
    theta4 = np.radians(270)
    R4 = np.array([[np.cos(theta4), 0, np.sin(theta4)], [0, 1, 0], [-np.sin(theta4), 0, np.cos(theta4)]])
    t4 = np.array([-ROOM_SIZE/2, 0.0, ROOM_SIZE/2]) 
    T4 = np.eye(4)
    T4[:3, :3] = R4
    T4[:3, 3] = t4
    pcd4 = process_wall(4, T4, scale4, all_cameras)
    if pcd4: 
        combined_pcd += pcd4

    # --- SAVE ---
    output_ply = OUTPUT_DIR / "merged_room.ply"
    o3d.io.write_point_cloud(str(output_ply), combined_pcd)
    
    output_json = OUTPUT_DIR / "all_cameras.json"
    final_json = {"cameras": all_cameras}
    with open(output_json, 'w') as f:
        json.dump(final_json, f, indent=2)
    
    print(f"Saved {output_ply} and {output_json}")


if __name__ == "__main__":
    main()

