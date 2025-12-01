#!/usr/bin/env python3
"""
Main execution script for Structure from Motion pipeline.

This script provides a command-line interface to run different stages of the SFM pipeline:
- Feature matching
- Two-view reconstruction
- Incremental SFM
"""

import argparse
import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src import (
    load_image_dataset, save_ply,
    find_and_filter_matches, get_matches, draw_matches,
    evaluate_pair, find_best_pair, incremental_sfm,
    build_intrinsic_matrix, triangulate_and_check,
    display_image_grid, visualize_matches, plot_3d_pointcloud, plot_2d_projections
)


def feature_matching(images, image_files, num_pairs=4):
    """Run feature matching on consecutive image pairs."""
    print("\n" + "="*60)
    print("FEATURE MATCHING")
    print("="*60)
    
    if len(images) < num_pairs + 1:
        print(f"Not enough images. Need at least {num_pairs + 1}, got {len(images)}")
        return
    
    for i in range(num_pairs):
        img1 = images[i]
        img2 = images[i+1]
        
        kp1, kp2, good_matches = find_and_filter_matches(img1, img2)
        match_visualization = draw_matches(img1, kp1, img2, kp2, good_matches)
        
        print(f"Pair {i+1} to {i+2}: Found {len(good_matches)} good matches")
        
        # Save visualization
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 10))
        plt.imshow(match_visualization)
        plt.title(f"Matches: Image {i+1} to {i+2} (Found {len(good_matches)} good matches)", fontsize=16)
        plt.axis('off')
        plt.savefig(f"outputs/matches_{i}_to_{i+1}.png", dpi=150, bbox_inches='tight')
        plt.close()


def two_view_reconstruction(images, image_files, output_dir="outputs"):
    """Run two-view reconstruction pipeline."""
    print("\n" + "="*60)
    print("TWO-VIEW RECONSTRUCTION")
    print("="*60)
    
    # Find best pair
    idx1, idx2, reconstruction_data = find_best_pair(images)
    
    if reconstruction_data is None:
        print("ERROR: Could not find suitable pair for reconstruction.")
        return
    
    img1 = images[idx1]
    img2 = images[idx2]
    
    # Visualize matches
    visualize_matches(img1, reconstruction_data['kp1'], img2, reconstruction_data['kp2'],
                     reconstruction_data['good_matches'],
                     title=f"Feature Matches: Image {idx1} to {idx2}")
    
    # Display reconstruction info
    print(f"\nIntrinsic Matrix K:\n{reconstruction_data['K']}")
    print(f"\nEssential Matrix E:\n{reconstruction_data['E']}")
    print(f"\nRotation Matrix R:\n{reconstruction_data['R']}")
    print(f"\nTranslation Vector t:\n{reconstruction_data['t']}")
    print(f"\nValid 3D Points: {reconstruction_data['points_3d'].shape[0]}")
    
    # Perform final reconstruction
    pts1, pts2, good_matches, kp1, kp2 = get_matches(img1, img2)
    print(f"\nGood matches: {len(good_matches)}")
    
    K = reconstruction_data['K']
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    R1, R2, t = cv2.decomposeEssentialMat(E)
    poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    
    best_pts = None
    best_count = -1
    
    for i, (R, t_vec) in enumerate(poses):
        pts3d, count = triangulate_and_check(R, t_vec, pts1, pts2, K)
        print(f"Pose {i+1}: {count} points in front")
        if count > best_count:
            best_count = count
            best_pts = pts3d
    
    print(f"Chosen pose has {best_count} points in front.")
    
    # Save point cloud
    output_path = os.path.join(output_dir, "pointcloud.ply")
    save_ply(output_path, best_pts)
    
    # Visualize
    plot_3d_pointcloud(best_pts, title="Two-View Reconstruction Point Cloud")
    plot_2d_projections(best_pts)
    
    return idx1, idx2, reconstruction_data


def incremental_sfm_reconstruction(images, image_files, init_idx0, init_idx1, 
                                  output_dir="outputs"):
    """Run incremental SFM reconstruction."""
    print("\n" + "="*60)
    print("INCREMENTAL STRUCTURE FROM MOTION")
    print("="*60)
    
    # Build K from first image
    K = build_intrinsic_matrix(images[0].shape)
    
    # Run incremental SFM
    poses, map_points = incremental_sfm(images, K, init_idx0, init_idx1)
    
    # Extract points and colors
    points_3d = np.array([p.xyz for p in map_points])
    colors = np.array([p.color for p in map_points])
    
    print(f"\nReconstructed {len(map_points)} 3D points")
    print(f"Estimated poses for {sum(1 for p in poses if p is not None)} images")
    
    # Save point cloud
    output_path = os.path.join(output_dir, "incremental_pointcloud.ply")
    save_ply(output_path, points_3d, colors)
    
    # Visualize
    plot_3d_pointcloud(points_3d, colors, title="Incremental SFM Point Cloud")
    plot_2d_projections(points_3d, colors)
    
    return poses, map_points


def main():
    parser = argparse.ArgumentParser(description='Structure from Motion Pipeline')
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='Path to image dataset folder')
    parser.add_argument('--pattern', type=str, default='img_*.jpeg',
                       help='Glob pattern for image files')
    parser.add_argument('--width', type=int, default=800,
                       help='Target width for image resizing')
    parser.add_argument('--mode', type=str, choices=['features', 'two-view', 'incremental', 'all'],
                       default='all', help='Pipeline mode to run')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--display-grid', action='store_true',
                       help='Display image grid')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load images
    print("Loading images...")
    images, image_files = load_image_dataset(args.dataset, args.pattern, args.width)
    
    if len(images) == 0:
        print(f"ERROR: No images found in {args.dataset} with pattern {args.pattern}")
        return
    
    # Display image grid if requested
    if args.display_grid:
        display_image_grid(images, image_files)
    
    # Run selected pipeline
    if args.mode in ['features', 'all']:
        feature_matching(images, image_files)
    
    if args.mode in ['two-view', 'all']:
        idx1, idx2, data = two_view_reconstruction(images, image_files, args.output)
    
    if args.mode in ['incremental', 'all']:
        if args.mode == 'incremental':
            # Find best pair first
            idx1, idx2, _ = find_best_pair(images)
        incremental_sfm_reconstruction(images, image_files, idx1, idx2, args.output)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

