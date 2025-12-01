#!/usr/bin/env python3
"""
View point cloud from PLY file.

This script loads and visualizes a 3D point cloud from a PLY file.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def load_ply(filepath):
    """Load point cloud from PLY file."""
    points = []
    colors = []
    
    with open(filepath, 'r') as f:
        # Skip header
        line = f.readline()
        while 'end_header' not in line:
            line = f.readline()
        
        # Read data
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                points.append([x, y, z])
                colors.append([r, g, b])
    
    return np.array(points), np.array(colors)


def view_pointcloud(ply_file, figsize=(15, 12), title=None):
    """
    Visualize 3D point cloud from PLY file.
    
    Args:
        ply_file (str): Path to PLY file
        figsize (tuple): Figure size
        title (str): Plot title
    """
    if not os.path.exists(ply_file):
        print(f"Error: File not found: {ply_file}")
        return
    
    print(f"Loading point cloud from {ply_file}...")
    points_3d, colors = load_ply(ply_file)
    
    if len(points_3d) == 0:
        print("Error: No points found in file")
        return
    
    print(f"Loaded {len(points_3d)} points")
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]
    
    # Use colors if available
    if colors is not None and len(colors) > 0:
        colors_rgb = colors[:, [2, 1, 0]] / 255.0  # BGR to RGB
        ax.scatter(x, y, z, c=colors_rgb, s=20, alpha=0.6)
    else:
        ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.6)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    
    if title is None:
        title = f"Point Cloud: {os.path.basename(ply_file)}"
    ax.set_title(title, fontsize=16, pad=20)
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed {len(points_3d)} 3D points")


def main():
    parser = argparse.ArgumentParser(description='View 3D point cloud from PLY file')
    parser.add_argument('ply_file', type=str, help='Path to PLY file')
    parser.add_argument('--title', type=str, default=None, help='Plot title')
    parser.add_argument('--figsize', type=int, nargs=2, default=[15, 12], 
                       help='Figure size (width height)')
    
    args = parser.parse_args()
    
    view_pointcloud(args.ply_file, figsize=tuple(args.figsize), title=args.title)


if __name__ == "__main__":
    main()

