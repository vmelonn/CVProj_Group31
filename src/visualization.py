"""
Visualization utilities for images, matches, and point clouds.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import List, Optional
from mpl_toolkits.mplot3d import Axes3D


def display_image_grid(images: List[np.ndarray], image_files: List[str], 
                      num_images: int = 12, rows: int = 3, cols: int = 4,
                      figsize: tuple = (20, 15)) -> None:
    """
    Display a grid of images.
    
    Args:
        images (List[np.ndarray]): List of images to display
        image_files (List[str]): List of image file paths (for titles)
        num_images (int): Number of images to display. Defaults to 12.
        rows (int): Number of rows in grid. Defaults to 3.
        cols (int): Number of columns in grid. Defaults to 4.
        figsize (tuple): Figure size. Defaults to (20, 15).
    """
    import os
    
    num_images = min(len(images), num_images)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle("Image Dataset Examples", fontsize=24)

    axes = axes.flatten()

    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Image {i+1} ({os.path.basename(image_files[i])})", fontsize=16)
        axes[i].axis('off')

    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def visualize_matches(img1: np.ndarray, kp1: List, img2: np.ndarray, kp2: List, 
                     matches: List, title: str = "Feature Matches", 
                     figsize: tuple = (20, 10)) -> None:
    """
    Visualize feature matches between two images.
    
    Args:
        img1 (np.ndarray): First image
        kp1 (List): Keypoints from first image
        img2 (np.ndarray): Second image
        kp2 (List): Keypoints from second image
        matches (List): List of matches
        title (str): Plot title. Defaults to "Feature Matches".
        figsize (tuple): Figure size. Defaults to (20, 10).
    """
    match_img = cv2.drawMatches(
        cv2.cvtColor(img1, cv2.COLOR_RGB2BGR), kp1,
        cv2.cvtColor(img2, cv2.COLOR_RGB2BGR), kp2,
        matches, None, flags=2
    )

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{title} ({len(matches)} matches)", fontsize=16)
    plt.axis('off')
    plt.show()


def plot_3d_pointcloud(points_3d: np.ndarray, colors: Optional[np.ndarray] = None,
                       figsize: tuple = (15, 12), title: str = "3D Point Cloud Visualization") -> None:
    """
    Visualize 3D point cloud in 3D space.
    
    Args:
        points_3d (np.ndarray): 3D points, shape (N, 3)
        colors (np.ndarray, optional): RGB colors, shape (N, 3). Defaults to None.
        figsize (tuple): Figure size. Defaults to (15, 12).
        title (str): Plot title. Defaults to "3D Point Cloud Visualization".
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    # Use colors if available, otherwise use z-coordinate for coloring
    if colors is not None and len(colors) > 0:
        # Convert BGR to RGB for display
        colors_rgb = colors[:, [2, 1, 0]] / 255.0
        ax.scatter(x, y, z, c=colors_rgb, s=20, alpha=0.6)
    else:
        ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.6)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
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


def plot_2d_projections(points_3d: np.ndarray, colors: Optional[np.ndarray] = None,
                        figsize: tuple = (18, 5)) -> None:
    """
    Visualize 3D point cloud projected onto 2D planes (XY, XZ, YZ projections).
    
    Args:
        points_3d (np.ndarray): 3D points, shape (N, 3)
        colors (np.ndarray, optional): RGB colors, shape (N, 3). Defaults to None.
        figsize (tuple): Figure size. Defaults to (18, 5).
    """
    x = points_3d[:, 0]
    y = points_3d[:, 1]
    z = points_3d[:, 2]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # XY projection
    if colors is not None and len(colors) > 0:
        colors_rgb = colors[:, [2, 1, 0]] / 255.0
        axes[0].scatter(x, y, c=colors_rgb, s=10, alpha=0.6)
        axes[1].scatter(x, z, c=colors_rgb, s=10, alpha=0.6)
        axes[2].scatter(y, z, c=colors_rgb, s=10, alpha=0.6)
    else:
        axes[0].scatter(x, y, c=z, cmap='viridis', s=10, alpha=0.6)
        axes[1].scatter(x, z, c=y, cmap='viridis', s=10, alpha=0.6)
        axes[2].scatter(y, z, c=x, cmap='viridis', s=10, alpha=0.6)

    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('Y', fontsize=12)
    axes[0].set_title('XY Projection', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')

    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('Z', fontsize=12)
    axes[1].set_title('XZ Projection', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')

    axes[2].set_xlabel('Y', fontsize=12)
    axes[2].set_ylabel('Z', fontsize=12)
    axes[2].set_title('YZ Projection', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_aspect('equal', adjustable='box')

    plt.suptitle('2D Scatter Plots of 3D Point Cloud', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"Displayed 2D projections of {len(points_3d)} 3D points")

