"""
File Utilities Module
Path handling and file operations
"""
import os
import glob
from pathlib import Path
from typing import List


def get_image_files(folder_path: str) -> List[str]:
    """
    Get all image files from a folder
    
    Args:
        folder_path: Path to folder containing images
        
    Returns:
        Sorted list of image file paths
    """
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.jpg")) + 
                         glob.glob(os.path.join(folder_path, "*.png")) +
                         glob.glob(os.path.join(folder_path, "*.jpeg")))
    return image_files


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

