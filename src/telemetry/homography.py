"""Map pixel coordinates (from video frame) to real-world track coordinates."""
import numpy as np
import cv2
import json


def load_homography_points(path):
    with open(path) as f:
        return json.load(f)


def compute_homography(image_points, world_points):
    """
    image_points: Nx2 array of pixel coords
    world_points: Nx2 array of corresponding real-world coords (meters)
    Returns 3x3 homography matrix.
    """
    H, _ = cv2.findHomography(np.array(image_points), np.array(world_points))
    return H


def pixel_to_world(H, point):
    px = np.array([point[0], point[1], 1.0])
    world = H @ px
    world /= world[2]
    return world[:2]
