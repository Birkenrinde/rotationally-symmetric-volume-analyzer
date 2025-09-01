import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from fractions import Fraction
from typing import Tuple, Optional, Any, List
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from numpy.polynomial.legendre import legval

# --- Type Aliases ---
Image = np.ndarray
Coords = np.ndarray
Centroid = Tuple[float, float]
FitCoeffs = np.ndarray
OptParams = List[float]

def format_radians_as_pi(x: float, pos: Any) -> str:
    """Formats a radian value as a multiple or fraction of pi for plot labels."""
    multiple = x / np.pi
    if np.isclose(multiple, 0): return "0"
    frac = Fraction(multiple).limit_denominator(10)
    num, den = frac.numerator, frac.denominator
    if den == 1:
        if num == 1: return "π"
        if num == -1: return "-π"
        return f"{num}π"
    if num == 1: return f"π/{den}"
    if num == -1: return f"-π/{den}"
    return f"{num}π/{den}"

def apply_canny(input_frame: Image, lower_threshold: int, upper_threshold: int) -> Coords:
    """Applies the Canny edge detector to an image."""
    edge_mask = cv2.Canny(input_frame, lower_threshold, upper_threshold, 3, L2gradient=True)
    y_coords, x_coords = np.where(edge_mask > 0)
    return np.column_stack((x_coords, y_coords))

def find_convex_hull(edge_coords: Coords, proximity_threshold: int) -> Coords:
    """Calculates the convex hull of a set of points, optionally including nearby points."""
    if edge_coords.shape[0] < 3: return np.array([[]])
    hull = ConvexHull(edge_coords)
    hull_points = edge_coords[hull.vertices]
    if proximity_threshold > 0 and hull_points.size > 0:
        distances = cdist(edge_coords, hull_points)
        min_dist_to_hull = np.min(distances, axis=1)
        nearby_indices = np.where(min_dist_to_hull <= proximity_threshold)[0]
        return np.unique(np.vstack((hull_points, edge_coords[nearby_indices])), axis=0)
    else:
        return hull_points

def calculate_centroid(coords: Coords) -> Optional[Centroid]:
    """Calculates the geometric centroid of a set of coordinates."""
    if coords.size == 0: return None
    return tuple(np.mean(coords, axis=0))

def polar_transform(edge_coords: Coords, centroid: Centroid) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms Cartesian coordinates to a polar system relative to a centroid."""
    delta = edge_coords - np.array(centroid)
    dx, dy = delta[:, 0], delta[:, 1]
    dy_upright = -dy  # Invert y-axis for standard angle convention (0 is up)
    thetas = np.arctan2(dx, dy_upright)
    radii = np.linalg.norm(delta, axis=1)
    return thetas, radii

def legendre_polynomial_series(theta: np.ndarray, *coeffs: float) -> np.ndarray:
    """Evaluates the Legendre series for a given theta and coefficients."""
    return legval(np.cos(theta), coeffs)

def get_shifted_centroid(base: Centroid, angle_rad: float, perp_shift: float, para_shift: float) -> Centroid:
    """Calculates a new centroid shifted along and perpendicular to a rotated axis."""
    dx = para_shift * np.sin(angle_rad) + perp_shift * np.cos(angle_rad)
    dy_up = para_shift * np.cos(angle_rad) - perp_shift * np.sin(angle_rad)
    return (base[0] + dx, base[1] - dy_up)

def core_compute_radius_over_theta(edge_coords: Coords, centroid: Centroid) -> Tuple[np.ndarray, np.ndarray]:
    delta = edge_coords - np.array(centroid)
    dx, dy = delta[:, 0], delta[:, 1]
    dy_upright = -dy 
    thetas = np.arctan2(dx, dy_upright)
    radii = np.linalg.norm(delta, axis=1)
    return thetas, radii