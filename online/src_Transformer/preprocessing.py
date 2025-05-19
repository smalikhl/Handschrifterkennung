import math
import numpy as np
from scipy.stats import linregress
from scipy.special import binom
import logging

logger = logging.getLogger(__name__)

def preprocess_handwriting(ink, args):
    """
    Applies a sequence of preprocessing steps to the ink data.
    Args:
        ink (np.array): Nx3 array [x, y, pen_up]
        args (list): List of preprocessing steps to apply (e.g., ["slope", "origin"]).
    Returns:
        np.array: Preprocessed ink data.
    """
    logger.debug(f"Preprocessing ink with args: {args}")
    processed_ink = ink.copy() # Work on a copy

    if "slope" in args:
        processed_ink = correct_slope(processed_ink)
    if "origin" in args:
        processed_ink = move_to_origin(processed_ink)
    if "flip_h" in args: # Added based on input
        processed_ink = flip_horizontally(processed_ink)
    if "slant" in args:
        processed_ink = correct_slant(processed_ink)
    if "height" in args:
        processed_ink = normalize_height(processed_ink)
    if "resample" in args:
        processed_ink = resampling(processed_ink)
    if "smooth" in args:
        processed_ink = smoothing(processed_ink)
    return processed_ink

def flip_horizontally(ink):
    """Flips the ink horizontally."""
    logger.debug("Applying horizontal flip.")
    ink[:,0] = (ink[:,0] - ink[:,0].max()) * -1
    return ink

def move_to_origin(ink):
    """
    Moves ink so that the lower left corner
    of its bounding box is the origin afterwards.
    """
    logger.debug("Moving ink to origin.")
    min_x = np.min(ink[:, 0])
    min_y = np.min(ink[:, 1])
    return ink - [min_x, min_y, 0]

def flip_vertically(ink):
    """
    Rotates ink by 180 degrees (NOTE: Likely not needed if correct_slope is used,
    but kept for completeness based on original code).
    """
    logger.debug("Applying vertical flip.")
    max_y = np.max(ink[:, 1])
    return np.array([[x, max_y - y, p] for [x, y, p] in ink])

def correct_slope(ink):
    """
    Rotates ink so that the regression line through
    all points is horizontal afterwards.
    """
    logger.debug("Correcting slope.")
    # Ensure there are enough points for linear regression
    if len(ink) < 2:
        logger.warning("Not enough points to correct slope.")
        return ink
    # Avoid issues if all points are identical
    if np.all(ink[:, 0] == ink[0, 0]) and np.all(ink[:, 1] == ink[0, 1]):
         logger.warning("All points are identical, cannot correct slope.")
         return ink
    # Avoid issues if all x are the same (vertical line)
    if np.all(ink[:, 0] == ink[0, 0]):
        logger.warning("Vertical line detected, skipping slope correction.")
        return ink

    try:
        # Perform linear regression only on x and y coordinates
        slope, intercept, r_value, p_value, std_err = linregress(ink[:, :2])

        # Handle potential NaN slope if points form a vertical line or are identical
        if np.isnan(slope):
             logger.warning("Slope calculation resulted in NaN, skipping slope correction.")
             return ink

        alpha = math.atan(-slope)
        cos_alpha = math.cos(alpha)
        sin_alpha = math.sin(alpha)

        # Center data around its minimum for stable rotation
        min_x = np.min(ink[:, 0])
        min_y = np.min(ink[:, 1])

        # Apply rotation matrix
        rot_x = min_x + cos_alpha * (ink[:, 0] - min_x) - sin_alpha * (ink[:, 1] - min_y)
        rot_y = min_y + sin_alpha * (ink[:, 0] - min_x) + cos_alpha * (ink[:, 1] - min_y)

        new_ink = np.column_stack((rot_x, rot_y, ink[:, 2]))

        # Move back to origin (optional, depending on desired normalization)
        new_min_x = np.min(new_ink[:, 0])
        new_min_y = np.min(new_ink[:, 1])
        return new_ink - [new_min_x, new_min_y, 0]

    except ValueError as e:
        logger.error(f"Error during slope correction: {e}. Skipping.")
        return ink


def correct_slant(ink):
    """
    Removes the most dominant slant-angle from the ink.
    """
    logger.debug("Correcting slant.")
    if len(ink) < 2:
        logger.warning("Not enough points to correct slant.")
        return ink

    last_point = ink[0]
    angles = []
    for cur_point in ink[1:]:
        if last_point[2] == 1: # Pen up
            last_point = cur_point
            continue
        delta_x = cur_point[0] - last_point[0]
        delta_y = cur_point[1] - last_point[1]

        if delta_x == 0:
            angle = 90 if delta_y > 0 else (-90 if delta_y < 0 else 0) # Vertical line or same point
        else:
            angle = math.degrees(math.atan(delta_y / delta_x))

        # Keep angle within [-90, 90] for consistency
        if angle > 90: angle -= 180
        if angle < -90: angle += 180

        angles.append(int(round(angle))) # Round to nearest degree for binning
        last_point = cur_point

    if not angles:
        logger.warning("No valid angles found for slant correction.")
        return ink

    # Shift angles to [0, 180] for histogram calculation
    angles_shifted = np.array(angles) + 90
    bins = np.bincount(angles_shifted, minlength=181) # Ensure bins 0 to 180 exist

    # Use a Gaussian kernel for weighting (more stable than binomial approximation)
    center_bin = 90  # Center on 0 degrees (90 in shifted space)
    sigma = 20       # Standard deviation for weighting
    weights = np.exp(-0.5 * ((np.arange(181) - center_bin) / sigma) ** 2)
    weights /= np.sum(weights) # Normalize weights

    # Apply weights and smooth
    weighted_bins = bins.astype(float) * weights
    smoothed = np.convolve(weighted_bins, [0.25, 0.5, 0.25], mode='same') # Simple smoothing

    # Find the dominant angle (peak in smoothed histogram)
    dominant_angle_shifted = np.argmax(smoothed)
    slant_angle = dominant_angle_shifted - 90 # Convert back to original range [-90, 90]

    logger.debug(f"Detected slant angle: {slant_angle} degrees.")

    # Apply shearing transformation to correct the slant
    tan_slant = math.tan(math.radians(slant_angle))
    min_y = np.min(ink[:, 1]) # Use min_y as reference for shearing

    sheared_x = ink[:, 0] + tan_slant * (ink[:, 1] - min_y)
    return np.column_stack((sheared_x, ink[:, 1], ink[:, 2]))


def resampling(ink, step_size=10):
    """
    Replaces given ink by a recalculated sequence of equidistant points.
    Uses linear interpolation.
    """
    logger.debug(f"Resampling ink with step size {step_size}.")
    if len(ink) < 2:
        logger.warning("Not enough points to resample.")
        return ink

    points = ink[:, :2]
    penups = ink[:, 2]

    # Calculate cumulative distance along the trajectory
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    total_distance = cumulative_distances[-1]

    if total_distance < step_size: # Avoid issues if ink is very short
        logger.warning("Total distance is less than step size. Returning original ink.")
        return ink

    # Define the new sampling points along the cumulative distance
    resample_distances = np.arange(0, total_distance, step_size)

    # Interpolate x, y coordinates
    resampled_x = np.interp(resample_distances, cumulative_distances, points[:, 0])
    resampled_y = np.interp(resample_distances, cumulative_distances, points[:, 1])

    # Interpolate pen-up status (find the original index closest to each resampled distance)
    # A simple approach: Assign pen-up status from the *preceding* original point.
    original_indices = np.searchsorted(cumulative_distances, resample_distances, side='right') - 1
    original_indices = np.maximum(0, original_indices) # Ensure indices are not negative
    resampled_penups = penups[original_indices]

    # Add the very last original point to ensure the trajectory ends correctly
    resampled_points = np.column_stack((resampled_x, resampled_y, resampled_penups))
    # Make sure the last point has pen-up=1 if the original did
    last_original_point = ink[-1, :].reshape(1, 3)

    # Check if the last resampled point is sufficiently close to the last original point
    if len(resampled_points) > 0 and np.linalg.norm(resampled_points[-1,:2] - last_original_point[0,:2]) > 1e-3:
         # Append the last original point only if it's different from the last resampled one
         resampled_points = np.vstack((resampled_points, last_original_point))
    elif len(resampled_points) == 0: # Handle cases where resampling yields no points
        resampled_points = last_original_point

    # Ensure the final point always has penup=1, reflecting the end of writing
    if len(resampled_points) > 0:
        resampled_points[-1, 2] = 1

    return resampled_points


def normalize_height(ink, new_height=120):
    """
    Scales ink so its vertical span (height) becomes new_height.
    Preserves aspect ratio unless height is zero.
    """
    logger.debug(f"Normalizing height to {new_height}.")
    min_y = np.min(ink[:, 1])
    max_y = np.max(ink[:, 1])
    old_height = max_y - min_y

    if old_height <= 1e-6: # Avoid division by zero or near-zero
        logger.warning(f"Ink has zero or near-zero height ({old_height}). Cannot normalize height.")
        return ink

    scale_factor = new_height / old_height
    # Scale x and y coordinates
    ink[:, :2] *= scale_factor
    # Recenter after scaling (optional, but often useful)
    ink = move_to_origin(ink)
    return ink


def smoothing(ink):
    """
    Applies Gaussian-like smoothing (0.25, 0.5, 0.25) to x, y coordinates.
    Keeps original pen-up flags. Handles boundaries.
    """
    logger.debug("Applying smoothing.")
    if ink.shape[0] < 3:
        logger.warning("Not enough points for smoothing.")
        return ink

    # Smooth x and y coordinates separately
    x_smooth = np.convolve(ink[:, 0], [0.25, 0.5, 0.25], mode='valid')
    y_smooth = np.convolve(ink[:, 1], [0.25, 0.5, 0.25], mode='valid')

    # Keep original first and last points (no smoothing applied there)
    smoothed_ink = np.zeros((ink.shape[0], 3))
    smoothed_ink[0, :] = ink[0, :]
    smoothed_ink[-1, :] = ink[-1, :]

    # Fill in the smoothed middle points
    smoothed_ink[1:-1, 0] = x_smooth
    smoothed_ink[1:-1, 1] = y_smooth

    # Preserve original pen-up flags for all points
    smoothed_ink[:, 2] = ink[:, 2]

    # Ensure the last point's penup is correct
    smoothed_ink[-1, 2] = ink[-1, 2]

    return smoothed_ink


def remove_delayed_strokes(ink):
    """
    Removes points of delayed strokes (segments between two penups)
    from the ink. A stroke is considered delayed if its rightmost point
    is to the left of the rightmost point of the last non-delayed stroke's
    endpoint.
    """
    logger.debug("Removing delayed strokes.")
    if len(ink) == 0:
        return np.array([]), np.array([])

    stroke_endpoints = np.where(ink[:, 2] == 1)[0]
    if len(stroke_endpoints) == 0: # Should not happen if last point is always penup=1
         logger.warning("No pen-up points found. Cannot remove delayed strokes.")
         # Assume single stroke
         return ink, np.array([])
    if stroke_endpoints[0] == 0: # Handle case where first point is penup=1
        stroke_endpoints = stroke_endpoints[1:]
        if len(stroke_endpoints) == 0: 
            return ink, np.array([]) # Only one point

    new_ink_list = []
    delayed_list = []
    start_idx = 0
    orientation_x = -np.inf # Keep track of the rightmost x of the confirmed strokes

    for end_idx in stroke_endpoints:
        stroke = ink[start_idx : end_idx + 1, :]
        if len(stroke) == 0:
            start_idx = end_idx + 1
            continue

        stroke_max_x = np.max(stroke[:, 0])

        # First stroke is never delayed
        if orientation_x == -np.inf:
            new_ink_list.append(stroke)
            # Update orientation_x based on the ENDPOINT of the first stroke
            orientation_x = stroke[-1, 0]
        elif stroke_max_x >= orientation_x:
            new_ink_list.append(stroke)
            # Update orientation_x based on the ENDPOINT of this accepted stroke
            orientation_x = stroke[-1, 0]
        else:
            # This stroke is considered delayed
            delayed_list.append(stroke)
            logger.debug(f"Delayed stroke detected (max_x={stroke_max_x} < orientation_x={orientation_x})")

        start_idx = end_idx + 1 # Move to the start of the next potential stroke

    # Concatenate the lists of arrays
    final_ink = np.vstack(new_ink_list) if new_ink_list else np.array([])
    final_delayed = np.vstack(delayed_list) if delayed_list else np.array([])

    return final_ink, final_delayed