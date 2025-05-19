# src_CRNN/preprocessing.py
import math
import numpy as np
from scipy.stats import linregress
from scipy.signal import convolve # Für Glättung
import logging

logger = logging.getLogger(__name__)

def preprocess_handwriting(ink, args):
    """Applies a sequence of preprocessing steps to the online ink data."""
    logger.debug(f"Applying preprocessing steps in order: {args}")
    processed_ink = ink.copy()
    for step in args:
        if processed_ink is None or processed_ink.shape[0] < 2:
             logger.warning(f"Skipping remaining preprocessing steps after previous step (ink has < 2 points).")
             break
        original_shape = processed_ink.shape
        func = PREPROCESSING_STEPS.get(step)
        if func:
            try:
                 processed_ink = func(processed_ink)
                 if processed_ink is None:
                      logger.error(f"Preprocessing step '{step}' failed and returned None.")
                      return None
                 if step != "resample" and processed_ink.shape[0] != original_shape[0]:
                     logger.debug(f"Step '{step}' changed point count: {original_shape[0]} -> {processed_ink.shape[0]}.")
            except Exception as e:
                 logger.error(f"Error during preprocessing step '{step}': {e}", exc_info=True)
                 return None
        else:
            logger.warning(f"Unknown preprocessing step '{step}'. Skipping.")
    if processed_ink is None or processed_ink.shape[0] < 2:
        logger.error("Preprocessing resulted in invalid ink (None or < 2 points).")
        return None
    return processed_ink

def move_to_origin(ink):
    """Moves ink so that the lower left corner (min_x, min_y) is at (0,0)."""
    if ink.shape[0] == 0: return ink
    min_coords = np.min(ink[:, :2], axis=0)
    ink[:, :2] -= min_coords
    return ink

def correct_slope(ink):
    """Rotates ink so the regression line through all points is horizontal."""
    if ink.shape[0] < 2: return ink
    valid_points = ink[np.isfinite(ink[:, 0]) & np.isfinite(ink[:, 1]), :2]
    if valid_points.shape[0] < 2: logger.warning("Slope Correction: Not enough valid points."); return ink
    # Check if x values are constant (vertical line)
    if np.allclose(valid_points[:, 0], valid_points[0, 0]):
        logger.warning("Slope Correction: Points form vertical line or single point. Skipping.")
        return ink

    try:
        slope, _, _, _, _ = linregress(valid_points)
        if np.isnan(slope) or not np.isfinite(slope): logger.warning(f"Slope Correction: Invalid slope ({slope})."); return ink
        angle_deg = math.degrees(math.atan(slope))
        if abs(angle_deg) < 0.5: logger.debug(f"Slope angle ({angle_deg:.2f} deg) near zero. Skipping rotation."); return ink

        logger.debug(f"Slope Correction: Rotating by {-angle_deg:.2f} degrees.")
        alpha = -math.atan(slope)
        cos_a, sin_a = math.cos(alpha), math.sin(alpha)
        mean_pt = np.mean(ink[:, :2], axis=0) # Rotate around mean
        centered = ink[:, :2] - mean_pt
        rot_x = mean_pt[0] + cos_a * centered[:, 0] - sin_a * centered[:, 1]
        rot_y = mean_pt[1] + sin_a * centered[:, 0] + cos_a * centered[:, 1]
        return np.column_stack((rot_x, rot_y, ink[:, 2]))
    except Exception as e: logger.error(f"Error during slope correction: {e}", exc_info=False); return ink

def correct_slant(ink):
    """Removes the dominant slant angle using histogram analysis and shearing."""
    if ink.shape[0] < 2: return ink
    angles = []
    for i in range(len(ink) - 1):
        p1, p2 = ink[i], ink[i+1]
        if p1[2] == 0: # Pen down segment
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            # Skip zero-length segments for angle calculation
            if abs(dx) < 1e-6 and abs(dy) < 1e-6: continue
            if abs(dx) < 1e-6: angle = 90.0 if dy >= 0 else -90.0 # Vertical or single point
            else: angle = math.degrees(math.atan(dy / dx))
            # Normalize angle to [-90, 90]
            while angle <= -90: angle += 180
            while angle > 90: angle -= 180
            angles.append(int(round(angle)))
    if not angles: logger.warning("Slant Correction: No valid segments found."); return ink

    angle_bins = np.array(angles) + 90 # Shift to [0, 180] for histogram
    hist, _ = np.histogram(angle_bins, bins=181, range=(0, 180))
    sigma = 15.0; center_bin = 90; bins_indices = np.arange(181)
    weights = np.exp(-0.5 * ((bins_indices - center_bin) / sigma) ** 2)
    weighted_hist = hist.astype(float) * weights
    smoothed_hist = convolve(weighted_hist, [0.25, 0.5, 0.25], mode='same')
    slant_angle_deg = np.argmax(smoothed_hist) - 90
    if abs(slant_angle_deg) < 1.0: logger.debug(f"Slant angle ({slant_angle_deg:.2f} deg) near zero. Skipping shear."); return ink

    logger.debug(f"Slant Correction: Shearing by {-slant_angle_deg:.2f} degrees.")
    tan_slant = math.tan(math.radians(-slant_angle_deg))
    ref_y = np.min(ink[:, 1]) # Use min_y as reference for shearing
    sheared_x = ink[:, 0] + tan_slant * (ink[:, 1] - ref_y)
    return np.column_stack((sheared_x, ink[:, 1], ink[:, 2]))

def normalize_height(ink, new_height=120):
    """Scales ink vertically to new_height, preserving aspect ratio."""
    if ink.shape[0] == 0: return ink
    min_y, max_y = np.min(ink[:, 1]), np.max(ink[:, 1])
    old_height = max_y - min_y
    if old_height < 1e-6: logger.warning("Height Normalization: Ink height near zero."); return ink
    scale_factor = new_height / old_height
    logger.debug(f"Height Normalization: Scaling by {scale_factor:.3f} (Old H={old_height:.1f})")
    min_x = np.min(ink[:, 0])
    # Apply scaling relative to min point to keep position somewhat relative
    ink[:, 0] = min_x + (ink[:, 0] - min_x) * scale_factor
    ink[:, 1] = min_y + (ink[:, 1] - min_y) * scale_factor
    return ink

def resampling(ink, step_size=10):
    """Resamples ink trajectory to have points (approximately) equidistant."""
    if ink.shape[0] < 2: return ink
    points = ink[:, :2]; penups = ink[:, 2]
    segment_diffs = np.diff(points, axis=0)
    segment_lengths = np.sqrt(np.sum(segment_diffs**2, axis=1))
    # Filter out zero-length segments before calculating cumulative sum
    valid_indices = np.where(segment_lengths > 1e-9)[0]
    if len(valid_indices) == 0: # All segments have zero length
        logger.warning("Resampling: All segments have near-zero length. Returning first point.")
        first_pt = ink[[0],:].copy(); first_pt[0,2] = 1.0; return first_pt

    valid_segment_lengths = segment_lengths[valid_indices]
    # Calculate cumulative distance ONLY for valid segments
    # Map valid indices back to original indices + 1 to align with diff output
    valid_orig_end_indices = valid_indices + 1
    # Keep original points corresponding to the start/end of valid segments
    points_for_interp = ink[np.unique(np.concatenate(([0], valid_orig_end_indices))), :2]
    # Re-calculate cumulative distance based only on the points used for interpolation
    if points_for_interp.shape[0] < 2: # Not enough points left for interpolation
        logger.warning("Resampling: Not enough points remaining after filtering zero-length segments."); return ink
    cumulative_distances = np.concatenate(([0], np.cumsum(np.sqrt(np.sum(np.diff(points_for_interp, axis=0)**2, axis=1)))))

    total_distance = cumulative_distances[-1]
    if total_distance < 1e-6: logger.warning("Resampling: Recalculated total distance near zero."); return ink[[0],:].copy()
    if total_distance < step_size: logger.warning(f"Resampling: Total distance ({total_distance:.2f}) < step size."); return ink.copy() # Return original

    num_steps = int(np.floor(total_distance / step_size))
    resample_distances = np.linspace(0, num_steps * step_size, num_steps + 1)

    # Interpolate based on the reduced point set and their cumulative distances
    resampled_x = np.interp(resample_distances, cumulative_distances, points_for_interp[:, 0])
    resampled_y = np.interp(resample_distances, cumulative_distances, points_for_interp[:, 1])

    # Penup interpolation needs original points/penups
    original_indices_for_penup = np.clip(np.searchsorted(np.concatenate(([0], np.cumsum(segment_lengths))), resample_distances, side='right') - 1, 0, len(penups) - 1)
    resampled_penups = penups[original_indices_for_penup]
    if len(resampled_penups) > 0: resampled_penups[0] = penups[0] # Ensure first point penup is correct

    resampled_ink_list = [np.column_stack((resampled_x, resampled_y, resampled_penups))]
    # Check distance to last original point (using the *original* last point)
    last_orig_point = ink[-1, :2]; last_res_point = np.array([resampled_x[-1], resampled_y[-1]])
    dist_to_last = np.linalg.norm(last_orig_point - last_res_point)
    # Append last original point if it's far OR if its penup state was important
    if dist_to_last > 1e-3 or (ink[-1, 2] == 1.0 and (len(resampled_penups)==0 or resampled_penups[-1] == 0.0)):
        resampled_ink_list.append(ink[[-1], :].copy()) # Append the last row

    final_resampled_ink = np.vstack(resampled_ink_list)
    if final_resampled_ink.shape[0] > 0: final_resampled_ink[-1, 2] = 1.0 # Ensure last point is penup
    logger.debug(f"Resampling: {len(ink)} -> {len(final_resampled_ink)} points (step={step_size})")
    return final_resampled_ink


def smoothing(ink):
    """Applies Gaussian-like smoothing (0.25, 0.5, 0.25 kernel) to x, y."""
    if ink.shape[0] < 3: return ink
    kernel = np.array([0.25, 0.5, 0.25])
    x_smooth = convolve(ink[:, 0], kernel, mode='valid')
    y_smooth = convolve(ink[:, 1], kernel, mode='valid')
    smoothed_ink = ink.copy()
    smoothed_ink[1:-1, 0] = x_smooth # Apply smoothed values to inner points
    smoothed_ink[1:-1, 1] = y_smooth
    # Preserve original penup flags, ensure last is 1
    smoothed_ink[:, 2] = ink[:, 2]
    smoothed_ink[-1, 2] = 1.0
    logger.debug("Smoothing applied.")
    return smoothed_ink

# Dictionary mapping step names to functions
PREPROCESSING_STEPS = {
    "slope": correct_slope,
    "origin": move_to_origin,
    "slant": correct_slant,
    "height": normalize_height,
    "resample": resampling,
    "smooth": smoothing,
}
