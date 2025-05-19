# src_CRNN/feature_extraction.py
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def moving_average(data_set, periods=3):
    """Calculates the moving average using convolution."""
    if not isinstance(periods, int) or periods <= 0: return data_set
    if data_set.ndim != 1 or len(data_set) == 0: return data_set # Handle empty case
    if periods >= len(data_set) or periods == 1: return data_set # Return original if period too large or 1
    padded_data = np.pad(data_set, (periods - 1, 0), mode='edge')
    weights = np.ones(periods) / periods
    return np.convolve(padded_data, weights, mode='valid')

def __normalize_vector(v):
    """Normalizes a vector using L2 norm. Handles zero norm."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else np.zeros_like(v) # Return explicit zero vector if norm is zero

# --- Feature Calculation Orchestration ---
def calculate_feature_vector_sequence(ink, args, delayed_strokes=None):
    """Calculates all features named in args for each point in the ink trajectory."""
    logger.debug(f"Calculating features: {args} for ink shape {ink.shape}")
    if ink is None or ink.ndim != 2 or ink.shape[1] != 3 or ink.shape[0] < 1:
         logger.error(f"Invalid ink for feature calculation: shape {ink.shape if ink is not None else 'None'}")
         return None
    if ink.shape[0] < 3: logger.warning(f"Ink has only {ink.shape[0]} points. Some features might be zero or based on limited context.")

    feature_vectors = []
    num_features_expected = _get_expected_num_features(args)
    if num_features_expected == 0: logger.warning("No features selected in FEAT_ARGS!"); return np.empty((ink.shape[0], 0), dtype=np.float32) # Return if no features

    # Pre-calculate MA difference if needed
    if "ma_x" in args:
        # Moving average needs at least 'periods' points
        ma_periods = 20
        if ink.shape[0] >= ma_periods:
             ma_x_values = moving_average(ink[:, 0], periods=ma_periods)
             ma_diff = (ink[:, 0] - ma_x_values).reshape(-1, 1)
        else:
             logger.debug(f"Not enough points ({ink.shape[0]}) for ma_x with period {ma_periods}. Using zeros.")
             ma_diff = np.zeros((ink.shape[0], 1)) # Fill with zeros if not enough points
        combined_data = np.hstack((ink, ma_diff))
    else:
        combined_data = ink

    for p in range(len(combined_data)):
        try:
            vec = __calculate_feature_vector(combined_data, p, args, delayed_strokes)
            if vec is None or len(vec) != num_features_expected:
                 logger.error(f"Feature calc failed or dimension mismatch at point {p}. Expected {num_features_expected}, Got {len(vec) if vec is not None else 'None'}.")
                 return None # Critical error for the sample
            feature_vectors.append(vec)
        except Exception as e:
             logger.error(f"Error calculating features for point {p}: {e}", exc_info=True)
             return None

    if not feature_vectors: logger.error("No valid feature vectors generated for this sample."); return np.empty((0, num_features_expected), dtype=np.float32)
    result_array = np.array(feature_vectors, dtype=np.float32)
    if result_array.shape[1] != num_features_expected: logger.error(f"Final feature array dim mismatch! Expected {num_features_expected}, got {result_array.shape[1]}."); return None
    if not np.all(np.isfinite(result_array)):
         nan_count = np.sum(np.isnan(result_array))
         inf_count = np.sum(np.isinf(result_array))
         logger.warning(f"Feature calculation resulted in {nan_count} NaN(s) and {inf_count} Inf(s)! Consider checking feature functions for stability (e.g., division by zero). Replacing with zeros.")
         result_array = np.nan_to_num(result_array, nan=0.0, posinf=0.0, neginf=0.0) # Replace non-finite with 0

    logger.debug(f"Feature calculation complete. Result shape: {result_array.shape}")
    return result_array

def _get_expected_num_features(args):
    """Helper to calculate expected feature dimension based on args list."""
    # Needs to exactly match the calculation in data_preparation.py
    return sum([
        1 if f in ["x_cor", "y_cor", "penup", "vic_aspect", "vic_curl", "vic_line", "vic_slope", "ma_x", "hat"] else # hat is 1D
        2 if f in ["dir", "curv"] else
        9 if f == "bitmap" else
        0 # Ignore unknown features
        for f in args
    ])

# --- Single Point Feature Vector Calculation ---
def __calculate_feature_vector(combined_data, point_index, args, delayed_strokes=None):
    """Calculates the feature vector for a single point using pre-combined data."""
    ink = combined_data[:, :3] # Original ink [x, y, penup]
    num_points = len(ink)
    if num_points == 0: return None # Should not happen if called from sequence func

    vicinity = __get_vicinity(ink, point_index, window_size=5)
    has_full_context_5 = vicinity.shape[0] == 5

    feat_vec = []
    try:
        if "x_cor" in args: feat_vec.append(float(ink[point_index, 0]))
        if "y_cor" in args: feat_vec.append(float(ink[point_index, 1]))
        if "penup" in args: feat_vec.append(float(ink[point_index, 2]))
        if "dir" in args: feat_vec.extend(__writing_direction(ink, point_index))
        if "curv" in args: feat_vec.extend(__curvature(ink, point_index))
        if "hat" in args: feat_vec.append(__hat(ink, point_index, delayed_strokes))
        if "vic_aspect" in args: feat_vec.append(__vicinity_aspect(vicinity) if has_full_context_5 else 0.0)
        if "vic_curl" in args: feat_vec.append(__vicinity_curliness(vicinity) if has_full_context_5 else 0.0)
        if "vic_line" in args: feat_vec.append(__vicinity_lineness(vicinity) if vicinity.shape[0] >= 3 else 0.0)
        if "vic_slope" in args: feat_vec.append(__vicinity_slope(ink, point_index))
        if "bitmap" in args: feat_vec.extend(__context_bitmap(ink, point_index))
        if "ma_x" in args: feat_vec.append(float(combined_data[point_index, 3]) if combined_data.shape[1] > 3 else 0.0)

        final_vec = __normalize_vector(np.array(feat_vec, dtype=np.float32))
        expected_dim = _get_expected_num_features(args)
        if len(final_vec) != expected_dim: raise ValueError(f"Dimension mismatch: Expected {expected_dim}, Got {len(final_vec)}")
        return final_vec
    except Exception as e:
         logger.error(f"Error calculating features for point {point_index}: {e}", exc_info=False)
         return None

# --- Individual Feature Implementations (Unverändert - Korrektheit vorausgesetzt) ---
# (Funktionen __writing_direction, __curvature, __hat, __get_vicinity,
#  __vicinity_aspect, __vicinity_curliness, __vicinity_lineness,
#  __vicinity_slope, __context_bitmap bleiben identisch wie im vorherigen Snippet)

def __writing_direction(ink, point_idx):
    n = len(ink)
    if n < 2: return [0.0, 0.0]
    if point_idx == 0: d = ink[1,:2] - ink[0,:2]
    elif point_idx == n - 1: d = ink[n-1,:2] - ink[n-2,:2]
    else: d = ink[point_idx+1,:2] - ink[point_idx-1,:2] # Central difference
    norm = np.linalg.norm(d)
    return (d / norm).tolist() if norm > 1e-9 else [0.0, 0.0]

def __curvature(ink, point_idx):
    n = len(ink)
    if n < 3: return [1.0, 0.0] # Assume straight
    if point_idx == 0: dir_in = [0.0, 0.0]
    else: dir_in = __writing_direction(ink, point_idx - 1)
    if point_idx >= n - 1: dir_out = [0.0, 0.0]
    else: dir_out = __writing_direction(ink, point_idx)
    if np.allclose(dir_in, 0.0) or np.allclose(dir_out, 0.0): return [1.0, 0.0]
    cos_in, sin_in = dir_in; cos_out, sin_out = dir_out
    curv_cos = np.clip(cos_out * cos_in + sin_out * sin_in, -1.0, 1.0)
    curv_sin = np.clip(sin_out * cos_in - cos_out * sin_in, -1.0, 1.0)
    return [curv_cos, curv_sin]

def __hat(ink, point_idx, delayed_strokes):
    """Checks if the point lies below a delayed stroke."""
    if delayed_strokes is None or not isinstance(delayed_strokes, np.ndarray) or delayed_strokes.shape[0] == 0: return 0.0
    point_x, point_y = ink[point_idx, :2]
    penup_indices = np.where(delayed_strokes[:, 2] == 1)[0]
    start = 0
    indices_to_iterate = np.unique(np.append(penup_indices, len(delayed_strokes)-1))
    for end in indices_to_iterate:
        if start > end: continue
        stroke_segment = delayed_strokes[start : end + 1]
        if stroke_segment.shape[0] > 0:
            try:
                min_x, max_x = np.min(stroke_segment[:, 0]), np.max(stroke_segment[:, 0])
                min_y = np.min(stroke_segment[:, 1])
                if min_x <= point_x <= max_x and point_y < min_y: return 1.0
            except Exception as e: logger.warning(f"Error in __hat delayed segment processing: {e}", exc_info=False)
        start = end + 1
    return 0.0

def __get_vicinity(ink, point_idx, window_size=5):
    n = len(ink)
    if n == 0: return np.empty((0, 2))
    half = window_size // 2
    start = max(0, point_idx - half)
    end = min(n, point_idx + half + 1)
    start = min(start, end) # Ensure start <= end
    return ink[start:end, :2] # Return x, y

def __vicinity_aspect(vicinity):
    if vicinity is None or vicinity.shape[0] < 2: return 0.0
    try:
        min_c, max_c = np.min(vicinity, axis=0), np.max(vicinity, axis=0)
        dx, dy = max_c[0] - min_c[0], max_c[1] - min_c[1]
        denom = dx + dy
        return np.clip((2.0 * dy / denom) - 1.0, -1.0, 1.0) if denom > 1e-9 else 0.0
    except Exception as e: logger.warning(f"Error in __vicinity_aspect: {e}"); return 0.0

def __vicinity_curliness(vicinity):
    if vicinity is None or vicinity.shape[0] < 2: return 0.0
    try:
        min_c, max_c = np.min(vicinity, axis=0), np.max(vicinity, axis=0)
        dx, dy = max_c[0] - min_c[0], max_c[1] - min_c[1]
        max_dim = max(dx, dy)
        if max_dim < 1e-9: return 0.0 # If no extent, curliness is 0
        path_len = np.sum(np.sqrt(np.sum(np.diff(vicinity, axis=0)**2, axis=1)))
        if path_len < 1e-9: return -2.0 # Zero path length means straight line essentially
        curl = (path_len / max_dim) - 2.0
        return np.clip(curl, -2.0, 10.0) # Allow some positive curliness, cap it
    except Exception as e: logger.warning(f"Error in __vicinity_curliness: {e}"); return 0.0

def __vicinity_lineness(vicinity):
    if vicinity is None or vicinity.shape[0] < 3: return 0.0
    try:
        p1, p_last = vicinity[0], vicinity[-1]
        x1, y1 = p1; x_last, y_last = p_last
        line_len_sq = (y_last - y1)**2 + (x_last - x1)**2
        if line_len_sq < 1e-12:
             if len(vicinity) == 0: return 0.0 # Should not happen with >=3 check
             return np.mean(np.sum((vicinity - p1)**2, axis=1))
        num = (y_last - y1) * vicinity[:, 0] - (x_last - x1) * vicinity[:, 1] + x_last * y1 - y_last * x1
        mean_sq_dist = np.mean(num**2 / line_len_sq)
        # Check for non-finite results and return 0 if necessary
        return mean_sq_dist if np.isfinite(mean_sq_dist) else 0.0
    except Exception as e: logger.warning(f"Error in __vicinity_lineness: {e}"); return 0.0


def __vicinity_slope(ink, point_idx):
    n = len(ink)
    if point_idx < 2 or point_idx >= n - 1: return 0.0 # Need p-2 and p+1
    try:
        p_first, p_last = ink[point_idx - 2, :2], ink[point_idx + 1, :2]
        dx, dy = p_last[0] - p_first[0], p_last[1] - p_first[1]
        length = math.hypot(dx, dy)
        # Return cosine of angle with x-axis
        return np.clip(dx / length, -1.0, 1.0) if length > 1e-9 else 1.0 # Default to horizontal if length is zero
    except Exception as e: logger.warning(f"Error in __vicinity_slope: {e}"); return 0.0

def __context_bitmap(ink, point_idx, bin_size=10, grid_size=3):
    if ink.shape[0] == 0: return np.zeros(grid_size*grid_size).tolist()
    try:
        center_x, center_y = ink[point_idx, :2]
        origin_x = center_x - (grid_size / 2.0) * bin_size
        origin_y = center_y - (grid_size / 2.0) * bin_size
        bitmap = np.zeros((grid_size, grid_size), dtype=float)
        count = 0
        # Define grid bounds for efficiency (optional, depends on data scale)
        min_bx, max_bx = origin_x, origin_x + grid_size * bin_size
        min_by, max_by = origin_y, origin_y + grid_size * bin_size

        # Iterate only through points potentially within the grid? Can be complex.
        # Simple iteration for now:
        for p_x, p_y, _ in ink:
            # Check if point is roughly within grid bounds first (optimization)
            # if not (min_bx <= p_x < max_bx and min_by <= p_y < max_by): continue

            # Calculate bin index robustly
            bin_x = math.floor((p_x - origin_x) / bin_size)
            bin_y = math.floor((p_y - origin_y) / bin_size)

            if 0 <= bin_x < grid_size and 0 <= bin_y < grid_size:
                bitmap[bin_y, bin_x] += 1; count += 1

        bitmap_flat = bitmap.flatten()
        # Normalize by total count in the grid
        bitmap_norm = bitmap_flat / count if count > 0 else bitmap_flat
        # Ensure finite values after normalization
        if not np.all(np.isfinite(bitmap_norm)):
            logger.warning(f"Bitmap for point {point_idx} contained non-finite values after normalization. Setting to zeros.")
            return np.zeros_like(bitmap_flat).tolist()
        return bitmap_norm.tolist()
    except Exception as e: logger.warning(f"Error in __context_bitmap: {e}"); return np.zeros(grid_size*grid_size).tolist()


