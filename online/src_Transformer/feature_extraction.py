import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

def moving_average(data_set, periods=3):
    """Calculates the moving average."""
    if periods <= 0:
        return data_set
    if periods >= len(data_set):
        # Return the mean of the whole dataset for each element if periods > len
        return np.full_like(data_set, np.mean(data_set))

    weights = np.ones(periods) / periods
    # Pad the start to get the same length output
    padded_data = np.pad(data_set, (periods - 1, 0), mode='edge')
    ma = np.convolve(padded_data, weights, mode='valid')
    return ma

def calculate_feature_vector_sequence(ink, args, delayed_strokes=None):
    """
    Calculates all features named in args for each point in the ink trajectory.

    Args:
        ink (np.array): Nx3 array [x, y, pen_up] (after preprocessing).
        args (list): List of feature names to calculate.
        delayed_strokes (np.array, optional): Array of delayed strokes (for 'hat' feature).

    Returns:
        np.array: Feature vector sequence [num_points, num_features].
    """
    logger.debug(f"Calculating features: {args}")
    if ink.shape[0] == 0:
        logger.warning("Cannot calculate features for empty ink.")
        # Determine expected number of features to return array of correct shape (0, num_features)
        num_features = 0
        if "x_cor" in args: num_features += 1
        if "y_cor" in args: num_features += 1
        if "penup" in args: num_features += 1
        if "dir" in args: num_features += 2
        if "curv" in args: num_features += 2
        if "hat" in args: num_features += 1
        if "vic_aspect" in args: num_features += 1
        if "vic_curl" in args: num_features += 1
        if "vic_line" in args: num_features += 1
        if "vic_slope" in args: num_features += 1
        if "bitmap" in args: num_features += 9 # Original has 9 bitmap features
        if "ma_x" in args: num_features += 1 # Moving Average X difference added
        return np.empty((0, num_features), dtype=np.float32)

    # Add moving average difference feature for x-coordinate if requested
    if "ma_x" in args:
        ma = moving_average(ink[:, 0], periods=20) # Using periods=20 as in original example
        ma_diff = (ink[:, 0] - ma).reshape(-1, 1)
    else:
        ma_diff = np.empty((ink.shape[0], 0)) # Empty array if not needed

    # Combine original ink with ma_diff for easier indexing
    combined_ink = np.hstack((ink, ma_diff))

    feature_vectors = []
    for p in range(len(combined_ink)):
        vec = __calculate_feature_vector(combined_ink, p, args, delayed_strokes)
        feature_vectors.append(vec)

    return np.array(feature_vectors, dtype=np.float32)


def __calculate_feature_vector(combined_ink, point_index, args, delayed_strokes=None):
    """Calculates the feature vector for a single point."""
    ink = combined_ink[:, :3] # Extract original ink part [x, y, penup]
    num_points = len(ink)
    
    # Calculate the number of features based on args
    num_features = 0
    if "x_cor" in args: num_features += 1
    if "y_cor" in args: num_features += 1
    if "penup" in args: num_features += 1
    if "dir" in args: num_features += 2
    if "curv" in args: num_features += 2
    if "hat" in args: num_features += 1
    if "vic_aspect" in args: num_features += 1
    if "vic_curl" in args: num_features += 1
    if "vic_line" in args: num_features += 1
    if "vic_slope" in args: num_features += 1
    if "bitmap" in args: num_features += 9 # 3x3 bitmap
    if "ma_x" in args: num_features += 1 # Moving average difference
    
    # Return zeros if not enough points for feature calculation
    if num_points < 3:
        return np.zeros(num_features, dtype=np.float32)
    
    feat_vec = []

    # Basic Coordinates & Pen status
    if "x_cor" in args:
        feat_vec.append(float(ink[point_index, 0])) # X coordinate
    if "y_cor" in args:
        feat_vec.append(float(ink[point_index, 1])) # Y coordinate
    if "penup" in args:
        feat_vec.append(__is_penup(ink, point_index))

    # Direction & Curvature (handle boundaries)
    if "dir" in args:
        if num_points < 2: feat_vec.extend([0.0, 0.0])
        else: feat_vec.extend(__writing_direction(ink, point_index))
    if "curv" in args:
        if num_points < 3: feat_vec.extend([0.0, 0.0]) # Curvature needs at least 3 points
        else: feat_vec.extend(__curvature(ink, point_index))

    # HAT Feature (requires delayed strokes)
    if "hat" in args:
        if delayed_strokes is None or len(delayed_strokes) == 0:
            feat_vec.append(0.0)
        else:
            feat_vec.append(__hat(ink, point_index, delayed_strokes))

    # Vicinity Features (require context, handle boundaries)
    vicinity_window = 5 # Total points needed (e.g., p-2 to p+2)
    has_enough_context = (point_index >= vicinity_window // 2) and \
                         (point_index < num_points - vicinity_window // 2)

    if "vic_aspect" in args:
        feat_vec.append(__vicinity_aspect(ink, point_index) if has_enough_context else 0.0)
    if "vic_curl" in args:
        feat_vec.append(__vicinity_curliness(ink, point_index) if has_enough_context else 0.0)
    if "vic_line" in args:
        feat_vec.append(__vicinity_lineness(ink, point_index) if has_enough_context else 0.0)
    if "vic_slope" in args:
        # Slope needs p-2 to p+1 -> context size 4 minimum
         has_slope_context = (point_index >= 2) and (point_index < num_points - 1)
         feat_vec.append(__vicinity_slope(ink, point_index) if has_slope_context else 0.0)

    # Context Bitmap
    if "bitmap" in args:
        feat_vec.extend(__context_bitmap(ink, point_index))

    # Moving Average Feature (from combined_ink)
    if "ma_x" in args:
        feat_vec.append(float(combined_ink[point_index, 3]))

    # Normalize the feature vector
    return __normalize_vector(np.array(feat_vec, dtype=np.float32))


def __normalize_vector(v):
    """Normalizes a vector using L2 norm."""
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

# --- Individual Feature Functions ---

def __is_penup(ink, point_idx):
    return float(ink[point_idx, 2])

def __writing_direction(ink, point_idx):
    """Calculates normalized direction vector."""
    num_points = len(ink)
    if num_points < 2: return [0.0, 0.0]

    if point_idx == 0:
        p_prev = ink[point_idx, :2]
        p_next = ink[point_idx + 1, :2]
        d = p_next - p_prev # Direction vector from point 0 to point 1
    elif point_idx == num_points - 1:
        p_prev = ink[point_idx - 1, :2]
        p_next = ink[point_idx, :2]
        d = p_next - p_prev # Direction vector from point n-2 to point n-1
    else:
        # Central difference for inner points
        p_prev = ink[point_idx - 1, :2]
        p_next = ink[point_idx + 1, :2]
        d = p_next - p_prev # Direction vector across point p_idx

    norm = np.linalg.norm(d)
    return (d / norm).tolist() if norm != 0 else [0.0, 0.0]

def __curvature(ink, point_idx):
    """Calculates curvature based on change in direction."""
    num_points = len(ink)
    if num_points < 3: return [0.0, 0.0] # Need at least 3 points for curvature

    # Ensure indices are valid for direction calculation
    idx_prev = max(0, point_idx - 1)
    idx_curr = point_idx
    idx_next = min(num_points - 1, point_idx + 1)

    # Get directions
    if point_idx == 0:
        # Use first two segments if possible
        if num_points >= 3:
            dir_in = __writing_direction(ink, 0) # Dir 0->1
            dir_out = __writing_direction(ink, 1) # Dir 1->2
        else: # Only 2 points
             return [1.0, 0.0] # No curvature measurable
    elif point_idx == num_points - 1:
        # Use last two segments
        if num_points >= 3:
             dir_in = __writing_direction(ink, num_points - 2)
             dir_out = __writing_direction(ink, num_points - 1)
        else:
            return [1.0, 0.0] # Straight line
    else:
        # Standard case: use directions around point_idx
        dir_in = __writing_direction(ink, point_idx - 1)
        dir_out = __writing_direction(ink, point_idx)

    cos_prev, sin_prev = dir_in
    cos_next, sin_next = dir_out

    # Calculate cosine and sine of the angle between the direction vectors
    curv_cos = cos_prev * cos_next + sin_prev * sin_next
    curv_sin = cos_prev * sin_next - sin_prev * cos_next

    # Clamp values to avoid potential floating point issues
    curv_cos = np.clip(curv_cos, -1.0, 1.0)
    curv_sin = np.clip(curv_sin, -1.0, 1.0)

    return [curv_cos, curv_sin]


def __hat(ink, point_idx, delayed_strokes):
    """Checks if the point lies below a delayed stroke ('hat' feature)."""
    if delayed_strokes is None or len(delayed_strokes) == 0:
        return 0.0

    point_x, point_y = ink[point_idx, :2]

    # Handle different possible formats of delayed_strokes
    if delayed_strokes.ndim == 2 and delayed_strokes.shape[0] > 0:
        # Find stroke segments within the delayed strokes
        penup_indices = np.where(delayed_strokes[:, 2] == 1)[0]
        start = 0
        for end in penup_indices:
            stroke_segment = delayed_strokes[start:end+1, :]
            if len(stroke_segment) > 0:
                minx = np.min(stroke_segment[:, 0])
                maxx = np.max(stroke_segment[:, 0])
                miny = np.min(stroke_segment[:, 1])
                # Check if point is horizontally within segment and vertically below it
                if minx <= point_x <= maxx and point_y < miny:
                    return 1.0
            start = end + 1
    # Handle the case where delayed_strokes is a list of arrays
    elif isinstance(delayed_strokes, list) and len(delayed_strokes) > 0:
        for stroke in delayed_strokes:
            if len(stroke) > 0:
                minx = np.min(stroke[:, 0])
                maxx = np.max(stroke[:, 0])
                miny = np.min(stroke[:, 1])
                if minx <= point_x <= maxx and point_y < miny:
                    return 1.0

    return 0.0


def __get_vicinity(ink, point_idx, window_size=5):
    """Safely extracts the vicinity window around a point."""
    num_points = len(ink)
    half_window = window_size // 2
    start = max(0, point_idx - half_window)
    end = min(num_points, point_idx + half_window + 1) # +1 for slice exclusion
    vicinity = ink[start:end, :2] # Only x, y coordinates
    return vicinity


def __vicinity_aspect(ink, point_idx):
    """Aspect ratio of the vicinity's bounding box."""
    # Uses fixed p-2:p+3, requires window size 5
    vicinity = __get_vicinity(ink, point_idx, window_size=5)
    if len(vicinity) < 2: return 0.0 # Need at least 2 points for aspect

    min_coords = np.min(vicinity, axis=0)
    max_coords = np.max(vicinity, axis=0)
    dx = max_coords[0] - min_coords[0]
    dy = max_coords[1] - min_coords[1]

    if dx + dy < 1e-6: # Avoid division by zero if points are identical
        return 0.0
    # Formula: 2 * dy / (dx + dy) - 1. Range [-1, 1]
    return (2.0 * dy / (dx + dy)) - 1.0


def __vicinity_curliness(ink, point_idx):
    """Ratio of path length to max bounding box dimension in vicinity."""
    vicinity = __get_vicinity(ink, point_idx, window_size=5)
    if len(vicinity) < 2: return 0.0

    min_coords = np.min(vicinity, axis=0)
    max_coords = np.max(vicinity, axis=0)
    dx = max_coords[0] - min_coords[0]
    dy = max_coords[1] - min_coords[1]
    max_dim = max(dx, dy)

    if max_dim < 1e-6: # Points are likely identical
        return 0.0

    # Calculate path length within the vicinity
    segment_lengths = np.sqrt(np.sum(np.diff(vicinity, axis=0)**2, axis=1))
    path_length = np.sum(segment_lengths)

    # Return path_length / max_dim - 2 (as in original)
    return (path_length / max_dim) - 2.0 if max_dim > 0 else 0.0


def __vicinity_lineness(ink, point_idx):
    """Mean squared distance of vicinity points to the line connecting start and end points."""
    v = __get_vicinity(ink, point_idx, window_size=5)
    if len(v) < 3: return 0.0 # Need at least 3 points for meaningful lineness

    p1 = v[0]   # First point in vicinity
    p2 = v[-1]  # Last point in vicinity

    x1, y1 = p1
    x2, y2 = p2

    # Length of the line segment connecting first and last point
    diag_line_length_sq = (y2 - y1)**2 + (x2 - x1)**2

    if diag_line_length_sq < 1e-12: # First and last points are very close
        # Calculate mean squared distance to the first point as an alternative
        distances_sq = np.sum((v - p1)**2, axis=1)
        return np.mean(distances_sq) if len(v) > 0 else 0.0

    # Calculate squared perpendicular distance of each point to the line (p1, p2)
    numerator_sq = ((y2 - y1) * v[:, 0] - (x2 - x1) * v[:, 1] + x2 * y1 - y2 * x1)**2
    squared_distances = numerator_sq / diag_line_length_sq

    # Return the average squared distance
    return np.mean(squared_distances)


def __vicinity_slope(ink, point_idx):
    """Cosine of the angle of the line connecting first and last points of a vicinity window."""
    # Uses fixed p-2:p+2 (window size 4)
    if point_idx < 2 or point_idx > len(ink) - 2: # Boundary check
        return 0.0

    p_first = ink[point_idx - 2, :2] # Point p-2
    p_last = ink[point_idx + 1, :2]  # Point p+1

    delta_x = p_last[0] - p_first[0]
    delta_y = p_last[1] - p_first[1]

    if abs(delta_x) < 1e-6: # Vertical line
        return 0.0 # Cos(90 degrees)

    # Calculate slope
    slope = delta_y / delta_x
    # Angle = atan(slope)
    angle = math.atan(slope)
    # Return cosine of the angle
    return math.cos(angle)


def __context_bitmap(ink, point_idx, bin_size=10, grid_size=3):
    """Creates a 3x3 bitmap centered around the point."""
    center_x, center_y = ink[point_idx, :2]
    
    # Adjust origin to center the middle bin [1,1] on the point
    window_origin_x = center_x - (grid_size // 2 + 0.5) * bin_size
    window_origin_y = center_y - (grid_size // 2 + 0.5) * bin_size

    # Initialize bitmap
    bitmap = np.zeros((grid_size, grid_size), dtype=float)
    num_points_in_grid = 0

    # Iterate through all points in the ink
    for p_x, p_y, _ in ink:
        # Calculate bin indices
        bin_x = int(math.floor((p_x - window_origin_x) / bin_size))
        bin_y = int(math.floor((p_y - window_origin_y) / bin_size))

        # Check if the point falls within the grid boundaries
        if 0 <= bin_x < grid_size and 0 <= bin_y < grid_size:
            bitmap[bin_y, bin_x] += 1
            num_points_in_grid += 1

    # Flatten and normalize the bitmap
    bitmap_flat = bitmap.flatten() # Shape (9,) for 3x3 grid

    # Ensure we don't divide by zero
    if num_points_in_grid > 0:
        bitmap_normalized = bitmap_flat / num_points_in_grid
    else:
        bitmap_normalized = bitmap_flat # Avoid division by zero

    return bitmap_normalized.tolist()