# src_Transformer/data_preparation.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import tarfile
from glob import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import logging
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # Optional for visualization
import multiprocessing as mp
from tqdm import tqdm

# Import our own modules
from preprocessing import preprocess_handwriting
from feature_extraction import calculate_feature_vector_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def ensure_dir(directory):
    """Creates the directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def get_project_root():
    """Finds the project root directory where 'src_Transformer', 'data' etc. are located."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level from 'src_Transformer' to the project root
    return os.path.dirname(current_dir)

def Pfad_Sicherung(project_root):
    """Sets the working directory to the project root."""
    os.chdir(project_root)
    logger.info(f"Working directory set to {project_root}.")

# -- XML Extraction --
def extract_strokes_from_xml(xml_file_path):
    """Extracts stroke points from an IAM XML file."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        stroke_set = root.find("StrokeSet")
        if stroke_set is None:
            logger.warning(f"No StrokeSet found in {xml_file_path}")
            return []

        strokes = []
        for stroke_node in stroke_set:
            point_count = 0
            for point in stroke_node:
                try:
                    x = int(point.attrib['x'])
                    y = int(point.attrib['y'])
                    # Add point with pen_down=0 initially
                    strokes.append([x, y, 0])
                    point_count += 1
                except KeyError:
                    logger.warning(f"Point missing x or y attribute in {xml_file_path}")
                except ValueError:
                     logger.warning(f"Non-integer coordinate in {xml_file_path}")

            # Mark the *last added* point of this stroke as pen_up=1
            if point_count > 0 and strokes:
                strokes[-1][2] = 1
            elif point_count == 0:
                 logger.warning(f"Empty stroke detected in {xml_file_path}")

        # Ensure the very last point in the entire file has penup=1
        if strokes:
             strokes[-1][2] = 1
        else:
             logger.warning(f"No points extracted from {xml_file_path}")

        return strokes
    except ET.ParseError:
        logger.error(f"Failed to parse XML file: {xml_file_path}")
        return []
    except Exception as e:
        logger.error(f"Error extracting strokes from {xml_file_path}: {e}")
        return []

def folder_has_data(folder_path, ext):
    """Checks if a folder contains files with a specific extension."""
    if not os.path.isdir(folder_path):
        return False
    # Using glob for efficiency
    pattern = os.path.join(folder_path, '**', f'*{ext}')
    return bool(glob(pattern, recursive=True))

def maybe_unpack_archives(data_folder):
    """
    Unpacks archives if necessary and needed.
    """
    project_root = get_project_root()
    data_folder_abs = os.path.join(project_root, data_folder)
    ascii_path = os.path.join(data_folder_abs, "ascii")
    line_strokes_path = os.path.join(data_folder_abs, "lineStrokes")

    ascii_has_txt = folder_has_data(ascii_path, ".txt")
    line_has_xml = folder_has_data(line_strokes_path, ".xml")

    if ascii_has_txt and line_has_xml:
        logger.info("ASCII (.txt) and lineStrokes (.xml) directories seem populated. Skipping unpacking.")
        return

    logger.info("Checking for archives to unpack...")

    archive_patterns = {
        "ascii": [os.path.join(data_folder_abs, "ascii-all.tar.gz"), os.path.join(data_folder_abs, "ascii-all.tar")],
        "lineStrokes": [os.path.join(data_folder_abs, "lineStrokes-all.tar.gz"), os.path.join(data_folder_abs, "lineStrokes-all.tar")]
    }

    needs_unpacking = {
        "ascii": not ascii_has_txt,
        "lineStrokes": not line_has_xml
    }

    for data_type, paths in archive_patterns.items():
        if needs_unpacking[data_type]:
            unpacked = False
            for archive_path in paths:
                if os.path.exists(archive_path):
                    mode = 'r:gz' if archive_path.endswith('.gz') else 'r:'
                    logger.info(f"Unpacking {archive_path} with mode {mode} into {data_folder_abs}...")
                    try:
                        with tarfile.open(archive_path, mode) as tar:
                            tar.extractall(path=data_folder_abs)
                        logger.info(f"Successfully unpacked {archive_path}.")
                        unpacked = True
                        break # Stop after unpacking one archive for this type
                    except tarfile.ReadError as e:
                        logger.error(f"Error reading archive {archive_path}: {e}")
                    except Exception as e:
                        logger.error(f"Failed to unpack {archive_path}: {e}")
            if not unpacked:
                 logger.warning(f"Could not find or unpack required archive for {data_type} data")
        else:
             logger.info(f"{data_type} data already present, skipping unpack.")

def create_initial_dataframe(data_folder):
    """
    Scans for .txt files and creates an initial DataFrame mapping XML paths to transcripts.
    """
    project_root = get_project_root()
    data_folder_abs = os.path.join(project_root, data_folder)
    ascii_path = os.path.join(data_folder_abs, 'ascii')
    line_strokes_path = os.path.join(data_folder_abs, 'lineStrokes')

    if not os.path.isdir(ascii_path):
        logger.error(f"ASCII directory not found: {ascii_path}. Cannot create DataFrame.")
        return pd.DataFrame(columns=['xml_file_path', 'transcript'])

    logger.info(f"Scanning for .txt files in {ascii_path}...")
    # Find all .txt files recursively
    txt_files = glob(os.path.join(ascii_path, '**', '*.txt'), recursive=True)
    logger.info(f"Found {len(txt_files)} .txt files.")

    if not txt_files:
        logger.warning("No .txt files found. Returning empty DataFrame.")
        return pd.DataFrame(columns=['xml_file_path', 'transcript'])

    data_list = []
    skipped_csr_count = 0
    file_read_errors = 0

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading file {txt_file}: {e}")
            file_read_errors += 1
            continue

        # Find the start of the transcriptions (CSR section) robustly
        csr_index = -1
        for idx, line in enumerate(lines):
            if line.strip().upper() == 'CSR:':
                csr_index = idx
                break

        if csr_index == -1:
            # Try slightly different variations if exact match fails
            for idx, line in enumerate(lines):
                 if "CSR:" in line.strip().upper():
                     csr_index = idx
                     logger.debug(f"Found potential CSR line variation in {txt_file}")
                     break

        if csr_index == -1:
            logger.warning(f"Could not find 'CSR:' marker in {txt_file}. Skipping file.")
            skipped_csr_count += 1
            continue

        # Transcripts usually start 2 lines after 'CSR:' (blank line, then first transcript)
        transcript_lines = lines[csr_index + 2:]

        # Derive base path for XML files
        relative_txt_path = os.path.relpath(txt_file, ascii_path)
        xml_base_path_rel = os.path.splitext(relative_txt_path)[0] # Remove .txt extension
        # Construct the corresponding path in lineStrokes directory
        xml_base_path_abs = os.path.join(line_strokes_path, xml_base_path_rel)

        for i, line in enumerate(transcript_lines):
            transcript = line.strip()
            if not transcript: # Skip empty lines
                continue

            # Construct the expected XML file path (e.g., a01-000u-01.xml)
            # The original naming convention seems to be file-XX.xml where XX is the line number
            # Let's assume line number corresponds to i+1
            expected_xml_filename = f"{os.path.basename(xml_base_path_abs)}-{i+1:02}.xml"
            xml_file_path_abs = os.path.join(os.path.dirname(xml_base_path_abs), expected_xml_filename)

            # Store the relative path from the project root for portability
            xml_file_path_rel = os.path.relpath(xml_file_path_abs, project_root)

            data_list.append({
                'xml_file_path': xml_file_path_rel.replace("\\", "/"), # Use forward slashes
                'transcript': transcript
            })

    logger.info(f"Finished scanning .txt files. Entries found: {len(data_list)}. Files skipped (no CSR): {skipped_csr_count}. File read errors: {file_read_errors}.")

    df = pd.DataFrame(data_list)
    if df.empty:
        logger.warning("DataFrame is empty after processing .txt files.")
    return df

def split_data(df, test_size=0.1, val_size=0.1, random_state=42):
    """
    Splits the DataFrame into train, validation, and test sets.
    Ensures that the validation and test sizes are fractions of the *original* data.
    """
    if df.empty:
        return df, df, df

    # Calculate actual fractions for train_test_split
    # First split into train and temp (val + test)
    train_frac = 1.0 - test_size - val_size
    temp_size = test_size + val_size # Size of the temporary set relative to original

    if train_frac <= 0 or train_frac >= 1:
        raise ValueError("Invalid test/val sizes. Ensure test_size + val_size < 1.")

    df_train, df_temp = train_test_split(
        df, test_size=temp_size, random_state=random_state, shuffle=True
    )

    # Calculate split ratio for val vs test
    val_size_of_temp = val_size / temp_size

    if len(df_temp) > 0: # Only split if temp set is not empty
        df_val, df_test = train_test_split(
            df_temp, test_size=(1.0 - val_size_of_temp), random_state=random_state, shuffle=True
        )
    else: # Should not happen with valid sizes, but handle defensively
        df_val, df_test = df_temp.copy(), df_temp.copy() # Return empty DataFrames if temp is empty

    logger.info(f"Data split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    # Sanity check percentages
    total = len(df)
    if total > 0:
        logger.info(f"Percentages: Train={len(df_train)/total:.1%}, Val={len(df_val)/total:.1%}, Test={len(df_test)/total:.1%}")

    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def process_sample(args):
    """
    Processes a single sample for parallel processing.
    Args:
        args: Tuple containing (row_data, project_root, bin_path, norm_args, feat_args)
    Returns:
        Dict with results or None if error
    """
    idx, row, project_root, bin_folder, norm_args, feat_args = args
    
    xml_path_rel = row['xml_file_path']
    xml_path_abs = os.path.join(project_root, xml_path_rel)
    transcript = row['transcript']

    # Create corresponding binary file path
    bin_path_rel = xml_path_rel.replace('lineStrokes', bin_folder).replace('.xml', '.bin')
    bin_path_abs = os.path.join(project_root, bin_path_rel)

    # Ensure the directory for the bin file exists
    os.makedirs(os.path.dirname(bin_path_abs), exist_ok=True)

    try:
        strokes_list = extract_strokes_from_xml(xml_path_abs)
        if not strokes_list:
            return None  # Skip if no strokes

        strokes_np = np.array(strokes_list, dtype=float)

        # --- Preprocessing ---
        ink = preprocess_handwriting(strokes_np, norm_args)

        # --- Feature Extraction ---
        features = calculate_feature_vector_sequence(ink, feat_args, delayed_strokes=None)

        if features.shape[0] == 0 or features.shape[1] == 0:
            return None  # Skip if empty features

        # --- Save Features ---
        features.astype(np.float32).tofile(bin_path_abs)

        return {
            'bin_file_path': bin_path_rel.replace("\\", "/"),
            'transcript': transcript
        }

    except Exception as e:
        return None

def process_and_save_features(df, dataset_name, data_folder, bin_folder, norm_args, feat_args, use_parallel=True):
    """
    Processes one partition (train, val, test) of the data.
    Reads XML, preprocesses, extracts features, and saves to .bin files.
    Returns a DataFrame with paths to the .bin files.
    Optionally uses parallel processing for speed.
    """
    project_root = get_project_root()
    bin_output_dir = os.path.join(project_root, data_folder, bin_folder)
    ensure_dir(bin_output_dir)

    logger.info(f"Processing {dataset_name} data ({len(df)} samples)...")

    if use_parallel and len(df) > 10:  # Only use parallel for larger datasets
        num_cpus = max(1, mp.cpu_count() - 1)  # Keep one CPU free
        logger.info(f"Using {num_cpus} CPU cores for parallel processing")
        
        # Prepare arguments for parallel processing
        args_list = [(idx, row, project_root, bin_folder, norm_args, feat_args) 
                    for idx, row in df.iterrows()]
        
        # Process in parallel
        with mp.Pool(processes=num_cpus) as pool:
            results = list(tqdm(
                pool.imap(process_sample, args_list),
                total=len(args_list),
                desc=f"Processing {dataset_name}"
            ))
        
        # Filter out None results (errors)
        processed_list = [r for r in results if r is not None]
        
        # Count stats
        error_count = sum(1 for r in results if r is None)
        
    else:
        # Sequential processing
        processed_list = []
        error_count = 0
        empty_stroke_count = 0
        empty_feature_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name}"):
            xml_path_rel = row['xml_file_path']
            xml_path_abs = os.path.join(project_root, xml_path_rel)
            transcript = row['transcript']

            # Create corresponding binary file path
            bin_path_rel = xml_path_rel.replace('lineStrokes', bin_folder).replace('.xml', '.bin')
            bin_path_abs = os.path.join(project_root, bin_path_rel)

            # Ensure the directory for the bin file exists
            os.makedirs(os.path.dirname(bin_path_abs), exist_ok=True)

            try:
                strokes_list = extract_strokes_from_xml(xml_path_abs)
                if not strokes_list:
                    logger.warning(f"Skipping {xml_path_rel}: No strokes extracted.")
                    empty_stroke_count += 1
                    continue

                strokes_np = np.array(strokes_list, dtype=float)

                # --- Preprocessing ---
                ink = preprocess_handwriting(strokes_np, norm_args)

                # --- Feature Extraction ---
                features = calculate_feature_vector_sequence(ink, feat_args, delayed_strokes=None)

                if features.shape[0] == 0 or features.shape[1] == 0:
                    logger.warning(f"Skipping {xml_path_rel}: Feature extraction resulted in empty array.")
                    empty_feature_count += 1
                    continue

                # --- Save Features ---
                features.astype(np.float32).tofile(bin_path_abs)

                processed_list.append({
                    'bin_file_path': bin_path_rel.replace("\\", "/"),
                    'transcript': transcript
                })

            except FileNotFoundError:
                logger.error(f"XML file not found: {xml_path_abs}. Skipping.")
                error_count += 1
            except Exception as e:
                logger.error(f"Error processing {xml_path_rel}: {e}")
                error_count += 1

    logger.info(f"Finished processing {dataset_name} data:")
    logger.info(f"  Successfully processed: {len(processed_list)}")
    logger.info(f"  Errors/Skipped: {error_count}")

    return pd.DataFrame(processed_list)

def main():
    # --- Configuration ---
    DATA_FOLDER = 'data'        # Relative to project root
    BIN_FOLDER = 'bin_files'    # Subfolder within DATA_FOLDER for .bin files
    OUTPUT_EXCEL_PREFIX = 'iam' # Prefix for the output Excel files

    # Preprocessing & Feature Args
    NORM_ARGS = ["origin", "smooth", "slope", "resample", "slant", "height"]
    FEAT_ARGS = [
        "x_cor", "y_cor", "penup",       # 3 basic features
        "dir",                           # 2 direction features (x,y components)
        "curv",                          # 2 curvature features (cos,sin)
        "vic_aspect", "vic_curl", "vic_line", "vic_slope", # 4 vicinity features
        "bitmap"                         # 9 bitmap features (3x3 grid)
    ]
    # Check: 1+1+1 + 2 + 2 + 4 + 9 = 20. Correct.

    SPLIT_RANDOM_STATE = 42
    TEST_SIZE = 0.10 # 10% for testing
    VAL_SIZE = 0.10  # 10% for validation (leaving 80% for training)
    USE_PARALLEL = True  # Use parallel processing for speed

    # --- Setup ---
    project_root = get_project_root()
    Pfad_Sicherung(project_root)
    data_folder_abs = os.path.join(project_root, DATA_FOLDER)

    # 1) Unpack Archives if needed
    maybe_unpack_archives(DATA_FOLDER)

    # 2) Create Initial DataFrame from TXT/XML mapping
    df_initial = create_initial_dataframe(DATA_FOLDER)

    # 3) Filter out entries where the XML file doesn't actually exist
    logger.info(f"Initial DataFrame size: {len(df_initial)}")
    df_initial['xml_exists'] = df_initial['xml_file_path'].apply(lambda p: os.path.exists(os.path.join(project_root, p)))
    df_filtered = df_initial[df_initial['xml_exists']].copy()
    logger.info(f"DataFrame size after checking XML existence: {len(df_filtered)}")
    if len(df_filtered) < len(df_initial):
        logger.warning(f"Removed {len(df_initial) - len(df_filtered)} entries due to missing XML files.")
    if df_filtered.empty:
        logger.error("No valid XML file paths found. Cannot proceed.")
        return
    df_filtered = df_filtered.drop(columns=['xml_exists'])

    # 4) Split Data into Train, Validation, Test sets (80/10/10)
    df_train, df_val, df_test = split_data(
        df_filtered, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=SPLIT_RANDOM_STATE
    )

    # 5) Process each split: Preprocessing, Feature Extraction, Saving .bin files
    df_train_processed = process_and_save_features(
        df_train, "Train", DATA_FOLDER, BIN_FOLDER, NORM_ARGS, FEAT_ARGS, use_parallel=USE_PARALLEL
    )
    df_val_processed = process_and_save_features(
        df_val, "Validation", DATA_FOLDER, BIN_FOLDER, NORM_ARGS, FEAT_ARGS, use_parallel=USE_PARALLEL
    )
    df_test_processed = process_and_save_features(
        df_test, "Test", DATA_FOLDER, BIN_FOLDER, NORM_ARGS, FEAT_ARGS, use_parallel=USE_PARALLEL
    )

    # 6) Save the final DataFrames (pointing to .bin files) as Excel files
    train_excel_path = os.path.join(data_folder_abs, f"{OUTPUT_EXCEL_PREFIX}_train.xlsx")
    val_excel_path = os.path.join(data_folder_abs, f"{OUTPUT_EXCEL_PREFIX}_val.xlsx")
    test_excel_path = os.path.join(data_folder_abs, f"{OUTPUT_EXCEL_PREFIX}_test.xlsx")

    try:
        df_train_processed.to_excel(train_excel_path, index=False)
        logger.info(f"Train data index saved to: {train_excel_path}")
        df_val_processed.to_excel(val_excel_path, index=False)
        logger.info(f"Validation data index saved to: {val_excel_path}")
        df_test_processed.to_excel(test_excel_path, index=False)
        logger.info(f"Test data index saved to: {test_excel_path}")
    except Exception as e:
        logger.error(f"Failed to save processed data indices to Excel: {e}")

    logger.info("Data preparation script finished.")

if __name__ == "__main__":
    main()