# src_CRNN/data_preparation.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # Für Mac/Intel Konflikte
import tarfile
from glob import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import logging
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from tqdm import tqdm
import sys

# Stelle sicher, dass Projekt-Root im Pfad ist für config etc.
project_root_for_import = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root_for_import not in sys.path:
     sys.path.insert(0, project_root_for_import)

# Importiere config von außerhalb
import config

# --- Direkte Relative Imports für Module im selben Paket ---
try:
    from preprocessing import preprocess_handwriting
    from feature_extraction import calculate_feature_vector_sequence, _get_expected_num_features
    # Importiere spezifische Funktionen aus utils
    from utils import create_directory, clean_relative_path # WICHTIG: create_directory verwenden!
except ImportError as e:
    # Wenn dieser Import fehlschlägt, liegt ein grundlegendes Problem vor
    # Initialisiere Logger hier, da config es eventuell noch nicht getan hat
    logging.basicConfig(level=logging.ERROR)
    logger_fallback = logging.getLogger(__name__)
    logger_fallback.error(f"Kritischer Fehler beim relativen Import in data_preparation: {e}. Paketstruktur/ PYTHONPATH prüfen!", exc_info=True)
    sys.exit(1)

# Logger holen (sollte jetzt von config initialisiert sein)
logger = logging.getLogger(__name__)


# --- XML Extraction (Unverändert) ---
def extract_strokes_from_xml(xml_file_path):
    """Extracts stroke points from an IAM XML file with enhanced error handling."""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        stroke_set = root.find("StrokeSet")
        if stroke_set is None: logger.warning(f"No StrokeSet found in {xml_file_path}"); return []
        strokes = []
        for stroke_node in stroke_set:
            point_count = 0
            for point in stroke_node:
                try:
                    x = int(point.attrib['x']); y = int(point.attrib['y'])
                    strokes.append([float(x), float(y), 0.0]) # Use float
                    point_count += 1
                except (KeyError, ValueError): continue # Skip invalid points silently after first warning?
                except Exception as point_e: logger.warning(f"Error reading point: {point_e}"); continue
            if point_count > 0 and strokes: strokes[-1][2] = 1.0 # Mark last added as pen_up
            elif point_count == 0: logger.debug(f"Empty stroke detected in {xml_file_path}")
        if strokes: strokes[-1][2] = 1.0; return strokes
        else: logger.warning(f"No valid points extracted from {xml_file_path}"); return []
    except ET.ParseError: logger.error(f"Failed to parse XML file: {xml_file_path}"); return []
    except FileNotFoundError: logger.error(f"XML file not found: {xml_file_path}"); return []
    except Exception as e: logger.error(f"Error extracting strokes from {xml_file_path}: {e}"); return []


# --- Archive/File Handling (Unverändert) ---
def folder_has_data(folder_path, ext):
    """Checks if a folder contains files with a specific extension recursively."""
    if not os.path.isdir(folder_path): return False
    return bool(glob(os.path.join(folder_path, '**', f'*{ext}'), recursive=True))

def maybe_unpack_archives(data_folder_abs):
    """Unpacks standard IAM-OnDB archives if needed."""
    ascii_path = os.path.join(data_folder_abs, "ascii")
    line_strokes_path = os.path.join(data_folder_abs, "lineStrokes")
    ascii_has_txt = os.path.isdir(ascii_path) and folder_has_data(ascii_path, ".txt")
    line_has_xml = os.path.isdir(line_strokes_path) and folder_has_data(line_strokes_path, ".xml")

    if ascii_has_txt and line_has_xml:
        logger.info("ASCII and lineStrokes directories seem populated. Skipping unpacking.")
        return True # Indicate success/completeness

    logger.info("Checking for archives to unpack...")
    archive_patterns = {
        "ascii": ["ascii-all.tar.gz", "ascii.tar.gz", "ascii.tgz"], # Suche nach beiden Endungen
        "lineStrokes": ["lineStrokes-all.tar.gz", "lineStrokes.tar.gz", "lines.tgz"] # auch lines.tgz prüfen?
    }
    needs_unpacking = {"ascii": not ascii_has_txt, "lineStrokes": not line_has_xml}
    unpacked_any = False

    for data_type, archives in archive_patterns.items():
        if needs_unpacking[data_type]:
            logger.info(f"Need to unpack data for '{data_type}'. Searching...")
            found_and_unpacked = False
            for arc_name in archives:
                archive_path = os.path.join(data_folder_abs, arc_name)
                if os.path.exists(archive_path):
                    # Mode-Erkennung verbessert
                    mode = 'r:gz' if archive_path.endswith('.gz') or archive_path.endswith('.tgz') else 'r:' if archive_path.endswith('.tar') else None
                    if mode is None: logger.warning(f"Unbekanntes Archivformat übersprungen: {arc_name}"); continue

                    logger.info(f"Found archive: {archive_path}. Unpacking (Mode: {mode})...")
                    try:
                        with tarfile.open(archive_path, mode) as tar:
                            tar.extractall(path=data_folder_abs)
                        logger.info(f"Successfully unpacked {archive_path}.")
                        unpacked_any = True; found_and_unpacked = True
                        # Verzeichnisprüfung direkt nach Entpacken (wichtig)
                        target_path = ascii_path if data_type == "ascii" else line_strokes_path
                        if not os.path.isdir(target_path):
                             logger.warning(f"Nach Entpacken von {arc_name}, Zielordner {target_path} nicht gefunden! Prüfe Archivstruktur.")
                        break # Stop after unpacking one for this type
                    except Exception as e: logger.error(f"Failed to unpack {archive_path}: {e}")
            if not found_and_unpacked:
                 logger.warning(f"Could not find a required archive for '{data_type}' data in {data_folder_abs}. Looked for: {archives}")

    # Verify again after attempted unpacking
    if unpacked_any:
        # Re-check directory status
        ascii_ok = os.path.isdir(ascii_path) and folder_has_data(ascii_path, ".txt")
        strokes_ok = os.path.isdir(line_strokes_path) and folder_has_data(line_strokes_path, ".xml")
        if not ascii_ok: logger.error("Unpacking attempted, but ASCII (.txt) data directory is still missing or empty."); return False
        if not strokes_ok: logger.error("Unpacking attempted, but lineStrokes (.xml) data directory is still missing or empty."); return False
        logger.info("Archive unpacking successful and verified.")
        return True
    elif needs_unpacking["ascii"] or needs_unpacking["lineStrokes"]:
        logger.error("Failed to find/unpack necessary archives. Cannot proceed without BOTH ascii and lineStrokes folders containing data.")
        return False # Indicate failure
    else:
        return True # Already had data


# --- DataFrame Creation & Splitting (Unverändert) ---
def create_initial_dataframe(data_folder_abs):
    """Scans for .txt, parses transcripts, maps to XML paths relative to BASE_PATH."""
    ascii_path = os.path.join(data_folder_abs, 'ascii')
    # line_strokes_folder = os.path.join(data_folder_abs, 'lineStrokes') # Nicht direkt benötigt hier
    if not os.path.isdir(ascii_path): logger.error(f"ASCII dir not found: {ascii_path}"); return pd.DataFrame()

    logger.info(f"Scanning for .txt files in {ascii_path}...")
    txt_files = glob(os.path.join(ascii_path, '**', '*.txt'), recursive=True)
    logger.info(f"Found {len(txt_files)} .txt files.")
    if not txt_files: return pd.DataFrame()

    data_list = []; skipped_csr = 0; file_errors = 0; skipped_empty = 0
    for txt_file in tqdm(txt_files, desc="Parsing TXT files"):
        try:
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f: lines = f.readlines()
        except Exception as e: file_errors += 1; logger.debug(f"Error reading {txt_file}: {e}"); continue
        csr_index = -1
        for idx, line in enumerate(lines):
            # Robusteres Finden von CSR:
            stripped_line = line.strip()
            if stripped_line == "CSR:" or stripped_line.startswith("CSR: "): # Auch Leerzeichen nach : erlauben
                 csr_index = idx
                 break
        # Handle file ending right after CSR:
        if csr_index == -1 or csr_index + 2 >= len(lines): skipped_csr += 1; logger.debug(f"CSR missing or too close to end in {txt_file}"); continue
        transcript_lines = lines[csr_index + 2:]
        rel_txt_path_from_ascii = os.path.relpath(txt_file, ascii_path)
        xml_base_rel = os.path.splitext(rel_txt_path_from_ascii)[0]

        for i, line in enumerate(transcript_lines):
            transcript = line.strip()
            if not transcript: skipped_empty += 1; continue
            # Derive XML path
            txt_basename = os.path.basename(xml_base_rel)
            # Handle potential variations like '-01' or just '01' - :02 formatiert immer mit führender 0
            xml_filename = f"{txt_basename}-{i+1:02}.xml"
            xml_dir_rel = os.path.dirname(xml_base_rel)
            # Pfad relativ zu BASE_PATH, basierend auf config.DATA_FOLDER_NAME
            xml_path_rel_to_base = os.path.join(config.DATA_FOLDER_NAME, 'lineStrokes', xml_dir_rel, xml_filename).replace("\\", "/")
            data_list.append({'xml_file_path': xml_path_rel_to_base, 'transcript': transcript})

    logger.info(f"Finished TXT scan. Entries: {len(data_list)}. Skipped CSR/EOF: {skipped_csr}. Read errors: {file_errors}. Skipped empty lines: {skipped_empty}.")
    df = pd.DataFrame(data_list)
    # Check for duplicates - important!
    duplicates = df[df.duplicated(subset=['xml_file_path'], keep=False)]
    if not duplicates.empty:
         logger.warning(f"Found {len(duplicates)} duplicate XML paths. Example: {duplicates['xml_file_path'].iloc[0]}. Dropping duplicates, keeping first.")
         df = df.drop_duplicates(subset=['xml_file_path'], keep='first').reset_index(drop=True)
    return df

def split_data(df, test_size=0.1, val_size=0.1, random_state=42):
    """Splits DataFrame into train/val/test."""
    if df.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    n_samples = len(df)
    if n_samples < 3: logger.warning(f"Dataset too small ({n_samples}) for split."); return df.copy(), pd.DataFrame(), pd.DataFrame()
    # Runden zur nächsten ganzen Zahl kann stabiler sein als int() bei kleinen Anteilen
    n_test = max(1, round(n_samples * test_size))
    n_val = max(1, round(n_samples * val_size))
    if n_test + n_val >= n_samples: # Adjust if needed
        logger.warning("Requested Test+Val size too large, adjusting...")
        n_test = min(n_test, n_samples - 1); n_val = max(0, n_samples - n_test - 1)
    test_frac = n_test / n_samples
    # Berechne val_frac für den *Rest* nach dem Test-Split
    val_frac_rem = n_val / (n_samples - n_test) if (n_samples - n_test) > 0 else 0
    logger.info(f"Splitting {n_samples} samples: Target Test={n_test}, Val={n_val}")

    # Ensure test_frac is within valid range [0, 1]
    test_frac = np.clip(test_frac, 0.0, 1.0)

    # Split test set
    df_train_val, df_test = train_test_split(df, test_size=test_frac, random_state=random_state, shuffle=True)

    # Split validation set from the rest
    if len(df_train_val) > 1 and val_frac_rem > 0:
         # Ensure val_frac_rem is within valid range [0, 1]
         val_frac_rem = np.clip(val_frac_rem, 0.0, 1.0)
         df_train, df_val = train_test_split(df_train_val, test_size=val_frac_rem, random_state=random_state, shuffle=True)
    elif len(df_train_val) > 0: # No val split needed or possible
        df_train = df_train_val
        df_val = pd.DataFrame(columns=df.columns) # Ensure empty DF has correct columns
    else: # Should not happen if n_samples >= 3, but safeguard
        df_train = pd.DataFrame(columns=df.columns)
        df_val = pd.DataFrame(columns=df.columns)

    logger.info(f"Split result: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    # Sanity check sum
    if len(df_train) + len(df_val) + len(df_test) != n_samples:
        logger.error("Consistency check failed: Sum of split sizes doesn't match original!")
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


# --- Feature Processing ---
def process_sample(args_tuple):
    """Processes a single sample: XML -> Preprocess -> Features -> Save Bin."""
    # Verwende direkt importierte Funktionen
    idx, row, base_path, bin_folder_rel, norm_args, feat_args = args_tuple
    xml_path_rel = row['xml_file_path']
    xml_path_abs = os.path.join(base_path, xml_path_rel)
    transcript = row['transcript']
    bin_path_rel = xml_path_rel.replace(
        f"{config.DATA_FOLDER_NAME}/lineStrokes", f"{config.DATA_FOLDER_NAME}/{bin_folder_rel}"
    ).replace('.xml', '.bin').replace("\\", "/")
    bin_path_abs = os.path.join(base_path, bin_path_rel)

    try:
        # Verwende die direkt importierte Funktion
        create_directory(os.path.dirname(bin_path_abs))
        strokes_list = extract_strokes_from_xml(xml_path_abs)
        if not strokes_list: return None
        strokes_np = np.array(strokes_list, dtype=float)
        if strokes_np.shape[0] < 2: logger.debug(f"Skipping {xml_path_rel}, < 2 points."); return None

        # Verwende direkt importierte Funktion
        ink = preprocess_handwriting(strokes_np, norm_args)
        if ink is None or ink.shape[0] < 2: logger.warning(f"Preprocessing failed for {xml_path_rel}"); return None

        # Verwende direkt importierte Funktion
        features = calculate_feature_vector_sequence(ink, feat_args, delayed_strokes=None)
        if features is None or features.size == 0: logger.warning(f"Feature extraction failed for {xml_path_rel}"); return None

        features.astype(np.float32).tofile(bin_path_abs)
        return {'bin_file_path': bin_path_rel, 'transcript': transcript}
    except Exception as e:
        logger.error(f"Error processing sample {idx} ({xml_path_rel}): {e}", exc_info=False)
        return None

def process_and_save_features(df, dataset_name, bin_folder_rel, norm_args, feat_args, use_parallel=True):
    """Processes a dataframe partition (train/val/test) using parallel workers."""
    base_path = config.BASE_PATH
    bin_output_dir_abs = os.path.join(base_path, config.DATA_FOLDER_NAME, bin_folder_rel)
    # Verwende die direkt importierte Funktion
    create_directory(bin_output_dir_abs)
    logger.info(f"Starting processing for {dataset_name} data ({len(df)} samples) -> Output base: {bin_output_dir_abs}")
    if df.empty: return pd.DataFrame(columns=['bin_file_path', 'transcript'])

    processed_list = []; error_count = 0
    args_list = [(idx, row, base_path, bin_folder_rel, norm_args, feat_args) for idx, row in df.iterrows()]

    actual_use_parallel = use_parallel and len(df) > 10
    if actual_use_parallel:
        num_cpus = max(1, mp.cpu_count() - 1)
        logger.info(f"Using {num_cpus} CPU cores for parallel processing of {dataset_name}.")
        try:
            # Set start method if needed (esp. on macOS/Windows)
            # try: mp.set_start_method('fork', force=True) # Oder 'spawn'
            # except RuntimeError: pass # Ignore if already set
            with mp.Pool(processes=num_cpus) as pool:
                results = list(tqdm(pool.imap_unordered(process_sample, args_list), total=len(args_list), desc=f"Processing {dataset_name}"))
            processed_list = [r for r in results if r is not None]
            error_count = len(results) - len(processed_list)
        except Exception as pool_e:
            logger.error(f"Error during parallel processing: {pool_e}", exc_info=True)
            logger.warning("Falling back to sequential processing.")
            actual_use_parallel = False # Fallback

    if not actual_use_parallel: # Sequential or Fallback
        logger.info(f"Using sequential processing for {dataset_name}.")
        for args_tuple in tqdm(args_list, desc=f"Processing {dataset_name}"):
            result = process_sample(args_tuple)
            if result is not None: processed_list.append(result)
            else: error_count += 1

    logger.info(f"Finished processing {dataset_name}: Success={len(processed_list)}, Errors/Skipped={error_count}")
    final_df = pd.DataFrame(processed_list)
    if final_df.empty and not df.empty: logger.error(f"Processing for {dataset_name} resulted in empty DataFrame!")
    return final_df


# --- Main Execution ---
def run_data_preparation():
    """Main function to run the entire online data preparation pipeline."""
    # --- Configuration (Adaptable) ---
    DATA_FOLDER_NAME_LOCAL = os.path.basename(config.DATA_FOLDER) # 'data'
    BIN_FOLDER_REL = config.BIN_FEATURE_FOLDER_NAME # 'bin_features'
    OUTPUT_EXCEL_PREFIX = config.DATA_PREFIX # 'iam_prepared'

    # Define Preprocessing & Feature Args (MATCH CONFIG.FEATURE_DIM!)
    NORM_ARGS = ["slope", "origin", "slant", "height", "resample"] # Example order
    FEAT_ARGS = ["x_cor", "y_cor", "penup", "dir", "curv", "vic_aspect", "vic_curl", "vic_line", "vic_slope", "bitmap"]

    # Verify FEATURE_DIM (verwende importierte Hilfsfunktion)
    try:
        # Verwende die direkt importierte Funktion
        calculated_dim = _get_expected_num_features(FEAT_ARGS)
    except NameError: # Falls _get_expected_num_features nicht importiert wurde (sollte nicht passieren)
        logger.warning("Konnte _get_expected_num_features nicht direkt verwenden, berechne manuell.")
        calculated_dim = sum([...]) # Manuelle Berechnung
    if calculated_dim != config.FEATURE_DIM:
        logger.error(f"FATAL: Calculated feature dim ({calculated_dim} from FEAT_ARGS) != config.FEATURE_DIM ({config.FEATURE_DIM}). Update config or FEAT_ARGS.")
        sys.exit(1)
    logger.info(f"Using feature dimension: {config.FEATURE_DIM}")
    logger.info(f"Using preprocessing steps: {NORM_ARGS}")
    logger.info(f"Using feature extraction steps: {FEAT_ARGS}")

    SPLIT_RANDOM_STATE = config.RANDOM_SEED
    TEST_SIZE = config.TEST_SPLIT
    VAL_SIZE = config.VAL_SPLIT
    USE_PARALLEL = (config.NUM_WORKERS > 1)

    # --- Setup ---
    base_path = config.BASE_PATH
    data_folder_abs = config.DATA_FOLDER # Verwende absoluten Pfad aus config
    output_bin_folder_abs = config.BIN_FEATURE_PATH # Verwende absoluten Pfad aus config
    # Verwende die direkt importierte Funktion create_directory
    create_directory(data_folder_abs)
    create_directory(output_bin_folder_abs)

    # --- Pipeline ---
    logger.info("--- Starting Online Data Preparation Pipeline ---")
    logger.info(f"Project Base Path: {base_path}")
    logger.info(f"Data Folder: {data_folder_abs}")
    logger.info(f"Output Bin Feature Folder: {output_bin_folder_abs}")

    if not maybe_unpack_archives(data_folder_abs): sys.exit(1) # Stop if critical data missing

    df_initial = create_initial_dataframe(data_folder_abs)
    if df_initial.empty: logger.error("Failed to create initial DataFrame. Exiting."); sys.exit(1)

    logger.info("Verifying existence of referenced XML files...")
    xml_exists_mask = df_initial['xml_file_path'].apply(lambda p: os.path.exists(os.path.join(base_path, p)))
    df_filtered = df_initial[xml_exists_mask].copy()
    num_removed = len(df_initial) - len(df_filtered)
    if num_removed > 0: logger.warning(f"Removed {num_removed} entries due to missing XML files.")
    if df_filtered.empty: logger.error("No valid XML file paths found after filtering. Exiting."); sys.exit(1)
    logger.info(f"DataFrame size after XML check: {len(df_filtered)}")

    logger.info("Splitting data into Train, Validation, and Test sets...")
    df_train, df_val, df_test = split_data(df_filtered, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=SPLIT_RANDOM_STATE)

    logger.info("Processing datasets (Preprocessing, Feature Extraction, Saving bin files)...")
    df_train_processed = process_and_save_features(df_train, "Train", BIN_FOLDER_REL, NORM_ARGS, FEAT_ARGS, use_parallel=USE_PARALLEL)
    df_val_processed = process_and_save_features(df_val, "Validation", BIN_FOLDER_REL, NORM_ARGS, FEAT_ARGS, use_parallel=USE_PARALLEL)
    df_test_processed = process_and_save_features(df_test, "Test", BIN_FOLDER_REL, NORM_ARGS, FEAT_ARGS, use_parallel=USE_PARALLEL)

    logger.info("Saving processed data indices to Excel manifest files...")
    # Verwende Pfade aus config
    train_excel_path = config.TRAIN_MANIFEST_FILE
    val_excel_path = config.VAL_MANIFEST_FILE
    test_excel_path = config.TEST_MANIFEST_FILE
    try:
        if not df_train_processed.empty: df_train_processed.to_excel(train_excel_path, index=False); logger.info(f"Train manifest saved: {train_excel_path}")
        else: logger.warning("Empty train processed df, not saving manifest.")
        if not df_val_processed.empty: df_val_processed.to_excel(val_excel_path, index=False); logger.info(f"Validation manifest saved: {val_excel_path}")
        else: logger.warning("Empty validation processed df, not saving manifest.")
        if not df_test_processed.empty: df_test_processed.to_excel(test_excel_path, index=False); logger.info(f"Test manifest saved: {test_excel_path}")
        else: logger.warning("Empty test processed df, not saving manifest.")
    except Exception as e: logger.error(f"Failed to save manifest Excel files: {e}", exc_info=True)

    logger.info("--- Online Data Preparation Pipeline Finished ---")

if __name__ == "__main__":

    run_data_preparation()
