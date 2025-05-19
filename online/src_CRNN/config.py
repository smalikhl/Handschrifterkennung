# --- START OF FILE config.py ---

import os
import random
import numpy as np
import torch
from datetime import datetime
import argparse
import logging
import time

# Minimales Argument Parsing, um Konflikte zu vermeiden
parser = argparse.ArgumentParser(description="Konfiguration der ONLINE-OCR-Pipeline (CRNN+CTC)")
_, _ = parser.parse_known_args()

# ---- Basis Pfad ---
# Annahme: config.py liegt außerhalb von src_CRNN, im Projekt-Root
try:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    # Überprüfe, ob wir im richtigen Verzeichnis sind
    if not os.path.exists(os.path.join(BASE_PATH, 'src_CRNN')) and os.path.basename(BASE_PATH) == 'src_CRNN':
         BASE_PATH = os.path.dirname(BASE_PATH) # Gehe eine Ebene hoch
except NameError: # Wenn __file__ nicht definiert ist (selten)
     BASE_PATH = os.getcwd()
     print(f"WARNUNG: __file__ nicht definiert, verwende CWD als BASE_PATH: {BASE_PATH}")


# ---- Hauptverzeichnis der Datensätze ---
# Ordner, der die IAM-OnDB-Daten (entpackte Archive) enthält (ascii, lineStrokes etc.)
# Beispiel: "data" relativ zum BASE_PATH
DATA_FOLDER_NAME = "data" # Name des Datenordners
DATA_FOLDER = os.path.join(BASE_PATH, DATA_FOLDER_NAME)
# Ordner, in dem die extrahierten binären Merkmalsdateien gespeichert werden
BIN_FEATURE_FOLDER_NAME = "bin_features" # Name des Unterordners für Features
BIN_FEATURE_PATH = os.path.join(DATA_FOLDER, BIN_FEATURE_FOLDER_NAME)

# ---- DATENQUELLE für Online-Daten (Manifeste) ----
# Pfad zu den Excel-Dateien, die relative Pfade zu den .bin-Dateien (relativ zu BASE_PATH)
# und Transkriptionen enthalten. Diese werden von data_preparation.py generiert.
# Beispiel: "data/iam_prepared_train.xlsx"
DATA_PREFIX = "iam_prepared" # Wird von data_preparation.py verwendet
TRAIN_MANIFEST_FILE = os.path.join(DATA_FOLDER, f"{DATA_PREFIX}_train.xlsx")
VAL_MANIFEST_FILE   = os.path.join(DATA_FOLDER, f"{DATA_PREFIX}_val.xlsx")
TEST_MANIFEST_FILE  = os.path.join(DATA_FOLDER, f"{DATA_PREFIX}_test.xlsx")

# Interne Variablen (zeigen auf die aktuell zu nutzenden Manifest-Dateien)
# Diese werden im Training/Evaluation Skript verwendet
CURRENT_TRAIN_MANIFEST = TRAIN_MANIFEST_FILE
CURRENT_VAL_MANIFEST = VAL_MANIFEST_FILE
CURRENT_TEST_MANIFEST = TEST_MANIFEST_FILE

# ---- Speicherorte für Modelle, Logs, Ergebnisse etc. ----
MODEL_TYPE_TAG = "online_crnn_ctc" # Klarheit für Online-CRNN-CTC Training
MODEL_VERSION = f"{MODEL_TYPE_TAG}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
BASE_OUTPUT_PATH = os.path.join(BASE_PATH, "outputs_online") # Eigener Hauptordner für Online
RUN_OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, MODEL_VERSION) # Eigener Ordner pro Lauf

MODEL_SAVE_PATH = os.path.join(RUN_OUTPUT_PATH, "models")
LOGS_PATH = os.path.join(RUN_OUTPUT_PATH, "logs")
CHECKPOINT_PATH = os.path.join(RUN_OUTPUT_PATH, "checkpoints")
RESULTS_PATH = os.path.join(RUN_OUTPUT_PATH, "results")
METRICS_PATH = os.path.join(RUN_OUTPUT_PATH, "metrics")

# ---- Daten-Splits (Verhältnisse für Neugenerierung durch data_preparation.py) ----
# Diese werden verwendet, wenn data_preparation.py neue Splits erstellt
TEST_SPLIT = 0.10  # 10% für Test
VAL_SPLIT = 0.10   # 10% für Validierung (Rest ist Training: 80%)

# ---- ONLINE Feature & Sequenz Dimensionen ----
# Dimension der extrahierten Merkmale pro Zeitstempel (Resultat von feature_extraction.py)
# Muss mit FEAT_ARGS in data_preparation.py übereinstimmen!
FEATURE_DIM = 20            # ANPASSEN, falls sich FEAT_ARGS ändern!
# Maximale Sequenzlänge (Anzahl Zeitstempel) nach Resampling/Verarbeitung.
# Dies wird die "Breite" des Pseudo-Bildes für die CNN.
MAX_SEQ_LEN = 1000           # ANPASSEN! Basierend auf Analyse der Datenlängen nach Resampling.
                            # Beeinflusst Speicher und Rechenaufwand stark.
# Die "Höhe" des Pseudo-Bildes für die CNN entspricht FEATURE_DIM.
# Die CNN erwartet Channels=1.
CNN_INPUT_HEIGHT = FEATURE_DIM # Die Höhe für die CNN entspricht der Feature-Dimension
CNN_INPUT_WIDTH = MAX_SEQ_LEN  # Die Breite für die CNN entspricht der max. Sequenzlänge
CNN_INPUT_CHANNELS = 1         # Wir behandeln die Feature-Sequenz als 1-Kanal-Bild

# ---- Trainingskonfiguration (weitgehend wie beim Offline-Modell) ----
LEARNING_RATE = 0.0001       # Lernrate
BATCH_SIZE    = 32           # Anpassen an GPU-Speicher (Online-Daten können lang sein)
EPOCHS        = 100          # Maximale Anzahl Epochen
OPTIMIZER     = 'adamw'      # 'adam' | 'adamw' | 'sgd'
WEIGHT_DECAY  = 1e-5         # Gewichtungszerfall
USE_MIXED_PRECISION = True   # Aktivieren für schnelleres Training (wenn GPU unterstützt)
SCHEDULER     = 'ReduceLROnPlateau' # 'ReduceLROnPlateau', 'CosineAnnealingLR', 'none'
SCHEDULER_PATIENCE = 7       # Geduld für ReduceLROnPlateau
SCHEDULER_METRIC = 'loss'    # Metrik für Scheduler (ohne 'val_') - loss, cer, wer, char_f1
EARLY_STOPPING_PATIENCE = 15 # Anzahl Epochen ohne Verbesserung
EARLY_STOPPING_METRIC = 'val_loss' # Metrik für Early Stopping (mit 'val_')
SAVE_CHECKPOINT_INTERVAL = 5 # Speichere History/Plots alle 5 Epochen

# ---- Zeichensatz (Anpassen an IAM-OnDB oder Ziel-Datensatz!) ----
# !!! WICHTIG: Überprüfen für IAM-OnDB! Enthält evtl. andere Sonderzeichen. !!!
#            Dieser Satz stammt aus dem Offline-Beispiel und MUSS angepasst werden.
# Beispiel (unvollständig, bitte prüfen!):
CHAR_LIST = list(" !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz[]_{|}~`^%")
# Füge spezielle Tokens hinzu
CHAR_LIST = CHAR_LIST + ['<UNK>'] # Optional: Unbekanntes Zeichen
# Sortieren ist nicht nötig, aber konsistent
CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHAR_LIST)}
# CTC Blank Token MUSS den HÖCHSTEN Index haben und darf NICHT im CHAR_LIST selbst sein
BLANK_TOKEN = '<CTC_BLANK>'
BLANK_IDX = len(CHAR_LIST) # Index nach dem letzten echten Zeichen
CHAR_TO_IDX[BLANK_TOKEN] = BLANK_IDX
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHAR_TO_IDX) # Anzahl der echten Zeichen + Blank Token

# ---- Label-Länge (Filter) ----
# Wird in data_preparation.py kaum noch benötigt, da Labels meist kürzer sind.
# Aber zur Sicherheit ein hoher Wert.
MAX_LABEL_LENGTH = 200

# ---- Sonstiges ----
LOGGING_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = min(4, os.cpu_count() if os.cpu_count() else 1) # Für DataLoader

# Zufallseeds setzen
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    # Ggf. für volle Reproduzierbarkeit (Performance-Kosten)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# --- Ordnerstruktur sicherstellen ---
PATH_CONFIG = {
    "run_output":        RUN_OUTPUT_PATH,
    "model_save":        MODEL_SAVE_PATH,
    "checkpoints":       CHECKPOINT_PATH,
    "results":           RESULTS_PATH,
    "metrics":           METRICS_PATH
}
for path in PATH_CONFIG.values():
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"[WARNUNG] Fehler beim Erstellen des Verzeichnisses {path}: {e}")

# --- Logging Konfiguration ---
log_file_path = os.path.join(RUN_OUTPUT_PATH, "run.log")
log_root = logging.getLogger()
# Alte Handler entfernen (wichtig bei mehrfachem Import)
for handler in log_root.handlers[:]:
    log_root.removeHandler(handler)
    handler.close()
logging.basicConfig(
    level=getattr(logging, LOGGING_LEVEL.upper(), logging.INFO),
    format='%(asctime)s [%(levelname)-7s] %(name)-25s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*50)
logger.info(f"Konfiguration für ONLINE CRNN+CTC geladen.")
logger.info(f"Basis-Pfad: {BASE_PATH}") # Logge den erkannten Basispfad
logger.info(f"Ausgabeordner: {RUN_OUTPUT_PATH}")
logger.info(f"Log-Datei: {log_file_path}")
logger.info(f"Trainiere auf Gerät: {DEVICE}")
logger.info(f"Datenquelle (Manifeste): {DATA_FOLDER}/{DATA_PREFIX}_*.xlsx")
logger.info(f"Feature-Ordner (.bin): {BIN_FEATURE_PATH}")
logger.info(f"CNN Input: H={CNN_INPUT_HEIGHT} (FeatureDim), W={CNN_INPUT_WIDTH} (MaxSeqLen), C={CNN_INPUT_CHANNELS}")
logger.info(f"Maximale Label-Länge (theoretisch): {MAX_LABEL_LENGTH}")
logger.info(f"Zeichensatzgröße: {len(CHAR_LIST)}, Anzahl Klassen (inkl. Blank): {NUM_CLASSES}")
logger.info(f"Scheduler Metrik: {SCHEDULER_METRIC}, EarlyStopping Metrik: {EARLY_STOPPING_METRIC}")
logger.info(f"Batch Size: {BATCH_SIZE}, Max Epochs: {EPOCHS}")
logger.info("="*50)
