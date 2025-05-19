import sys
import os
import argparse
import logging


from datetime import datetime

# --- Pfad Setup ---
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Stelle sicher, dass der Root zum Pfad hinzugefügt wird
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Setze Arbeitsverzeichnis (wichtig für relative Pfade in config etc.)
    os.chdir(project_root)
    print(f"DEBUG: Project Root (main.py): {project_root}")
    print(f"DEBUG: Current Working Dir (main.py): {os.getcwd()}")
except Exception as path_e:
     print(f"[FEHLER] Kritischer Fehler beim Pfad-Setup in main.py: {path_e}")
     sys.exit(1)

# --- Modulimporte (NACH Pfad Setup) ---
try:
    import config # Lädt config.py aus dem Root-Verzeichnis
    # Importiere Module aus dem src_CRNN Paket
    import utils as online_utils
    import data_preparation as online_data_preparation
    import training_online as online_training
    import evaluation_online as online_evaluation
    # model_online_crnn wird von training/evaluation importiert
except ImportError as e:
     # Wenn config schon geladen hat, versuche zu loggen
     try:
         logger_fallback = logging.getLogger(__name__)
         logger_fallback.critical(f"Kritischer Importfehler in main.py: {e}. Prüfe Pfade und Modulnamen.", exc_info=True)
     except Exception:
          print(f"[FEHLER] Kritischer Importfehler in main.py: {e}")
          print(f"Python-Pfad: {sys.path}")
     sys.exit(1)
except Exception as config_e:
     # Fehler beim Laden von config selbst
     print(f"[FEHLER] Kritischer Fehler beim Laden der Konfiguration (config.py): {config_e}")
     sys.exit(1)

# Logger holen (sollte jetzt von config initialisiert sein)
logger = logging.getLogger(__name__)

# =============================================================================
# main.py - Orchestrierung der ONLINE Zeilen-OCR Pipeline (CRNN+CTC)
# =============================================================================
def main():
    # (Restlicher Code von main.py bleibt identisch wie im vorherigen Snippet)
    pipeline_start_time = datetime.now()
    logger.info("==============================================================")
    logger.info("=== STARTE ONLINE HANDSCHRIFT-OCR PIPELINE (CRNN+CTC) ===")
    logger.info(f"=== Startzeit: {pipeline_start_time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info("==============================================================")

    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Orchestriert die Online CRNN+CTC OCR Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- Hauptaktionen / Pipeline-Schritte ---
    parser.add_argument('--prepare_online_data', action='store_true',
                        help="[Schritt 1] Führt die komplette Online-Datenvorbereitung durch.")
    parser.add_argument('--train_online', action='store_true',
                        help="[Schritt 2] Trainiert das Online-CRNN+CTC-Modell.")
    parser.add_argument('--evaluate_online', action='store_true',
                        help="[Schritt 3] Evaluiert das beste trainierte Online-Modell.")
    # --- Optionale Überschreibungen ---
    parser.add_argument('--lr', type=float, default=None, help=f"Überschreibt config.LEARNING_RATE ({config.LEARNING_RATE}).")
    parser.add_argument('--batch_size', type=int, default=None, help=f"Überschreibt config.BATCH_SIZE ({config.BATCH_SIZE}).")
    parser.add_argument('--epochs', type=int, default=None, help=f"Überschreibt config.EPOCHS ({config.EPOCHS}).")
    parser.add_argument('--feature_dim', type=int, default=None, help=f"Überschreibt config.FEATURE_DIM ({config.FEATURE_DIM}).")
    parser.add_argument('--max_seq_len', type=int, default=None, help=f"Überschreibt config.MAX_SEQ_LEN ({config.MAX_SEQ_LEN}).")
    parser.add_argument('--early_stopping', type=int, default=None, help=f"Überschreibt config.EARLY_STOPPING_PATIENCE ({config.EARLY_STOPPING_PATIENCE}).")
    parser.add_argument('--max_label_len', type=int, default=None, help=f"Überschreibt config.MAX_LABEL_LENGTH ({config.MAX_LABEL_LENGTH}).")
    parser.add_argument('--output_dir', type=str, default=None, help="Überschreibt das Basis-Ausgabeverzeichnis.")
    parser.add_argument('--data_folder', type=str, default=None, help="Überschreibt den Pfad zum Datenordner.")
    parser.add_argument('--manifest_prefix', type=str, default=None, help="Überschreibt das Prefix der Manifest-Dateien.")
    # --- Spezifische Argumente ---
    parser.add_argument('--load_model_for_eval', type=str, default=None,
                        help="Pfad zu einer spezifischen '.pth' Modelldatei für '--evaluate_online'.")
    args = parser.parse_args()

    logger.info("--- Kommandozeilenargumente ---")
    for arg, value in sorted(vars(args).items()): logger.info(f"  --{arg}: {value}")
    logger.info("-------------------------------")

    # --- Konfiguration überschreiben ---
    config_updated_messages = []
    if args.lr is not None and args.lr != config.LEARNING_RATE: config.LEARNING_RATE = args.lr; config_updated_messages.append(f"LEARNING_RATE -> {args.lr}")
    if args.batch_size is not None and args.batch_size != config.BATCH_SIZE: config.BATCH_SIZE = args.batch_size; config_updated_messages.append(f"BATCH_SIZE -> {args.batch_size}")
    if args.epochs is not None and args.epochs != config.EPOCHS: config.EPOCHS = args.epochs; config_updated_messages.append(f"EPOCHS -> {args.epochs}")
    if args.feature_dim is not None and args.feature_dim != config.FEATURE_DIM: config.FEATURE_DIM = args.feature_dim; config.CNN_INPUT_HEIGHT = args.feature_dim; config_updated_messages.append(f"FEATURE_DIM / CNN_INPUT_HEIGHT -> {args.feature_dim} (VORSICHT!)")
    if args.max_seq_len is not None and args.max_seq_len != config.MAX_SEQ_LEN: config.MAX_SEQ_LEN = args.max_seq_len; config.CNN_INPUT_WIDTH = args.max_seq_len; config_updated_messages.append(f"MAX_SEQ_LEN / CNN_INPUT_WIDTH -> {args.max_seq_len} (VORSICHT!)")
    if args.early_stopping is not None and args.early_stopping != config.EARLY_STOPPING_PATIENCE: config.EARLY_STOPPING_PATIENCE = args.early_stopping; config_updated_messages.append(f"EARLY_STOPPING_PATIENCE -> {args.early_stopping}")
    if args.max_label_len is not None and args.max_label_len != config.MAX_LABEL_LENGTH: config.MAX_LABEL_LENGTH = args.max_label_len; config_updated_messages.append(f"MAX_LABEL_LENGTH -> {args.max_label_len}")
    if args.data_folder is not None:
        config.DATA_FOLDER = os.path.join(config.BASE_PATH, args.data_folder) # Update absolute path
        config.BIN_FEATURE_PATH = os.path.join(config.DATA_FOLDER, config.BIN_FEATURE_FOLDER_NAME)
        prefix = args.manifest_prefix if args.manifest_prefix is not None else config.DATA_PREFIX
        config.CURRENT_TRAIN_MANIFEST = os.path.join(config.DATA_FOLDER, f"{prefix}_train.xlsx")
        config.CURRENT_VAL_MANIFEST = os.path.join(config.DATA_FOLDER, f"{prefix}_val.xlsx")
        config.CURRENT_TEST_MANIFEST = os.path.join(config.DATA_FOLDER, f"{prefix}_test.xlsx")
        config_updated_messages.append(f"DATA_FOLDER -> {config.DATA_FOLDER}")
    if args.manifest_prefix is not None and args.manifest_prefix != config.DATA_PREFIX:
         config.DATA_PREFIX = args.manifest_prefix
         if args.data_folder is None:
            config.CURRENT_TRAIN_MANIFEST = os.path.join(config.DATA_FOLDER, f"{config.DATA_PREFIX}_train.xlsx")
            config.CURRENT_VAL_MANIFEST = os.path.join(config.DATA_FOLDER, f"{config.DATA_PREFIX}_val.xlsx")
            config.CURRENT_TEST_MANIFEST = os.path.join(config.DATA_FOLDER, f"{config.DATA_PREFIX}_test.xlsx")
         config_updated_messages.append(f"DATA_PREFIX -> {config.DATA_PREFIX}")
    if args.output_dir is not None:
         custom_base_output_path = os.path.join(config.BASE_PATH, args.output_dir)
         model_version_dir_name = os.path.basename(config.RUN_OUTPUT_PATH)
         config.RUN_OUTPUT_PATH = os.path.join(custom_base_output_path, model_version_dir_name)
         config.MODEL_SAVE_PATH = os.path.join(config.RUN_OUTPUT_PATH, "models")
         config.LOGS_PATH = os.path.join(config.RUN_OUTPUT_PATH, "logs") # This won't redirect logs already started
         config.CHECKPOINT_PATH = os.path.join(config.RUN_OUTPUT_PATH, "checkpoints")
         config.RESULTS_PATH = os.path.join(config.RUN_OUTPUT_PATH, "results")
         config.METRICS_PATH = os.path.join(config.RUN_OUTPUT_PATH, "metrics")
         config.PATH_CONFIG = { "run_output": config.RUN_OUTPUT_PATH, "model_save": config.MODEL_SAVE_PATH,
                                "checkpoints": config.CHECKPOINT_PATH, "results": config.RESULTS_PATH, "metrics": config.METRICS_PATH }
         for path in config.PATH_CONFIG.values(): online_utils.create_directory(path)
         config_updated_messages.append(f"Ausgabeordner Basis auf {custom_base_output_path} gesetzt -> Aktueller Lauf: {config.RUN_OUTPUT_PATH}")

    if config_updated_messages:
         logger.info("--- Konfigurations-Überschreibungen ---")
         for msg in config_updated_messages: logger.info(f"  - {msg}")
         logger.info("------------------------------------")
    else:
        logger.info("Keine Konfigurationswerte durch Kommandozeilenargumente überschrieben.")

    # --- Modell für Evaluation bestimmen ---
    evaluation_model_path_to_use = None
    best_model_in_checkpoint = os.path.join(config.CHECKPOINT_PATH, "best_online_crnn_model.pth")
    if args.load_model_for_eval:
        specified_model_path = args.load_model_for_eval
        if not os.path.isabs(specified_model_path): specified_model_path = os.path.join(config.BASE_PATH, specified_model_path)
        if os.path.exists(specified_model_path) and specified_model_path.endswith(".pth"):
            evaluation_model_path_to_use = specified_model_path; logger.info(f"Spezifisches Modell für Eval: '{evaluation_model_path_to_use}'")
        else: logger.error(f"Spez. Modell '{specified_model_path}' für Eval nicht gefunden/gültig."); # (Abbruch/Weiter im Eval Schritt)
    else:
        if os.path.exists(best_model_in_checkpoint): evaluation_model_path_to_use = best_model_in_checkpoint; logger.info(f"Standard-Modell für Eval: '{evaluation_model_path_to_use}'")
        else: evaluation_model_path_to_use = None

    # --- Pipeline Schritte ausführen ---
    logger.info("\n" + "="*30 + " STARTE ONLINE PIPELINE SCHRITTE " + "="*30)
    executed_steps = []
    # Schritt 1: Daten vorbereiten
    if args.prepare_online_data:
        logger.info("[Schritt 1/3] Starte Online-Datenvorbereitung...")
        executed_steps.append("Prepare Online Data")
        try: 
            online_data_preparation.run_data_preparation()
            logger.info("[Schritt 1/3] Online-Datenvorbereitung abgeschlossen.")
        except SystemExit as sys_exit:
            if sys_exit.code != 0: logger.critical("[Schritt 1/3] Datenvorbereitung mit Fehler beendet. Abbruch."); return
        except Exception as step1_e: logger.exception("[Schritt 1/3] FEHLER bei Online-Datenvorbereitung!", exc_info=True); return
    # Schritt 2: Training
    if args.train_online:
        logger.info("[Schritt 2/3] Starte Online-Modelltraining (CRNN+CTC)...")
        executed_steps.append("Train Online Model")
        if not os.path.exists(config.CURRENT_TRAIN_MANIFEST): logger.error(f"Trainings-Manifest '{config.CURRENT_TRAIN_MANIFEST}' fehlt. Zuerst '--prepare_online_data' ausführen."); return
        try: online_training.train_online_model(); logger.info("[Schritt 2/3] Online-Training abgeschlossen.")
        except Exception as step2_e: logger.exception("[Schritt 2/3] FEHLER während Online-Training!", exc_info=True)
    # Schritt 3: Evaluation
    if args.evaluate_online:
        logger.info("[Schritt 3/3] Starte Evaluation auf allen Splits (Online CRNN+CTC)...")
        executed_steps.append("Evaluate Online Model")
        if not os.path.exists(config.CURRENT_TEST_MANIFEST): logger.warning(f"Test-Manifest '{config.CURRENT_TEST_MANIFEST}' fehlt. Eval auf Testset nicht möglich.")
        if evaluation_model_path_to_use:
            try: online_evaluation.evaluate_online_model(model_path_to_evaluate=evaluation_model_path_to_use); logger.info("[Schritt 3/3] Online-Evaluation abgeschlossen.")
            except Exception as step3_e: logger.exception("[Schritt 3/3] FEHLER während Online-Evaluation!", exc_info=True)
        else: logger.error("[Schritt 3/3] Evaluation übersprungen: Kein gültiges Modell gefunden.")

    # --- Abschluss ---
    logger.info("\n" + "="*30 + " ONLINE PIPELINE BEENDET " + "="*30)
    if executed_steps: logger.info(f"Ausgeführte Schritte: {', '.join(executed_steps)}")
    else: logger.info("Keine Pipeline-Schritte ausgewählt."); logger.info("Nutze --prepare_online_data, --train_online oder --evaluate_online.")
    pipeline_end_time = datetime.now(); total_duration = pipeline_end_time - pipeline_start_time
    logger.info(f"Gesamte Orchestrierungsdauer: {str(total_duration).split('.')[0]}")
    logger.info(f"Ausgabeordner für diesen Lauf: {config.RUN_OUTPUT_PATH}")
    logger.info("==============================================================")

# --- Main Guard ---
if __name__ == "__main__":
    try: main(); sys.exit(0)
    except SystemExit as se:
         if se.code != 0: logger.critical(f"Pipeline wurde mit Fehlercode {se.code} beendet.")
         else: logger.info("Pipeline normal beendet.")
    except Exception as global_e: logger.critical("Unerwarteter Fehler auf oberster Ebene!", exc_info=True); sys.exit(1)
