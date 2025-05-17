
import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import sys
import time

# Lokale Imports (angepasst)
import config # Lädt ONLINE config
# Direkte relative Imports für sibling Module
try:
    import utils as online_utils
    import model_online_crnn
    from training_online import OnlineHandwritingDataset, online_collate_fn # Brauchen Dataset/Collate aus Training
except ImportError as e:
    logging.basicConfig(level=logging.ERROR)
    logger_fallback = logging.getLogger(__name__)
    logger_fallback.error(f"Kritischer Fehler beim relativen Import in evaluation_online: {e}. Paketstruktur/ PYTHONPATH prüfen!", exc_info=True)
    sys.exit(1)


logger = logging.getLogger(__name__)

# --- Plotting & Reporting (weitgehend unverändert) ---
# (Funktionen save_sample_predictions_text, plot_error_distribution, plot_length_vs_metric bleiben identisch wie im vorherigen Snippet)
def save_sample_predictions_text(relative_sample_paths, true_labels, pred_labels, split_name, num_samples=30):
    """Speichert eine Textdatei mit Beispielvorhersagen."""
    online_utils.create_directory(config.RESULTS_PATH) # Util Funktion korrekt aufrufen
    txt_path = os.path.join(config.RESULTS_PATH, f"{split_name}_sample_predictions_final.txt")
    if len(relative_sample_paths) > num_samples: indices = np.random.choice(len(relative_sample_paths), num_samples, replace=False)
    else: indices = range(len(relative_sample_paths)); num_samples = len(relative_sample_paths)
    if num_samples == 0: logger.warning(f"Keine Samples für Text-Report ({split_name})."); return

    content = f"Sample Predictions ({split_name.capitalize()} Split) - {num_samples} Samples\n"
    content += "="*50 + "\n"
    for i in indices:
        rel_path = relative_sample_paths[i]
        true_label = true_labels[i] if true_labels[i] else "[LEER]"
        pred_label = pred_labels[i] if pred_labels[i] else "[LEER]"
        is_correct = true_label == pred_label; match_indicator = "✅" if is_correct else "❌"
        content += f"Sample (Rel. Pfad: ...{rel_path[-50:]}) {match_indicator}\n"
        content += f"  GT  : '{true_label}'\n"
        content += f"  PRED: '{pred_label}'\n"
        content += "-"*20 + "\n"
    try:
        with open(txt_path, 'w', encoding='utf-8') as f: f.write(content)
        logger.info(f"Beispielvorhersagen (Text) gespeichert: {txt_path}")
    except Exception as e: logger.error(f"Fehler beim Speichern der Text-Datei {txt_path}: {e}")

def plot_error_distribution(all_true_strs, all_pred_strs, split_name):
    """Plottet Histogramme und Boxplots der CER und WER."""
    if not all_true_strs or not all_pred_strs: logger.warning(f"Leere Listen für Fehlerverteilungsplot ({split_name})."); return
    cer_values = [online_utils.compute_cer(p, t) for p, t in zip(all_pred_strs, all_true_strs)]
    wer_values = [online_utils.compute_wer(p, t) for p, t in zip(all_pred_strs, all_true_strs)]
    online_utils.create_directory(config.RESULTS_PATH) # Util Funktion korrekt aufrufen
    try:
        # Histogramme
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)); sns.histplot(cer_values, bins=30, kde=True, color='skyblue', ax=axes[0])
        axes[0].set_title(f'CER Verteilung ({split_name.capitalize()})'); axes[0].set_xlabel('CER'); axes[0].set_ylabel('Häufigkeit'); axes[0].grid(axis='y', alpha=0.7); axes[0].set_xlim(left=0)
        sns.histplot(wer_values, bins=30, kde=True, color='salmon', ax=axes[1])
        axes[1].set_title(f'WER Verteilung ({split_name.capitalize()})'); axes[1].set_xlabel('WER'); axes[1].set_ylabel('Häufigkeit'); axes[1].grid(axis='y', alpha=0.7); axes[1].set_xlim(left=0)
        hist_path = os.path.join(config.RESULTS_PATH, f"{split_name}_error_distribution_hist.png"); plt.tight_layout(); plt.savefig(hist_path); plt.close(fig)
        logger.info(f"Fehlerverteilungs-Histogramme ({split_name}) gespeichert: {hist_path}")
        # Boxplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5)); sns.boxplot(y=cer_values, color='skyblue', ax=axes[0]); axes[0].set_title(f'CER Boxplot ({split_name.capitalize()})'); axes[0].set_ylabel('CER'); axes[0].grid(axis='y', alpha=0.7); axes[0].set_ylim(bottom=0)
        sns.boxplot(y=wer_values, color='salmon', ax=axes[1]); axes[1].set_title(f'WER Boxplot ({split_name.capitalize()})'); axes[1].set_ylabel('WER'); axes[1].grid(axis='y', alpha=0.7); axes[1].set_ylim(bottom=0)
        boxplot_path = os.path.join(config.RESULTS_PATH, f"{split_name}_error_boxplot.png"); plt.tight_layout(); plt.savefig(boxplot_path); plt.close(fig)
        logger.info(f"Fehlerverteilungs-Boxplots ({split_name}) gespeichert: {boxplot_path}")
    except Exception as e: logger.error(f"Fehler beim Erstellen der Fehlerverteilungsplots für {split_name}: {e}", exc_info=True)
    finally: plt.close('all')

def plot_length_vs_metric(all_true_strs, all_pred_strs, metric_name, split_name):
    """Analysiert Metrik vs. Länge der wahren Sequenz (Zeichen)."""
    if not all_true_strs or not all_pred_strs: return
    lengths = [len(t) for t in all_true_strs]; metric_values = []
    if metric_name.lower() == 'cer': metric_values = [online_utils.compute_cer(p, t) for p, t in zip(all_pred_strs, all_true_strs)]
    elif metric_name.lower() == 'wer': metric_values = [online_utils.compute_wer(p, t) for p, t in zip(all_pred_strs, all_true_strs)]
    elif metric_name.lower() == 'sentence_accuracy': metric_values = [1.0 if p == t else 0.0 for p, t in zip(all_pred_strs, all_true_strs)]
    else: logger.warning(f"Metrik '{metric_name}' für Längenanalyse nicht unterstützt."); return
    if not metric_values: return
    df = pd.DataFrame({'length': lengths, 'metric': metric_values})
    grouped = df.groupby('length')['metric'].agg(['mean', 'count']).reset_index(); min_samples = 5
    grouped_filtered = grouped[grouped['count'] >= min_samples]
    if grouped_filtered.empty: logger.warning(f"Nicht genug Datenpunkte (min {min_samples}) für Längenanalyse ({metric_name}, {split_name})."); return
    try:
        plt.figure(figsize=(14, 6)); sns.barplot(x='length', y='mean', data=grouped_filtered, palette='viridis')
        plot_title = metric_name.replace('_', ' ').upper(); plt.title(f'{plot_title} nach Sequenzlänge ({split_name.capitalize()}) (min {min_samples} Samples/Länge)')
        plt.xlabel('Länge des wahren Labels (Zeichen)'); plt.ylabel(f'Durchschnittliche {plot_title}'); tick_spacing = max(1, len(grouped_filtered['length']) // 20)
        plt.xticks(ticks=np.arange(0, len(grouped_filtered['length']), tick_spacing), labels=grouped_filtered['length'].iloc[::tick_spacing], rotation=45, ha='right'); plt.grid(axis='y', alpha=0.7); plt.tight_layout()
        plot_path = os.path.join(config.RESULTS_PATH, f"{split_name}_length_vs_{metric_name.lower()}.png"); plt.savefig(plot_path); plt.close()
        logger.info(f"Plot Länge vs. {plot_title} ({split_name}) gespeichert: {plot_path}")
    except Exception as e: logger.error(f"Fehler beim Plotten Länge vs. {metric_name.upper()} für {split_name}: {e}", exc_info=True)
    finally: plt.close('all')

# --- Haupt-Evaluationsfunktion ---
def evaluate_model_on_split(model, data_loader, criterion, device, split_name):
    """Führt Evaluation auf einem Split durch (Online CRNN+CTC)."""
    # (Keine Änderung am inneren Code dieser Funktion nötig)
    if data_loader is None or len(data_loader.dataset) == 0:
        logger.warning(f"DataLoader für Split '{split_name}' leer. Eval übersprungen.")
        return ({'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'sentence_accuracy': 0.0,
                 'bleu': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'char_f1': 0.0}, [], [], [])
    model.eval(); all_true_strings = []; all_pred_strings = []; all_relative_paths_collected = []
    running_loss = 0.0; sample_count = 0; processed_batches = 0

    logger.info(f"Starte detaillierte Evaluation auf '{split_name}' Split...")
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Eval {split_name}", leave=False, unit="batch")
        for batch_idx, batch_data in enumerate(progress_bar):
            if not isinstance(batch_data, tuple) or len(batch_data) != 4: logger.error(f"Unerwartetes Format vom DataLoader Batch {batch_idx}. Skip."); continue
            batch_images, batch_labels_flat, batch_label_lengths, batch_rel_paths_from_collate = batch_data
            if batch_images.nelement() == 0: logger.warning(f"Leerer Feature-Tensor Batch {batch_idx}. Skip."); continue
            batch_size = batch_images.size(0)
            all_relative_paths_collected.extend(batch_rel_paths_from_collate) # Sammle Pfade aus Collate

            batch_images = batch_images.to(device); batch_labels_flat_dev = batch_labels_flat.to(device)
            batch_label_lengths_dev = batch_label_lengths.to(device)

            try:
                logits = model(batch_images) # (SeqLen, Batch, Classes)
                log_probs = F.log_softmax(logits, dim=2)
                T_out, N_out, _ = log_probs.size()
                if N_out != batch_size: logger.error(f"Batch Size Mismatch Eval: In={batch_size}, Out={N_out}. Skip."); continue
                input_lengths = torch.full(size=(N_out,), fill_value=T_out, dtype=torch.long, device=device)
                loss = criterion(log_probs, batch_labels_flat_dev, input_lengths, batch_label_lengths_dev)
                if not torch.isnan(loss) and not torch.isinf(loss): running_loss += loss.item() * batch_size
                else: logger.warning(f"Ungültiger Loss ({loss.item()}) Eval Batch {batch_idx}.")
                pred_strs = online_utils.decode_ctc_output(logits, config.IDX_TO_CHAR)
                all_pred_strings.extend(pred_strs)
                current_idx = 0
                for i in range(batch_size):
                    length = batch_label_lengths[i].item()
                    end_idx = current_idx + length
                    label_indices = batch_labels_flat[current_idx:end_idx].tolist()
                    true_str = online_utils.decode_label_list(label_indices, config.IDX_TO_CHAR)
                    all_true_strings.append(true_str)
                    current_idx = end_idx
                sample_count += batch_size; processed_batches += 1
            except Exception as eval_err:
                 logger.error(f"Fehler bei Evaluation (Split {split_name}, Batch {batch_idx}): {eval_err}", exc_info=True)
                 all_pred_strings.extend(["EVAL_ERROR"] * batch_size)
                 all_true_strings.extend(["EVAL_ERROR"] * batch_size)
                 sample_count += batch_size
                 continue
            current_avg_loss = running_loss / sample_count if sample_count > 0 else 0
            progress_bar.set_postfix({'Loss': f"{current_avg_loss:.4f}"})

    if processed_batches == 0 or sample_count == 0:
         logger.error(f"Keine Batches erfolgreich für Split '{split_name}' verarbeitet.")
         return ({'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'sentence_accuracy': 0.0, 'bleu': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'char_f1': 0.0}, [], [], [])

    num_evaluated_samples = min(len(all_true_strings), len(all_pred_strings), len(all_relative_paths_collected))
    if num_evaluated_samples < sample_count:
        logger.warning(f"Inkonsistente Längen nach Eval '{split_name}': Samples={sample_count}, Results={num_evaluated_samples}. Kürze Listen.")
        all_true_strings = all_true_strings[:num_evaluated_samples]
        all_pred_strings = all_pred_strings[:num_evaluated_samples]
        final_paths = all_relative_paths_collected[:num_evaluated_samples]
    else:
         final_paths = all_relative_paths_collected


    if num_evaluated_samples == 0: # If filtering removed everything
         logger.error("Keine konsistenten Samples nach Validation übrig.")
         return ({'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'sentence_accuracy': 0.0, 'bleu': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'char_f1': 0.0}, [], [], [])


    logger.info(f"Berechne finale Metriken für {num_evaluated_samples} Samples auf Split '{split_name}'.")
    avg_loss = running_loss / sample_count if sample_count > 0 else float('inf')
    metrics = online_utils.compute_all_metrics(all_true_strings, all_pred_strings)
    metrics['loss'] = avg_loss
    metrics_log_str = ", ".join([f"{k.replace('_',' ').upper()}={v:.4f}" for k, v in metrics.items()])
    logger.info(f"Gesamtmetriken '{split_name}': {metrics_log_str}")
    online_utils.save_metrics_to_file(metrics, split_name, filename_suffix="final_eval")

    if num_evaluated_samples > 0:
        try:
            plot_error_distribution(all_true_strings, all_pred_strings, split_name)
            plot_length_vs_metric(all_true_strings, all_pred_strings, 'cer', split_name)
            plot_length_vs_metric(all_true_strings, all_pred_strings, 'sentence_accuracy', split_name) # Korrigierter Name
            save_sample_predictions_text(final_paths, all_true_strings, all_pred_strings, split_name)
        except Exception as plot_e: logger.error(f"Fehler beim Erstellen der Plots/Text-Report für {split_name}: {plot_e}", exc_info=True)

    return metrics, all_true_strings, all_pred_strings, final_paths

# --- Haupt-Evaluationsfunktion ---
def evaluate_online_model(model_path_to_evaluate=None):
    """ Hauptfunktion zur Evaluation des Online CRNN+CTC Modells. """
    eval_start_time = time.time()
    try:
        device = torch.device(config.DEVICE); logger.info(f"Starte Online-Evaluation auf: {device}")

        # Modellpfad bestimmen (unverändert)
        if model_path_to_evaluate and os.path.exists(model_path_to_evaluate): model_path = model_path_to_evaluate
        else: model_path = os.path.join(config.CHECKPOINT_PATH, "best_online_crnn_model.pth")
        if not os.path.exists(model_path): logger.error(f"Kein Evaluationsmodell gefunden unter {model_path}. Abbruch."); return
        logger.info(f"Verwende Modell: {model_path}")

        # Lade Modell (unverändert)
        model = model_online_crnn.build_online_crnn_model()
        if model is None: raise RuntimeError("Modellerstellung fehlgeschlagen.")
        try: model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as load_err: logger.error(f"Fehler beim Laden des Model State Dict: {load_err}", exc_info=True); return
        model.to(device); model.eval(); logger.info("Modell geladen und in Eval-Modus.")

        criterion = nn.CTCLoss(blank=config.BLANK_IDX, reduction='mean', zero_infinity=True)

        # Lade Datenmanifeste (unverändert)
        logger.info("Lade Datenmanifeste für Train, Val und Test Splits...")
        df_train = online_utils.load_manifest(config.CURRENT_TRAIN_MANIFEST)
        df_val = online_utils.load_manifest(config.CURRENT_VAL_MANIFEST)
        df_test = online_utils.load_manifest(config.CURRENT_TEST_MANIFEST)
        if df_train.empty and df_val.empty and df_test.empty: logger.error("Keine Daten in Manifesten gefunden. Abbruch."); return
        logger.info(f"Manifeste geladen: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

        # Erstelle Datasets und Loader (unverändert)
        base_path = config.BASE_PATH
        train_dataset = OnlineHandwritingDataset(df_train, base_path) if not df_train.empty else None
        val_dataset = OnlineHandwritingDataset(df_val, base_path) if not df_val.empty else None
        test_dataset = OnlineHandwritingDataset(df_test, base_path) if not df_test.empty else None
        eval_batch_size = max(1, config.BATCH_SIZE); logger.info(f"Verwende Eval-Batchsize: {eval_batch_size}")
        train_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=online_collate_fn, num_workers=config.NUM_WORKERS) if train_dataset else None
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=online_collate_fn, num_workers=config.NUM_WORKERS) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=online_collate_fn, num_workers=config.NUM_WORKERS) if test_dataset else None

        # --- Evaluation auf allen Splits (unverändert) ---
        all_split_metrics = {}; all_split_predictions = {}
        for split_name, data_loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
            if data_loader is None: logger.warning(f"Überspringe Evaluation für '{split_name}', da keine Daten."); continue
            try:
                metrics, true_strs, pred_strs, paths = evaluate_model_on_split(model, data_loader, criterion, device, split_name)
                all_split_metrics[split_name] = metrics
                if paths: all_split_predictions[split_name] = pd.DataFrame({'Relative Path': paths, 'True Label': true_strs, 'Predicted Label': pred_strs})
                else: all_split_predictions[split_name] = pd.DataFrame()
            except Exception as split_eval_err:
                 logger.error(f"Fehler während Evaluation von Split '{split_name}': {split_eval_err}", exc_info=True)
                 all_split_metrics[split_name] = {'error': str(split_eval_err)}; all_split_predictions[split_name] = pd.DataFrame()

        # --- Zusammenfassende Berichte (unverändert) ---
        if not all_split_metrics: logger.error("Keine Metriken berechnet. Breche Berichtgenerierung ab."); return
        logger.info("Erstelle zusammenfassende Berichte und Plots...")
        all_metrics_path = os.path.join(config.METRICS_PATH, "all_split_metrics_final.json")
        try:
             serializable_metrics_all = {split: online_utils.convert_metrics_to_serializable(m) for split, m in all_split_metrics.items()}
             with open(all_metrics_path, 'w', encoding='utf-8') as f: json.dump(serializable_metrics_all, f, indent=4)
             logger.info(f"Finale Split-Metriken gespeichert: {all_metrics_path}")
        except Exception as json_e: logger.error(f"Fehler beim Speichern der finalen Split-Metriken: {json_e}")
        try: online_utils.plot_metrics_comparison(all_split_metrics)
        except Exception as plot_comp_e: logger.error(f"Fehler beim Erstellen des Metriken-Vergleichsplots: {plot_comp_e}")
        online_utils.create_directory(config.RESULTS_PATH)
        for split_name, predictions_df in all_split_predictions.items():
             if not predictions_df.empty and 'error' not in predictions_df.columns:
                samples_csv_path = os.path.join(config.RESULTS_PATH, f"{split_name}_all_predictions_final.csv")
                try: predictions_df.to_csv(samples_csv_path, index=False, encoding='utf-8'); logger.info(f"Vollständige Vorhersagen für '{split_name}' gespeichert: {samples_csv_path}")
                except Exception as csv_e: logger.error(f"Fehler beim Speichern der Vorhersagen-CSV für {split_name}: {csv_e}")
        report = { # (Report-Inhalt unverändert) ...
            'model_info': {'evaluated_model_path': model_path, 'model_type': 'CRNN+CTC (Online)'},
            'dataset_info': { 'train_size': len(df_train), 'val_size': len(df_val), 'test_size': len(df_test),
                              'feature_dim': config.FEATURE_DIM, 'max_seq_len': config.MAX_SEQ_LEN,
                              'char_list_size': len(config.CHAR_LIST), 'num_classes_incl_blank': config.NUM_CLASSES },
            'evaluation_metrics': serializable_metrics_all,
            'evaluation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        report_path = os.path.join(config.RESULTS_PATH, "evaluation_report_final.json")
        try: 
            with open(report_path, 'w', encoding='utf-8') as f: json.dump(report, f, indent=4); logger.info(f"Finaler Evaluationsbericht gespeichert: {report_path}")
        except Exception as report_e: logger.error(f"Fehler beim Speichern des finalen Evaluationsberichts: {report_e}")

        logger.info("Online-Evaluation auf allen Splits erfolgreich abgeschlossen.")

    except FileNotFoundError as fnf_e: logger.error(f"Datei nicht gefunden während Evaluation: {fnf_e}")
    except ImportError as imp_e: logger.error(f"Import Fehler (fehlt python-Levenshtein?): {imp_e}")
    except Exception as e: logger.exception(f"Kritischer Fehler während der Online-Evaluation: {e}", exc_info=True)
    finally:
        eval_duration = time.time() - eval_start_time
        logger.info(f"Gesamte Evaluationsdauer: {eval_duration:.2f} Sekunden.")

# --- Direktaufruf (unverändert) ---
if __name__ == "__main__":
    if not logging.getLogger().hasHandlers(): logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-7s] %(name)-25s: %(message)s')
    logger.info("Online Evaluations-Skript wird direkt ausgeführt.")
    parser_eval = argparse.ArgumentParser(description="Online CRNN+CTC Modell Evaluation")
    parser_eval.add_argument('--model', type=str, default=None, help="Optional: Pfad zum Modell (.pth). Standard: best_online_crnn_model.pth")
    args_eval = parser_eval.parse_args()
    try:
        evaluate_online_model(model_path_to_evaluate=args_eval.model)
    except Exception as main_eval_e: logger.critical(f"Evaluation fehlgeschlagen: {main_eval_e}", exc_info=True); exit(1)
    exit(0)