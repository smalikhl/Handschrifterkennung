
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
# Verwende torch.amp statt torch.cuda.amp
from torch.cuda.amp import GradScaler # GradScaler bleibt in torch.cuda.amp
from torch.amp import autocast # NEU: Korrekter Import für autocast
import numpy as np
import logging
import pandas as pd
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import time
import random
import sys

# Lokale Modulimporte (angepasst auf relativ)
import config # Lädt ONLINE config
try:
    import utils as online_utils
    import model_online_crnn
except ImportError as e:
    # Fallback für Logger, falls config nicht initialisiert
    logging.basicConfig(level=logging.ERROR)
    logger_fallback = logging.getLogger(__name__)
    logger_fallback.error(f"Kritischer Fehler beim relativen Import in training_online: {e}. Paketstruktur/ PYTHONPATH prüfen!", exc_info=True)
    sys.exit(1)

logger = logging.getLogger(__name__)

class OnlineHandwritingDataset(Dataset):
    """ PyTorch Dataset für Online-Features (.bin) und deren Labels."""
    # (Code unverändert wie zuvor)
    def __init__(self, manifest_df, base_path):
        super().__init__()
        required_cols = ['bin_file_path', 'transcript']
        if not all(col in manifest_df.columns for col in required_cols):
            raise ValueError(f"Manifest DataFrame fehlen Spalten: {required_cols}")
        self.manifest = manifest_df.dropna(subset=required_cols).copy()
        self.manifest['transcript'] = self.manifest['transcript'].astype(str)
        self.base_path = base_path
        self.fail_count = 0
        if len(self.manifest) == 0: logger.warning("OnlineHandwritingDataset initialisiert mit 0 Samples.")
        logger.info(f"OnlineHandwritingDataset: {len(self.manifest)} Samples. Base Path: {base_path}")
    def __len__(self): return len(self.manifest)
    def __getitem__(self, idx):
        if not 0 <= idx < len(self.manifest):
            logger.error(f"Index {idx} ungültig (Länge {len(self.manifest)})"); return self._get_fallback_item()
        row = self.manifest.iloc[idx]; relative_bin_path = row['bin_file_path']; label_str = row['transcript']
        absolute_bin_path = os.path.join(self.base_path, relative_bin_path).replace("\\", "/")
        feature_pseudo_image = online_utils.load_feature_pseudo_image(absolute_bin_path)
        if feature_pseudo_image is None:
            self.fail_count += 1
            if self.fail_count <= 20 or self.fail_count % 100 == 0: logger.warning(f"Fehler beim Laden von {absolute_bin_path}. Fallback. (# {self.fail_count})")
            return self._get_fallback_item()
        try:
            encoded_label = online_utils.encode_label(label_str, config.CHAR_TO_IDX)
            if not encoded_label: logger.warning(f"Leeres kodiertes Label für '{label_str}' ({relative_bin_path}). Filter in Collate."); return feature_pseudo_image, [], relative_bin_path
        except Exception as enc_e: logger.error(f"Fehler Kodieren '{label_str}': {enc_e}"); return self._get_fallback_item()
        return feature_pseudo_image, encoded_label, relative_bin_path
    def _get_fallback_item(self):
        logger.debug("Generiere Fallback-Item.")
        fallback_image = torch.zeros((1, config.CNN_INPUT_HEIGHT, config.CNN_INPUT_WIDTH), dtype=torch.float32)
        fallback_label = []; fallback_path = "FALLBACK/PATH/ERROR"; return fallback_image, fallback_label, fallback_path

def online_collate_fn(batch):
    """ Angepasster CollateFn für Online-Pseudo-Bilder und Labels."""
    # (Code unverändert wie zuvor)
    valid_batch = []
    original_batch_size = len(batch)
    for item in batch:
        if isinstance(item, tuple) and len(item) == 3:
             img_tensor, label_indices, rel_path = item
             if isinstance(img_tensor, torch.Tensor) and img_tensor.ndim == 3 and \
                img_tensor.shape == (config.CNN_INPUT_CHANNELS, config.CNN_INPUT_HEIGHT, config.CNN_INPUT_WIDTH) and \
                isinstance(label_indices, list) and len(label_indices) > 0 and \
                isinstance(rel_path, str) and "FALLBACK" not in rel_path:
                  valid_batch.append(item)
    if not valid_batch:
         return (torch.empty((0, config.CNN_INPUT_CHANNELS, config.CNN_INPUT_HEIGHT, config.CNN_INPUT_WIDTH), dtype=torch.float),
                 torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), [])
    images, labels, rel_paths = zip(*valid_batch)
    label_lengths = [len(l) for l in labels]
    try: batch_images = torch.stack(images, dim=0)
    except Exception as stack_e:
        logger.error(f"Stacking Fehler: {stack_e}, Shapes: {[img.shape for img in images]}"); return (torch.empty((0, config.CNN_INPUT_CHANNELS, config.CNN_INPUT_HEIGHT, config.CNN_INPUT_WIDTH), dtype=torch.float), torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), [])
    try: batch_labels_flat = torch.tensor([idx for sublist in labels for idx in sublist], dtype=torch.long); batch_label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    except Exception as label_e: logger.error(f"Label Tensor Fehler: {label_e}, Labels: {labels}"); return (torch.empty((0, config.CNN_INPUT_CHANNELS, config.CNN_INPUT_HEIGHT, config.CNN_INPUT_WIDTH), dtype=torch.float), torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long), [])
    final_batch_size = len(valid_batch)
    if final_batch_size < original_batch_size: logger.debug(f"CollateFn: Batch von {original_batch_size} auf {final_batch_size} reduziert.")
    return batch_images, batch_labels_flat, batch_label_lengths, list(rel_paths)

def load_online_data_manifests():
    """Lädt die Pfade und Labels aus den Train/Val Manifest-Dateien."""
    logger.info("Lade Train/Val Manifeste (Excel-Dateien)...")
    df_train = online_utils.load_manifest(config.CURRENT_TRAIN_MANIFEST)
    df_val = online_utils.load_manifest(config.CURRENT_VAL_MANIFEST)
    return df_train, df_val

def log_validation_samples(true_strings, pred_strings, rel_paths, num_samples=5, epoch_num=None):
    """ Loggt Validierungsbeispiele (True vs. Pred). """
    # (Code unverändert wie zuvor)
    if not true_strings or not pred_strings or not rel_paths: return
    actual_num_samples = min(num_samples, len(true_strings)); indices = range(actual_num_samples)
    log_header = f"--- Validierungsbeispiele Epoche {epoch_num or '?'} (Top {actual_num_samples}) ---"
    logger.info(log_header)
    for i in indices:
        true_s = true_strings[i]; pred_s = pred_strings[i]; path = rel_paths[i]
        is_correct = "✅" if true_s == pred_s else "❌"
        logger.info(f"  Sample (Datei: ...{path[-40:]}) {is_correct}")
        logger.info(f"    GT  : '{true_s}'")
        logger.info(f"    PRED: '{pred_s}'")
    logger.info("-" * len(log_header))

def validate_epoch(model, data_loader, device, criterion, current_epoch_num, split_name="val"):
    """ Validierungsfunktion für Online CRNN+CTC mit korrigiertem autocast. """
    if data_loader is None or len(data_loader.dataset) == 0:
         logger.warning(f"Kein Val-Loader ('{split_name}'), Validierung übersprungen.")
         return {'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'sentence_accuracy': 0.0, 'bleu': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'char_f1': 0.0}
    model.eval(); running_loss = 0.0; all_true_strings = []; all_pred_strings = []; all_rel_paths = []
    sample_count = 0; processed_batches = 0
    logger.info(f"Starte Validation '{split_name}' (Epoche {current_epoch_num})...")
    device_type_str = config.DEVICE.split(':')[0] # 'cuda' oder 'cpu'
    # Prüfe ob AMP für Eval genutzt werden soll (kann performance bringen, aber nicht essenziell)
    amp_is_enabled_for_eval = config.USE_MIXED_PRECISION and device_type_str == "cuda"

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc=f"Eval {split_name} Ep {current_epoch_num}", leave=False, unit="batch")
        for batch_images, batch_labels_flat, batch_label_lengths, batch_rel_paths in progress_bar:
            if batch_images.nelement() == 0: continue
            batch_size = batch_images.size(0)
            all_rel_paths.extend(batch_rel_paths)
            batch_images = batch_images.to(device); batch_labels_flat_dev = batch_labels_flat.to(device); batch_label_lengths_dev = batch_label_lengths.to(device)

            try:
                # *** KORRIGIERTER AUTOCAST AUFRUF ***
                with autocast(device_type=device_type_str, dtype=torch.float16 if amp_is_enabled_for_eval else None, enabled=amp_is_enabled_for_eval):
                     logits = model(batch_images) # (SeqLen, Batch, Classes)
                     log_probs = F.log_softmax(logits, dim=2)

                T_out = logits.size(0); N_out = logits.size(1)
                if N_out != batch_size: logger.error(f"Batch Size Mismatch Validation: In={batch_size}, Out={N_out}"); continue
                input_lengths = torch.full(size=(N_out,), fill_value=T_out, dtype=torch.long, device=device)
                loss = criterion(log_probs, batch_labels_flat_dev, input_lengths, batch_label_lengths_dev)
                if not torch.isnan(loss) and not torch.isinf(loss): running_loss += loss.item() * batch_size
                else: logger.warning(f"Ungültiger Loss ({loss.item():.4f}) in Val.")
                pred_strs = online_utils.decode_ctc_output(logits, config.IDX_TO_CHAR)
                all_pred_strings.extend(pred_strs)
                processed_batches += 1; sample_count += batch_size
            except Exception as eval_err:
                 logger.error(f"Fehler im Validation Forward/Loss ({split_name}): {eval_err}", exc_info=False)
                 all_pred_strings.extend(["FORWARD/LOSS_ERROR"] * batch_size); sample_count += batch_size; continue

            current_idx = 0
            try:
                for length in batch_label_lengths.tolist():
                    end_idx = current_idx + length; label_indices = batch_labels_flat[current_idx:end_idx].tolist()
                    true_str = online_utils.decode_label_list(label_indices, config.IDX_TO_CHAR); all_true_strings.append(true_str); current_idx = end_idx
            except Exception as true_dec_err:
                 logger.error(f"Fehler beim GT-Dekodieren ({split_name}): {true_dec_err}")
                 missing_count = batch_size - (len(all_true_strings) % batch_size if batch_size else 0); all_true_strings.extend(["TRUTH_DECODING_ERROR"] * missing_count)
            current_avg_loss = running_loss / sample_count if sample_count > 0 else 0
            progress_bar.set_postfix({'Val Loss': f"{current_avg_loss:.4f}"})

    # --- Nach der Schleife ---
    # (Restliche Logik von validate_epoch bleibt unverändert)
    if processed_batches == 0 or sample_count == 0:
         logger.error(f"Keine Batches erfolgreich in Validation '{split_name}' verarbeitet."); return {'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'sentence_accuracy': 0.0, 'bleu': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'char_f1': 0.0}
    num_eval = min(len(all_true_strings), len(all_pred_strings), len(all_rel_paths))
    if num_eval < sample_count: logger.warning(f"Inkonsistente Längen '{split_name}': Expected={sample_count}, Got={num_eval}. Kürze für Metriken."); all_true_strings, all_pred_strings, all_rel_paths = all_true_strings[:num_eval], all_pred_strings[:num_eval], all_rel_paths[:num_eval]
    if num_eval == 0: logger.error("Keine konsistenten Samples nach Validation übrig."); return {'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'sentence_accuracy': 0.0, 'bleu': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'char_f1': 0.0}
    avg_loss = running_loss / sample_count if sample_count > 0 else float('inf')
    metrics = online_utils.compute_all_metrics(all_true_strings, all_pred_strings); metrics['loss'] = avg_loss
    metrics_log_str = ", ".join([f"{k.replace('_', ' ').upper()}={v:.4f}" for k, v in metrics.items()]); logger.info(f"Validation '{split_name}' (Ep {current_epoch_num}) Metrics: {metrics_log_str}")
    log_validation_samples(all_true_strings, all_pred_strings, all_rel_paths, num_samples=5, epoch_num=current_epoch_num)
    return metrics

def save_training_history(history_list, filename="training_history"):
    # (Code unverändert wie zuvor)
    if not history_list: return
    df_history = pd.DataFrame(history_list)
    history_csv_path = os.path.join(config.RESULTS_PATH, f"{filename}.csv")
    history_json_path = os.path.join(config.RESULTS_PATH, f"{filename}.json")
    online_utils.create_directory(config.RESULTS_PATH)
    try: df_history.to_csv(history_csv_path, index=False); logger.info(f"Trainingshistorie (CSV) gespeichert: {history_csv_path}")
    except Exception as e: logger.error(f"Fehler beim Speichern der History CSV: {e}", exc_info=True)
    try:
        history_plain = online_utils.convert_metrics_to_serializable(history_list)
        with open(history_json_path, 'w', encoding='utf-8') as f: json.dump(history_plain, f, indent=2)
        logger.info(f"Trainingshistorie (JSON) gespeichert: {history_json_path}")
    except Exception as e: logger.error(f"Fehler beim Speichern der History JSON: {e}", exc_info=True)

def plot_training_curves(history_list):
    # (Code unverändert wie zuvor)
    if not history_list or not isinstance(history_list, list) or 'epoch' not in history_list[0]: logger.warning("Unvollständige History für Plot."); return
    try:
        df_history = pd.DataFrame(history_list)
        if df_history.empty: logger.warning("History DataFrame leer."); return
        epochs = df_history['epoch']; metrics_to_plot = set();
        for col in df_history.columns:
            if col.startswith(('train_', 'val_')) and col != 'val_learning_rate' and col != 'train_learning_rate': metrics_to_plot.add('_'.join(col.split('_')[1:]))
        if not metrics_to_plot: logger.warning("Keine plotbaren Metriken gefunden."); return
        preferred_order = ['loss', 'cer', 'wer', 'char_f1', 'sentence_accuracy', 'bleu', 'char_precision', 'char_recall']
        sorted_metrics = sorted(list(metrics_to_plot), key=lambda x: preferred_order.index(x) if x in preferred_order else float('inf'))
        num_metrics = len(sorted_metrics); n_cols = 3; n_rows = (num_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False); axes = axes.flatten(); plot_index = 0
        for metric in sorted_metrics:
            ax = axes[plot_index]; has_data = False; train_col, val_col = f'train_{metric}', f'val_{metric}'
            if train_col in df_history.columns and df_history[train_col].notna().any(): ax.plot(epochs, df_history[train_col], marker='.', linestyle='-', label=f"Train {metric.upper()}"); has_data = True
            if val_col in df_history.columns and df_history[val_col].notna().any(): ax.plot(epochs, df_history[val_col], marker='.', linestyle='--', label=f"Val {metric.upper()}"); has_data = True
            if has_data:
                plot_title = metric.replace('_', ' ').upper(); ax.set_title(f"{plot_title} Verlauf"); ax.set_xlabel("Epoche"); ax.set_ylabel(plot_title); ax.legend(); ax.grid(True, linestyle=':', alpha=0.6)
                if metric in ['cer', 'wer', 'sentence_accuracy', 'char_precision', 'char_recall', 'char_f1']: ax.set_ylim(bottom=0, top=max(1.1, ax.get_ylim()[1]))
            else: ax.set_title(f"{metric.replace('_', ' ').upper()} Verlauf (Keine Daten)")
            plot_index += 1
        for i in range(plot_index, len(axes)): fig.delaxes(axes[i])
        plt.tight_layout(pad=2.0); plot_save_path = os.path.join(config.RESULTS_PATH, "training_curves.png")
        plt.savefig(plot_save_path); logger.info(f"Trainingskurven gespeichert: {plot_save_path}")
    except Exception as e: logger.error(f"Fehler beim Erstellen der Trainingskurven: {e}", exc_info=True)
    finally: plt.close('all')


# --- Haupt-Trainingsfunktion ---
def train_online_model():
    """ Hauptfunktion zum Trainieren des Online-CRNN+CTC-Modells mit korrigiertem autocast."""
    train_start_time = time.time()
    logger.info("="*50); logger.info("=== STARTE ONLINE CRNN+CTC MODELLTRAINING ==="); logger.info("="*50)
    try:
        device = torch.device(config.DEVICE); logger.info(f"Verwende Gerät: {device}")
        device_type_str = device.type # 'cuda' oder 'cpu'
        model = model_online_crnn.build_online_crnn_model();
        if model is None: logger.error("Modellerstellung fehlgeschlagen."); return
        model = model.to(device); logger.info(f"Modell '{type(model).__name__}' erstellt und auf {device} verschoben.")
        criterion = nn.CTCLoss(blank=config.BLANK_IDX, reduction='mean', zero_infinity=True); logger.info(f"CTC-Loss initialisiert (Blank Index: {config.BLANK_IDX})")
        lr, wd = config.LEARNING_RATE, config.WEIGHT_DECAY; opt_name = config.OPTIMIZER.lower()
        if opt_name == 'adam': optimizer = optim.Adam(model.parameters(), lr=lr)
        elif opt_name == 'adamw': optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'sgd': optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
        else: logger.error(f"Unbek. Optimizer '{opt_name}'. Nutze AdamW."); optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        logger.info(f"Optimizer: {type(optimizer).__name__} (LR={lr}, WD={wd if opt_name=='adamw' else 'N/A'})")
        scheduler = None; sched_name = config.SCHEDULER.lower(); scheduler_metric = config.SCHEDULER_METRIC
        scheduler_mode = 'min' if 'loss' in scheduler_metric or 'cer' in scheduler_metric or 'wer' in scheduler_metric else 'max'
        if sched_name == 'reducelronplateau': scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=0.2, patience=config.SCHEDULER_PATIENCE, verbose=True, min_lr=1e-7)
        elif sched_name == 'cosineannealinglr': scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-7)
        if scheduler: logger.info(f"Scheduler: {type(scheduler).__name__} (Mode='{scheduler_mode}', Metric='{scheduler_metric}')")
        else: logger.info("Kein LR Scheduler verwendet.")
        scaler = GradScaler(enabled=(config.USE_MIXED_PRECISION and device_type_str == "cuda")); logger.info(f"Mixed Precision Training (AMP) aktiviert: {scaler.is_enabled()}")
        df_train, df_val = load_online_data_manifests()
        if df_train.empty: logger.error("Trainings-Manifest leer. Training nicht möglich."); return
        logger.info("Erstelle Datasets und DataLoader...")
        base_path = config.BASE_PATH
        train_dataset = OnlineHandwritingDataset(df_train, base_path)
        val_dataset = OnlineHandwritingDataset(df_val, base_path) if not df_val.empty else None
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=online_collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, collate_fn=online_collate_fn, num_workers=config.NUM_WORKERS, pin_memory=True) if val_dataset else None
        logger.info(f"DataLoader erstellt: Train Batches={len(train_loader)}, Val Batches={len(val_loader) if val_loader else 0}")
        logger.info(f"Samples: Train={len(train_dataset)}, Val={len(val_dataset or [])}")
        logger.info("="*50); logger.info(f"=== STARTE TRAININGSLOOP (Max {config.EPOCHS} Epochen) ==="); logger.info("="*50)
        es_metric_key = config.EARLY_STOPPING_METRIC; is_loss_metric = 'loss' in es_metric_key or 'cer' in es_metric_key or 'wer' in es_metric_key
        metric_mode = 'min' if is_loss_metric else 'max'; best_val_metric_value = float('inf') if metric_mode == 'min' else float('-inf')
        patience_counter = 0; best_epoch = -1; training_history = []
        logger.info(f"Early Stopping: Metric='{es_metric_key}', Patience={config.EARLY_STOPPING_PATIENCE}, Mode='{metric_mode}'")
        for epoch in range(1, config.EPOCHS + 1):
            epoch_start_time = time.time(); logger.info(f"--- Epoche {epoch}/{config.EPOCHS} ---")
            model.train(); running_train_loss = 0.0; train_samples_processed = 0
            progress_bar_train = tqdm(train_loader, desc=f"Epoche {epoch} Train", leave=False, unit=" Batch")
            for batch_idx, (batch_images, batch_labels_flat, batch_label_lengths, _) in enumerate(progress_bar_train):
                if batch_images.nelement() == 0: continue
                batch_size = batch_images.size(0)
                batch_images = batch_images.to(device, non_blocking=True); batch_labels_flat = batch_labels_flat.to(device, non_blocking=True); batch_label_lengths = batch_label_lengths.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                try:
                    # *** KORRIGIERTER AUTOCAST AUFRUF ***
                    with torch.amp.autocast(device_type=device_type_str, dtype=torch.float16 if scaler.is_enabled() else None, enabled=scaler.is_enabled()):
                        logits = model(batch_images) # (SeqLen, Batch, Classes)
                        log_probs = F.log_softmax(logits, dim=2)
                        T_out, N_out, _ = log_probs.shape
                        if N_out != batch_size: logger.error(f"Batch Size Mismatch Train: In={batch_size}, Out={N_out}. Skip."); optimizer.zero_grad(set_to_none=True); continue
                        input_lengths = torch.full((N_out,), T_out, dtype=torch.long, device=device)
                        # Handle zero-length targets for CTC Loss if they occur (should be filtered by collate)
                        if batch_label_lengths.min() <= 0:
                            logger.warning(f"Batch {batch_idx} contains zero-length targets. Skipping loss calculation for this batch.")
                            optimizer.zero_grad(set_to_none=True) # Ensure grads are zero before skipping
                            continue
                        loss = criterion(log_probs, batch_labels_flat, input_lengths, batch_label_lengths)

                    if torch.isnan(loss) or torch.isinf(loss): logger.warning(f"Ungültiger Loss ({loss.item():.4f}) Ep {epoch} Batch {batch_idx}. Skip."); optimizer.zero_grad(set_to_none=True); continue
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    running_train_loss += loss.item() * batch_size; train_samples_processed += batch_size
                    current_avg_loss = running_train_loss / train_samples_processed if train_samples_processed > 0 else 0
                    progress_bar_train.set_postfix({'Train Loss': f"{current_avg_loss:.4f}"})
                except RuntimeError as rt_error:
                    if "CUDA out of memory" in str(rt_error): logger.critical("CUDA OOM!", exc_info=False); raise rt_error
                    elif "zero-length labels" in str(rt_error): # Specific catch for CTC zero length label error
                        logger.error(f"CTC Zero-length label error in Batch {batch_idx}. Skipping. Label lengths: {batch_label_lengths.tolist()}")
                        optimizer.zero_grad(set_to_none=True) # Zero grad before skipping
                        continue
                    else: logger.error(f"Runtime Error Train (Ep {epoch}, Batch {batch_idx}): {rt_error}", exc_info=True); continue
            avg_train_loss = running_train_loss / train_samples_processed if train_samples_processed > 0 else float('inf')
            logger.info(f"Epoche {epoch}: Durchschnittlicher Train Loss = {avg_train_loss:.4f}")
            # --- Validierung, History, Logging, Scheduler, Early Stopping, Checkpointing ---
            # (Restlicher Code der Trainingsschleife bleibt unverändert)
            current_val_metrics = {}
            if val_loader:
                try: current_val_metrics = validate_epoch(model, val_loader, device, criterion, epoch, 'val')
                except Exception as val_e: logger.error(f"Fehler Validierung Ep {epoch}: {val_e}", exc_info=True); current_val_metrics = {'loss': float('inf'), 'cer': 1.0, 'wer': 1.0, 'sentence_accuracy': 0.0, 'bleu': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'char_f1': 0.0}
            else: current_val_metrics = {k: None for k in ['loss', 'cer', 'wer', 'sentence_accuracy', 'bleu', 'char_precision', 'char_recall', 'char_f1']}
            epoch_summary = {'epoch': epoch, 'train_loss': avg_train_loss}; current_lr = optimizer.param_groups[0]['lr']; epoch_summary['learning_rate'] = current_lr
            for key, value in current_val_metrics.items():
                if value is not None: epoch_summary[f'val_{key}'] = value
            training_history.append(epoch_summary); log_string = f"Epoche {epoch}: Train Loss={avg_train_loss:.4f}"
            if val_loader:
                val_log_parts = []; log_order = ['loss', 'cer', 'wer', 'char_f1', 'sentence_accuracy', 'bleu', 'char_precision', 'char_recall']
                for key in log_order:
                    if key in current_val_metrics and current_val_metrics[key] is not None: val_log_parts.append(f"Val {key.replace('_',' ').upper()}={current_val_metrics[key]:.4f}")
                for key, value in current_val_metrics.items():
                     if key not in log_order and value is not None: val_log_parts.append(f"Val {key.replace('_',' ').upper()}={value:.4f}")
                log_string += " | " + " | ".join(val_log_parts)
            log_string += f" | LR={current_lr:.7f}"; logger.info(log_string)
            if scheduler and val_loader:
                metric_name_for_scheduler = scheduler_metric.replace('val_', ''); metric_value = current_val_metrics.get(metric_name_for_scheduler)
                if metric_value is None or not np.isfinite(metric_value): logger.warning(f"Metrik '{scheduler_metric}' für Scheduler ungültig. Skip.")
                else:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau): scheduler.step(metric_value)
                    else: scheduler.step()
            early_stop_triggered = False
            if val_loader and config.EARLY_STOPPING_PATIENCE > 0:
                 current_metric_value = epoch_summary.get(es_metric_key)
                 if current_metric_value is None or not np.isfinite(current_metric_value): logger.warning(f"ES Metrik '{es_metric_key}' ungültig Ep {epoch}.")
                 else:
                      improved = False; tolerance = 1e-5
                      if metric_mode == 'min' and current_metric_value < best_val_metric_value - tolerance: improved = True
                      elif metric_mode == 'max' and current_metric_value > best_val_metric_value + tolerance: improved = True
                      if improved:
                          improvement_diff = abs(current_metric_value - best_val_metric_value); best_val_metric_value = current_metric_value; patience_counter = 0; best_epoch = epoch
                          logger.info(f"Verbesserung bei '{es_metric_key}': {current_metric_value:.4f}. Speichere bestes Modell.")
                          best_model_save_path = os.path.join(config.CHECKPOINT_PATH, "best_online_crnn_model.pth"); torch.save(model.state_dict(), best_model_save_path)
                      else:
                          patience_counter += 1; logger.info(f"Keine Verbesserung bei '{es_metric_key}'. Geduld: {patience_counter}/{config.EARLY_STOPPING_PATIENCE} (Best: {best_val_metric_value:.4f} @ Ep {best_epoch})")
                          if patience_counter >= config.EARLY_STOPPING_PATIENCE: logger.warning("EARLY STOPPING!"); early_stop_triggered = True
            epoch_duration = time.time() - epoch_start_time; logger.info(f"Epoche {epoch} Dauer: {epoch_duration:.2f} Sek.")
            save_interval = config.SAVE_CHECKPOINT_INTERVAL
            if save_interval > 0 and (epoch % save_interval == 0 or epoch == config.EPOCHS or early_stop_triggered): logger.info(f"Speichere History/Plots (Epoche {epoch})..."); save_training_history(training_history); plot_training_curves(training_history)
            if early_stop_triggered: break
        # --- Ende Trainingsschleife ---
        logger.info("="*50); logger.info("=== TRAINING ABGESCHLOSSEN ===");
        if early_stop_triggered: logger.info(f"Grund: Early Stopping nach Epoche {epoch}.")
        else: logger.info(f"Grund: Maximale Epochenzahl ({config.EPOCHS}) erreicht.")
        if best_epoch != -1: logger.info(f"Bestes Modell in Epoche {best_epoch} mit {es_metric_key}={best_val_metric_value:.4f} gespeichert.")
        else: logger.warning("Keine Verbesserung während des Trainings beobachtet.")
        logger.info("Speichere finale History/Plots..."); save_training_history(training_history, filename="training_history_final"); plot_training_curves(training_history)
    except RuntimeError as rt_oom:
        if "CUDA out of memory" in str(rt_oom): logger.critical("Training wegen CUDA OOM abgebrochen.", exc_info=False)
        else: logger.critical(f"Kritischer Runtime Fehler im Training: {rt_oom}", exc_info=True)
        if 'training_history' in locals() and training_history: save_training_history(training_history, filename="training_history_error")
    except Exception as e:
        logger.critical(f"Kritischer Fehler im Trainingsprozess: {e}", exc_info=True)
        if 'training_history' in locals() and training_history: save_training_history(training_history, filename="training_history_error")
    finally: total_training_time = time.time() - train_start_time; logger.info(f"Gesamte Trainingsdauer: {total_training_time / 60:.2f} Minuten."); logger.info("="*50)

if __name__ == "__main__":
    if not logging.getLogger().hasHandlers(): logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-7s] %(name)-25s: %(message)s')
    logger.info("Online Training Skript wird direkt ausgeführt.")
    try: train_online_model()
    except Exception as main_e: logger.critical(f"Training fehlgeschlagen: {main_e}", exc_info=True); exit(1)
    exit(0)
