
import os
import csv
import logging
import numpy as np
import json
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein
import pandas as pd

import config

logger = logging.getLogger(__name__)

def create_directory(path):
    """Erstellt einen Ordner (rekursiv), wenn nicht vorhanden."""
    # (Code unverändert)
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    except OSError as e:
        if not os.path.isdir(path):
            logger.error(f"Fehler beim Erstellen des Verzeichnisses {path}: {e}", exc_info=True)
            raise

def clean_relative_path(raw_path):
    """Bereinigt relative Pfade."""
    # (Code unverändert)
    if not isinstance(raw_path, str):
        logger.warning(f"Ungültiger Pfad-Typ an clean_relative_path: {type(raw_path)}. Gebe leer zurück.")
        return ""
    cleaned_path = raw_path.strip().replace("\\", "/")
    if cleaned_path.startswith('/'):
        logger.debug(f"Entferne führenden Slash von Pfad: '{raw_path}' -> '{cleaned_path.lstrip('/')}'")
        cleaned_path = cleaned_path.lstrip('/')
    return cleaned_path

def load_feature_pseudo_image(absolute_bin_path):
    """Lädt Merkmale aus .bin und formt zu Pseudo-Bild Tensor (1, H, W)."""
    # (Code unverändert)
    target_h = config.CNN_INPUT_HEIGHT; target_w = config.CNN_INPUT_WIDTH
    if not absolute_bin_path or not isinstance(absolute_bin_path, str) or not os.path.exists(absolute_bin_path):
        logger.error(f"Feature-Datei nicht gefunden oder ungültiger Pfad: {absolute_bin_path}"); return None
    if os.path.getsize(absolute_bin_path) == 0: logger.warning(f"Feature-Datei ist leer: {absolute_bin_path}"); return None
    try:
        with open(absolute_bin_path, 'rb') as f: raw_bytes = f.read()
        expected_bytes_multiple = 4 * config.FEATURE_DIM
        if len(raw_bytes) % expected_bytes_multiple != 0: logger.error(f"Datei {os.path.basename(absolute_bin_path)}: Größe ({len(raw_bytes)}) != multiple of (4 * {config.FEATURE_DIM}). Corrupted?"); return None
        if len(raw_bytes) == 0: logger.warning(f"Feature-Datei ist nach Leseversuch leer: {absolute_bin_path}"); return None
        features_flat = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32)
        num_timesteps = features_flat.numel() // config.FEATURE_DIM
        if num_timesteps == 0: logger.warning(f"0 Zeitstempel nach Reshape: {absolute_bin_path}"); return None
        features = features_flat.reshape(num_timesteps, config.FEATURE_DIM) # [T, F]
        if num_timesteps > 1:
             mean = torch.mean(features, dim=0, keepdim=True); std = torch.std(features, dim=0, keepdim=True)
             std = torch.where(std < 1e-6, torch.ones_like(std), std); features = (features - mean) / std
        if not torch.all(torch.isfinite(features)): logger.error(f"Non-finite values NACH Normalisierung für {absolute_bin_path}. Ersetze durch 0."); features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features_transposed = features.transpose(0, 1) # [F, T]
        current_h, current_w = features_transposed.shape
        if current_h != target_h: logger.error(f"Feature Dim mismatch: {current_h} != {target_h} für {absolute_bin_path}!"); return None
        final_features = torch.zeros((target_h, target_w), dtype=torch.float32); w_to_copy = min(current_w, target_w)
        final_features[:, :w_to_copy] = features_transposed[:, :w_to_copy]
        if current_w > target_w: logger.warning(f"Feature sequence truncated W={current_w} -> {target_w} for {os.path.basename(absolute_bin_path)}")
        pseudo_image_tensor = final_features.unsqueeze(0) # [1, H, W]
        if pseudo_image_tensor.shape != (1, target_h, target_w): logger.error(f"Finale Tensor-Shape {pseudo_image_tensor.shape} != (1, {target_h}, {target_w})!"); return None
        return pseudo_image_tensor
    except Exception as e: logger.exception(f"Fehler beim Laden/Verarbeiten von {absolute_bin_path}: {e}"); return None

# --- Label Handling ---
def encode_label(label_str, char_to_idx):
    """Wandelt Label-String in Index-Liste um, überspringt unbekannte."""
    # (Code unverändert)
    encoded = []; unknown_chars = set()
    for char in label_str:
        idx = char_to_idx.get(char)
        if idx is not None and idx != config.BLANK_IDX: encoded.append(idx)
        elif char not in unknown_chars and idx != config.BLANK_IDX :
            unknown_chars.add(char); logger.warning(f"Überspringe unbekanntes Zeichen '{char}' in Label '{label_str}'. Füge CHAR_LIST hinzu?")
    return encoded

def decode_label_list(encoded_label_list, idx_to_char):
    """Dekodiert Index-Liste zu String, ignoriert Blank."""
    # (Code unverändert)
    return "".join([idx_to_char[idx] for idx in encoded_label_list if idx != config.BLANK_IDX and idx in idx_to_char])

def decode_ctc_output(predictions_tensor, idx_to_char):
    """Dekodiert CTC Output (Best Path)."""
    # (Code unverändert)
    blank_idx = config.BLANK_IDX
    if isinstance(predictions_tensor, torch.Tensor): predictions_np = predictions_tensor.detach().cpu().numpy()
    elif isinstance(predictions_tensor, np.ndarray): predictions_np = predictions_tensor
    else: logger.error(f"Ungültiger Typ für decode_ctc_output: {type(predictions_tensor)}"); return ["DECODE_INPUT_ERROR"]
    try: best_path_indices = np.argmax(predictions_np, axis=2)
    except Exception as e: logger.error(f"argmax Fehler: {e}. Shape: {predictions_np.shape}"); return ["DECODE_ARGMAX_ERROR"] * predictions_np.shape[1]
    best_path_indices_t = best_path_indices.T
    result_strings = []
    for sequence in best_path_indices_t:
        merged_sequence = [];
        if len(sequence)>0: merged_sequence.append(sequence[0])
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]: merged_sequence.append(sequence[i])
        decoded_chars = []
        for idx in merged_sequence:
            if idx != blank_idx and idx in idx_to_char: decoded_chars.append(idx_to_char[idx])
            elif idx != blank_idx: decoded_chars.append('?')
        result_strings.append("".join(decoded_chars))
    return result_strings

# --- Metriken (Unverändert) ---
# (Code für levenshtein_distance, compute_cer, compute_wer, calculate_bleu_score, compute_all_metrics bleibt identisch)
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = np.arange(len(s2) + 1, dtype=np.int32)
    for i, c1 in enumerate(s1):
        current_row = np.zeros(len(s2) + 1, dtype=np.int32); current_row[0] = i + 1
        for j, c2 in enumerate(s2):
            ins = previous_row[j + 1] + 1; dels = current_row[j] + 1; subs = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(ins, dels, subs)
        previous_row = current_row
    return previous_row[-1]
def compute_cer(pred_str, true_str):
    if true_str is None or pred_str is None: return 1.0
    true_len = len(true_str);
    if true_len == 0: return 0.0 if len(pred_str) == 0 else 1.0
    return levenshtein_distance(pred_str, true_str) / float(true_len)
def compute_wer(pred_str, true_str):
    if true_str is None or pred_str is None: return 1.0
    pred_words = pred_str.split(); true_words = true_str.split(); true_word_len = len(true_words)
    if true_word_len == 0: return 0.0 if len(pred_words) == 0 else 1.0
    return levenshtein_distance(pred_words, true_words) / float(true_word_len)
def calculate_bleu_score(pred_str, true_str):
    if true_str is None or pred_str is None or not true_str: return 0.0
    try:
        ref = [true_str.split()]; hyp = pred_str.split(); smoothing = SmoothingFunction().method1
        weights = (0.25, 0.25, 0.25, 0.25); hyp_len = len(hyp)
        if hyp_len < 4: weights = tuple([1.0/hyp_len]*hyp_len) + tuple([0.0]*(4-hyp_len)) if hyp_len > 0 else (0.0, 0.0, 0.0, 0.0)
        return sentence_bleu(ref, hyp, smoothing_function=smoothing, weights=weights)
    except Exception: return 0.0 # Catch potential errors
def compute_all_metrics(all_true_strs, all_pred_strs):
    metrics = {}; num = len(all_true_strs)
    if num == 0 or len(all_pred_strs) != num: return {'cer': 1.0, 'wer': 1.0, 'sentence_accuracy': 0.0, 'bleu': 0.0, 'char_precision': 0.0, 'char_recall': 0.0, 'char_f1': 0.0}
    tot_cer, tot_wer, tot_bleu, exact = 0.0, 0.0, 0.0, 0; tot_ins, tot_del, tot_sub, tot_true = 0, 0, 0, 0
    for true_s, pred_s in zip(all_true_strs, all_pred_strs):
        tot_cer += compute_cer(pred_s, true_s); tot_wer += compute_wer(pred_s, true_s); tot_bleu += calculate_bleu_score(pred_s, true_s)
        if true_s == pred_s: exact += 1
        try:
            if true_s is not None and pred_s is not None:
                 ops = Levenshtein.editops(pred_s, true_s); tot_ins += sum(1 for op in ops if op[0] == 'insert'); tot_del += sum(1 for op in ops if op[0] == 'delete'); tot_sub += sum(1 for op in ops if op[0] == 'replace'); tot_true += len(true_s)
            elif true_s is not None: tot_true += len(true_s)
        except Exception as e: logger.debug(f"Levenshtein Error: {e}")
    metrics['cer'] = tot_cer / num; metrics['wer'] = tot_wer / num; metrics['bleu'] = tot_bleu / num; metrics['sentence_accuracy'] = float(exact) / num
    tot_corr = max(0, tot_true - tot_del - tot_sub); den_p = float(tot_corr + tot_ins + tot_sub); den_r = float(tot_true)
    cp = (tot_corr / den_p) if den_p > 0 else (1.0 if tot_true == 0 else 0.0)
    cr = (tot_corr / den_r) if den_r > 0 else (1.0 if den_p == 0 else 0.0)
    cf1 = (2*cp*cr / (cp+cr)) if cp+cr > 1e-8 else 0.0
    metrics['char_precision'] = cp; metrics['char_recall'] = cr; metrics['char_f1'] = cf1
    return metrics


# --- Split Handling / Manifest Laden ---
def load_manifest(manifest_path):
    """Lädt Pfade und Transkripte aus einer Manifest-Datei (Excel)."""
    # (Code unverändert)
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest-Datei nicht gefunden: {manifest_path}")
        return pd.DataFrame(columns=['bin_file_path', 'transcript'])
    try:
        df = pd.read_excel(manifest_path)
        path_col, label_col = None, None
        poss_p = ['bin_file_path', 'file_path', 'filename', 'path', 'image']; poss_l = ['transcript', 'label', 'text', 'ground_truth']
        df_cols = {col.lower().strip(): col for col in df.columns}
        for p in poss_p:
             if p in df_cols: path_col = df_cols[p]; break
        for l in poss_l:
             if l in df_cols: label_col = df_cols[l]; break
        if not path_col or not label_col:
            logger.error(f"Benötigte Spalten nicht in {manifest_path} gefunden. Header: {list(df.columns)}"); return pd.DataFrame()
        df_final = df[[path_col, label_col]].copy(); df_final.rename(columns={path_col: 'bin_file_path', label_col: 'transcript'}, inplace=True)
        orig_len = len(df_final); df_final['bin_file_path'] = df_final['bin_file_path'].apply(clean_relative_path); df_final['transcript'] = df_final['transcript'].astype(str).str.strip()
        df_final = df_final.dropna(subset=['bin_file_path', 'transcript'])
        df_final = df_final[(df_final['transcript'] != '') & (df_final['bin_file_path'] != '')]
        filt_len = len(df_final)
        if filt_len < orig_len: logger.warning(f"{orig_len - filt_len} Zeilen aus {os.path.basename(manifest_path)} entfernt (leer/ungültig).")
        logger.info(f"Manifest '{os.path.basename(manifest_path)}' geladen: {filt_len} gültige Einträge.")
        return df_final
    except Exception as e: logger.exception(f"Fehler beim Laden des Manifests {manifest_path}: {e}"); return pd.DataFrame()


# --- Sonstiges (Metriken speichern, Plotting) ---
# (Funktionen convert_metrics_to_serializable, save_metrics_to_file, plot_metrics_comparison bleiben identisch)
def convert_metrics_to_serializable(data):
    if isinstance(data, dict): return {k: convert_metrics_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list): return [convert_metrics_to_serializable(item) for item in data]
    elif isinstance(data, (np.generic, np.number)): return data.item()
    elif isinstance(data, torch.Tensor): return data.item()
    else: return data

def save_metrics_to_file(metrics_dict, split_name, output_dir=None, filename_suffix=""):
    if output_dir is None: output_dir = config.METRICS_PATH
    create_directory(output_dir); base_fn = f"{split_name}_metrics";
    if filename_suffix: base_fn += f"_{filename_suffix}"; output_f = os.path.join(output_dir, f"{base_fn}.json")
    try:
        serializable = convert_metrics_to_serializable(metrics_dict)
        with open(output_f, 'w', encoding='utf-8') as f: json.dump(serializable, f, indent=4)
        logger.info(f"Metriken '{split_name}'{(' ('+filename_suffix+')') if filename_suffix else ''} gespeichert: {output_f}")
        return True
    except Exception as e: logger.exception(f"Fehler Speichern Metriken '{split_name}': {e}"); return False

def plot_metrics_comparison(all_split_metrics):
    try: import matplotlib.pyplot as plt; import pandas as pd
    except ImportError: logger.error("matplotlib/pandas benötigt für plot_metrics_comparison."); return
    if not all_split_metrics or not isinstance(all_split_metrics, dict): logger.warning("Keine Daten für Metriken-Vergleichsplot."); return
    splits = list(all_split_metrics.keys()); metrics = ['loss', 'cer', 'wer', 'sentence_accuracy', 'bleu', 'char_precision', 'char_recall', 'char_f1']
    p_data = {m: [all_split_metrics.get(s,{}).get(m, np.nan) for s in splits] for m in metrics}
    df = pd.DataFrame(p_data, index=splits); valid_m = [m for m in metrics if df[m].notna().any()]
    if not valid_m: logger.warning("Keine validen Metriken zum Plotten."); return
    num_m=len(valid_m); n_c=3; n_r=(num_m+n_c-1)//n_c; fig,ax=plt.subplots(n_r,n_c,figsize=(6*n_c,4*n_r),squeeze=False); ax=ax.flatten(); p_idx=0
    for m in valid_m:
        ax_cur=ax[p_idx]; title=m.replace('_',' ').upper()
        if df[m].notna().any():
            df[m].plot(kind='bar', ax=ax_cur, rot=0, legend=False); ax_cur.set(title=f'{title} Vergleich', ylabel=title, xlabel='Split'); ax_cur.grid(axis='y',ls=':',alpha=0.7)
            try:
                 for c in ax_cur.containers: ax_cur.bar_label(c,fmt='%.4f',label_type='edge',padding=3)
            except Exception: pass
        else: ax_cur.set_title(f"{title} Vergleich (Keine Daten)")
        p_idx+=1
    for i in range(p_idx, len(ax)): fig.delaxes(ax[i])
    plt.tight_layout(pad=2.0); plot_path = os.path.join(config.RESULTS_PATH, "metrics_comparison.png")
    try: plt.savefig(plot_path); logger.info(f"Metriken-Vergleichsplot gespeichert: {plot_path}")
    except Exception as e: logger.error(f"Fehler Speichern Metriken-Vergleichsplot: {e}")
    finally: plt.close('all')


