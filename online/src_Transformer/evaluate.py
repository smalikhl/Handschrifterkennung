# src_Transformer/evaluate.py
import os
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from collections import Counter

# Import our own modules
from utils import (
    get_timestamp, ensure_dir, set_seeds, get_device, get_project_root,
    VectorizeChar, IAMDataset, iam_collate_fn, wer, cer, compute_f1,
    compute_bleu, compute_recall, compute_precision
)
from model import Transformer

def decode_prediction(token_ids, idx_to_char, special_token_ids):
    """
    Properly decodes a sequence of token IDs into text, stopping at the first end token.
    
    Args:
        token_ids: Array/list of token IDs
        idx_to_char: Mapping from token IDs to characters
        special_token_ids: Tuple of (start_token_id, end_token_id, pad_token_id)
        
    Returns:
        Decoded text string
    """
    start_token_id, end_token_id, pad_token_id = special_token_ids
    
    # Find the first end token
    end_pos = -1
    for i, idx in enumerate(token_ids):
        if idx == end_token_id:
            end_pos = i
            break
            
    # Create text from character IDs (excluding special tokens)
    text = ""
    for idx in token_ids[:end_pos if end_pos >= 0 else None]:
        if idx not in [start_token_id, end_token_id, pad_token_id]:
            text += idx_to_char.get(idx, "?")
            
    return text
###############################################################################
# Logging Setup
###############################################################################
log_file = "evaluation.log"
# Ensure logs are appended if evaluate.py is called multiple times
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s (%(filename)s:%(lineno)d)',
    handlers=[
        logging.FileHandler(log_file, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


###############################################################################
# Path Setup & Helpers
###############################################################################
try:
    PROJECT_ROOT = get_project_root()
    # Don't change working dir here, assume it's run from project root or rely on absolute paths
    logger.info(f"Project Root detected: {PROJECT_ROOT}")
    # Define core directories relative to project root for saving plots if needed
    PLOT_DIR = os.path.join(PROJECT_ROOT, "Plots", "Evaluation") # Subfolder for evaluation plots
    ensure_dir(PLOT_DIR)
except Exception as e:
    logger.exception("Error setting up project paths!", exc_info=True)
    exit(1)

###############################################################################
# Evaluation Functions
###############################################################################

@torch.no_grad()
def compute_metrics(model, dataloader, vectorizer, device):
    """Computes CER, WER and F1 using improved token generation."""
    model.eval()
    total_cer, total_wer, total_f1 = 0.0, 0.0, 0.0
    total_count = 0
    idx_to_char = vectorizer.get_idx_to_char_map()
    start_token = vectorizer.start_token_id
    end_token = vectorizer.end_token_id
    pad_token = vectorizer.pad_token_id

    progress_bar = tqdm(dataloader, desc="Computing Metrics", unit="batch", leave=False)
    predictions_list = []

    for batch in progress_bar:
        source = batch["source"].to(device)
        target_ids = batch["target"]

        # Generate predictions with temperature and top-k sampling
        pred_ids = model.generate(
            source, 
            start_token_idx=start_token, 
            end_token_idx=end_token,
            temperature=0.8,  # Add temperature for better diversity
            top_k=5           # Use top-k sampling
        )
        pred_ids_cpu = pred_ids.cpu().numpy()
        target_ids_cpu = target_ids.cpu().numpy()

        batch_size = source.size(0)
        for i in range(batch_size):
    # Decode target and prediction texts
            special_token_ids = (start_token, end_token, pad_token)
            target_text = decode_prediction(target_ids_cpu[i], idx_to_char, special_token_ids)
            pred_text = decode_prediction(pred_ids_cpu[i], idx_to_char, special_token_ids)
            
            # Calculate metrics
            current_cer = cer(target_text, pred_text)
            current_wer = wer(target_text, pred_text)
            current_f1 = compute_f1(target_text, pred_text)
            
            total_cer += current_cer
            total_wer += current_wer
            total_f1 += current_f1
            total_count += 1
            
            # Store for later analysis
            predictions_list.append({
                "target_text": target_text,
                "prediction_text": pred_text,
                "cer": current_cer,
                "wer": current_wer,
                "f1": current_f1
            })
        
        # Update progress bar
        progress_bar.set_postfix({
            "Avg CER": f"{total_cer/max(1, total_count):.4f}",
            "Avg WER": f"{total_wer/max(1, total_count):.4f}",
            "Avg F1": f"{total_f1/max(1, total_count):.4f}"
        })
    
    if total_count == 0:
        logger.warning("No samples evaluated for metrics.")
        return 1.0, 1.0, 0.0, pd.DataFrame()
    
    return (
        total_cer / total_count,
        total_wer / total_count,
        total_f1 / total_count,
        pd.DataFrame(predictions_list)
    )

def run_evaluation(model, dataloader, vectorizer, device):
    """
    Performs evaluation and calculates CER, WER, F1.
    Returns a DataFrame with detailed results.
    """
    model.eval()
    results_list = []
    idx_to_char = vectorizer.get_idx_to_char_map()
    start_token = vectorizer.start_token_id
    end_token = vectorizer.end_token_id
    pad_token = vectorizer.pad_token_id

    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    for batch in progress_bar:
        source = batch["source"].to(device)
        target_ids = batch["target"] # Keep on CPU: [B, Tt]

        # Generate predictions
        pred_ids = model.generate(
            source, 
            start_token_idx=start_token, 
            end_token_idx=end_token,
            temperature=0.8,
            top_k=5
        )
        pred_ids_cpu = pred_ids.cpu().numpy()
        target_ids_cpu = target_ids.cpu().numpy()

        batch_size = source.size(0)
        for i in range(batch_size):
            # Decode target and prediction texts using the shared helper function
            special_token_ids = (start_token, end_token, pad_token)
            target_text = decode_prediction(target_ids_cpu[i], idx_to_char, special_token_ids)
            pred_text = decode_prediction(pred_ids_cpu[i], idx_to_char, special_token_ids)

            # Calculate Metrics
            current_cer = cer(target_text, pred_text)
            current_wer = wer(target_text, pred_text)
            current_f1 = compute_f1(target_text, pred_text)
            current_bleu = compute_bleu(target_text, pred_text)
            current_recall = compute_recall(target_text, pred_text)
            current_precision = compute_precision(target_text, pred_text)
            results_list.append({
                "target_text": target_text,
                "prediction_text": pred_text,
                "cer": current_cer,
                "wer": current_wer,
                "f1": current_f1,
                "bleu": current_bleu,
                "recall": current_recall,
                "precision": current_precision,
                "target_length": len(target_text),
                "prediction_length": len(pred_text),
                "target_raw_ids": target_ids_cpu[i].tolist(), # Store raw IDs for potential analysis
                "pred_raw_ids": pred_ids_cpu[i].tolist(),
            })

        # Optional: update progress bar with running averages
        if results_list:
            avg_cer = np.mean([r['cer'] for r in results_list])
            avg_wer = np.mean([r['wer'] for r in results_list])
            avg_f1 = np.mean([r['f1'] for r in results_list])
            avg_bleu = np.mean([r['bleu'] for r in results_list])
            progress_bar.set_postfix({
                "Avg CER": f"{avg_cer:.4f}",
                "Avg WER": f"{avg_wer:.4f}",
                "Avg F1": f"{avg_f1:.4f}",
                "Avg BLEU": f"{avg_bleu:.4f}"})

    df_result = pd.DataFrame(results_list)
    return df_result

###############################################################################
# Plotting Functions for Evaluation (30 separate plots)
###############################################################################

def plot_evaluation_distributions(df_result, output_dir, prefix=""):
    """Creates histograms and boxplots for core metrics."""
    ensure_dir(output_dir)
    metrics = ['cer', 'wer', 'f1']
    colors = ['skyblue', 'salmon', 'lightgreen']
    titles = ['Character Error Rate (CER)', 'Word Error Rate (WER)', 'F1 Score (Character Level)']

    # Histograms
    for metric, color, title in zip(metrics, colors, titles):
        plt.figure(figsize=(8, 5))
        plt.hist(df_result[metric].dropna(), bins=30, color=color, edgecolor='black', alpha=0.8)
        plt.title(f'Distribution of {title}')
        plt.xlabel(metric.upper())
        plt.ylabel('Frequency')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}hist_{metric}.png'))
        plt.close()

    # Boxplots
    for metric, color, title in zip(metrics, colors, titles):
        plt.figure(figsize=(6, 5))
        plt.boxplot(df_result[metric].dropna(), patch_artist=True, showmeans=True,
                    boxprops=dict(facecolor=color, alpha=0.8))
        plt.title(f'Boxplot of {title}')
        plt.ylabel(metric.upper())
        plt.xticks([1], [metric.upper()])
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}box_{metric}.png'))
        plt.close()

    logger.info(f"Distribution plots saved to {output_dir} with prefix '{prefix}'")

def plot_evaluation_correlations(df_result, output_dir, prefix=""):
    """Creates scatterplots between metrics."""
    ensure_dir(output_dir)

    # CER vs WER
    plt.figure(figsize=(7, 6))
    plt.scatter(df_result['cer'], df_result['wer'], alpha=0.4, c='purple', s=10)
    plt.title('CER vs WER')
    plt.xlabel('CER')
    plt.ylabel('WER')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}scatter_cer_wer.png'))
    plt.close()

    # CER vs F1
    plt.figure(figsize=(7, 6))
    plt.scatter(df_result['cer'], df_result['f1'], alpha=0.4, c='orange', s=10)
    plt.title('CER vs F1 Score')
    plt.xlabel('CER')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}scatter_cer_f1.png'))
    plt.close()

    # WER vs F1
    plt.figure(figsize=(7, 6))
    plt.scatter(df_result['wer'], df_result['f1'], alpha=0.4, c='darkgreen', s=10)
    plt.title('WER vs F1 Score')
    plt.xlabel('WER')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}scatter_wer_f1.png'))
    plt.close()

    logger.info(f"Correlation plots saved to {output_dir} with prefix '{prefix}'")

def plot_length_analysis(df_result, output_dir, prefix=""):
    """Creates plots regarding text lengths and metrics."""
    ensure_dir(output_dir)
    df_result['length_diff'] = df_result['target_length'] - df_result['prediction_length']
    df_result['abs_length_diff'] = df_result['length_diff'].abs()

    # Histogram Length Difference
    plt.figure(figsize=(8, 5))
    plt.hist(df_result['length_diff'].dropna(), bins=50, color='teal', edgecolor='black', alpha=0.8)
    plt.title('Distribution of Length Difference (Target - Prediction)')
    plt.xlabel('Length Difference')
    plt.ylabel('Frequency')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}hist_length_diff.png'))
    plt.close()

    # Scatter CER vs Length Difference
    plt.figure(figsize=(7, 6))
    plt.scatter(df_result['length_diff'], df_result['cer'], alpha=0.4, c='navy', s=10)
    plt.title('Length Difference vs CER')
    plt.xlabel('Length Difference (Target - Prediction)')
    plt.ylabel('CER')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}scatter_lenDiff_cer.png'))
    plt.close()

    # Scatter WER vs Length Difference
    plt.figure(figsize=(7, 6))
    plt.scatter(df_result['length_diff'], df_result['wer'], alpha=0.4, c='darkred', s=10)
    plt.title('Length Difference vs WER')
    plt.xlabel('Length Difference (Target - Prediction)')
    plt.ylabel('WER')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}scatter_lenDiff_wer.png'))
    plt.close()

    # Scatter F1 vs Length Difference
    plt.figure(figsize=(7, 6))
    plt.scatter(df_result['length_diff'], df_result['f1'], alpha=0.4, c='darkgreen', s=10)
    plt.title('Length Difference vs F1 Score')
    plt.xlabel('Length Difference (Target - Prediction)')
    plt.ylabel('F1 Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}scatter_lenDiff_f1.png'))
    plt.close()

    # Scatter CER vs Target Length
    plt.figure(figsize=(7, 6))
    plt.scatter(df_result['target_length'], df_result['cer'], alpha=0.4, c='blue', s=10)
    plt.title('Target Length vs CER')
    plt.xlabel('Target Length')
    plt.ylabel('CER')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}scatter_targetLen_cer.png'))
    plt.close()

     # Scatter WER vs Target Length
    plt.figure(figsize=(7, 6))
    plt.scatter(df_result['target_length'], df_result['wer'], alpha=0.4, c='red', s=10)
    plt.title('Target Length vs WER')
    plt.xlabel('Target Length')
    plt.ylabel('WER')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}scatter_targetLen_wer.png'))
    plt.close()

    logger.info(f"Length analysis plots saved to {output_dir} with prefix '{prefix}'")

def plot_all_evaluation_plots(df_result, output_dir, prefix="eval_"):
    """Wrapper function to generate all standard evaluation plots."""
    plot_evaluation_distributions(df_result, output_dir, prefix)
    plot_evaluation_correlations(df_result, output_dir, prefix)
    plot_length_analysis(df_result, output_dir, prefix)
    # Add calls to more specific plotting functions if needed
    logger.info("All evaluation plots generated.")


###############################################################################
# Main Evaluation Flow
###############################################################################
def main(args):
    # --- Setup ---
    set_seeds(args.seed)
    device = get_device(force_gpu=False) # Allow CPU for evaluation if GPU not needed/available
    timestamp = get_timestamp()

    run_name = f"{os.path.basename(args.model_path).replace('.pt', '')}_on_{os.path.basename(args.data_path).replace('.xlsx','')}_{timestamp}"
    output_dir = os.path.join(args.output_dir, run_name)
    ensure_dir(output_dir)
    logger.info(f"Starting evaluation run: {run_name}")
    logger.info(f"Output will be saved to: {output_dir}")

    # --- Load Data ---
    data_abs_path = os.path.join(PROJECT_ROOT, args.data_path)
    try:
        df_eval = pd.read_excel(data_abs_path)
        logger.info(f"Loaded evaluation data: {len(df_eval)} samples from {data_abs_path}")
    except FileNotFoundError:
        logger.error(f"Evaluation data file not found: {data_abs_path}")
        return
    except Exception as e:
        logger.error(f"Error loading evaluation data: {e}", exc_info=True)
        return

    # --- Vectorizer & Dataloader ---
    # Vectorizer must match the trained model
    vectorizer = VectorizeChar(max_len=args.target_maxlen)
    vocab_size = vectorizer.get_vocab_size()
    logger.info(f"Vectorizer initialized with vocab size {vocab_size} and max_len {args.target_maxlen}.")

    eval_dataset = IAMDataset(df_eval, vectorizer, feature_dim=args.feature_dim)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size, # Batch size for evaluation
        shuffle=False,
        collate_fn=iam_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    # --- Load Model ---
    model_abs_path = os.path.join(PROJECT_ROOT, args.model_path)
    try:
        # Model architecture must match the saved weights!
        model = Transformer(
            num_hid=args.embed_dim, # Must match training
            num_head=args.num_heads, # Must match training
            num_feed_forward=args.ffn_dim, # Must match training
            input_features=args.feature_dim,
            target_maxlen=args.target_maxlen,
            num_layers_enc=args.encoder_layers, # Must match training
            num_layers_dec=args.decoder_layers, # Must match training
            num_classes=vocab_size # Must match vectorizer
        ).to(device)

        state_dict = torch.load(model_abs_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Model state dict loaded successfully from: {model_abs_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_abs_path}")
        return
    except Exception as e:
        logger.error(f"Error loading model state dict: {e}. Ensure model architecture parameters match the checkpoint.", exc_info=True)
        return

    # --- Perform Evaluation ---
    logger.info("Starting evaluation loop...")
    df_results = run_evaluation(model, eval_loader, vectorizer, device)

    if df_results.empty:
        logger.error("Evaluation resulted in an empty DataFrame. No results to save or plot.")
        return

    # --- Calculate and Save Results ---
    avg_cer = df_results['cer'].mean()
    avg_wer = df_results['wer'].mean()
    avg_f1 = df_results['f1'].mean()
    avg_bleu = df_results['bleu'].mean()
    avg_recall = df_results['recall'].mean()
    avg_precision = df_results['precision'].mean()
    std_cer = df_results['cer'].std()
    std_wer = df_results['wer'].std()
    std_f1 = df_results['f1'].std()
    std_bleu = df_results['bleu'].std()
    std_recall = df_results['recall'].std()
    std_precision = df_results['precision'].std()

    logger.info("--- Evaluation Summary ---")
    logger.info(f"  Average CER: {avg_cer:.5f} (+/- {std_cer:.5f})")
    logger.info(f"  Average WER: {avg_wer:.5f} (+/- {std_wer:.5f})")
    logger.info(f"  Average F1:  {avg_f1:.5f} (+/- {std_f1:.5f})")
    logger.info(f"  Average BLEU: {avg_bleu:.5f} (+/- {std_bleu:.5f})")
    logger.info(f"  Average Recall: {avg_recall:.5f} (+/- {std_recall:.5f})")
    logger.info(f"  Average Precision: {avg_precision:.5f} (+/- {std_precision:.5f})")
    logger.info("--------------------------")

    # Save detailed results (Excel)
    results_excel_path = os.path.join(output_dir, 'evaluation_details.xlsx')
    try:
        df_results.to_excel(results_excel_path, index=False)
        logger.info(f"Detailed evaluation results saved to: {results_excel_path}")
    except Exception as e:
        logger.error(f"Error saving results to Excel: {e}")

    # Save summary (text file)
    summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
    try:
        with open(summary_path, 'w') as f:
            f.write("Evaluation Summary\n")
            f.write("====================\n")
            f.write(f"Model Path: {args.model_path}\n")
            f.write(f"Data Path: {args.data_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Number of Samples: {len(df_results)}\n")
            f.write("--------------------\n")
            f.write(f"Average CER: {avg_cer:.5f} (Std: {std_cer:.5f})\n")
            f.write(f"Average WER: {avg_wer:.5f} (Std: {std_wer:.5f})\n")
            f.write(f"Average F1:  {avg_f1:.5f} (Std: {std_f1:.5f})\n")
            f.write(f"Average BLEU: {avg_bleu:.5f} (Std: {std_bleu:.5f})\n")
            f.write(f"Average Recall: {avg_recall:.5f} (Std: {std_recall:.5f})\n")
            f.write(f"Average Precision: {avg_precision:.5f} (Std: {std_precision:.5f})\n")
            f.write("====================\n")
        logger.info(f"Evaluation summary saved to: {summary_path}")
    except Exception as e:
        logger.error(f"Error saving summary file: {e}")

    # --- Generate Plots ---
    logger.info("Generating evaluation plots...")
    plot_dir_for_run = os.path.join(PLOT_DIR, run_name) # Store plots in a subfolder named after the run
    ensure_dir(plot_dir_for_run)
    plot_all_evaluation_plots(df_results, plot_dir_for_run, prefix="") # Use empty prefix for cleaner filenames

    logger.info("Evaluation script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Transformer model for IAM Handwriting Recognition")

    # Required paths
    parser.add_argument('--model_path', type=str, required=True, help="Relative path (from project root) to the trained model checkpoint (.pt)")
    parser.add_argument('--data_path', type=str, required=True, help="Relative path (from project root) to the evaluation data Excel file (e.g., data/iam_test.xlsx)")
    parser.add_argument('--output_dir', type=str, required=True, help="Relative path (from project root) to the directory where evaluation results will be saved")

    # Model Architecture parameters (MUST match the loaded model)
    parser.add_argument('--feature_dim', type=int, default=20, help='Dimension of input features')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension (num_hid)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ffn_dim', type=int, default=1024, help='Hidden dimension in FeedForward layers')
    parser.add_argument('--encoder_layers', type=int, default=4, help='Number of Transformer encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of Transformer decoder layers')
    parser.add_argument('--target_maxlen', type=int, default=100, help='Maximum length for target sequences (used by Vectorizer)')

    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    main(args)