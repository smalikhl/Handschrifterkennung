# src_Transformer/train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
import argparse
import subprocess
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import our own modules
from utils import (
    get_timestamp, ensure_dir, set_seeds, get_device, get_project_root,
    VectorizeChar, IAMDataset, iam_collate_fn, wer, cer, compute_f1
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
log_file = "training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)-8s] %(message)s (%(filename)s:%(lineno)d)',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

###############################################################################
# Path Setup
###############################################################################
try:
    PROJECT_ROOT = get_project_root()
    os.chdir(PROJECT_ROOT)
    logger.info(f"Working directory set to: {PROJECT_ROOT}")
    # Define core directories relative to project root
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "Models")
    RESULT_DIR = os.path.join(PROJECT_ROOT, "Results")
    PLOT_DIR = os.path.join(PROJECT_ROOT, "Plots")
    TB_DIR = os.path.join(PROJECT_ROOT, "runs")
    # Ensure directories exist
    ensure_dir(DATA_DIR)
    ensure_dir(MODEL_DIR)
    ensure_dir(RESULT_DIR)
    ensure_dir(PLOT_DIR)
    ensure_dir(TB_DIR)
except Exception as e:
    logger.exception("Error setting up project paths!", exc_info=True)
    exit(1) # Exit if basic setup fails

###############################################################################
# LR Scheduler 
###############################################################################
class CustomSchedule(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, init_lr=1e-5, lr_after_warmup=1e-3, final_lr=1e-5,
                 warmup_epochs=15, decay_epochs=85, steps_per_epoch=100, last_epoch=-1):
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch
        super().__init__(optimizer, last_epoch)
        logger.info(f"CustomSchedule initialized: warmup={warmup_epochs} epochs ({warmup_epochs*steps_per_epoch} steps), decay={decay_epochs} epochs")

    def get_lr(self):
        # Current step
        current_step = self.last_epoch + 1 # LRScheduler step is called *before* optimizer.step
        current_epoch = current_step / self.steps_per_epoch # Can be fractional

        # Warmup phase
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        if current_step <= warmup_steps and self.warmup_epochs > 0:
            lr = self.init_lr + (self.lr_after_warmup - self.init_lr) * (current_step / warmup_steps)

        # Decay phase
        else:
            decay_start_step = warmup_steps
            total_decay_steps = self.decay_epochs * self.steps_per_epoch
            steps_into_decay = current_step - decay_start_step

            if steps_into_decay >= total_decay_steps or total_decay_steps <= 0:
                lr = self.final_lr # Reached final LR or no decay phase
            else:
                # Cosine annealing decay from lr_after_warmup to final_lr
                cosine_factor = 0.5 * (1 + np.cos(np.pi * steps_into_decay / total_decay_steps))
                lr = self.final_lr + cosine_factor * (self.lr_after_warmup - self.final_lr)
                # Cosine annealing decay from lr_after_warmup to final_lr
                cosine_factor = 0.5 * (1 + np.cos(np.pi * steps_into_decay / total_decay_steps))
                lr = self.final_lr + cosine_factor * (self.lr_after_warmup - self.final_lr)
                lr = max(lr, self.final_lr) # Ensure LR doesn't go below final_lr

        return [lr for _ in self.optimizer.param_groups]

###############################################################################
# Training & Evaluation Helper Functions
###############################################################################

def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler):
    """Conducts one epoch of training with gradient clipping."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")

    for i, batch in enumerate(progress_bar):
        source = batch["source"].to(device)
        target = batch["target"].to(device)

        # Prepare for teacher forcing
        decoder_input = target[:, :-1]
        ground_truth = target[:, 1:]

        optimizer.zero_grad()

        # Forward pass
        logits = model(src_Transformer=source, tgt=decoder_input)

        # Calculate loss
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), ground_truth.reshape(B * T))

        # Backward pass with gradient clipping
        loss.backward()
        # Apply gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.6f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


@torch.no_grad()
def evaluate_one_epoch(model, dataloader, criterion, device):
    """Performs one evaluation epoch (calculates only loss)."""
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

    for batch in progress_bar:
        source = batch["source"].to(device)
        target = batch["target"].to(device)
        decoder_input = target[:, :-1]
        ground_truth = target[:, 1:]

        logits = model(src_Transformer=source, tgt=decoder_input)

        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), ground_truth.reshape(B * T))
        total_loss += loss.item()
        progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


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

def display_random_predictions(pred_df: pd.DataFrame, num_samples=5):
    """Displays random examples of target texts and predictions from the DataFrame."""
    if pred_df.empty or len(pred_df) < num_samples:
        logger.warning("Not enough predictions to display random samples.")
        return

    logger.info("--- Random Prediction Samples ---")
    samples = pred_df.sample(min(num_samples, len(pred_df)))
    for i, (_, row) in enumerate(samples.iterrows()):
        print(f"Sample {i+1}:")
        print(f"  Target:     '{row['target_text']}'")
        print(f"  Prediction: '{row['prediction_text']}'")
        print(f"  (CER: {row['cer']:.4f}, WER: {row['wer']:.4f}, F1: {row['f1']:.4f})")
        print("-" * 10)
    logger.info("---------------------------------")

###############################################################################
# Plotting Functions
###############################################################################

def plot_training_history(history_df, plot_path):
    """Plots Loss, CER, WER, F1 from the history DataFrame."""
    try:
        epochs = history_df['epoch']
        plt.figure(figsize=(18, 6))

        # Loss Plot
        plt.subplot(1, 4, 1)
        plt.plot(epochs, history_df['train_loss'], label='Train Loss', marker='o')
        plt.plot(epochs, history_df['val_loss'], label='Val Loss', marker='x')
        plt.title("Loss vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # CER Plot
        plt.subplot(1, 4, 2)
        plt.plot(epochs, history_df['val_cer'], label='Val CER', marker='s', color='red')
        plt.title("Validation CER vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("CER")
        plt.legend()
        plt.grid(True)

        # WER Plot
        plt.subplot(1, 4, 3)
        plt.plot(epochs, history_df['val_wer'], label='Val WER', marker='^', color='purple')
        plt.title("Validation WER vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("WER")
        plt.legend()
        plt.grid(True)

        # F1 Plot
        plt.subplot(1, 4, 4)
        plt.plot(epochs, history_df['val_f1'], label='Val F1', marker='d', color='green')
        plt.title("Validation F1 vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        ensure_dir(os.path.dirname(plot_path))
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Training history plot saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error plotting training history: {e}", exc_info=True)


def plot_detailed_metrics(val_cers, val_wers, val_f1s, base_filename):
    """Creates detailed histograms and boxplots of validation metrics over epochs."""
    try:
        ensure_dir(PLOT_DIR)

        # Combined Histogram
        plt.figure(figsize=(12, 5))
        plt.hist(val_cers, bins=15, alpha=0.7, label='CER', color='skyblue')
        plt.hist(val_wers, bins=15, alpha=0.7, label='WER', color='salmon')
        plt.hist(val_f1s, bins=15, alpha=0.7, label='F1', color='lightgreen')
        plt.title('Distribution of Validation Metrics over Epochs')
        plt.xlabel('Metric Value')
        plt.ylabel('Frequency (Epochs)')
        plt.legend()
        plt.grid(axis='y')
        plt.savefig(os.path.join(PLOT_DIR, f'{base_filename}_metrics_hist.png'))
        plt.close()

        # Combined Boxplot
        plt.figure(figsize=(8, 5))
        plt.boxplot([val_cers, val_wers, val_f1s],
                    labels=['CER', 'WER', 'F1'],
                    patch_artist=True,
                    showmeans=True,
                    boxprops=dict(facecolor='lightblue'))
        plt.title('Boxplot of Validation Metrics over Epochs')
        plt.ylabel('Metric Value')
        plt.grid(axis='y')
        plt.savefig(os.path.join(PLOT_DIR, f'{base_filename}_metrics_boxplot.png'))
        plt.close()

        logger.info(f"Detailed metric plots saved with prefix: {base_filename}")

    except Exception as e:
         logger.error(f"Error plotting detailed metrics: {e}", exc_info=True)

###############################################################################
# Main Training Flow
###############################################################################
def main(args):
    # --- Setup ---
    set_seeds(args.seed)
    device = get_device(force_gpu=True) # Force GPU as requested
    timestamp = get_timestamp()

    run_name = f"{args.model_name}_{timestamp}"
    model_dir = os.path.join(MODEL_DIR, run_name)
    ensure_dir(model_dir)
    logger.info(f"Starting training run: {run_name}")
    logger.info(f"Models will be saved to: {model_dir}")

    # --- Data Loading ---
    train_data_path = os.path.join(DATA_DIR, "iam_train.xlsx")
    val_data_path = os.path.join(DATA_DIR, "iam_val.xlsx")

    try:
        df_train = pd.read_excel(train_data_path)
        df_val = pd.read_excel(val_data_path)
        logger.info(f"Loaded training data: {len(df_train)} samples from {train_data_path}")
        logger.info(f"Loaded validation data: {len(df_val)} samples from {val_data_path}")
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}. Did you run data_preparation.py first?")
        return
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        return

    # --- Vectorizer & Dataloaders ---
    vectorizer = VectorizeChar(max_len=args.target_maxlen)
    vocab_size = vectorizer.get_vocab_size()
    pad_idx = vectorizer.pad_token_id # Important for loss calculation

    train_dataset = IAMDataset(df_train, vectorizer, feature_dim=args.feature_dim)
    val_dataset = IAMDataset(df_val, vectorizer, feature_dim=args.feature_dim)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=iam_collate_fn,
        num_workers=args.num_workers, # Use multiple workers for loading
        pin_memory=True # If using GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size, # Can often use larger batch size for validation
        shuffle=False,
        collate_fn=iam_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    logger.info(f"Dataloaders created. Vocab size: {vocab_size}, Pad Index: {pad_idx}")
    # Add this after vectorizer initialization
    # Verify token handling is correct
    sample_text = "test handwriting"
    encoded_ids = vectorizer(sample_text)
    decoded_text = decode_prediction(encoded_ids, vectorizer.get_idx_to_char_map(), 
                                    (vectorizer.start_token_id, vectorizer.end_token_id, vectorizer.pad_token_id))
    logger.info(f"Token verification: '{sample_text}' → ids: {encoded_ids} → decoded: '{decoded_text}'")
    if sample_text != decoded_text:
        logger.warning(f"Token encoding/decoding mismatch! This may affect model training.")
    # --- Model ---
    model = Transformer(
        num_hid=args.embed_dim,
        num_head=args.num_heads,
        num_feed_forward=args.ffn_dim,
        input_features=args.feature_dim,
        target_maxlen=args.target_maxlen,
        num_layers_enc=args.encoder_layers,
        num_layers_dec=args.decoder_layers,
        num_classes=vocab_size,
        dropout_rate=args.dropout
    ).to(device)

    # Log model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {type(model).__name__}")
    logger.info(f"  Total Parameters: {total_params:,}")
    logger.info(f"  Trainable Parameters: {trainable_params:,}")

    # --- Loss & Optimizer ---
    # Use CrossEntropyLoss, ignoring the padding index
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)  # Added label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = len(train_loader)
    scheduler = CustomSchedule(
        optimizer=optimizer,
        init_lr=args.warmup_init_lr,
        lr_after_warmup=args.learning_rate,
        final_lr=args.final_lr,
        warmup_epochs=args.warmup_epochs,
        decay_epochs=args.decay_epochs,
        steps_per_epoch=steps_per_epoch
    )

    logger.info("Optimizer and Scheduler configured.")

    # --- TensorBoard Setup ---
    tb_log_dir = os.path.join(TB_DIR, run_name)
    ensure_dir(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)
    logger.info(f"TensorBoard logging to: {tb_log_dir}")

    # Optional: Log model graph (might fail for complex models)
    try:
         dummy_batch = next(iter(train_loader))
         dummy_source = dummy_batch["source"].to(device)
         dummy_target = dummy_batch["target"].to(device)[:, :-1] # Prepare decoder input
         with torch.no_grad():
             writer.add_graph(model, (dummy_source, dummy_target), verbose=False)
         logger.info("Model graph logged to TensorBoard.")
    except Exception as e:
         logger.warning(f"Could not log model graph to TensorBoard: {e}")

    # --- Training Loop ---
    best_val_loss = float('inf')
    best_val_metrics = None
    epochs_without_improvement = 0
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'val_cer': [], 'val_wer': [], 'val_f1': [],
        'learning_rate': []
    }

    logger.info(f"Starting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch}/{args.epochs} ---")

        # Training Phase
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        current_lr = scheduler.get_last_lr()[0]
        logger.info(f"Epoch {epoch}: Average Train Loss = {train_loss:.4f}, LR = {current_lr:.6f}")
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)

        # Validation Phase
        val_loss = evaluate_one_epoch(model, val_loader, criterion, device)
        logger.info(f"Epoch {epoch}: Average Validation Loss = {val_loss:.4f}")
        writer.add_scalar("Loss/Validation", val_loss, epoch)

        # Compute Metrics (CER, WER, F1) on Validation Set
        val_cer, val_wer, val_f1, val_pred_df = compute_metrics(model, val_loader, vectorizer, device)
        logger.info(f"Epoch {epoch}: Validation CER = {val_cer:.4f}, WER = {val_wer:.4f}, F1 = {val_f1:.4f}")
        writer.add_scalar("Metrics/Val_CER", val_cer, epoch)
        writer.add_scalar("Metrics/Val_WER", val_wer, epoch)
        writer.add_scalar("Metrics/Val_F1", val_f1, epoch)

        # Log history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_cer'].append(val_cer)
        history['val_wer'].append(val_wer)
        history['val_f1'].append(val_f1)
        history['learning_rate'].append(current_lr)

        # Display some predictions from this epoch's validation
        display_random_predictions(val_pred_df)

        # Early Stopping & Model Checkpointing (based on validation loss)
        current_val_loss = val_loss
        if current_val_loss < best_val_loss - args.min_delta:
            improvement = best_val_loss - current_val_loss
            best_val_loss = current_val_loss
            best_val_metrics = (val_cer, val_wer, val_f1)
            epochs_without_improvement = 0
            
            # Save the best model
            best_model_path = os.path.join(model_dir, f'{args.model_name}_best.pt')
            try:
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"Validation loss improved by {improvement:.6f} to {best_val_loss:.6f}. Best model saved to {best_model_path}")
            except Exception as e:
                logger.error(f"Error saving best model: {e}")
                
            # Also save model weights at certain epochs for potential ensemble later
            if epoch % 10 == 0 or epoch == args.epochs:
                epoch_model_path = os.path.join(model_dir, f'{args.model_name}_epoch{epoch}.pt')
                try:
                    torch.save(model.state_dict(), epoch_model_path)
                    logger.info(f"Model checkpoint saved at epoch {epoch}: {epoch_model_path}")
                except Exception as e:
                    logger.error(f"Error saving epoch checkpoint: {e}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No significant validation loss improvement ({best_val_loss - current_val_loss:.6f} < {args.min_delta}). Count: {epochs_without_improvement}/{args.patience}")
            if epochs_without_improvement >= args.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

    # --- Post-Training ---
    logger.info("Training finished.")
    writer.close() # Close TensorBoard writer

    # Save final model
    final_model_path = os.path.join(model_dir, f'{args.model_name}_final.pt')
    try:
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")

    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(RESULT_DIR, f'training_history_{run_name}.csv')
    try:
        history_df.to_csv(history_path, index=False)
        logger.info(f"Training history saved to {history_path}")
    except Exception as e:
        logger.error(f"Error saving training history CSV: {e}")

    # Plot training history
    plot_path = os.path.join(PLOT_DIR, f'training_history_{run_name}.png')
    plot_training_history(history_df, plot_path)

    # Plot detailed validation metric distributions
    if history['val_cer']: # Check if metrics were recorded
        plot_detailed_metrics(history['val_cer'], history['val_wer'], history['val_f1'], run_name)

    # Print final best metrics
    if best_val_metrics:
        logger.info("--- Best Model Performance (on validation) ---")
        logger.info(f"  Best Validation Loss: {best_val_loss:.6f}")
        logger.info(f"  Best CER: {best_val_metrics[0]:.6f}")
        logger.info(f"  Best WER: {best_val_metrics[1]:.6f}")
        logger.info(f"  Best F1:  {best_val_metrics[2]:.6f}")

    # Automatically run evaluation on the TEST set using the BEST model
    logger.info("--- Starting Final Evaluation on Test Set ---")
    best_model_path_final = os.path.join(model_dir, f'{args.model_name}_best.pt')
    test_data_path = os.path.join(DATA_DIR, "iam_test.xlsx")
    evaluate_script_path = os.path.join(PROJECT_ROOT, 'src_Transformer', 'evaluate.py')
    eval_output_dir = os.path.join(RESULT_DIR, "Evaluation", run_name + "_best_on_test")
    ensure_dir(eval_output_dir)

    if os.path.exists(best_model_path_final) and os.path.exists(test_data_path) and os.path.exists(evaluate_script_path):
        logger.info(f"Running evaluation script: {evaluate_script_path}")
        logger.info(f"  Model: {best_model_path_final}")
        logger.info(f"  Test Data: {test_data_path}")
        logger.info(f"  Output Dir: {eval_output_dir}")
        evaluate_command = [
            'python', evaluate_script_path,
            '--model_path', best_model_path_final,
            '--data_path', test_data_path,     # Use TEST data
            '--output_dir', eval_output_dir,
            '--feature_dim', str(args.feature_dim),
            '--target_maxlen', str(args.target_maxlen),
            '--embed_dim', str(args.embed_dim),
            '--num_heads', str(args.num_heads),
            '--ffn_dim', str(args.ffn_dim),
            '--encoder_layers', str(args.encoder_layers),
            '--decoder_layers', str(args.decoder_layers),
            '--batch_size', '32'  # Fixed batch size for evaluation
        ]
        try:
            # Run evaluation script as a separate process
            subprocess.run(evaluate_command, check=True)
            logger.info("Evaluation script finished.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Evaluation script failed with exit code {e.returncode}.")
        except Exception as e:
            logger.error(f"Error running evaluation script: {e}")
    else:
        logger.warning("Could not run final evaluation: Best model, test data, or evaluate script not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer model for IAM Handwriting Recognition")

    # Paths & Names
    parser.add_argument('--model_name', type=str, required=True, help="Base name for saving the model and logs (timestamp added automatically)")
    # Data Loading
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers for data loading')
    # Model Architecture
    parser.add_argument('--feature_dim', type=int, default=20, help='Dimension of input features (from data_preparation)')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension (num_hid)')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--ffn_dim', type=int, default=1024, help='Hidden dimension in FeedForward layers')
    parser.add_argument('--encoder_layers', type=int, default=4, help='Number of Transformer encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=2, help='Number of Transformer decoder layers')
    parser.add_argument('--target_maxlen', type=int, default=100, help='Maximum length for target sequences')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # Training Hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Peak learning rate after warmup')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW weight decay (L2 regularization)')
    # LR Scheduler
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs for LR scheduler')
    parser.add_argument('--decay_epochs', type=int, default=95, help='Number of decay epochs for LR scheduler')
    parser.add_argument('--warmup_init_lr', type=float, default=1e-6, help='Initial LR during warmup')
    parser.add_argument('--final_lr', type=float, default=1e-6, help='Final LR after decay')
    # Early Stopping
    parser.add_argument('--patience', type=int, default=15, help='Epochs without improvement to wait before early stopping')
    parser.add_argument('--min_delta', type=float, default=1e-4, help='Minimum improvement in val_loss to be considered significant')
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    main(args)