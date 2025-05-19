# src_Transformer/utils.py
import os
import random
import string
import logging
from datetime import datetime
from collections import Counter
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

# --- General Utilities ---

def get_timestamp():
    """Generates a timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_dir(directory):
    """Creates the directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)

def set_seeds(seed=42):
    """Sets all relevant seeds for reproducible experiments."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # for multi-GPU
        # For reproducibility, may affect performance:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    logger.info(f"Seeds set to {seed}")


def get_device(force_gpu=True):
    """Returns the PyTorch device (GPU if available, otherwise CPU)."""
    if force_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"GPU found: {torch.cuda.get_device_name(device.index)}")
            return device
        else:
            logger.error("GPU forced, but no GPU found! Aborting.")
            raise RuntimeError("No GPU found, but GPU usage was forced.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"GPU found and selected: {torch.cuda.get_device_name(device.index)}")
        return device
    else:
        device = torch.device("cpu")
        logger.info("No GPU found, using CPU.")
        return device

def get_project_root():
    """Finds the project root directory where 'src_Transformer', 'data' etc. are located."""
    src_Transformer_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(src_Transformer_dir)


# --- Text Vectorization ---

class VectorizeChar:
    """
    Transforms text into a list of integer indices based on a vocabulary.
    Adds start (<) and end (>) tokens. Unknown characters are mapped to '#'.
    """
    def __init__(self, max_len=100):
        # Vocabulary: special tokens + printable characters
        # 0: Padding
        # 1: Unknown ('#')
        # 2: Start ('<')
        # 3: End ('>')
        # 4: Hyphen ('-')
        self.special_tokens = ["<PAD>", "<UNK>", "<START>", "<END>", "-"]
        self.vocab_chars = list(string.printable[:95]) # Base characters

        # Remove special tokens from base characters
        for token in self.special_tokens[1:]: # Skip PAD
            if token in self.vocab_chars:
                self.vocab_chars.remove(token)

        self.vocab = self.special_tokens + sorted(self.vocab_chars) # Alphabetically sorted for consistency
        self.max_len = max_len # Max sequence length (can be used for truncation/padding)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        # Define IDs for special tokens
        self.pad_token_id = self.char_to_idx["<PAD>"]
        self.unk_token_id = self.char_to_idx["<UNK>"]
        self.start_token_id = self.char_to_idx["<START>"]
        self.end_token_id = self.char_to_idx["<END>"]

        logger.info(f"Vectorizer initialized. Vocab size: {len(self.vocab)}")


    # Replace the current __call__ method in utils.py (around line 103)
    def __call__(self, text: str) -> list[int]:
        """
        Converts a text into a list of integer indices,
        with proper handling of special tokens.
        """
        # Start with just the start token ID
        indices = [self.start_token_id]
        
        # Add character IDs for the main text
        for ch in text:
            indices.append(self.char_to_idx.get(ch, self.unk_token_id))
        
        # Add end token ID
        indices.append(self.end_token_id)

        # Truncate if max_len is exceeded
        if len(indices) > self.max_len:
            indices = indices[:self.max_len-1] + [self.end_token_id]

        return indices

    def get_vocabulary(self) -> list[str]:
        """Returns the vocabulary (list of characters/tokens)."""
        return self.vocab

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return len(self.vocab)

    def get_idx_to_char_map(self) -> dict[int, str]:
        """Returns the mapping table from index to character."""
        return self.idx_to_char


# --- Feature Loading ---

def path_to_features(path: str, expected_features=20) -> torch.Tensor:
    """
    Reads a binary file (.bin) with float32 features,
    reshapes to [T, num_features] and applies improved normalization.
    """
    project_root = get_project_root()
    full_path = os.path.join(project_root, path)
    try:
        with open(full_path, 'rb') as f:
            raw = f.read()
        # Convert bytes to float32 Tensor
        x = torch.frombuffer(bytearray(raw), dtype=torch.float32)

        # Verify feature dimension
        num_elements = x.numel()
        if num_elements % expected_features != 0:
            logger.error(f"File {path}: Number of elements ({num_elements}) not divisible by expected features ({expected_features}). Corrupted file?")
            return torch.empty((0, expected_features), dtype=torch.float32)

        T = num_elements // expected_features
        x = x.reshape(T, expected_features)

        # Apply robust feature normalization (globally per feature)
        if T > 0:
            # Calculate mean and std per feature dimension
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)
            
            # Replace zeros in std with 1.0 to avoid division by zero
            std = torch.where(std < 1e-6, torch.ones_like(std), std)
            
            # Normalize
            x = (x - mean) / std
            
            # Clip extreme values
            x = torch.clamp(x, min=-5.0, max=5.0)
        else:
            logger.warning(f"Feature tensor is empty after reshape for {path}.")

        return x

    except FileNotFoundError:
        logger.error(f"Feature file not found: {full_path}")
        return torch.empty((0, expected_features), dtype=torch.float32)
    except Exception as e:
        logger.error(f"Error reading or processing feature file {full_path}: {e}")
        return torch.empty((0, expected_features), dtype=torch.float32)


# --- PyTorch Dataset ---

class IAMDataset(Dataset):
    """
    PyTorch Dataset for loading prepared IAM data (.bin features).
    """
    def __init__(self, data_df: pd.DataFrame, vectorizer: VectorizeChar, feature_dim: int = 20):
        """
        Args:
            data_df: DataFrame with columns 'bin_file_path' and 'transcript'.
            vectorizer: Instance of VectorizeChar for text vectorization.
            feature_dim: Expected number of features per time step.
        """
        super().__init__()
        # Filter out rows with missing or invalid bin_file_path or transcript
        self.data = data_df.dropna(subset=['bin_file_path', 'transcript']).reset_index(drop=True)
        self.vectorizer = vectorizer
        self.feature_dim = feature_dim
        logger.info(f"IAMDataset initialized with {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Loads a sample: feature tensor and target IDs."""
        if idx >= len(self.data):
            raise IndexError("Index out of bounds")

        row = self.data.iloc[idx]
        bin_path = row['bin_file_path']
        transcript = str(row['transcript']) # Ensure transcript is string

        # Load features from .bin path
        x = path_to_features(bin_path, expected_features=self.feature_dim) # [T, feature_dim], float

        # Vectorize the text
        y_ids = self.vectorizer(transcript) # List of int
        y = torch.tensor(y_ids, dtype=torch.long) # [T_y], long

        # Return empty features with valid transcript if feature loading failed
        if x.shape[0] == 0:
            logger.warning(f"Sample {idx} ({bin_path}) resulted in empty features. Returning dummy data.")
            x = torch.zeros((1, self.feature_dim), dtype=torch.float32) # Single dummy time step

        return x, y


# --- PyTorch Collate Function ---

def iam_collate_fn(batch):
    """
    Collate function for dynamic padding of batches.
    Filters out samples where the features couldn't be loaded (shape[0]==0).
    """
    # Filter out invalid samples where x might be empty due to loading errors
    valid_batch = [(x, y) for x, y in batch if x is not None and x.shape[0] > 0]

    if not valid_batch:
        # Return empty dictionary if the entire batch is invalid
        logger.warning("Collate function received an empty or fully invalid batch.")
        return {"source": torch.empty(0), "target": torch.empty(0)}

    xs, ys = zip(*valid_batch)

    # Pad source sequences (features)
    xs_padded = pad_sequence(xs, batch_first=True, padding_value=0.0) # [B, T_max_x, feature_dim]

    # Pad target sequences (transcripts)
    pad_token_id = 0 # Standard assumption for padding token
    ys_padded = pad_sequence(ys, batch_first=True, padding_value=pad_token_id) # [B, T_max_y]

    return {"source": xs_padded, "target": ys_padded}


# --- Evaluation Metrics ---

def levenshtein(a, b):
    """
    Calculates the Levenshtein distance between two sequences (strings or lists).
    Optimized implementation (uses less memory).
    """
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    if n > m:
        a, b = b, a
        n, m = m, n

    current_row = list(range(n + 1))
    for i in range(1, m + 1):
        previous_row = current_row
        current_row = [i] + [0] * n
        for j in range(1, n + 1):
            add = previous_row[j] + 1
            delete = current_row[j - 1] + 1
            change = previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]


def wer(ground_truth: str, prediction: str) -> float:
    """
    Calculates the Word Error Rate (WER).
    Tokens are separated by whitespace.
    """
    # Normalize whitespace and split into words
    gt_words = ground_truth.strip().split()
    pred_words = prediction.strip().split()

    # Handle empty sequences
    if not gt_words:
        return 1.0 if pred_words else 0.0 # Error is 1 if prediction is not empty, 0 if both are empty

    distance = levenshtein(gt_words, pred_words)
    error_rate = distance / len(gt_words)
    return min(error_rate, 1.0) # Cap error rate at 1.0


def cer(ground_truth: str, prediction: str) -> float:
    """
    Calculates the Character Error Rate (CER).
    Doesn't ignore any characters (direct comparison).
    """
     # Handle empty sequences
    if not ground_truth:
        return 1.0 if prediction else 0.0 # Error is 1 if prediction is not empty, 0 if both are empty

    distance = levenshtein(ground_truth, prediction)
    error_rate = distance / len(ground_truth)
    return min(error_rate, 1.0) # Cap error rate at 1.0


def compute_f1(target: str, prediction: str) -> float:
    """
    Calculates the F1-score at character level between target and prediction.
    Treats strings as sequences of characters.
    """
    target_counter = Counter(target)
    prediction_counter = Counter(prediction)

    # Intersection of counters gives common characters and their minimum counts
    common_chars = target_counter & prediction_counter
    num_common = sum(common_chars.values())

    # Precision: common / predicted_length
    pred_len = len(prediction)
    if pred_len == 0:
        precision = 0.0 # Empty prediction with non-empty target = 0 precision
    else:
        precision = num_common / pred_len

    # Recall: common / target_length
    target_len = len(target)
    if target_len == 0:
        recall = 1.0 if num_common == 0 else 0.0 # Perfect recall if target is empty and prediction matched
    else:
        recall = num_common / target_len

    # F1 Score: 2 * (precision * recall) / (precision + recall)
    if precision + recall == 0:
        return 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1


def compute_precision(target: str, prediction: str) -> float:
    """
    Calculates Precision at character level between target and prediction.
    Precision = Number of correctly predicted characters / Total number of predicted characters.
    """
    target_counter = Counter(target)
    prediction_counter = Counter(prediction)

    # Intersection of counters gives common characters and their minimum counts
    common_chars = target_counter & prediction_counter
    num_common = sum(common_chars.values())

    # Precision: common / predicted_length
    pred_len = len(prediction)
    if pred_len == 0:
        return 0.0 # Empty prediction with non-empty target = 0 precision
    else:
        return num_common / pred_len


def compute_recall(target: str, prediction: str) -> float:
    """
    Calculates Recall at character level between target and prediction.
    Recall = Number of correctly predicted characters / Total number of target characters.
    """
    target_counter = Counter(target)
    prediction_counter = Counter(prediction)

    # Intersection of counters gives common characters and their minimum counts
    common_chars = target_counter & prediction_counter
    num_common = sum(common_chars.values())

    # Recall: common / target_length
    target_len = len(target)
    if target_len == 0:
        return 1.0 if num_common == 0 else 0.0 # Perfect recall if target is empty and prediction matched
    else:
        return num_common / target_len


def compute_bleu(target: str, prediction: str, n_gram=4, weights=None) -> float:
    """
    Calculates the BLEU score between target and prediction.
    BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of texts
    translated by machine translation systems.
    
    Args:
        target: Target text (reference)
        prediction: Predicted text
        n_gram: Maximum n-gram size (default: 4)
        weights: Weights for the n-grams (default: uniform)
    
    Returns:
        BLEU score between 0 and 1
    """
    if not target or not prediction:
        return 0.0
    
    # Default weights if not specified
    if weights is None:
        weights = [1/n_gram] * n_gram
    
    # Normalize weights
    weights_sum = sum(weights)
    if weights_sum != 1.0:
        weights = [w / weights_sum for w in weights]
    
    # Tokenize texts (at character level)
    target_tokens = list(target)
    prediction_tokens = list(prediction)
    
    # Calculate Brevity Penalty
    bp = min(1.0, math.exp(1 - len(target_tokens) / max(1, len(prediction_tokens))))
    
    # Calculate n-gram precisions
    precisions = []
    for i in range(1, n_gram + 1):
        if i > len(prediction_tokens):
            precisions.append(0.0)
            continue
        
        # Generate n-grams
        target_ngrams = _get_ngrams(target_tokens, i)
        prediction_ngrams = _get_ngrams(prediction_tokens, i)
        
        # Count matching n-grams
        matches = sum((target_ngrams & prediction_ngrams).values())
        total = sum(prediction_ngrams.values())
        
        # Calculate precision for this n-gram
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)
    
    # Calculate weighted geometric mean of precisions
    if all(p > 0 for p in precisions):
        s = sum(w * math.log(p) for w, p in zip(weights, precisions))
        bleu = bp * math.exp(s)
    else:
        bleu = 0.0
    
    return bleu


def _get_ngrams(tokens, n):
    """
    Helper function to generate n-grams from a token list.
    
    Args:
        tokens: List of tokens
        n: n-gram size
    
    Returns:
        Counter of n-grams
    """
    ngrams = Counter()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams[ngram] += 1
    return ngrams