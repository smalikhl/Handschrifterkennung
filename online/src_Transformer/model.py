# src_Transformer/model.py (improved version)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

def merge_two_last_dims(x: torch.Tensor) -> torch.Tensor:
    """Reshapes [B, T, F, C] to [B, T, F*C]."""
    if x.dim() != 4:
        logger.warning(f"merge_two_last_dims expected 4D tensor, got {x.dim()}D. Returning unchanged.")
        return x
    b, t, f, c = x.shape
    return x.view(b, t, f * c)

class PositionalEncoding(nn.Module):
    """
    Fixed positional encoding as in the original Transformer paper.
    This improves over the simple embedding by providing better generalization.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create constant positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension [1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class TokenEmbedding(nn.Module):
    """
    Combines Token-Embedding and Positional-Embedding.
    """
    def __init__(self, num_vocab: int, maxlen: int, embed_dim: int, dropout_rate=0.1):
        """
        Args:
            num_vocab (int): Vocabulary size.
            maxlen (int): Maximum sequence length for positional embeddings.
            embed_dim (int): Embedding dimension.
            dropout_rate (float): Dropout rate applied to embeddings.
        """
        super().__init__()
        self.token_emb = nn.Embedding(num_vocab, embed_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embed_dim, maxlen)
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout_rate)
        logger.info(f"TokenEmbedding initialized: vocab_size={num_vocab}, maxlen={maxlen}, embed_dim={embed_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input Tensor with Token-IDs [batch_size, seq_len].
        Returns:
            torch.Tensor: Output Tensor [batch_size, seq_len, embed_dim].
        """
        # Get token embeddings
        token_embeddings = self.token_emb(x)
        
        # Scale embeddings as per the original Transformer paper
        token_embeddings = token_embeddings * math.sqrt(self.embed_dim)
        
        # Add positional encoding
        embeddings = self.positional_encoding(token_embeddings)
        
        # Apply dropout
        return self.dropout(embeddings)

class Conv2dSubsampling(nn.Module):
    """
    Reduces temporal and feature dimensions through stacked Conv2D layers.
    Input: [B, T, F_in]
    Output: [B, T_out, F_out]
    """
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.1):
        """
        Args:
            in_features (int): Input feature dimension (F_in).
            out_features (int): Output feature dimension (F_out).
            dropout_rate (float): Dropout rate after activation.
        """
        super().__init__()
        # Verify feature dimensions
        if in_features != 20:
            logger.warning(f"Conv2dSubsampling expects in_features=20, got {in_features}. Model may not work correctly.")
        
        # Improved convolutional architecture
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate the output dimension after convolution
        final_conv_width = self._calculate_output_width(in_features)
        merged_dim = final_conv_width * 64
        
        # Linear projection to match the required embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(merged_dim, out_features),
            nn.LayerNorm(out_features)
        )
        
        logger.info(f"Conv2dSubsampling initialized: in={in_features}, conv_filters=64, "
                   f"merged_dim={merged_dim}, out={out_features}")

    def _calculate_output_width(self, input_width, kernel_size=3, stride=2, padding=1):
        """Helper to calculate output width of Conv2D layers."""
        w = input_width
        # Conv1
        w = math.floor((w + 2 * padding - kernel_size) / stride + 1)
        # Conv2
        w = math.floor((w + 2 * padding - kernel_size) / stride + 1)
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input features [B, T, F_in].
        Returns:
            torch.Tensor: Subsampled features [B, T_out, F_out].
        """
        # Input shape: [B, T, F_in]
        batch_size, time_steps, _ = x.shape
        
        # Add channel dimension for Conv2D: [B, 1, T, F_in]
        x = x.unsqueeze(1)
        
        # Apply convolutional layers
        x = self.conv_layers(x)  # [B, 64, T/4, F_in/4]
        
        # Rearrange for linear projection: [B, T/4, F_in/4, 64]
        x = x.permute(0, 2, 3, 1).contiguous()
        
        # Merge feature dimensions
        x = merge_two_last_dims(x)  # [B, T/4, F_in/4 * 64]
        
        # Apply projection to get desired output dimension
        x = self.projection(x)  # [B, T/4, out_features]
        
        return x

class TransformerEncoderLayer(nn.Module):
    """
    Single layer of the Transformer Encoder.
    Multi-Head Self-Attention followed by Feed-Forward network.
    Includes LayerNorm and Dropout with pre-norm architecture.
    """
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        # Attention mechanism
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feed_forward_dim, embed_dim)
        )
        
        # Layer normalization (applied before attention and ffn - "pre-norm")
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input [B, T, D].
            mask (torch.Tensor, optional): Attention mask.
        Returns:
            torch.Tensor: Output [B, T, D].
        """
        # Pre-norm architecture for better training stability
        # 1. Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(
            x_norm, x_norm, x_norm, 
            key_padding_mask=mask, 
            attn_mask=None
        )
        x = x + self.dropout1(attn_output)
        
        # 2. Feed-forward with residual connection
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + self.dropout2(ffn_output)
        
        return x

class TransformerDecoderLayer(nn.Module):
    """
    Single layer of the Transformer Decoder.
    Masked Multi-Head Self-Attention, Multi-Head Cross-Attention, Feed-Forward.
    Uses pre-norm architecture for better training stability.
    """
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        # Self-attention mechanism
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        
        # Cross-attention mechanism
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feed_forward_dim, embed_dim)
        )
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        # Dropout for residual connections
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor = None,
                memory_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None
               ) -> torch.Tensor:
        """
        Args:
            tgt (torch.Tensor): Target sequence [B, T_dec, D].
            memory (torch.Tensor): Encoder output [B, T_enc, D].
            tgt_mask (torch.Tensor, optional): Causal mask for self-attention [T_dec, T_dec].
            memory_key_padding_mask (torch.Tensor, optional): Padding mask for encoder output [B, T_enc].
            tgt_key_padding_mask (torch.Tensor, optional): Padding mask for target input [B, T_dec].
        Returns:
            torch.Tensor: Output [B, T_dec, D].
        """
        # 1. Self-attention with pre-norm
        tgt_norm = self.norm1(tgt)
        self_attn_output, _ = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(self_attn_output)
        
        # 2. Cross-attention with pre-norm
        tgt_norm = self.norm2(tgt)
        cross_attn_output, _ = self.cross_attn(
            tgt_norm, memory, memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(cross_attn_output)
        
        # 3. Feed-forward with pre-norm
        tgt_norm = self.norm3(tgt)
        ffn_output = self.ffn(tgt_norm)
        tgt = tgt + self.dropout3(ffn_output)
        
        return tgt

class Transformer(nn.Module):
    """
    Transformer model for Handwriting Recognition.
    Combines Encoder and Decoder with improved architecture.
    """
    def __init__(self,
                 num_hid: int = 256,           # Embedding dimension
                 num_head: int = 8,            # Number of attention heads
                 num_feed_forward: int = 1024,  # Hidden dim in FFN
                 input_features: int = 20,     # Dimension of input features
                 target_maxlen: int = 100,     # Max length for target sequence
                 num_layers_enc: int = 3,      # Number of encoder layers
                 num_layers_dec: int = 3,      # Number of decoder layers
                 num_classes: int = 98,        # Vocabulary size
                 dropout_rate: float = 0.1     # Dropout rate
                ):
        super().__init__()
        self.num_hid = num_hid
        self.num_classes = num_classes
        self.target_maxlen = target_maxlen
        self.pad_token_id = 0
        
        logger.info("Initializing Transformer Model with improved architecture...")
        logger.info(f"  Embed Dim: {num_hid}")
        logger.info(f"  Num Heads: {num_head}")
        logger.info(f"  FFN Dim: {num_feed_forward}")
        logger.info(f"  Input Features: {input_features}")
        logger.info(f"  Target Maxlen: {target_maxlen}")
        logger.info(f"  Encoder Layers: {num_layers_enc}")
        logger.info(f"  Decoder Layers: {num_layers_dec}")
        logger.info(f"  Output Classes: {num_classes}")
        logger.info(f"  Dropout Rate: {dropout_rate}")
        
        # Encoder Input: Conv2D Subsampling
        self.encoder_input_layer = Conv2dSubsampling(
            in_features=input_features,
            out_features=num_hid,
            dropout_rate=dropout_rate
        )
        
        # Decoder Input: Token + Positional Embedding
        self.decoder_input_layer = TokenEmbedding(
            num_vocab=num_classes,
            maxlen=target_maxlen,
            embed_dim=num_hid,
            dropout_rate=dropout_rate
        )
        
        # Encoder Stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                num_hid, num_head, num_feed_forward, dropout_rate
            ) for _ in range(num_layers_enc)
        ])
        
        # Encoder final norm
        self.encoder_norm = nn.LayerNorm(num_hid)
        
        # Decoder Stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                num_hid, num_head, num_feed_forward, dropout_rate
            ) for _ in range(num_layers_dec)
        ])
        
        # Decoder final norm
        self.decoder_norm = nn.LayerNorm(num_hid)
        
        # Final Classifier Layer
        self.classifier = nn.Linear(num_hid, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info("Transformer Model initialized successfully.")

    def _initialize_weights(self):
        """Applies Xavier uniform initialization to linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.num_hid**-0.5)
                if module.padding_idx is not None:
                    nn.init.constant_(module.weight[module.padding_idx], 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generates a square causal mask for decoder self-attention."""
        # Create boolean mask: True = ignore position (upper triangular)
        mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)
        return mask

    def _create_padding_mask(self, sequence: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """Creates a padding mask (True where padded)."""
        return (sequence == pad_token_id)

    def encode(self, src_Transformer: torch.Tensor, src_Transformer_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encodes the source sequence.
        Args:
            src_Transformer (torch.Tensor): Source features [B, T_src_Transformer, F_in].
            src_Transformer_padding_mask (torch.Tensor, optional): Padding mask for source [B, T_src_Transformer].
        Returns:
            torch.Tensor: Encoder output [B, T_out, D].
        """
        # Apply convolutional subsampling
        x = self.encoder_input_layer(src_Transformer)
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask=src_Transformer_padding_mask)
        
        # Apply final normalization
        x = self.encoder_norm(x)
        
        return x

    def decode(self,
               tgt: torch.Tensor,
               memory: torch.Tensor,
               tgt_mask: torch.Tensor,
               memory_key_padding_mask: torch.Tensor = None,
               tgt_key_padding_mask: torch.Tensor = None
              ) -> torch.Tensor:
        """
        Decodes using target sequence and encoder memory.
        Args:
            tgt (torch.Tensor): Target token IDs [B, T_dec].
            memory (torch.Tensor): Encoder output [B, T_enc_out, D].
            tgt_mask (torch.Tensor): Causal mask [T_dec, T_dec].
            memory_key_padding_mask (torch.Tensor, optional): Encoder output padding mask [B, T_enc_out].
            tgt_key_padding_mask (torch.Tensor, optional): Target padding mask [B, T_dec].
        Returns:
            torch.Tensor: Decoder output logits [B, T_dec, num_classes].
        """
        # Get token embeddings
        y = self.decoder_input_layer(tgt)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            y = layer(
                tgt=y,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
        
        # Apply final normalization
        y = self.decoder_norm(y)
        
        # Apply classifier to get logits
        logits = self.classifier(y)
        
        return logits

    def forward(self, src_Transformer: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for training.
        Args:
            src_Transformer (torch.Tensor): Source features [B, T_src_Transformer, F_in].
            tgt (torch.Tensor): Target token IDs (shifted right) [B, T_dec].
        Returns:
            torch.Tensor: Output logits [B, T_dec, num_classes].
        """
        # Create masks
        tgt_seq_len = tgt.size(1)
        causal_mask = self._generate_causal_mask(tgt_seq_len, tgt.device)
        tgt_padding_mask = self._create_padding_mask(tgt, self.pad_token_id)
        
        # Encode source
        memory = self.encode(src_Transformer)
        
        # Decode target
        logits = self.decode(
            tgt, memory, causal_mask, 
            memory_key_padding_mask=None,
            tgt_key_padding_mask=tgt_padding_mask
        )
        
        return logits

    # Replace the generate method in model.py (around line 545)
    @torch.no_grad()
    def generate(self,
                src_Transformer: torch.Tensor,
                start_token_idx: int,
                end_token_idx: int,
                max_len: int = None,
                temperature: float = 0.7,
                top_k: int = 10,
                beam_size: int = 3
            ) -> torch.Tensor:
        """
        Generates target sequence with improved handling of end tokens.
        """
        self.eval()
        device = src_Transformer.device
        batch_size = src_Transformer.size(0)
        
        if max_len is None:
            max_len = self.target_maxlen
        
        # Encode source sequence
        memory = self.encode(src_Transformer)
        
        # Use beam search if beam_size > 1
        if beam_size > 1:
            return self._beam_search_decode(
                memory, start_token_idx, end_token_idx, max_len, beam_size
            )
        
        # Initialize with start token
        decoded_ids = torch.full(
            (batch_size, 1), start_token_idx, dtype=torch.long, device=device
        )
        
        # Track which sequences have finished generating
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens one by one
        for step in range(max_len - 1):
            # Skip if all sequences are finished
            if finished_sequences.all():
                break
                
            # Get current sequence length
            current_len = decoded_ids.size(1)
            
            # Create masks
            causal_mask = self._generate_causal_mask(current_len, device)
            tgt_padding_mask = self._create_padding_mask(decoded_ids, self.pad_token_id)
            
            # Get logits for next token
            logits = self.decode(
                decoded_ids, memory, causal_mask, 
                memory_key_padding_mask=None,
                tgt_key_padding_mask=tgt_padding_mask
            )
            
            # Get logits for just the last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Mask logits for finished sequences to force padding token
            next_token_logits = next_token_logits.masked_fill(
                finished_sequences.unsqueeze(1), float('-inf')
            )
            next_token_logits[:, self.pad_token_id] = next_token_logits[:, self.pad_token_id].masked_fill(
                finished_sequences, 0.0
            )
            
            # Apply top-k sampling
            if top_k > 0:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, indices, values)
            
            # Convert to probabilities and sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Mark sequences that generated an end token as finished
            newly_finished = (next_token == end_token_idx)
            finished_sequences = finished_sequences | newly_finished
            
            # Append to sequence
            decoded_ids = torch.cat([decoded_ids, next_token.unsqueeze(1)], dim=1)
        
        # Clean up sequences - replace tokens after the first end token with pad tokens
        cleaned_sequences = []
        for i, seq in enumerate(decoded_ids):
            # Find first end token position
            end_positions = (seq == end_token_idx).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                # Get position of first end token
                first_end = end_positions[0].item()
                # Keep sequence up to and including first end token, pad the rest
                cleaned_seq = torch.cat([
                    seq[:first_end+1],
                    torch.full((len(seq) - first_end - 1,), self.pad_token_id, device=device)
                ])
                cleaned_sequences.append(cleaned_seq)
            else:
                cleaned_sequences.append(seq)
        
        return torch.stack(cleaned_sequences)

    def _beam_search_decode(self, memory, start_token_idx, end_token_idx, max_len, beam_width):
        """
        Implements beam search decoding with proper length handling.
        """
        device = memory.device
        batch_size = memory.size(0)
        
        # Initialize with batch_size sequences, each containing just start_token
        beams = []
        for b in range(batch_size):
            beams.append([{
                "sequence": torch.tensor([[start_token_idx]], device=device),
                "score": 0.0,
                "finished": False
            }])
        
        # Process each sequence in the batch
        result_sequences = []
        max_result_len = 0  # Track maximum sequence length for padding
        
        for b in range(batch_size):
            memory_slice = memory[b:b+1]  # [1, T_enc, D]
            current_beams = beams[b]
            
            # Generate tokens
            for step in range(max_len - 1):
                candidates = []
                
                # Process each beam
                for beam in current_beams:
                    if beam["finished"]:
                        candidates.append(beam)
                        continue
                    
                    # Get current sequence
                    seq = beam["sequence"]  # [1, t]
                    
                    # Create masks
                    current_len = seq.size(1)
                    causal_mask = self._generate_causal_mask(current_len, device)
                    tgt_padding_mask = self._create_padding_mask(seq, self.pad_token_id)
                    
                    # Get logits for next token
                    logits = self.decode(
                        seq, memory_slice, causal_mask, 
                        memory_key_padding_mask=None,
                        tgt_key_padding_mask=tgt_padding_mask
                    )
                    
                    # Get logits for just the last token
                    next_token_logits = logits[0, -1, :]  # [vocab_size]
                    
                    # Convert to log probabilities
                    log_probs = F.log_softmax(next_token_logits, dim=-1)
                    
                    # Get top-k next tokens
                    values, indices = torch.topk(log_probs, beam_width)
                    
                    # Create new candidates
                    for i in range(beam_width):
                        token = indices[i].unsqueeze(0).unsqueeze(0)  # [1, 1]
                        log_prob = values[i].item()
                        
                        # Create new sequence by extending current one
                        new_seq = torch.cat([seq, token], dim=1)  # [1, t+1]
                        new_score = beam["score"] + log_prob
                        
                        # Length normalization to prevent bias toward shorter sequences
                        # Dividing by length^alpha (alpha=0.7 is empirically good)
                        length_penalty = ((new_seq.size(1))**0.7)
                        normalized_score = new_score / length_penalty
                        
                        # Check if sequence is finished
                        finished = (token.item() == end_token_idx)
                        
                        candidates.append({
                            "sequence": new_seq,
                            "score": new_score,
                            "normalized_score": normalized_score,
                            "finished": finished
                        })
                
                # Sort candidates by normalized score and select top beams
                current_beams = sorted(candidates, key=lambda x: x["normalized_score"], reverse=True)[:beam_width]
                
                # Check if all beams are finished
                if all(beam["finished"] for beam in current_beams):
                    break
            
            # Get the highest scoring beam
            best_seq = current_beams[0]["sequence"].squeeze(0)  # Remove batch dimension: [t]
            
            # Find first end token position and truncate/clean sequence
            end_positions = (best_seq == end_token_idx).nonzero(as_tuple=True)[0]
            if len(end_positions) > 0:
                # Get position of first end token (include the end token)
                first_end = end_positions[0].item() + 1
                # Keep sequence up to and including first end token
                cleaned_seq = best_seq[:first_end]
            else:
                cleaned_seq = best_seq
            
            # Update max sequence length for padding later
            max_result_len = max(max_result_len, len(cleaned_seq))
            result_sequences.append(cleaned_seq)
        
        # Ensure all sequences have the same length by padding
        padded_sequences = []
        for seq in result_sequences:
            if len(seq) < max_result_len:
                # Pad sequence to match max_result_len
                padding = torch.full((max_result_len - len(seq),), self.pad_token_id, 
                                    dtype=torch.long, device=device)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq.unsqueeze(0))  # Add batch dimension back
        
        # Now all sequences have same length and can be safely concatenated
        return torch.cat(padded_sequences, dim=0)  # [B, max_result_len]