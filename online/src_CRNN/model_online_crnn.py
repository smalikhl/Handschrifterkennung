
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Lokale Imports
import config # Lädt die ONLINE Konfiguration
# Importiere utils relativ
try:
    import utils as online_utils
except ImportError:
    import utils as online_utils 
    logging.warning("Relativer Import für utils in model_online_crnn fehlgeschlagen. Versuche direkten Import.")


logger = logging.getLogger(__name__)

class BidirectionalLSTM(nn.Module):
    """ Standard BiLSTM-Schicht mit Linearer Projektion. (Unverändert) """
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
    def forward(self, x):
        # self.rnn.flatten_parameters() # Kann bei DP helfen, sonst oft nicht nötig
        rnn_output, _ = self.rnn(x)
        if self.dropout: rnn_output = self.dropout(rnn_output)
        return self.fc(rnn_output)

class CRNN(nn.Module):
    """
    Angepasstes CRNN für ONLINE Daten (Merkmals-Pseudo-Bilder).
    Nimmt Input [B, C, H, W] wobei C=1, H=FEATURE_DIM, W=MAX_SEQ_LEN.
    """
    def __init__(self, num_classes=config.NUM_CLASSES, rnn_hidden_size=256, rnn_dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_dropout = rnn_dropout
        self.cnn_input_channels = config.CNN_INPUT_CHANNELS
        self.cnn_input_height = config.CNN_INPUT_HEIGHT
        self.cnn_input_width = config.CNN_INPUT_WIDTH

        cnn_layers = []
        current_channels = self.cnn_input_channels
        layer_config = [
            {'type': 'conv', 'out': 64, 'k': 3, 's': 1, 'p': 1}, {'type': 'bn'}, {'type': 'relu'},
            {'type': 'pool', 'k': 2, 's': 2}, # H->H/2, W->W/2
            {'type': 'conv', 'out': 128, 'k': 3, 's': 1, 'p': 1}, {'type': 'bn'}, {'type': 'relu'},
            {'type': 'pool', 'k': 2, 's': 2}, # H->H/4, W->W/4
            {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1}, {'type': 'bn'}, {'type': 'relu'},
            {'type': 'conv', 'out': 256, 'k': 3, 's': 1, 'p': 1}, {'type': 'bn'}, {'type': 'relu'},
            {'type': 'pool', 'k': (2, 1), 's': (2, 1)}, # H->H/8, W->W/4
            {'type': 'conv', 'out': 512, 'k': 3, 's': 1, 'p': 1}, {'type': 'bn'}, {'type': 'relu'},
            {'type': 'conv', 'out': 512, 'k': 3, 's': 1, 'p': 1}, {'type': 'bn'}, {'type': 'relu'},
            {'type': 'dropout', 'p': 0.2},
            {'type': 'pool', 'k': (2, 1), 's': (2, 1)}, # H->H/16, W->W/4
            {'type': 'conv', 'out': 512, 'k': 3, 's': 1, 'p': 1}, {'type': 'bn'}, {'type': 'relu'},
            {'type': 'dropout', 'p': 0.2}
        ]
        for layer_cfg in layer_config:
            l_type = layer_cfg['type']
            if l_type == 'conv': cnn_layers.append(nn.Conv2d(current_channels, layer_cfg['out'], layer_cfg['k'], layer_cfg['s'], layer_cfg['p'])); current_channels = layer_cfg['out']
            elif l_type == 'bn': cnn_layers.append(nn.BatchNorm2d(current_channels))
            elif l_type == 'relu': cnn_layers.append(nn.ReLU(inplace=True))
            elif l_type == 'pool': cnn_layers.append(nn.MaxPool2d(layer_cfg['k'], layer_cfg['s']))
            elif l_type == 'dropout': cnn_layers.append(nn.Dropout(layer_cfg['p']))
        self.cnn_part1 = nn.Sequential(*cnn_layers)

        dummy_input = torch.zeros(1, self.cnn_input_channels, self.cnn_input_height, self.cnn_input_width)
        logger.info(f"CRNN: Dummy Input Shape für CNN-Check: {dummy_input.shape}")
        with torch.no_grad(): cnn_out_part1 = self.cnn_part1(dummy_input)
        _b, cnn_out_channels, cnn_out_h, cnn_out_w = cnn_out_part1.size()
        logger.info(f"CRNN: Shape nach CNN Part 1: B={_b}, C={cnn_out_channels}, H={cnn_out_h}, W={cnn_out_w}")

        if cnn_out_h <= 0:
             logger.error(f"FATAL: CNN-Ausgabehöhe ist <= 0 ({cnn_out_h}).")
             raise ValueError("CNN Ausgabehöhe <= 0")
        final_cnn_kernel_h = cnn_out_h
        final_cnn_out_channels = 512
        self.cnn_part2 = nn.Sequential(
             nn.Conv2d(cnn_out_channels, final_cnn_out_channels,
                       kernel_size=(final_cnn_kernel_h, 2), stride=(1, 1), padding=(0, 1)),
             nn.BatchNorm2d(final_cnn_out_channels),
             nn.ReLU(inplace=True)
        )
        current_channels = final_cnn_out_channels

        with torch.no_grad(): final_cnn_output = self.cnn_part2(cnn_out_part1)
        _b, self.final_cnn_channels, self.cnn_output_height, self.cnn_output_width = final_cnn_output.size()
        logger.info(f"CRNN: Shape nach CNN Part 2: B={_b}, C={self.final_cnn_channels}, H={self.cnn_output_height}, W={self.cnn_output_width}")
        if self.cnn_output_height != 1:
            logger.error(f"FATAL: Finale CNN-Ausgabehöhe ist {self.cnn_output_height} (erwartet 1).")
            raise ValueError("Finale CNN Ausgabehöhe ist nicht 1")

        self.rnn_input_size = self.final_cnn_channels
        self.sequence_length = self.cnn_output_width
        logger.info(f"CRNN: Berechnete RNN Input Features: {self.rnn_input_size}")
        logger.info(f"CRNN: Berechnete RNN/CTC Sequenzlänge (aus CNN Breite): {self.sequence_length}")

        self.rnn = nn.Sequential(
            BidirectionalLSTM(self.rnn_input_size, self.rnn_hidden_size, self.rnn_hidden_size, dropout_prob=self.rnn_dropout),
            BidirectionalLSTM(self.rnn_hidden_size, self.rnn_hidden_size, self.num_classes, dropout_prob=self.rnn_dropout)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                 for name, param in m.named_parameters():
                      if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
                      elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
                      elif 'bias' in name:
                          param.data.fill_(0)
                          # Optional: Forget Gate Bias auf 1 setzen
                          # n = param.size(0); start, end = n // 4, n // 2
                          # param.data[start:end].fill_(1.)

    def forward(self, x):
        logger.debug(f"CRNN Input Shape: {x.shape}")
        conv_features_p1 = self.cnn_part1(x)
        conv_features = self.cnn_part2(conv_features_p1)
        logger.debug(f"CRNN Shape nach CNN: {conv_features.shape}")
        batch, channels, height, width = conv_features.size()
        if height != 1: raise ValueError(f"Höhe nach CNN ist {height}, muss 1 sein!")
        rnn_input = conv_features.squeeze(2).permute(0, 2, 1).contiguous()
        logger.debug(f"CRNN Shape für RNN Input: {rnn_input.shape}")
        rnn_output = self.rnn(rnn_input)
        logger.debug(f"CRNN Shape nach RNN: {rnn_output.shape}")
        output_ctc = rnn_output.permute(1, 0, 2)
        logger.debug(f"CRNN Finale Output Shape für CTC: {output_ctc.shape}")
        return output_ctc

def build_online_crnn_model():
    """ Baut das Online-CRNN-Modell. """
    logger.info("Initialisiere ONLINE CRNN-Modell...")
    try:
        model = CRNN(
            num_classes=config.NUM_CLASSES,
            rnn_hidden_size=256,
            rnn_dropout=0.3
        )
        logger.info("Online CRNN-Modell erfolgreich initialisiert.")
        logger.info(f"  Input H={model.cnn_input_height}, W={model.cnn_input_width}, C={model.cnn_input_channels}")
        logger.info(f"  Klassen (inkl. Blank): {config.NUM_CLASSES}, RNN Hidden: {model.rnn_hidden_size}, SeqLen (CNN out): {model.sequence_length}")
        try:
            num_params = sum(p.numel() for p in model.parameters())
            num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"  Parameter: {num_params:,} (Trainierbar: {num_trainable:,})")
            # Nutze utils zum Erstellen des Verzeichnisses
            online_utils.create_directory(config.CHECKPOINT_PATH)
            arch_path = os.path.join(config.CHECKPOINT_PATH, "model_architecture_online.txt")
            with open(arch_path, 'w') as f:
                f.write(str(model))
                f.write("\n\n--- Parameters ---\n")
                f.write(f"Total: {num_params:,}\nTrainable: {num_trainable:,}\n")
            logger.info(f"  Modellarchitektur gespeichert: {arch_path}")
        except Exception as arch_e: logger.warning(f"Modellzusammenfassung nicht speicherbar: {arch_e}")
        return model
    except Exception as e:
        logger.exception(f"Schwerwiegender Fehler beim Erstellen des Online-CRNN-Modells: {e}", exc_info=True)
        return None
