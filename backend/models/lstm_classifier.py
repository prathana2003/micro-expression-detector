"""
Phase 2 — Temporal Micro-Expression / Cognitive Load Model (scaffold)

This is intentionally a *skeleton*: it defines the model shape and the data
contract so you can plug in real training once you have CASME II / SAMM
(or your own logged data from phase1_blink_and_au_poc.py) preprocessed.

Input contract:
    x: FloatTensor of shape (batch, seq_len, num_features)
       seq_len   = number of frames in the sliding window (e.g. 15-30)
       num_features = per-frame feature vector length. If you extend
                       phase1's [ear_avg, au4_brow_dist, au12_mouth_dist]
                       to more AUs, num_features grows accordingly.

Output:
    - expression_logits: (batch, num_classes) — e.g. [neutral, micro-anxiety,
      micro-suppression, ...] depending on how you label your dataset
    - cognitive_load: (batch, 1) — a continuous 0-1 "load" score

Requires: pip install torch  (see requirements.txt, Phase 2 section)
"""

import torch
import torch.nn as nn


class MicroExpressionLSTM(nn.Module):
    def __init__(self, num_features=3, hidden_size=64, num_layers=2,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        lstm_out_dim = hidden_size * 2  # bidirectional

        self.classifier_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

        self.cognitive_load_head = nn.Sequential(
            nn.Linear(lstm_out_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),  # keeps the load score in [0, 1]
        )

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the final timestep's output as the sequence summary.
        last_step = lstm_out[:, -1, :]  # (batch, hidden_size * 2)

        expression_logits = self.classifier_head(last_step)
        cognitive_load = self.cognitive_load_head(last_step)
        return expression_logits, cognitive_load


if __name__ == "__main__":
    # Smoke test with random data — confirms shapes wire up correctly
    # before you touch real data.
    batch_size, seq_len, num_features, num_classes = 8, 30, 3, 4

    model = MicroExpressionLSTM(num_features=num_features, num_classes=num_classes)
    dummy_input = torch.randn(batch_size, seq_len, num_features)

    logits, load_score = model(dummy_input)
    print("expression_logits shape:", logits.shape)      # (8, 4)
    print("cognitive_load shape:   ", load_score.shape)   # (8, 1)
