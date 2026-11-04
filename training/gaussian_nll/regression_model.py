import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.models.encoder import BertEncoder


class RegressionModel(nn.Module):
    def __init__(self, freeze_bert=True, hidden_dim=256):
        super().__init__()
        self.encoder = BertEncoder(freeze=freeze_bert)
        self.shared = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.mean_head = nn.Linear(128, 1)
        self.log_var = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask)
        h = self.shared(features)
        mean = self.mean_head(h).squeeze(-1)
        return mean
