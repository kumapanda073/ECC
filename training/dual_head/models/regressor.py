import torch.nn as nn
from common.utils.dataset import NUM_DENSITY_CLASSES


class CrowdRegressor(nn.Module):
    def __init__(self, input_dim: int = 768, hidden1_dim: int = 256, hidden2_dim: int = 128, hidden3_dim: int = 32, num_classes: int = None):
        super().__init__()
        out_dim = num_classes if num_classes is not None else NUM_DENSITY_CLASSES
        # 共有層なし: 分類ヘッドと分散ヘッドを完全分離
        self.density_head = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(0.2),        #過学習を防ぐ
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2_dim, out_dim),
        )
        self.variance_head = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.ReLU(),
            nn.Linear(hidden3_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        logits = self.density_head(x)
        variance = self.variance_head(x).squeeze(-1)
        return logits, variance
