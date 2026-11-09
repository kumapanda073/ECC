import torch.nn as nn
from common.utils.dataset import NUM_DENSITY_CLASSES


class CrowdRegressor(nn.Module):

    def __init__(self, input_dim: int = 768, hidden1_dim: int = 256, hidden2_dim: int = 128,
                 hidden3_dim: int = 32, num_classes: int = None, stop_gradient: bool = True):
        super().__init__()
        out_dim = num_classes if num_classes is not None else NUM_DENSITY_CLASSES
        self.stop_gradient = stop_gradient

        # 共有 MLP
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        # 分類ヘッド
        self.classification_head = nn.Linear(hidden2_dim, out_dim)
        self.variance_head = nn.Sequential(
            nn.Linear(hidden2_dim, hidden3_dim),
            nn.ReLU(),
            nn.Linear(hidden3_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        g = self.shared(x)
        logits = self.classification_head(g)
        g_var = g.detach() if self.stop_gradient else g
        variance = self.variance_head(g_var).squeeze(-1)
        return logits, variance
