import torch.nn as nn
from common.models.encoder import BertEncoder
from training.dual_head.models.regressor import CrowdRegressor


class CrowdDensityModel(nn.Module):
    def __init__(self, freeze_bert: bool = True, hidden1_dim: int = 256, num_classes: int = None):
        super().__init__()
        self.encoder = BertEncoder(freeze=freeze_bert)
        self.regressor = CrowdRegressor(hidden1_dim=hidden1_dim, num_classes=num_classes)

    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask)
        return self.regressor(features)
