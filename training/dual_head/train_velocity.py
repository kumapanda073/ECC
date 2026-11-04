import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR  = _THIS_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR))

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import BertJapaneseTokenizer
from training.dual_head.models.model_factory import CrowdDensityModel
from common.utils.dataset import get_dataloader, NUM_VELOCITY_CLASSES

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
DATA_PATH = ROOT_DIR / 'data' / 'train_vel.json'
VAL_PATH  = ROOT_DIR / 'data' / 'val_vel.json'
# Deep Ensembles 用: 環境変数 SEED を指定するとシードを固定し、
# メンバーごとに別フォルダ (checkpoints_velocity_ens/memberN) へ保存する。
# 未指定なら従来通り checkpoints_velocity/ に保存する (既存の挙動)。
SEED = os.environ.get("SEED")
if SEED is not None:
    SEED = int(SEED)
    CHECKPOINT_DIR = _THIS_DIR / "checkpoints_velocity_ens" / f"member{SEED}"
else:
    CHECKPOINT_DIR = _THIS_DIR / "checkpoints_velocity"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
NULL_WEIGHT = 1.0
VARIANCE_WEIGHT = 1.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train():
    device = get_device()
    print(f"Using device: {device}")
    if SEED is not None:
        set_seed(SEED)
        print(f"[Ensemble] SEED={SEED} -> {CHECKPOINT_DIR}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    dataloader     = get_dataloader(DATA_PATH, tokenizer, label_field="velocity", batch_size=BATCH_SIZE)
    val_dataloader = get_dataloader(VAL_PATH,  tokenizer, label_field="velocity", batch_size=BATCH_SIZE, shuffle=False)

    def combined_loss(logits, variance_pred, soft_labels, target_variance, is_null):
        log_probs = F.log_softmax(logits, dim=-1)
        per_sample_loss = -(soft_labels * log_probs).sum(dim=-1)
        weights = torch.where(is_null, torch.tensor(NULL_WEIGHT, device=logits.device), torch.ones(1, device=logits.device))
        cls_loss = (weights * per_sample_loss).mean()
        var_loss = F.mse_loss(variance_pred, target_variance)
        return cls_loss + VARIANCE_WEIGHT * var_loss

    model = CrowdDensityModel(freeze_bert=True, num_classes=NUM_VELOCITY_CLASSES).to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in dataloader:
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            soft_labels     = batch['soft_label'].to(device)
            is_null         = batch['is_null'].to(device)
            target_variance = batch['target_variance'].to(device)

            optimizer.zero_grad()
            logits, variance_pred = model(input_ids, attention_mask)
            loss = combined_loss(logits, variance_pred, soft_labels, target_variance, is_null)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids       = batch['input_ids'].to(device)
                attention_mask  = batch['attention_mask'].to(device)
                soft_labels     = batch['soft_label'].to(device)
                is_null         = batch['is_null'].to(device)
                target_variance = batch['target_variance'].to(device)
                logits, variance_pred = model(input_ids, attention_mask)
                val_loss += combined_loss(logits, variance_pred, soft_labels, target_variance, is_null).item()
        avg_val_loss = val_loss / len(val_dataloader)

        print(f"Epoch {epoch}/{EPOCHS} - train loss: {avg_loss:.4f} / val loss: {avg_val_loss:.4f}")

        ckpt_path = CHECKPOINT_DIR / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    train()
