from pathlib import Path
import sys
import random
import numpy as np
import torch
from torch.optim import Adam
from transformers import BertJapaneseTokenizer

_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR  = _THIS_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR))

from training.gaussian_nll.regression_model import RegressionModel
from training.gaussian_nll.regression_dataset import get_dataloader

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
DATA_PATH = ROOT_DIR / 'data' / 'train_den.json'
VAL_PATH  = ROOT_DIR / 'data' / 'val_den.json'
CHECKPOINT_DIR = _THIS_DIR / 'checkpoints_regression_density'
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
SEED = 0


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def nll_loss(model, mean_pred, target, is_null):
    m = ~is_null
    if not m.any():
        return None
    log_var = model.log_var
    return (0.5 * (log_var + (target[m] - mean_pred[m]) ** 2 / log_var.exp())).mean()


def run_epoch(model, loader, device, optimizer=None):
    total, n = 0.0, 0
    for batch in loader:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target         = batch['target'].to(device)
        is_null        = batch['is_null'].to(device)

        if optimizer is not None:
            optimizer.zero_grad()
        mean_pred = model(input_ids, attention_mask)
        loss = nll_loss(model, mean_pred, target, is_null)
        if loss is None:
            continue
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def train():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    dataloader     = get_dataloader(DATA_PATH, tokenizer, label_field="density", batch_size=BATCH_SIZE)
    val_dataloader = get_dataloader(VAL_PATH,  tokenizer, label_field="density", batch_size=BATCH_SIZE, shuffle=False)

    model = RegressionModel(freeze_bert=True).to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_val = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = run_epoch(model, dataloader, device, optimizer)
        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(model, val_dataloader, device)

        var = model.log_var.exp().item()
        print(f"Epoch {epoch}/{EPOCHS} - train {train_loss:.4f} / val {val_loss:.4f} / 学習した分散 {var:.5f}")

        torch.save(model.state_dict(), CHECKPOINT_DIR / f"model_epoch{epoch}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "model_best.pt")
            print(f"  → val 最良 (model_best.pt を更新)")


if __name__ == "__main__":
    train()