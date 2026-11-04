import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR  = _THIS_DIR.parents[1]
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'
from sklearn.metrics import f1_score
from transformers import BertJapaneseTokenizer
from training.dual_head.models.model_factory import CrowdDensityModel
from common.utils.dataset import (
    NUM_DENSITY_CLASSES, NUM_VELOCITY_CLASSES,
    DENSITY_MEANS, VELOCITY_MEANS,
    density_to_label, velocity_to_label,
)
from common.utils.preprocess import clean_text

FIGURE_DIR = _THIS_DIR / 'figure'

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
DENSITY_CHECKPOINT  = _THIS_DIR / 'checkpoints' / 'model_epoch10.pt'
VELOCITY_CHECKPOINT = _THIS_DIR / 'checkpoints_velocity' / 'model_epoch10.pt'
DENSITY_VAL_PATH    = ROOT_DIR / 'data' / 'val_den.json'
VELOCITY_VAL_PATH   = ROOT_DIR / 'data' / 'val_vel.json'


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(checkpoint, val_path, label_field, num_classes, to_label_fn, class_means, device, tokenizer):
    model = CrowdDensityModel(freeze_bert=True, num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    with open(val_path, encoding="utf-8") as f:
        data = json.load(f)

    true_labels, pred_labels = [], []

    for item in data:
        val = item.get(label_field)
        if val is None:
            continue
        true_label = to_label_fn(val)
        text = clean_text(item["text"])
        enc = tokenizer(text, max_length=128, padding="max_length",
                        truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits, _ = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        pred_labels.append(int(torch.argmax(logits, dim=-1)))
        true_labels.append(true_label)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    accuracy = (true_labels == pred_labels).mean()

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t, p] += 1

    return accuracy, cm, true_labels, pred_labels


def plot_confusion_matrix(label_field, accuracy, cm, class_means):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_means)
    tick_labels = [f"{v:.2f}" for v in class_means]

    norm_cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm_cm, vmin=0, vmax=1, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Proportion of Predictions")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(f"{label_field.upper()}  Accuracy {accuracy*100:.1f}%")

    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if norm_cm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]}\n({norm_cm[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=9, color=color)

    plt.tight_layout()
    out_path = FIGURE_DIR / f"confusion_matrix_{label_field}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → 保存: {out_path}")


def print_results(label_field, accuracy, cm, class_means, true_labels, pred_labels):
    num_classes = len(class_means)
    total = cm.sum()
    correct = int((true := np.diag(cm)).sum())
    print(f"\n{'='*55}")
    print(f"  {label_field.upper()} Model Evaluation")
    print(f"{'='*55}")
    print(f"  Overall Accuracy: {accuracy*100:.1f}%  ({correct}/{total} correct predictions)")

    per_class_f1 = f1_score(true_labels, pred_labels, average=None, labels=list(range(num_classes)))
    macro_f1     = f1_score(true_labels, pred_labels, average='macro')
    print(f"  Macro F1: {macro_f1:.3f}")

    print(f"\n  Confusion Matrix  (行=正解クラス, 列=予測クラス)")
    header = "        " + "".join(f" Pred{i}" for i in range(num_classes))
    print(header)
    for i in range(num_classes):
        row = f"  True{i} "
        for j in range(num_classes):
            cell = f"[{cm[i,j]:3d}]" if i == j else f"  {cm[i,j]:3d} "
            row += cell
        per_class_acc = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(row + f"   Acc {per_class_acc*100:.0f}%  F1 {per_class_f1[i]:.2f}  (≈{class_means[i]:.2f})")

    return per_class_f1, macro_f1


def plot_f1(label_field, per_class_f1, macro_f1, class_means):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    tick_labels = [f"{v:.2f}" for v in class_means]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(class_means)), per_class_f1, color="steelblue")
    ax.axhline(macro_f1, color="tomato", linestyle="--", label=f"Macro F1: {macro_f1:.3f}")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(class_means)))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"{label_field.upper()}  Per-class F1 score")
    ax.legend()

    for bar, val in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    out_path = FIGURE_DIR / f"f1_{label_field}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → 保存: {out_path}")


def main():
    device = get_device()
    print(f"device: {device}")
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

    acc, cm, true_labels, pred_labels = evaluate(
        DENSITY_CHECKPOINT, DENSITY_VAL_PATH,
        "density", NUM_DENSITY_CLASSES, density_to_label, DENSITY_MEANS,
        device, tokenizer,
    )
    per_class_f1, macro_f1 = print_results("density", acc, cm, DENSITY_MEANS, true_labels, pred_labels)
    plot_confusion_matrix("density", acc, cm, DENSITY_MEANS)
    plot_f1("density", per_class_f1, macro_f1, DENSITY_MEANS)

    acc, cm, true_labels, pred_labels = evaluate(
        VELOCITY_CHECKPOINT, VELOCITY_VAL_PATH,
        "velocity", NUM_VELOCITY_CLASSES, velocity_to_label, VELOCITY_MEANS,
        device, tokenizer,
    )
    per_class_f1, macro_f1 = print_results("velocity", acc, cm, VELOCITY_MEANS, true_labels, pred_labels)
    plot_confusion_matrix("velocity", acc, cm, VELOCITY_MEANS)
    plot_f1("velocity", per_class_f1, macro_f1, VELOCITY_MEANS)


if __name__ == "__main__":
    main()
