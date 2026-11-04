import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR  = _THIS_DIR.parents[1] 
sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertJapaneseTokenizer
from training.dual_head.models.model_factory import CrowdDensityModel
from common.utils.dataset import VELOCITY_MEANS, NUM_VELOCITY_CLASSES
from common.utils.preprocess import clean_text

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
CHECKPOINT_PATH = _THIS_DIR / 'checkpoints_velocity' / 'model_epoch10.pt'
MAX_LENGTH = 128


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device) -> CrowdDensityModel:
    model = CrowdDensityModel(freeze_bert=True, num_classes=NUM_VELOCITY_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(text: str, model: CrowdDensityModel, tokenizer, device) -> dict:
    text = clean_text(text)
    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids      = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits, head_var = model(input_ids, attention_mask)

    probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_class = int(np.argmax(probs))
    v_mean = float(VELOCITY_MEANS[pred_class])
    dist_var = float(np.dot(probs, (VELOCITY_MEANS - v_mean) ** 2))
    v_var = dist_var + float(head_var.cpu().numpy())

    return {
        'velocity_mean': v_mean,
        'velocity_var':  v_var,
    }


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(CHECKPOINT_PATH, device)

    print("歩行速度推定システム（終了: q）")
    while True:
        text = input("\nつぶやきを入力: ")
        if text.strip().lower() == 'q':
            break
        result = predict(text, model, tokenizer, device)
        print(f"推定速度: {result['velocity_mean']:.3f} m/s  (分散: {result['velocity_var']:.4f})")
