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
from common.utils.dataset import DENSITY_MEANS
from common.utils.preprocess import clean_text

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
CHECKPOINT_PATH = _THIS_DIR / 'checkpoints' / 'model_epoch10.pt'
MAX_LENGTH = 128


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, device) -> CrowdDensityModel:
    model = CrowdDensityModel(freeze_bert=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(text: str, model: CrowdDensityModel, tokenizer, device) -> dict:
    """
    Returns:
        density_mean : 推定密度(argmaxクラスの中央値)
        dist_var     : 分類ヘッドのみの分散(softmax分布の広がり)
        head_var     : 分散ヘッドの出力
        proposed_var : 提案手法の分散(dist_var + head_var)
        probs        : softmax確率(5クラス)
    """
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

    density_probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_class = int(np.argmax(density_probs))
    d_mean    = float(DENSITY_MEANS[pred_class])
    dist_var  = float(np.dot(density_probs, (DENSITY_MEANS - d_mean) ** 2))  # 分類ヘッドのみ
    head_var  = float(head_var.cpu().numpy())                                # 分散ヘッド

    return {
        'density_mean': d_mean,
        'dist_var':     dist_var,               # 分類ヘッドのみ
        'head_var':     head_var,               # 分散ヘッドの寄与
        'proposed_var': dist_var + head_var,    # 提案(分類ヘッド + 分散ヘッド)
        'probs':        density_probs,
    }


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(CHECKPOINT_PATH, device)

    print("人流状況推定システム（終了: q）")
    while True:
        text = input("\nつぶやきを入力: ")
        if text.strip().lower() == 'q':
            break
        result = predict(text, model, tokenizer, device)
        probs_str = ", ".join(f"{p:.2f}" for p in result['probs'])
        print(f"推定密度: {result['density_mean']:.3f}   softmax=[{probs_str}]")
        print(f"  [分類ヘッドのみ]   分散 = {result['dist_var']:.4f}   (dist_var)")
        print(f"  [提案 分類+分散]   分散 = {result['proposed_var']:.4f}   (dist_var {result['dist_var']:.4f} + head_var {result['head_var']:.4f})")
