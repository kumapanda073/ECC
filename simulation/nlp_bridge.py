import json
import random
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import BertJapaneseTokenizer
from training.dual_head.models.model_factory import CrowdDensityModel
from common.utils.dataset import (
    DENSITY_MEANS, VELOCITY_MEANS, 
    NUM_VELOCITY_CLASSES, density_to_label, 
    velocity_to_label
)
from common.utils.preprocess import clean_text

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
_SIM_DIR = Path(__file__).resolve().parent
POOL_PATH       = _SIM_DIR / 'text_pool.json'
NOISY_POOL_PATH = _SIM_DIR / 'noisy_pool.json'


class NLPBridge:

    def __init__(self, density_checkpoint: str, velocity_checkpoint: str, device=None):
        if device is None:          #どのデバイスのGPUを使うかを判定する
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

        density_model = CrowdDensityModel(freeze_bert=True)
        density_model.load_state_dict(torch.load(density_checkpoint, map_location=device))
        density_model.to(device)
        density_model.eval()

        velocity_model = CrowdDensityModel(freeze_bert=True, num_classes=NUM_VELOCITY_CLASSES)
        velocity_model.load_state_dict(torch.load(velocity_checkpoint, map_location=device))
        velocity_model.to(device)
        velocity_model.eval()

        self.density_pool       = self._build_pool(POOL_PATH,       'density',  density_model, tokenizer, density_to_label,  DENSITY_MEANS)
        self.velocity_pool      = self._build_pool(POOL_PATH,       'velocity', velocity_model, tokenizer, velocity_to_label, VELOCITY_MEANS)
        self.noisy_density_pool = self._build_pool(NOISY_POOL_PATH, 'density',  density_model, tokenizer, density_to_label,  DENSITY_MEANS)
        self.noisy_velocity_pool= self._build_pool(NOISY_POOL_PATH, 'velocity', velocity_model, tokenizer, velocity_to_label, VELOCITY_MEANS)

        print(f"[NLPBridge] Ready: density {sum(len(v) for v in self.density_pool.values())} / "
              f"velocity {sum(len(v) for v in self.velocity_pool.values())} / "
              f"noisy_density {sum(len(v) for v in self.noisy_density_pool.values())} / "
              f"noisy_velocity {sum(len(v) for v in self.noisy_velocity_pool.values())}")

    def _build_pool(self, pool_path, type_key, model, tokenizer, to_label_fn, means) -> dict:
        with open(pool_path, 'r', encoding='utf-8') as f:
            pool_data = json.load(f)
        items  = [item for item in pool_data if item.get('type') == type_key]
        texts  = [clean_text(item['text']) for item in items]
        results = self._batch_infer(model, tokenizer, texts, means)
        pool = {}
        for item, result in zip(items, results):
            key = to_label_fn(item[type_key])
            pool.setdefault(key, []).append(result)
        return pool

    def _batch_infer(self, model, tokenizer, texts: list, means: np.ndarray, batch_size: int = 32) -> list:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            input_ids      = enc['input_ids'].to(self.device)
            attention_mask = enc['attention_mask'].to(self.device)

            with torch.no_grad():
                logits, head_var = model(input_ids, attention_mask)

            probs    = F.softmax(logits, dim=-1).cpu().numpy()
            head_var = head_var.cpu().numpy()

            for j in range(len(batch)):
                pred_class = int(np.argmax(probs[j]))
                mean     = float(means[pred_class])
                dist_var = float(np.dot(probs[j], (means - mean) ** 2))
                var      = dist_var + float(head_var[j])
                results.append({'mean': mean, 'var': var, 'dist_var': dist_var})
        return results

    def _select_from_pool(self, pool, key):
        if key in pool:
            return random.choice(pool[key])
        all_results = [r for rs in pool.values() for r in rs]
        return random.choice(all_results)

    def observe(self, perceived_density: float, perceived_velocity: float,
                error_rate: float = 0.0, noise_rate: float = 0.0) -> dict:
        dk = density_to_label(perceived_density)
        vk = velocity_to_label(perceived_velocity)

        # 密度テキスト選択
        if error_rate > 0.0 and random.random() < error_rate:
            wrong_keys = [k for k in self.density_pool if k != dk]
            if wrong_keys:
                dk = random.choice(wrong_keys)

        d_noisy = noise_rate > 0.0 and random.random() < noise_rate
        d_pool = self.noisy_density_pool if d_noisy else self.density_pool
        d_result = self._select_from_pool(d_pool, dk)

        # 速度テキスト選択
        if error_rate > 0.0 and random.random() < error_rate:
            wrong_keys = [k for k in self.velocity_pool if k != vk]
            if wrong_keys:
                vk = random.choice(wrong_keys)

        v_noisy = noise_rate > 0.0 and random.random() < noise_rate
        v_pool = self.noisy_velocity_pool if v_noisy else self.velocity_pool
        v_result = self._select_from_pool(v_pool, vk)

        return {
            'density_mean':  d_result['mean'],
            'density_var':   d_result['var'],
            'density_dist_var':  d_result['dist_var'],
            'velocity_mean': v_result['mean'],
            'velocity_var':  v_result['var'],
            'velocity_dist_var': v_result['dist_var'],
            'density_noisy': d_noisy,
            'velocity_noisy': v_noisy,
        }
