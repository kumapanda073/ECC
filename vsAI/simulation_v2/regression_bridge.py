import json
import random
import numpy as np
import torch

from paths import SIM_DIR

from transformers import BertJapaneseTokenizer
from training.gaussian_nll.regression_model import RegressionModel
from common.utils.dataset import density_to_label, velocity_to_label
from common.utils.preprocess import clean_text

MODEL_NAME = "cl-tohoku/bert-base-japanese-v3"
POOL_PATH       = SIM_DIR / 'text_pool.json'
NOISY_POOL_PATH = SIM_DIR / 'noisy_pool.json'


class RegressionBridge:
    def __init__(self, density_checkpoint, velocity_checkpoint, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

        density_model = RegressionModel(freeze_bert=True)
        density_model.load_state_dict(torch.load(density_checkpoint, map_location=device))
        density_model.to(device)
        density_model.eval()

        velocity_model = RegressionModel(freeze_bert=True)
        velocity_model.load_state_dict(torch.load(velocity_checkpoint, map_location=device))
        velocity_model.to(device)
        velocity_model.eval()

        # ★ 学習した全テキスト共通の分散を、そのまま KF の R として使う
        self.density_R  = density_model.log_var.exp().item()
        self.velocity_R = velocity_model.log_var.exp().item()

        self.density_pool       = self._build_pool(POOL_PATH,       'density',  density_model, tokenizer, density_to_label)
        self.velocity_pool      = self._build_pool(POOL_PATH,       'velocity', velocity_model, tokenizer, velocity_to_label)
        self.noisy_density_pool = self._build_pool(NOISY_POOL_PATH, 'density',  density_model, tokenizer, density_to_label)
        self.noisy_velocity_pool= self._build_pool(NOISY_POOL_PATH, 'velocity', velocity_model, tokenizer, velocity_to_label)

        print(f"[RegressionBridge] Ready: density {sum(len(v) for v in self.density_pool.values())} / "
              f"velocity {sum(len(v) for v in self.velocity_pool.values())} / "
              f"noisy_density {sum(len(v) for v in self.noisy_density_pool.values())} / "
              f"noisy_velocity {sum(len(v) for v in self.noisy_velocity_pool.values())}")
        print(f"[RegressionBridge] R: density {self.density_R:.5f} / velocity {self.velocity_R:.5f}")  # ★

    def _build_pool(self, pool_path, type_key, model, tokenizer, to_label_fn):
        with open(pool_path, 'r', encoding='utf-8') as f:
            pool_data = json.load(f)
        items = [item for item in pool_data if item.get('type') == type_key]
        texts = [clean_text(item['text']) for item in items]
        results = self._batch_infer(model, tokenizer, texts)
        pool = {}
        for item, result in zip(items, results):
            key = to_label_fn(item[type_key])
            pool.setdefault(key, []).append(result)
        return pool

    def _batch_infer(self, model, tokenizer, texts, batch_size=32):
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
                means = model(input_ids, attention_mask)
            for m in means.cpu().numpy().tolist():
                results.append({'mean': m})
        return results

    def _select_from_pool(self, pool, key):
        if key in pool:
            return random.choice(pool[key])
        all_results = [r for rs in pool.values() for r in rs]
        return random.choice(all_results)

    def observe(self, perceived_density, perceived_velocity,
                error_rate=0.0, noise_rate=0.0):
        dk = density_to_label(perceived_density)
        vk = velocity_to_label(perceived_velocity)

        if error_rate > 0.0 and random.random() < error_rate:
            wrong_keys = [k for k in self.density_pool if k != dk]
            if wrong_keys:
                dk = random.choice(wrong_keys)

        d_noisy = noise_rate > 0.0 and random.random() < noise_rate
        d_pool = self.noisy_density_pool if d_noisy else self.density_pool
        d_result = self._select_from_pool(d_pool, dk)

        if error_rate > 0.0 and random.random() < error_rate:
            wrong_keys = [k for k in self.velocity_pool if k != vk]
            if wrong_keys:
                vk = random.choice(wrong_keys)

        v_noisy = noise_rate > 0.0 and random.random() < noise_rate
        v_pool = self.noisy_velocity_pool if v_noisy else self.velocity_pool
        v_result = self._select_from_pool(v_pool, vk)

        return {
            'density_mean':  d_result['mean'],
            'density_var':   self.density_R * getattr(self, 'density_var_scale', 1.0),
            'velocity_mean': v_result['mean'],
            'velocity_var':  self.velocity_R * getattr(self, 'velocity_var_scale', 1.0),
            'density_noisy': d_noisy,
            'velocity_noisy': v_noisy,
        }