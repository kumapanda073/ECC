import json
import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.utils.preprocess import clean_text

TARGET_VARIANCE_NULL = 0.9
TARGET_VARIANCE_RELEVANT = 0.05


class RegressionDataset(Dataset):
    def __init__(self, json_path, tokenizer, label_field="density", max_length=128):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.texts = [clean_text(item['text']) for item in data]
        self.is_null = [item.get(label_field) is None for item in data]
        self.targets = [item.get(label_field, 0.0) or 0.0 for item in data]
        self.target_variances = [
            TARGET_VARIANCE_NULL if n else TARGET_VARIANCE_RELEVANT
            for n in self.is_null
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32),
            'is_null': torch.tensor(self.is_null[idx], dtype=torch.bool),
            'target_variance': torch.tensor(self.target_variances[idx], dtype=torch.float32),
        }


def get_dataloader(json_path, tokenizer, label_field="density",
                   batch_size=16, shuffle=True):
    dataset = RegressionDataset(json_path, tokenizer, label_field=label_field)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
