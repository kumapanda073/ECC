import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertJapaneseTokenizer
from common.utils.preprocess import clean_text

NUM_DENSITY_CLASSES = 5
DENSITY_MAX = 2.0
DENSITY_WIDTH = DENSITY_MAX / NUM_DENSITY_CLASSES
DENSITY_MEANS = np.array([(k + 0.5) * DENSITY_WIDTH for k in range(NUM_DENSITY_CLASSES)])

NUM_VELOCITY_CLASSES = 5
VELOCITY_MAX = 1.8
VELOCITY_WIDTH = VELOCITY_MAX / NUM_VELOCITY_CLASSES
VELOCITY_MEANS = np.array([(k + 0.5) * VELOCITY_WIDTH for k in range(NUM_VELOCITY_CLASSES)])


def density_to_label(density: float) -> int:
    return min(int(density / DENSITY_WIDTH), NUM_DENSITY_CLASSES - 1)


def velocity_to_label(velocity: float) -> int:
    return min(int(velocity / VELOCITY_WIDTH), NUM_VELOCITY_CLASSES - 1)


UNIFORM_DENSITY_LABEL = np.ones(NUM_DENSITY_CLASSES, dtype=np.float32) / NUM_DENSITY_CLASSES
UNIFORM_VELOCITY_LABEL = np.ones(NUM_VELOCITY_CLASSES, dtype=np.float32) / NUM_VELOCITY_CLASSES


TARGET_VARIANCE_NULL     = 0.7
TARGET_VARIANCE_RELEVANT = 0.04


def density_to_soft_label(density) -> np.ndarray:
    if density is None:
        return UNIFORM_DENSITY_LABEL.copy()
    label = np.zeros(NUM_DENSITY_CLASSES, dtype=np.float32)
    label[density_to_label(density)] = 1.0
    return label


def velocity_to_soft_label(velocity) -> np.ndarray:
    if velocity is None:
        return UNIFORM_VELOCITY_LABEL.copy()
    label = np.zeros(NUM_VELOCITY_CLASSES, dtype=np.float32)
    label[velocity_to_label(velocity)] = 1.0
    return label


class CrowdDensityDataset(Dataset):
    def __init__(self, json_path: str, tokenizer: BertJapaneseTokenizer,
                 label_field: str = "density", max_length: int = 128):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.texts = [clean_text(item['text']) for item in data]

        if label_field == "density":
            self.soft_labels = [density_to_soft_label(item.get('density')) for item in data]
            self.is_null = [item.get('density') is None for item in data]
            self.target_variances = [TARGET_VARIANCE_NULL if n else TARGET_VARIANCE_RELEVANT for n in self.is_null]
        elif label_field == "velocity":
            self.soft_labels = [velocity_to_soft_label(item.get('velocity')) for item in data]
            self.is_null = [item.get('velocity') is None for item in data]
            self.target_variances = [TARGET_VARIANCE_NULL if n else TARGET_VARIANCE_RELEVANT for n in self.is_null]
        else:
            raise ValueError(f"Unknown label_field: {label_field}")

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
            'soft_label': torch.tensor(self.soft_labels[idx], dtype=torch.float32),
            'is_null': torch.tensor(self.is_null[idx], dtype=torch.bool),
            'target_variance': torch.tensor(self.target_variances[idx], dtype=torch.float32),
        }


def get_dataloader(json_path: str, tokenizer: BertJapaneseTokenizer,
                   label_field: str = "density",
                   batch_size: int = 16, shuffle: bool = True) -> DataLoader:
    dataset = CrowdDensityDataset(json_path, tokenizer, label_field=label_field)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
