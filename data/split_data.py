"""
データセットを ./train_*.json に全部入れたら、val_*.json に分割するスクリプト
"""

import os
import json
import random
import argparse
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def split(input_path: str, train_path: str, val_path: str, test_path: str,
          label_field: str = "density",
          train_ratio: float = 0.8, val_ratio: float = 0.2):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    groups = defaultdict(list)
    for item in data:
        key = item.get(label_field)
        if key is None:
            key = 'null'
        groups[key].append(item)

    train, val, test = [], [], []

    for _, items in sorted(groups.items(), key=lambda x: (x[0] == 'null', x[0] if x[0] != 'null' else 0)):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train += items[:n_train]
        val   += items[n_train:n_train + n_val]
        test  += items[n_train + n_val:]

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    for path, subset in [(train_path, train), (val_path, val), (test_path, test)]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)

    print(f"合計: {len(data)}件")
    print(f"train: {len(train)}件 → {train_path}")
    print(f"val:   {len(val)}件   → {val_path}")
    print(f"test:  {len(test)}件  → {test_path}")
    print()
    print("【カテゴリ別内訳】")
    for name, subset in [("train", train), ("val", val), ("test", test)]:
        counts = defaultdict(int)
        for item in subset:
            counts[item.get(label_field)] += 1
        print(f"  {name}: " + ", ".join(
            f"{k}→{v}件" for k, v in sorted(counts.items(), key=lambda x: (x[0] is None, x[0] or 0))
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["density", "velocity"], default="density")
    args = parser.parse_args()

    if args.mode == "density":
        split(
            input_path=os.path.join(DATA_DIR, "train_den.json"),
            train_path=os.path.join(DATA_DIR, "train_den.json"),
            val_path=os.path.join(DATA_DIR, "val_den.json"),
            test_path=os.path.join(DATA_DIR, "test_den.json"),
            label_field="density",
        )
    elif args.mode == "velocity":
        split(
            input_path=os.path.join(DATA_DIR, "train_vel.json"),
            train_path=os.path.join(DATA_DIR, "train_vel.json"),
            val_path=os.path.join(DATA_DIR, "val_vel.json"),
            test_path=os.path.join(DATA_DIR, "test_vel.json"),
            label_field="velocity",
        )
