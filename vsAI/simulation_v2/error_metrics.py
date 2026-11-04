import numpy as np

import paths
from config import num_sections

METRICS = ('RMSE', 'MAE')


def calculate_rmse(true, est):
    return np.sqrt(np.mean((true - est) ** 2, axis=1))


def calculate_mae(true, est):
    return np.mean(np.abs(true - est), axis=1)


METRIC_FUNCS = {'RMSE': calculate_rmse, 'MAE': calculate_mae}


class MetricRecorder:
    # 手法ごと・指標ごとに試行結果を貯めて、平均と標準偏差を出す

    def __init__(self, methods):
        self.methods = list(methods)
        self._values = {m: {k: [] for k in METRICS} for m in self.methods}

    def add(self, method, x_true, x_est):
        for k in METRICS:
            self._values[method][k].append(METRIC_FUNCS[k](x_true, x_est))

    def mean(self, method, metric):
        return np.mean(self._values[method][metric], axis=0)

    def std(self, method, metric):
        return np.std(self._values[method][metric], axis=0)

    def print_summary(self, labels):
        width = max(len(labels[m]) for m in self.methods)
        for k in METRICS:
            print(f"\n--- 全区間平均 {k} ---")
            for m in self.methods:
                avg = self.mean(m, k)
                print(f"  {labels[m]:<{width}} 密度: {avg[:num_sections].mean():.4f}  "
                      f"速度: {avg[num_sections:].mean():.4f}")
