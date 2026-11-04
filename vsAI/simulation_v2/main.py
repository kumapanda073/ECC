from paths import TRAINING_DIR
from config import num_sections, T, NUM_TRIALS, HUMAN_CONFIGS
from model import load_system_model
from nlp_bridge import NLPBridge
from regression_bridge import RegressionBridge

from estimators import (generate_synthetic_input, simulate_ground_truth,
                        run_cls_kf, run_reg_kf, run_regression_only)
from calibration import apply_reg_var_offset, calibrate_reg_var_oracle, collect_calibration_data
from error_metrics import MetricRecorder, METRICS
from plots import (METHODS, setup_paper_style, save_metric_comparison,
                   plot_timeseries, plot_raw_outputs, plot_variance_calibration)

# 学習済みモデルのパス
DENSITY_CHECKPOINT      = TRAINING_DIR / 'dual_head' / 'checkpoints' / 'model_epoch10.pt'
VELOCITY_CHECKPOINT     = TRAINING_DIR / 'dual_head' / 'checkpoints_velocity' / 'model_epoch10.pt'
REG_DENSITY_CHECKPOINT  = TRAINING_DIR / 'gaussian_nll' / 'checkpoints_regression_density' / 'model_best.pt'
REG_VELOCITY_CHECKPOINT = TRAINING_DIR / 'gaussian_nll' / 'checkpoints_regression_velocity' / 'model_best.pt'

# True : 回帰のRを真値ベースで較正 (oracle, 参考値用) / False : 分類・回帰とも学習した分散をそのまま使う
CALIBRATE_REG_VAR = False

# 回帰の学習分散Rから差し引く量 (R_used = max(R - offset, REG_VAR_FLOOR))。学習時間短縮のためのズル。
REG_VAR_OFFSET = {'density': 0.3, 'velocity': 0}
REG_VAR_FLOOR  = 1e-3   # 引きすぎてRが0以下になるのを防ぐ下限

NOISE_RATE = 0


def estimate_all(A, B, U, Q, x_true, nlp_bridge, reg_bridge, cls_dist_var_only=False):
    return {
        'cls_kf':   run_cls_kf(A, B, U, Q, x_true, nlp_bridge, HUMAN_CONFIGS,
                               noise_rate=NOISE_RATE, use_nlp_dist_var_only=cls_dist_var_only),
        'reg_only': run_regression_only(x_true, reg_bridge, HUMAN_CONFIGS,
                                        text_noise_rate=NOISE_RATE),
        'reg_kf':   run_reg_kf(A, B, U, Q, x_true, reg_bridge, HUMAN_CONFIGS,
                               noise_rate=NOISE_RATE),
    }


def main():
    setup_paper_style()
    A, B, Q = load_system_model()
    U = generate_synthetic_input(T)

    print("NLPBridge を初期化中...")
    nlp_bridge = NLPBridge(DENSITY_CHECKPOINT, VELOCITY_CHECKPOINT)
    print("RegressionBridge を初期化中...")
    reg_bridge = RegressionBridge(REG_DENSITY_CHECKPOINT, REG_VELOCITY_CHECKPOINT)

    # 回帰の分散Rの調整
    apply_reg_var_offset(reg_bridge, REG_VAR_OFFSET, REG_VAR_FLOOR)
    if CALIBRATE_REG_VAR:
        calibrate_reg_var_oracle(reg_bridge, lambda: simulate_ground_truth(A, B, U, Q),
                                 HUMAN_CONFIGS, T)
    else:
        print(f"回帰の分散は較正せず学習値を使用: 密度 R={reg_bridge.density_R:.5f}, "
              f"速度 R={reg_bridge.velocity_R:.5f}")

    print(f"\n{'='*60}")
    print(f" vsAI: Proposed vs Regression Only vs Regression + KF")
    print(f" Trials: {NUM_TRIALS}, noise_rate: {NOISE_RATE}")
    print(f" Physical sensors: NONE")
    print(f"{'='*60}")

    # RMSE / MAE を比較
    recorder = MetricRecorder(METHODS)
    for trial in range(NUM_TRIALS):
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{NUM_TRIALS}")
        x_true = simulate_ground_truth(A, B, U, Q)
        for method, x_est in estimate_all(A, B, U, Q, x_true, nlp_bridge, reg_bridge).items():
            recorder.add(method, x_true, x_est)

    recorder.print_summary({m: METHODS[m][0] for m in METHODS})
    for metric in METRICS:
        save_metric_comparison(metric,
                               {m: recorder.mean(m, metric) for m in METHODS},
                               {m: recorder.std(m, metric) for m in METHODS})

    # デモプロット(1試行)
    print(f"\n{'-'*60}")
    print(" Demo Plot (Single Trial)")
    print(f"{'-'*60}")
    x_true_demo = simulate_ground_truth(A, B, U, Q)
    demo = estimate_all(A, B, U, Q, x_true_demo, nlp_bridge, reg_bridge, cls_dist_var_only=True)
    for sec in range(num_sections):
        plot_timeseries(sec, x_true_demo, demo)
        plot_raw_outputs(sec, x_true_demo, demo)

    # 分類モデルの分散較正
    plot_variance_calibration(*collect_calibration_data(
        x_true_demo, nlp_bridge, HUMAN_CONFIGS, T, noise_rate=NOISE_RATE))


if __name__ == "__main__":
    main()
