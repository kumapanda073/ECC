"""
  python main.py             : 誤差率スイープ + noise率スイープ + 時系列デモ (全部)
  python main.py error_rate  : 誤差率スイープだけ
  python main.py noise_rate  : noise率スイープ(+キャリブレーション)だけ
  python main.py timeseries  : 時系列デモだけ
  python main.py replot      : 保存済みnpzから論文用図だけ再描画
"""
import argparse

from config import (
    T, ERROR_RATES, NOISE_RATES,
    DENSITY_CHECKPOINT_PATH, VELOCITY_CHECKPOINT_PATH,
)
from plotting.style import FIGURE_DIR
from plotting.sweep import replot_sweep_from_npz
from plotting.calibration import CALIB_SUBDIR, replot_calibration_from_npz

MODES = ('all', 'error_rate', 'noise_rate', 'timeseries', 'replot')


def main(mode='all'):
    # torch / BERT を読み込むものは、replotのときに読み込まないようmainの中でimportする
    from model import load_system_model, generate_synthetic_input
    from nlp_bridge import NLPBridge
    from experiments.error_rate import run_all_evaluations
    from experiments.noise_rate import run_noise_evaluations
    from experiments.timeseries import run_timeseries_demo

    try:
        A_mat, B_mat, Q_mat = load_system_model()
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        return

    print("NLPBridge を初期化中...")
    try:
        nlp_bridge = NLPBridge(DENSITY_CHECKPOINT_PATH, VELOCITY_CHECKPOINT_PATH)
    except Exception as e:
        print(f"NLPBridge 初期化エラー: {e}")
        return

    U_sim = generate_synthetic_input(T)

    print("-" * 60)
    print(f" Simulation & Evaluation Start (mode={mode})")
    print("-" * 60)

    if mode in ('all', 'error_rate'):
        run_all_evaluations(A_mat, B_mat, Q_mat, U_sim, nlp_bridge, ERROR_RATES)
    if mode in ('all', 'noise_rate'):
        run_noise_evaluations(A_mat, B_mat, Q_mat, U_sim, nlp_bridge, NOISE_RATES)
    if mode in ('all', 'timeseries'):
        run_timeseries_demo(A_mat, B_mat, Q_mat, U_sim, nlp_bridge)


def replot():
    if (FIGURE_DIR / 'noise_rate' / 'noise_rate_raw_rmse.npz').exists():
        replot_sweep_from_npz('noise_rate', 'noise_rate',
                              ['nlp_fixed', 'distonly', 'nlp'], 'Ratio of irrelevant texts (%)',
                              show_std=False)
    replot_sweep_from_npz('error_rate', 'error_rate',
                          ['phy', 'nlp_fixed', 'nlp'], 'Text error rate (%)')
    if (FIGURE_DIR / CALIB_SUBDIR / 'calibration_raw.npz').exists():
        replot_calibration_from_npz()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="人センサ + カルマンフィルタのシミュレーション評価")
    parser.add_argument('mode', nargs='?', default='all', choices=MODES,
                        help="実行する実験 (省略時は all)")
    args = parser.parse_args()

    if args.mode == 'replot':
        replot()
    else:
        main(args.mode)
