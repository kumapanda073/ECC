"""テキスト選択ミス率 (error rate) を変えたモンテカルロ評価。"""
import numpy as np

from config import num_sections, NUM_TRIALS, PHY_SENSORS, HUMAN_CONFIGS
from simulation import run_kalman_simulation_affine
from metrics import calculate_rmse, calculate_mae
from plotting.sweep import (
    _save_per_errorrate_outputs, _save_errorrate_comparison,
    _save_sweep_summary, _save_sweep_paper,
)


def run_all_evaluations(A_mat, B_mat, Q_mat, U_sim, nlp_bridge, error_rates):
    """1回のモンテカルロループで全誤差率を同時評価する。
    Phy Only・Fixed Var は誤差率に依存しないため各試行1回だけ実行する。"""
    print(f"\n[統合評価] {NUM_TRIALS}回のモンテカルロ試行 × {len(error_rates)}誤差率")

    all_rmse_phy       = []
    all_rmse_nlp_fixed = {er: [] for er in error_rates}
    all_rmse_nlp       = {er: [] for er in error_rates}
    all_mae_phy        = []
    all_mae_nlp_fixed  = {er: [] for er in error_rates}
    all_mae_nlp        = {er: [] for er in error_rates}

    for trial in range(NUM_TRIALS):
        if (trial + 1) % 10 == 0:
            print(f"  Trial {trial + 1}/{NUM_TRIALS}")

        x_true_shared, _, _, _ = run_kalman_simulation_affine(
            A_mat, B_mat, U_sim, Q_mat,
            use_human_sensor=False, active_phy_sensors=[]
        )
        _, xe_phy, _, _ = run_kalman_simulation_affine(
            A_mat, B_mat, U_sim, Q_mat,
            use_human_sensor=False,
            active_phy_sensors=PHY_SENSORS,
            x_true=x_true_shared
        )
        all_rmse_phy.append(calculate_rmse(x_true_shared, xe_phy))
        all_mae_phy.append(calculate_mae(x_true_shared, xe_phy))

        for er in error_rates:
            _, xe_nlp_fixed, _, _ = run_kalman_simulation_affine(
                A_mat, B_mat, U_sim, Q_mat,
                use_human_sensor=True,
                human_sensor_configs=HUMAN_CONFIGS,
                active_phy_sensors=PHY_SENSORS,
                x_true=x_true_shared,
                use_nlp=True,
                nlp_bridge=nlp_bridge,
                text_error_rate=er,
                text_noise_rate=0.0,
                use_nlp_fixed_var=True,
            )
            all_rmse_nlp_fixed[er].append(calculate_rmse(x_true_shared, xe_nlp_fixed))
            all_mae_nlp_fixed[er].append(calculate_mae(x_true_shared, xe_nlp_fixed))

            _, xe_nlp, _, _ = run_kalman_simulation_affine(
                A_mat, B_mat, U_sim, Q_mat,
                use_human_sensor=True,
                human_sensor_configs=HUMAN_CONFIGS,
                active_phy_sensors=PHY_SENSORS,
                x_true=x_true_shared,
                use_nlp=True,
                nlp_bridge=nlp_bridge,
                text_error_rate=er,
                text_noise_rate=0.0,
            )
            all_rmse_nlp[er].append(calculate_rmse(x_true_shared, xe_nlp))
            all_mae_nlp[er].append(calculate_mae(x_true_shared, xe_nlp))

    avg_phy = np.mean(all_rmse_phy, axis=0)
    std_phy = np.std(all_rmse_phy,  axis=0)
    mae_avg_phy = np.mean(all_mae_phy, axis=0)
    mae_std_phy = np.std(all_mae_phy,  axis=0)

    results = {}
    mae_results = {}
    for er in error_rates:
        avg_nlp_fixed = np.mean(all_rmse_nlp_fixed[er], axis=0)
        std_nlp_fixed = np.std(all_rmse_nlp_fixed[er],  axis=0)
        avg_nlp = np.mean(all_rmse_nlp[er], axis=0)
        std_nlp = np.std(all_rmse_nlp[er],  axis=0)
        results[er] = {
            'avg_phy': avg_phy, 'std_phy': std_phy,
            'avg_nlp_fixed': avg_nlp_fixed, 'std_nlp_fixed': std_nlp_fixed,
            'avg_nlp': avg_nlp, 'std_nlp': std_nlp,
            'raw_phy':       np.array(all_rmse_phy),
            'raw_nlp_fixed': np.array(all_rmse_nlp_fixed[er]),
            'raw_nlp':       np.array(all_rmse_nlp[er]),
        }
        mae_results[er] = {
            'avg_nlp_fixed': np.mean(all_mae_nlp_fixed[er], axis=0),
            'std_nlp_fixed': np.std(all_mae_nlp_fixed[er],  axis=0),
            'avg_nlp': np.mean(all_mae_nlp[er], axis=0),
            'std_nlp': np.std(all_mae_nlp[er],  axis=0),
        }

    for er in error_rates:
        r = results[er]
        _save_per_errorrate_outputs(
            avg_phy, std_phy,
            r['avg_nlp_fixed'], r['std_nlp_fixed'],
            r['avg_nlp'], r['std_nlp'], er,
        )
        m = mae_results[er]
        _save_per_errorrate_outputs(
            mae_avg_phy, mae_std_phy,
            m['avg_nlp_fixed'], m['std_nlp_fixed'],
            m['avg_nlp'], m['std_nlp'], er, metric='MAE',
        )
        label = f"(誤差率={int(er*100)}%)" if er > 0 else ""
        print(f"\n--- 全区間平均 RMSE {label} ---")
        print(f"  物理のみ          密度: {avg_phy[:num_sections].mean():.4f}  速度: {avg_phy[num_sections:].mean():.4f}")
        print(f"  NLP固定分散HB     密度: {r['avg_nlp_fixed'][:num_sections].mean():.4f}  速度: {r['avg_nlp_fixed'][num_sections:].mean():.4f}")
        print(f"  NLP動的分散HB     密度: {r['avg_nlp'][:num_sections].mean():.4f}  速度: {r['avg_nlp'][num_sections:].mean():.4f}")
        print(f"--- 全区間平均 MAE {label} ---")
        print(f"  物理のみ          密度: {mae_avg_phy[:num_sections].mean():.4f}  速度: {mae_avg_phy[num_sections:].mean():.4f}")
        print(f"  NLP固定分散HB     密度: {m['avg_nlp_fixed'][:num_sections].mean():.4f}  速度: {m['avg_nlp_fixed'][num_sections:].mean():.4f}")
        print(f"  NLP動的分散HB     密度: {m['avg_nlp'][:num_sections].mean():.4f}  速度: {m['avg_nlp'][num_sections:].mean():.4f}")

    _save_errorrate_comparison(results, error_rates)
    _save_sweep_summary(results, error_rates, 'error_rate',
                        {'Phy Only': 'phy', 'Proposed': 'nlp'},
                        subdir='error_rate')
    _save_sweep_paper(results, error_rates, 'error_rate',
                      ['phy', 'nlp_fixed', 'nlp'],
                      subdir='error_rate', xlabel='Text error rate (%)')
    return results
