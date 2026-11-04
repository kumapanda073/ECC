"""無関係テキストの割合 (noise rate) を変えたモンテカルロ評価とキャリブレーション。"""
import numpy as np

from config import num_sections, NUM_TRIALS, PHY_SENSORS, HUMAN_CONFIGS
from simulation import run_kalman_simulation_affine
from metrics import calculate_rmse, _collect_obs
from plotting.sweep import _save_noiserate_comparison, _save_sweep_paper, _save_R_vs_noise
from plotting.calibration import CALIB_KEYS, _save_calibration


def run_noise_evaluations(A_mat, B_mat, Q_mat, U_sim, nlp_bridge, noise_rates):
    """error_rate=0固定でnoise_rateを変えながら評価する。"""
    print(f"\n[Noise評価] {NUM_TRIALS}回のモンテカルロ試行 × {len(noise_rates)}noise率")

    all_rmse_phy       = []
    all_rmse_nlp_fixed = {nr: [] for nr in noise_rates}
    all_rmse_distonly  = {nr: [] for nr in noise_rates}  # 分類ヘッドのみ(dist_var), 折れ線図用
    all_rmse_nlp       = {nr: [] for nr in noise_rates}

    # 人センサの観測ごとに (付けた R_k^H, 実際の誤差^2) を貯める
    # (Case1: nlp_fixed / Case2: distonly / Case3: nlp)
    all_obs = {k: {nr: {p: {'R': [], 'err2': []} for p in ('Density', 'Velocity')}
                   for nr in noise_rates}
               for k in CALIB_KEYS}

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

        for nr in noise_rates:
            _, xe_nlp_fixed, hobs_fixed, _ = run_kalman_simulation_affine(
                A_mat, B_mat, U_sim, Q_mat,
                use_human_sensor=True,
                human_sensor_configs=HUMAN_CONFIGS,
                active_phy_sensors=PHY_SENSORS,
                x_true=x_true_shared,
                use_nlp=True,
                nlp_bridge=nlp_bridge,
                text_error_rate=0.0,
                text_noise_rate=nr,
                use_nlp_fixed_var=True,
            )
            all_rmse_nlp_fixed[nr].append(calculate_rmse(x_true_shared, xe_nlp_fixed))
            _collect_obs(hobs_fixed, x_true_shared, all_obs['nlp_fixed'][nr])

            _, xe_nlp, hobs_nlp, _ = run_kalman_simulation_affine(
                A_mat, B_mat, U_sim, Q_mat,
                use_human_sensor=True,
                human_sensor_configs=HUMAN_CONFIGS,
                active_phy_sensors=PHY_SENSORS,
                x_true=x_true_shared,
                use_nlp=True,
                nlp_bridge=nlp_bridge,
                text_error_rate=0.0,
                text_noise_rate=nr,
            )
            all_rmse_nlp[nr].append(calculate_rmse(x_true_shared, xe_nlp))
            _collect_obs(hobs_nlp, x_true_shared, all_obs['nlp'][nr])

            # 分類ヘッドのみ (dist_varのみ, 分散ヘッド不使用) — 折れ線図の比較対象
            _, xe_distonly, hobs_do, _ = run_kalman_simulation_affine(
                A_mat, B_mat, U_sim, Q_mat,
                use_human_sensor=True,
                human_sensor_configs=HUMAN_CONFIGS,
                active_phy_sensors=PHY_SENSORS,
                x_true=x_true_shared,
                use_nlp=True,
                nlp_bridge=nlp_bridge,
                text_error_rate=0.0,
                text_noise_rate=nr,
                use_nlp_dist_var_only=True,
            )
            all_rmse_distonly[nr].append(calculate_rmse(x_true_shared, xe_distonly))
            _collect_obs(hobs_do, x_true_shared, all_obs['distonly'][nr])

    avg_phy = np.mean(all_rmse_phy, axis=0)
    std_phy = np.std(all_rmse_phy,  axis=0)

    results = {}
    for nr in noise_rates:
        avg_nlp_fixed = np.mean(all_rmse_nlp_fixed[nr], axis=0)
        std_nlp_fixed = np.std(all_rmse_nlp_fixed[nr],  axis=0)
        avg_nlp = np.mean(all_rmse_nlp[nr], axis=0)
        std_nlp = np.std(all_rmse_nlp[nr],  axis=0)
        results[nr] = {
            'avg_phy': avg_phy, 'std_phy': std_phy,
            'avg_nlp_fixed': avg_nlp_fixed, 'std_nlp_fixed': std_nlp_fixed,
            'avg_distonly': np.mean(all_rmse_distonly[nr], axis=0),
            'std_distonly': np.std(all_rmse_distonly[nr],  axis=0),
            'avg_nlp': avg_nlp, 'std_nlp': std_nlp,
            'raw_phy':       np.array(all_rmse_phy),
            'raw_nlp_fixed': np.array(all_rmse_nlp_fixed[nr]),
            'raw_distonly':  np.array(all_rmse_distonly[nr]),
            'raw_nlp':       np.array(all_rmse_nlp[nr]),
        }
        print(f"\n--- 全区間平均 RMSE (noise_rate={int(nr*100)}%) ---")
        print(f"  物理のみ          密度: {avg_phy[:num_sections].mean():.4f}  速度: {avg_phy[num_sections:].mean():.4f}")
        print(f"  NLP固定分散HB     密度: {avg_nlp_fixed[:num_sections].mean():.4f}  速度: {avg_nlp_fixed[num_sections:].mean():.4f}")
        print(f"  NLP動的分散HB     密度: {avg_nlp[:num_sections].mean():.4f}  速度: {avg_nlp[num_sections:].mean():.4f}")

    _save_noiserate_comparison(results, noise_rates)
    _save_sweep_paper(results, noise_rates, 'noise_rate',
                      ['nlp_fixed', 'distonly', 'nlp'],
                      subdir='noise_rate', xlabel='Ratio of irrelevant texts (%)',
                      show_std=False)
    all_R = {k: {nr: {p: all_obs[k][nr][p]['R'] for p in ('Density', 'Velocity')}
                 for nr in noise_rates}
             for k in ('distonly', 'nlp')}
    _save_R_vs_noise(all_R, noise_rates)
    _save_calibration(all_obs, noise_rates)
    return results
