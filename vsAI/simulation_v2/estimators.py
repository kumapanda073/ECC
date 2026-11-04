"""
比較する3手法と真値生成を、それぞれ1関数で呼べるようにまとめる。
  - simulate_ground_truth : 真値の軌跡を生成
  - run_cls_kf            : Classification + KF (提案手法、NLPBridgeの観測をKFで融合)
  - run_reg_kf            : Regression + KF     (RegressionBridgeの観測をKFで融合)
  - run_regression_only   : Regression Only     (KFなし、回帰の出力を区間ごとに平均)
"""
import numpy as np

import paths
from config import (
    num_sections, n_states, T, inflow_main, inflow_u2, inflow_u3, inflow_v1, inflow_v2, inflow_v3,
    x_min_density, x_max_density, x_min_velocity, x_max_velocity,
)
from simulation import run_kalman_simulation_affine
from sensing import sensors_per_section, perceive


def generate_synthetic_input(T):
    U_sim = np.zeros((T, 6))
    U_sim[:, 0] = inflow_main
    U_sim[:, 1] = inflow_u2
    U_sim[:, 2] = inflow_u3
    U_sim[:, 3] = inflow_v1
    U_sim[:, 4] = inflow_v2
    U_sim[:, 5] = inflow_v3
    return U_sim


def simulate_ground_truth(A, B, U, Q):
    x_true, _, _, _ = run_kalman_simulation_affine(
        A, B, U, Q, use_human_sensor=False, active_phy_sensors=[])
    return x_true


def _run_text_kf(A, B, U, Q, x_true, bridge, human_sensor_configs, noise_rate, **kwargs):
    # テキスト観測のみをKFで融合した推定値を返す。物理センサなし。
    _, x_est, _, _ = run_kalman_simulation_affine(
        A, B, U, Q,
        use_human_sensor=True,
        human_sensor_configs=human_sensor_configs,
        active_phy_sensors=[],
        x_true=x_true,
        use_nlp=True,
        nlp_bridge=bridge,
        text_noise_rate=noise_rate,
        **kwargs,
    )
    return x_est


def run_cls_kf(A, B, U, Q, x_true, nlp_bridge, human_sensor_configs,
               noise_rate=0.0, use_nlp_dist_var_only=False):
    # Classification + KF (提案手法)
    return _run_text_kf(A, B, U, Q, x_true, nlp_bridge, human_sensor_configs, noise_rate,
                        use_nlp_dist_var_only=use_nlp_dist_var_only)


def run_reg_kf(A, B, U, Q, x_true, reg_bridge, human_sensor_configs, noise_rate=0.0):
    # Regression + KF
    return _run_text_kf(A, B, U, Q, x_true, reg_bridge, human_sensor_configs, noise_rate)


def run_regression_only(x_true, reg_bridge, human_sensor_configs,
                        text_error_rate=0.0, text_noise_rate=0.0):
    # Regression Only (KF なし)
    x_est = np.zeros((n_states, T))
    x_est[:, 0] = x_true[:, 0]

    for t in range(1, T):
        counts = sensors_per_section(human_sensor_configs, t)

        for sec_idx, count in counts.items():
            rho_true = x_true[sec_idx, t]
            vel_true = x_true[sec_idx + num_sections, t]

            rho_estimates, vel_estimates = [], []
            for _ in range(count):
                z_rho, z_vel = perceive(rho_true, vel_true)
                result = reg_bridge.observe(z_rho, z_vel,
                                            error_rate=text_error_rate,
                                            noise_rate=text_noise_rate)
                rho_estimates.append(result['density_mean'])
                vel_estimates.append(result['velocity_mean'])

            x_est[sec_idx, t] = np.mean(rho_estimates)
            x_est[sec_idx + num_sections, t] = np.mean(vel_estimates)

        for sec_idx in range(num_sections):
            if sec_idx not in counts:
                x_est[sec_idx, t] = x_est[sec_idx, t - 1]
                x_est[sec_idx + num_sections, t] = x_est[sec_idx + num_sections, t - 1]

        x_est[:num_sections, t] = np.clip(x_est[:num_sections, t], x_min_density, x_max_density)
        x_est[num_sections:, t] = np.clip(x_est[num_sections:, t], x_min_velocity, x_max_velocity)

    return x_est
