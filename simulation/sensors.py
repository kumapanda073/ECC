"""物理センサと人センサの観測を生成する"""

import numpy as np
from config import (
    num_sections,
    r_phy_density, r_phy_velocity,
    r_human_den_perception, r_human_vel_perception,
    x_min_density, x_max_density,
    x_min_velocity, x_max_velocity,
    num_human_categories, category_thresholds, category_means,
    vel_thresholds, vel_means,
    r_human_quantization, r_vel_quantization,
)


def observe_physical(x_true, t, active_phy_sensors):
    y_phy = {}
    for sec_idx in active_phy_sensors:
        idx_rho = sec_idx
        idx_vel = sec_idx + num_sections

        rho_true = x_true[idx_rho, t]
        vel_true = x_true[idx_vel, t]

        y_phy_rho = rho_true + np.sqrt(r_phy_density) * np.random.randn()
        y_phy_vel = vel_true + np.sqrt(r_phy_velocity) * np.random.randn()

        y_phy[sec_idx] = {'density': y_phy_rho, 'velocity': y_phy_vel}
    return y_phy


def observe_human(
    x_true, t, human_sensor_configs,
    use_nlp=False, nlp_bridge=None,
    text_error_rate=0.0, text_noise_rate=0.0,
    use_nlp_fixed_var=False, use_nlp_dist_var_only=False,
):
    """variances=None のときはkalman.pyで固定分散を使う"""
    human_obs = {}
    active_configs = [c for c in human_sensor_configs if t >= c['offset'] and (t - c['offset']) % c['interval'] == 0]

    sensors_per_section = {}
    for c in active_configs:
        sec = c['section']
        sensors_per_section[sec] = sensors_per_section.get(sec, 0) + 1

    for sec_idx, count in sensors_per_section.items():
        if sec_idx >= num_sections: continue
        idx_rho = sec_idx
        idx_vel = sec_idx + num_sections

        rho_true = x_true[idx_rho, t]
        vel_true = x_true[idx_vel, t]

        if use_nlp and nlp_bridge is not None:
            # NLP: 1人→知覚→テキスト選択→NLP推論(密度・速度)
            rho_values_to_use = []
            rho_variances     = []
            vel_values_to_use = []
            vel_variances     = []

            for _ in range(count):
                z_rho = rho_true + np.sqrt(r_human_den_perception) * np.random.randn()
                z_rho = np.clip(z_rho, x_min_density, x_max_density)

                z_vel = vel_true + np.sqrt(r_human_vel_perception) * np.random.randn()
                z_vel = np.clip(z_vel, x_min_velocity, x_max_velocity)

                result = nlp_bridge.observe(z_rho, z_vel, error_rate=text_error_rate, noise_rate=text_noise_rate)
                rho_values_to_use.append(result['density_mean'])
                vel_values_to_use.append(result['velocity_mean'])
                if use_nlp_fixed_var:
                    rho_variances.append(r_human_quantization)
                    vel_variances.append(r_vel_quantization)
                elif use_nlp_dist_var_only:
                    rho_variances.append(result['density_dist_var'])
                    vel_variances.append(result['velocity_dist_var'])
                else:
                    rho_variances.append(result['density_var'])
                    vel_variances.append(result['velocity_var'])

            human_obs[sec_idx] = {
                'density':  {'values': rho_values_to_use, 'variances': rho_variances},
                'velocity': {'values': vel_values_to_use, 'variances': vel_variances},
            }

        else:
            # 固定分散モード: カテゴリ中央値を観測値として使用
            measurements = []
            for _ in range(count):
                z = rho_true + np.sqrt(r_human_den_perception) * np.random.randn()
                z = np.clip(z, x_min_density, x_max_density)

                cat_val = category_means[0]
                for k in range(num_human_categories - 1, -1, -1):
                    if z >= category_thresholds[k]:
                        cat_val = category_means[k]
                        break
                measurements.append(cat_val)

            measurements = np.array(measurements)

            vel_measurements = []
            for _ in range(count):
                z_v = vel_true + np.sqrt(r_human_vel_perception) * np.random.randn()
                z_v = np.clip(z_v, x_min_velocity, x_max_velocity)

                cat_val_vel = vel_means[0]
                for k in range(len(vel_thresholds) - 1):
                    if z_v >= vel_thresholds[len(vel_thresholds)-2 - k]:
                        cat_val_vel = vel_means[len(vel_thresholds)-2 - k]
                        break
                vel_measurements.append(cat_val_vel)

            vel_measurements = np.array(vel_measurements)

            human_obs[sec_idx] = {
                'density':  {'values': measurements,     'variances': None},
                'velocity': {'values': vel_measurements, 'variances': None},
            }
    return human_obs
