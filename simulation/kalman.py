import numpy as np
from config import (
    num_sections, n_states,
    r_phy_density, r_phy_velocity,
    r_human_quantization, r_vel_quantization,
    x_min_density, x_max_density,
    x_min_velocity, x_max_velocity,
)


def _clip_state(x):
    x[:num_sections] = np.clip(x[:num_sections], x_min_density, x_max_density)
    x[num_sections:] = np.clip(x[num_sections:], x_min_velocity, x_max_velocity)
    return x


def kalman_step(x_prev_est, P, u_t, A_mat, B_mat, Q_mat, w_mean, y_phy, human_obs):
    dim_x = n_states

    # 時間更新
    x_pred = A_mat @ x_prev_est + B_mat @ u_t + w_mean
    x_pred = _clip_state(x_pred)        # モデルと同様のクリッピング
    P_pred = A_mat @ P @ A_mat.T + Q_mat

    H_list, R_list, y_list = [], [], []

    # 物理センサ
    for sec_idx, val_dict in y_phy.items():
        # 密度
        row = np.zeros(dim_x)
        row[sec_idx] = 1.0
        H_list.append(row)
        R_list.append(r_phy_density)
        y_list.append(val_dict['density'])
        # 速度
        row_vel = np.zeros(dim_x)
        row_vel[sec_idx + num_sections] = 1.0
        H_list.append(row_vel)
        R_list.append(r_phy_velocity)
        y_list.append(val_dict['velocity'])

    # 人センサ
    for sec_idx, data in human_obs.items():
        # 密度
        if 'density' in data:
            rho_values    = data['density']['values']
            rho_variances = data['density']['variances']
            for i, val in enumerate(rho_values):
                row = np.zeros(dim_x)
                row[sec_idx] = 1.0
                H_list.append(row)
                R_list.append(rho_variances[i] if rho_variances is not None else r_human_quantization)
                y_list.append(val)

        # 速度
        if 'velocity' in data:
            vel_values    = data['velocity']['values']
            vel_variances = data['velocity']['variances']
            for i, val in enumerate(vel_values):
                row = np.zeros(dim_x)
                row[sec_idx + num_sections] = 1.0
                H_list.append(row)
                R_list.append(vel_variances[i] if vel_variances is not None else r_vel_quantization)
                y_list.append(val)

    # 観測更新
    if y_list:
        H_mat = np.array(H_list)
        R_mat = np.diag(R_list)
        y_vec_obs = np.array(y_list)

        S = H_mat @ P_pred @ H_mat.T + R_mat
        K = P_pred @ H_mat.T @ np.linalg.inv(S)

        x_est = x_pred + K @ (y_vec_obs - H_mat @ x_pred)
        P = (np.eye(dim_x) - K @ H_mat) @ P_pred
    else:
        x_est = x_pred
        P = P_pred

    # 推定値クリッピング
    x_est = _clip_state(x_est)
    return x_est, P
