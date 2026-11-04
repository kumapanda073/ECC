import numpy as np
from config import (
    num_sections, n_states, T,
    x_min_density, x_max_density,
    x_min_velocity, x_max_velocity,
    W_MEAN_DENSITY, W_MEAN_VELOCITY,
    x0_density, x0_velocity, P0_scale,
)
from sensors import observe_physical, observe_human
from kalman import kalman_step


def build_w_mean():
    w_mean = np.zeros(n_states)
    for sec, v in W_MEAN_DENSITY.items():
        w_mean[sec] = v
    for sec, v in W_MEAN_VELOCITY.items():
        w_mean[num_sections + sec] = v
    return w_mean


def run_kalman_simulation_affine(
    A_mat, B_mat, U_arr, Q_mat,
    use_human_sensor=True,
    human_sensor_configs=None,
    active_phy_sensors=None,
    x_true=None,
    use_nlp=False,          # TrueのときNLPで観測値と動的分散を生成
    nlp_bridge=None,        # NLPBridgeインスタンス (use_nlp=Trueのとき必須)
    text_error_rate=0.0,    # テキスト選択ミス率
    text_noise_rate=0.0,    # noisyプール選択率
    use_nlp_fixed_var=False, # Trueのとき分散は固定値を使用
    use_nlp_dist_var_only=False, # Trueのとき分散は動的だがヘッドはdist_varのみ
):
    
    if human_sensor_configs is None:
        human_sensor_configs = []
    if active_phy_sensors is None:
        active_phy_sensors = []

    if U_arr.shape[0] != T and U_arr.shape[1] == T:
        U_arr = U_arr.T

    dim_x = n_states

    # ------------------
    # 1. 真値の準備
    # ------------------
    generate_new_truth = (x_true is None)

    # 外乱の平均ベクトル
    w_mean = build_w_mean()

    if generate_new_truth:
        x_true = np.zeros((dim_x, T))
        x_true[:num_sections, 0] = x0_density
        x_true[num_sections:, 0] = x0_velocity

    # 観測データ格納用
    y_phy_data = {}
    human_obs_data = {}

    # -------------------------------------------------
    # 2. メインループ (真値生成 -> 観測生成 -> 推定)
    # -------------------------------------------------
    x_est = np.zeros((dim_x, T))
    x_est[:, 0] = x_true[:, 0] # 初期値は真値と同じと仮定

    P = np.eye(dim_x) * P0_scale
    P_history = np.zeros((dim_x, T))
    P_history[:, 0] = np.diag(P)

    for t in range(1, T):
        # 真値の更新
        if generate_new_truth:
            w = np.random.multivariate_normal(w_mean, Q_mat)
            x_next = A_mat @ x_true[:, t-1] + B_mat @ U_arr[t-1] + w

            x_next[:num_sections] = np.clip(x_next[:num_sections], x_min_density, x_max_density)
            x_next[num_sections:] = np.clip(x_next[num_sections:], x_min_velocity, x_max_velocity)

            x_true[:, t] = x_next

        # 物理センサ観測の生成
        y_phy_data[t] = observe_physical(x_true, t, active_phy_sensors)

        # 人センサ観測の生成
        human_obs_data[t] = {}
        if use_human_sensor and human_sensor_configs:
            human_obs_data[t] = observe_human(
                x_true, t, human_sensor_configs,
                use_nlp=use_nlp, nlp_bridge=nlp_bridge,
                text_error_rate=text_error_rate, text_noise_rate=text_noise_rate,
                use_nlp_fixed_var=use_nlp_fixed_var,
                use_nlp_dist_var_only=use_nlp_dist_var_only,
            )

        # カルマンフィルタ 
        x_est[:, t], P = kalman_step(
            x_est[:, t-1], P, U_arr[t-1], A_mat, B_mat, Q_mat, w_mean,
            y_phy_data[t], human_obs_data[t],
        )

        P_history[:, t] = np.diag(P)

    return x_true, x_est, human_obs_data, P_history
