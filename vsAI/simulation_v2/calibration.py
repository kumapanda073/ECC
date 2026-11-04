import numpy as np

from sensing import iter_observations


def apply_reg_var_offset(reg_bridge, offsets, floor):
    for name in ('density', 'velocity'):
        attr = f'{name}_R'
        R_raw = getattr(reg_bridge, attr)
        R_new = R_raw - offsets[name]
        if R_new < floor:
            print(f"[警告] {name}: R={R_raw:.5f} から {offsets[name]} を引くと {R_new:.5f} になるため "
                  f"下限 {floor} に置き換えます")
            R_new = floor
        setattr(reg_bridge, attr, R_new)
        print(f"回帰の分散 ({name}): {R_raw:.5f} - {offsets[name]} -> {R_new:.5f}")


def collect_calibration_data(x_true, bridge, human_sensor_configs, T, noise_rate=0.0):
    density_var, density_sqerr, velocity_var, velocity_sqerr = [], [], [], []
    for _, _, rho_true, vel_true, result in iter_observations(
            x_true, bridge, human_sensor_configs, T, noise_rate=noise_rate):
        density_var.append(result['density_var'])
        density_sqerr.append((result['density_mean'] - rho_true) ** 2)
        velocity_var.append(result['velocity_var'])
        velocity_sqerr.append((result['velocity_mean'] - vel_true) ** 2)
    return density_var, density_sqerr, velocity_var, velocity_sqerr


def calibrate_reg_var_oracle(reg_bridge, simulate_truth, human_sensor_configs, T, n_runs=10):
    den_v, den_e, vel_v, vel_e = [], [], [], []
    for _ in range(n_runs):
        x_cal = simulate_truth()
        dv, de, vv, ve = collect_calibration_data(x_cal, reg_bridge, human_sensor_configs, T,
                                                  noise_rate=0)
        den_v += dv; den_e += de; vel_v += vv; vel_e += ve
    reg_bridge.density_var_scale  = np.mean(den_e) / np.mean(den_v)
    reg_bridge.velocity_var_scale = np.mean(vel_e) / np.mean(vel_v)
    print(f"回帰の分散の倍率: 密度 x{reg_bridge.density_var_scale:.2f}, "
          f"速度 x{reg_bridge.velocity_var_scale:.2f}")


def bin_calibration(pred_var, sqerr, n_bins=10):
    var_arr = np.array(pred_var)
    sqerr_arr = np.array(sqerr)
    bin_edges = np.unique(np.quantile(var_arr, np.linspace(0, 1, n_bins + 1)))
    bin_idx = np.clip(np.digitize(var_arr, bin_edges[1:-1]), 0, len(bin_edges) - 2)

    bin_var_means, bin_sqerr_means = [], []
    for b in range(len(bin_edges) - 1):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_var_means.append(var_arr[mask].mean())
        bin_sqerr_means.append(sqerr_arr[mask].mean())
    return bin_var_means, bin_sqerr_means
