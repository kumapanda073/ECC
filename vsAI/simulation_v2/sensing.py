import numpy as np

import paths
from config import (
    num_sections,
    r_human_den_perception, r_human_vel_perception,
    x_min_density, x_max_density, x_min_velocity, x_max_velocity,
)


def sensors_per_section(human_sensor_configs, t):
    counts = {}
    for c in human_sensor_configs:
        if t >= c['offset'] and (t - c['offset']) % c['interval'] == 0:
            counts[c['section']] = counts.get(c['section'], 0) + 1
    return {sec: n for sec, n in counts.items() if sec < num_sections}


def perceive(rho_true, vel_true):
    z_rho = rho_true + np.sqrt(r_human_den_perception) * np.random.randn()
    z_rho = np.clip(z_rho, x_min_density, x_max_density)
    z_vel = vel_true + np.sqrt(r_human_vel_perception) * np.random.randn()
    z_vel = np.clip(z_vel, x_min_velocity, x_max_velocity)
    return z_rho, z_vel


def iter_observations(x_true, bridge, human_sensor_configs, T,
                      error_rate=0.0, noise_rate=0.0):
    for t in range(1, T):
        for sec_idx, count in sensors_per_section(human_sensor_configs, t).items():
            rho_true = x_true[sec_idx, t]
            vel_true = x_true[sec_idx + num_sections, t]
            for _ in range(count):
                z_rho, z_vel = perceive(rho_true, vel_true)
                result = bridge.observe(z_rho, z_vel,
                                        error_rate=error_rate, noise_rate=noise_rate)
                yield t, sec_idx, rho_true, vel_true, result


def collect_raw_outputs(x_true, bridge, human_sensor_configs, T, noise_rate=0.0):
    def empty():
        return {sec: [[] for _ in range(T)] for sec in range(num_sections)}

    raw_density, raw_velocity, noisy_density, noisy_velocity = empty(), empty(), empty(), empty()
    for t, sec, _, _, result in iter_observations(x_true, bridge, human_sensor_configs, T,
                                                  noise_rate=noise_rate):
        raw_density[sec][t].append(result['density_mean'])
        raw_velocity[sec][t].append(result['velocity_mean'])
        noisy_density[sec][t].append(result.get('density_noisy', False))
        noisy_velocity[sec][t].append(result.get('velocity_noisy', False))
    return raw_density, raw_velocity, noisy_density, noisy_velocity
