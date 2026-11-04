import numpy as np
from config import num_sections


def calculate_rmse(true, est):
    return np.sqrt(np.mean((true - est) ** 2, axis=1))


def calculate_mae(true, est):
    return np.mean(np.abs(true - est), axis=1)


def _collect_obs(human_obs_data, x_true, store):
    for t, secs in human_obs_data.items():
        for sec, d in secs.items():
            for part, key, idx in (('Density', 'density', sec),
                                   ('Velocity', 'velocity', sec + num_sections)):
                var = d[key]['variances']
                if var is None:
                    continue
                vals = np.asarray(d[key]['values'], dtype=float)
                store[part]['R'].extend(var)
                store[part]['err2'].extend((vals - x_true[idx, t]) ** 2)
