import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    num_sections, n_states, q_disturbance,
    inflow_main, inflow_u2, inflow_u3, inflow_v1, inflow_v2, inflow_v3,
    A_MATRIX_CSV, dim_u, INPUT_STATE_INDICES,
    q_density_base, q_velocity_base, DISTURBANCE_STATE_INDICES,
)
def build_B_matrix_block(dim_x, dim_u):
    B = np.zeros((dim_x, dim_u))
    for j, idx in enumerate(INPUT_STATE_INDICES):
        B[idx, j] = 1.0
    return B

def load_system_model():
    BASE_DIR = Path(__file__).resolve().parent
    csv_path_A = BASE_DIR / "csv" / A_MATRIX_CSV
    if not csv_path_A.exists():
        raise FileNotFoundError(f"{csv_path_A} が見つからない。")

    df_A = pd.read_csv(csv_path_A, index_col=0)
    A = df_A.values

    B = build_B_matrix_block(n_states, dim_u)

    Q = np.zeros((n_states, n_states))

    for i in range(num_sections):
        Q[i, i] = q_density_base

    for i in range(num_sections):
        idx_vel = i + num_sections
        Q[idx_vel, idx_vel] = q_velocity_base

    for idx in DISTURBANCE_STATE_INDICES:
        Q[idx, idx] += q_disturbance

    return A, B, Q


def generate_synthetic_input(T):
    U_sim = np.zeros((T, dim_u))
    U_sim[:, 0] = inflow_main
    U_sim[:, 1] = inflow_u2
    U_sim[:, 2] = inflow_u3
    U_sim[:, 3] = inflow_v1
    U_sim[:, 4] = inflow_v2
    U_sim[:, 5] = inflow_v3
    return U_sim
