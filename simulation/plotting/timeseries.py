"""1試行分の時系列図。"""

import numpy as np
import matplotlib.pyplot as plt

from config import num_sections, T
from plotting.style import _fig_path, COLORS


def plot_timeseries(sec_idx, x_true, xe_phy, xe_nlp):
    idx_rho   = sec_idx
    idx_vel   = sec_idx + num_sections
    time_axis = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(3.6, 8.2), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.8, 3, 1.8]})
    specs = [
        (axes[0], axes[1], idx_rho, 'Density [ped/m²]', '(a)', '(b)'),
        (axes[2], axes[3], idx_vel, 'Speed [m/s]',   '(c)', '(d)'),
    ]
    for ax_ts, ax_err, idx, ylabel, panel_ts, panel_err in specs:
        unit = ylabel.split('[')[1].rstrip(']')

        ax_ts.plot(time_axis, x_true[idx],   label='Ground Truth',  color=COLORS['true'],     linestyle='--', linewidth=2.0)
        ax_ts.plot(time_axis, xe_phy[idx],   label='Phy Only',      color=COLORS['phy'],       linewidth=1.5, alpha=0.8)
        ax_ts.plot(time_axis, xe_nlp[idx],   label='NLP Dyn Var',   color=COLORS['nlp_dyn'],   linewidth=2.0)
        ax_ts.set_title(f'{panel_ts}  Section {sec_idx + 1}: {ylabel.split(" [")[0]}', loc='left', fontweight='bold')
        ax_ts.set_ylabel(ylabel)
        ax_ts.legend(loc='upper right', ncol=2)
        ax_ts.grid(True)

        err_phy   = np.abs(xe_phy[idx]   - x_true[idx])
        err_nlp   = np.abs(xe_nlp[idx]   - x_true[idx])
        ax_err.plot(time_axis, err_phy,   color=COLORS['phy'],       linewidth=1.5, alpha=0.8,  label='Phy Only')
        ax_err.plot(time_axis, err_nlp,   color=COLORS['nlp_dyn'],   linewidth=2.0,             label='NLP Dyn Var')

        ax_err.set_title(f'{panel_err}  Absolute Error ({ylabel.split(" [")[0]})', loc='left', fontweight='bold')
        ax_err.set_ylabel(f'|Error| [{unit}]')
        ax_err.legend(loc='upper right', ncol=3)
        ax_err.grid(True)

    axes[3].set_xlabel('Time Step')
    fname = f"timeseries_sec{sec_idx + 1}.png"
    path = _fig_path("timeseries", fname)
    plt.savefig(path)
    print(f"時系列グラフ保存: {path}")
    plt.close()
