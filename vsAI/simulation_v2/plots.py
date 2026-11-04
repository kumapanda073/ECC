import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from paths import result_path
from config import num_sections, T
from calibration import bin_calibration

COLORS = {
    'cls_kf(proposed)': '#F57C00',
    'reg_only':         '#2E7D32',
    'reg_kf':           '#1565C0',
    'ens_kf':           '#6A1B9A',  # Deep Ensembles + KF用
    'true':             '#222222',
}

METHODS = {
    'cls_kf':   ('Classification + KF', COLORS['cls_kf(proposed)'], 1.0),
    'reg_only': ('Regression Only',     COLORS['reg_only'],         1.0),
    'reg_kf':   ('Regression + KF',     COLORS['reg_kf'],           0.35),
}

BAR_ORDER = ('reg_only', 'reg_kf', 'cls_kf')
LINE_ORDER = ('cls_kf', 'reg_only', 'reg_kf')


def setup_paper_style():
    matplotlib.rcParams.update({
        'font.family':        'Hiragino Sans',
        'font.size':          13,
        'axes.titlesize':     14,
        'axes.labelsize':     13,
        'xtick.labelsize':    12,
        'ytick.labelsize':    12,
        'legend.fontsize':    11,
        'legend.framealpha':  0.85,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.linewidth':     1.2,
        'grid.alpha':         0.35,
        'grid.linestyle':     '--',
        'lines.linewidth':    2.0,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
    })


def _save(fig_path, message):
    plt.savefig(fig_path)
    print(f"{message}: {fig_path}")
    plt.close()


def save_metric_comparison(metric, avg, std):
    # 区間別の指標を手法ごとの棒グラフで比較する

    x_pos  = np.arange(num_sections)
    labels = [f"Sec {i+1}" for i in range(num_sections)]
    n_bars = len(BAR_ORDER)
    w = 0.8 / n_bars
    err_kw = dict(elinewidth=1.2, capsize=4, capthick=1.2, ecolor='#444444')
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * w

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    for ax, sl, title, panel in [
        (axes[0], slice(None, num_sections), 'Density', '(a)'),
        (axes[1], slice(num_sections, None), 'Speed',   '(b)'),
    ]:
        for off, m in zip(offsets, BAR_ORDER):
            label, color, _ = METHODS[m]
            ax.bar(x_pos + off, avg[m][sl], yerr=std[m][sl], width=w,
                   label=label, color=color, alpha=0.75, error_kw=err_kw)
        ax.set_title(f'{panel}  {metric} by Section ({title})', loc='left', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, axis='y')
        ax.set_axisbelow(True)
    plt.tight_layout(pad=1.5)
    _save(result_path(metric.lower(), f"vsai_{metric.lower()}_comparison.png"), "グラフ保存")


def _plot_section_estimates(sec_idx, x_true, estimates, tsl, figsize, error_legend):
    # 1区間の 密度/速度 の推定線と絶対誤差を4段で描く
    time_axis = np.arange(T)[tsl]

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.8, 3, 1.8]})
    specs = [
        (axes[0], axes[1], sec_idx,                'Density [ped/m²]', '(a)', '(b)'),
        (axes[2], axes[3], sec_idx + num_sections, 'Speed [m/s]',      '(c)', '(d)'),
    ]
    for ax_ts, ax_err, idx, ylabel, panel_ts, panel_err in specs:
        unit = ylabel.split('[')[1].rstrip(']')
        name = ylabel.split(' [')[0]
        truth = x_true[idx, tsl]

        ax_ts.plot(time_axis, truth, label='Ground Truth', color=COLORS['true'],
                   linestyle='--', linewidth=2.0)
        for m in LINE_ORDER:
            label, color, alpha = METHODS[m]
            est = estimates[m][idx, tsl]
            ax_ts.plot(time_axis, est, label=label, color=color, linewidth=2.0, alpha=alpha)
            ax_err.plot(time_axis, np.abs(est - truth), label=label, color=color,
                        linewidth=2.0, alpha=alpha)

        ax_ts.set_title(f'{panel_ts}  Section {sec_idx + 1}: {name}', loc='left', fontweight='bold')
        ax_ts.set_ylabel(ylabel)
        ax_ts.legend(loc='upper right', ncol=2)
        ax_ts.grid(True)

        ax_err.set_title(f'{panel_err}  Absolute Error ({name})', loc='left', fontweight='bold')
        ax_err.set_ylabel(f'|Error| [{unit}]')
        if error_legend:
            ax_err.legend(loc='upper right', ncol=2)
        ax_err.grid(True)

    axes[3].set_xlabel('Time Step')
    return axes


def plot_timeseries(sec_idx, x_true, estimates, t_max=40):
    axes = _plot_section_estimates(sec_idx, x_true, estimates, slice(0, t_max + 1),
                                   figsize=(11, 12), error_legend=True)
    axes[3].set_xlim(0, t_max)
    plt.tight_layout(pad=1.5)
    _save(result_path("timeseries", f"vsai_timeseries_sec{sec_idx + 1}.png"), "時系列グラフ保存")


def plot_raw_outputs(sec_idx, x_true, estimates):
    _plot_section_estimates(sec_idx, x_true, estimates, slice(None),
                            figsize=(11, 13), error_legend=False)
    plt.tight_layout(pad=1.5)
    _save(result_path("raw_outputs", f"vsai_raw_outputs_sec{sec_idx + 1}.png"), "生出力グラフ保存")


def plot_variance_calibration(den_var, den_sqerr, vel_var, vel_sqerr, label='Classification'):
    fig, axes = plt.subplots(2, 1, figsize=(8, 11))
    specs = [
        (axes[0], den_var, den_sqerr, 'Density', '(a)'),
        (axes[1], vel_var, vel_sqerr, 'Speed',   '(b)'),
    ]
    for ax, var, sqerr, title, panel in specs:
        bin_var, bin_sqerr = bin_calibration(var, sqerr)
        ax.plot(bin_var, bin_sqerr, 'o-', color=COLORS['cls_kf(proposed)'], label=label, markersize=7)

        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], color=COLORS['true'], linestyle='--',
                linewidth=1.5, label='y = x (perfect calibration)')
        ax.set_xlabel('Predicted Variance')
        ax.set_ylabel('Actual Squared Error')
        ax.set_title(f'{panel}  Variance Calibration ({title})', loc='left', fontweight='bold')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(pad=1.5)
    _save(result_path("raw_outputs", "vsai_variance_calibration.png"), "分散較正プロット保存")
