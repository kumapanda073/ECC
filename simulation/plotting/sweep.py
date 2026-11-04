import numpy as np
import matplotlib.pyplot as plt

from config import num_sections
from plotting.style import FIGURE_DIR, _fig_path, COLORS, PAPER_RC, PAPER_STYLE


def _save_per_errorrate_outputs(avg_phy, std_phy, avg_nlp_fixed, std_nlp_fixed,
                                avg_nlp, std_nlp, text_error_rate, metric='RMSE'):
    suffix = f"_err{int(text_error_rate * 100)}pct" if text_error_rate > 0 else ""
    x_pos  = np.arange(num_sections)
    labels = [f"Sec {i+1}" for i in range(num_sections)]
    n_bars = 3
    w = 0.7 / n_bars

    err_kw = dict(elinewidth=1.2, capsize=4, capthick=1.2, ecolor='#444444')
    panel_labels = ['(a)', '(b)']

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 10))
    for i, (ax, sl, title) in enumerate([
        (axes[0], slice(None, num_sections), 'Density'),
        (axes[1], slice(num_sections, None), 'Velocity'),
    ]):
        offsets = np.linspace(-(n_bars-1)/2, (n_bars-1)/2, n_bars) * w
        ax.bar(x_pos + offsets[0], avg_phy[sl],       yerr=std_phy[sl],       width=w,
               label='Phy Only',      color=COLORS['phy'],       alpha=0.75, error_kw=err_kw)
        ax.bar(x_pos + offsets[1], avg_nlp_fixed[sl], yerr=std_nlp_fixed[sl], width=w,
               label='NLP Fixed Var', color=COLORS['nlp_fixed'], alpha=0.75, error_kw=err_kw)
        ax.bar(x_pos + offsets[2], avg_nlp[sl],       yerr=std_nlp[sl],       width=w,
               label='NLP Dyn Var',   color=COLORS['nlp_dyn'],   alpha=0.75, error_kw=err_kw)
        ax.set_title(f'{panel_labels[i]}  {metric} by Section ({title})', loc='left', fontweight='bold')
        ax.set_ylabel(metric)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, axis='y')
        ax.set_axisbelow(True)
    fname = f"accuracy_{metric.lower()}_comparison{suffix}.png"
    path = _fig_path(f"accuracy_{metric.lower()}", fname)
    plt.savefig(path)
    print(f"グラフ保存: {path}")
    plt.close()


def _save_sweep_summary(results, rates, rate_name, methods, subdir):
    # sweep実験の数値まとめ。CSV・生データ(npz)・幅(std)の折れ線グラフを保存する。
    import csv
    parts = [('Density', slice(None, num_sections)), ('Speed', slice(num_sections, None))]
    rows = []
    for rate in rates:
        for label, key in methods.items():
            raw = results[rate][f'raw_{key}']
            n = raw.shape[0]
            for part, sl in parts:
                per_sec = raw[:, sl] 
                sec_avg = per_sec.mean(axis=1)
                trial_std = sec_avg.std(ddof=1)
                rows.append({
                    rate_name: int(round(rate * 100)), 'method': label, 'state': part,
                    'mean':      round(float(sec_avg.mean()), 5),
                    'band_std':  round(float(per_sec.std(axis=0).mean()), 5),
                    'trial_std': round(float(trial_std), 5),
                    'ci95':      round(float(1.96 * trial_std / np.sqrt(n)), 5),
                    'min':       round(float(sec_avg.min()), 5),
                    'max':       round(float(sec_avg.max()), 5),
                })

    csv_path = _fig_path(subdir, f"{rate_name}_summary.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"数値まとめ保存: {csv_path}")

    # 生データ : 後から箱ひげ図や検定をするとき用
    raw_path = _fig_path(subdir, f"{rate_name}_raw_rmse.npz")
    np.savez(raw_path, **{f"{key}_r{int(round(rate * 100))}": results[rate][f'raw_{key}']
                          for rate in rates for key in methods.values()})
    print(f"生データ保存: {raw_path}")

    # 幅そのものの折れ線グラフ
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.0))
    x_arr = np.array([int(round(r * 100)) for r in rates])
    for i, (ax, (part, _)) in enumerate(zip(axes, parts)):
        for label, key in methods.items():
            ys = [r['band_std'] for r in rows if r['method'] == label and r['state'] == part]
            ax.plot(x_arr, ys, marker='o', color=COLORS[{'phy': 'phy', 'nlp_fixed': 'nlp_fixed', 'distonly': 'nlp_distonly', 'nlp': 'nlp_dyn'}[key]], label=label)
        ax.set_title(f"{['(a)', '(b)'][i]}  Std of RMSE vs. {rate_name} ({part})", loc='left', fontweight='bold')
        ax.set_xlabel(f'{rate_name} (%)')
        ax.set_ylabel('Std of RMSE across trials')
        ax.set_xticks(x_arr)
        ax.set_xticklabels([f'{v}%' for v in x_arr])
        ax.legend()
        ax.grid(True)
    path = _fig_path(subdir, f"{rate_name}_std.png")
    plt.savefig(path)
    print(f"グラフ保存: {path}")
    plt.close()


def _save_sweep_paper(results, rates, rate_name, keys, subdir, xlabel, show_std=True):
    parts = [('Density', slice(None, num_sections)), ('Speed', slice(num_sections, None))]
    x_arr = np.array([int(round(r * 100)) for r in rates])

    with plt.rc_context(PAPER_RC):
        if show_std:
            fig, axes = plt.subplots(
                4, 1, figsize=(3.6, 8.2), sharex=True,
                gridspec_kw={'height_ratios': [1.6, 1.15, 1.6, 1.15], 'hspace': 0.18},
            )
        else:
            fig, axes = plt.subplots(
                2, 1, figsize=(3.6, 5.0), sharex=True,
                gridspec_kw={'hspace': 0.18},
            )
        for j, (part, sl) in enumerate(parts):
            if show_std:
                ax_top, ax_bot = axes[2 * j], axes[2 * j + 1]
            else:
                ax_top, ax_bot = axes[j], None
            for key in keys:
                st = PAPER_STYLE[key]
                sec_avg = np.stack([results[r][f'raw_{key}'][:, sl].mean(axis=1) for r in rates])  # (rate数, 試行数)
                m, sd = sec_avg.mean(axis=1), sec_avg.std(axis=1, ddof=1)
                ci = 1.96 * sd / np.sqrt(sec_avg.shape[1])
                kw = dict(color=st['color'], marker=st['marker'], ls=st['ls'],
                          lw=st.get('lw', 1.1), zorder=st.get('zorder', 3), markeredgewidth=0)
                ax_top.plot(x_arr, m, label=st['label'], **kw)
                ax_top.fill_between(x_arr, m - ci, m + ci, color=st['color'], alpha=0.18, lw=0,
                                    zorder=st.get('zorder', 3) - 1)   # 95% CI 帯 (RMSE図内に残す)
                if ax_bot is not None:
                    ax_bot.plot(x_arr, sd, **kw)
            ax_top.set_title(f"({'ab'[j]}) {part}", loc='left', fontweight='bold', pad=4)
            ax_top.legend()
            ax_top.set_ylabel('RMSE')
            ax_top.grid(True, lw=0.3, alpha=0.5)
            if ax_bot is not None:
                ax_bot.set_ylim(bottom=0)
                ax_bot.margins(y=0.12)
                ax_bot.set_xticks(x_arr)
                ax_bot.set_ylabel('Std. of RMSE')
                ax_bot.grid(True, lw=0.3, alpha=0.5)
            else:
                ax_top.set_xticks(x_arr)
        axes[-1].set_xlabel(xlabel)
        for ext in ('pdf', 'png'):
            path = _fig_path(subdir, f"{rate_name}_paper.{ext}")
            fig.savefig(path, dpi=400, bbox_inches='tight')
            print(f"論文用図保存: {path}")
        plt.close(fig)


def replot_sweep_from_npz(subdir, rate_name, keys, xlabel, show_std=True):
    # replot用
    data = np.load(FIGURE_DIR / subdir / f"{rate_name}_raw_rmse.npz")
    rates = sorted({int(k.rsplit('_r', 1)[1]) / 100 for k in data.files})
    results = {r: {f'raw_{k}': data[f"{k}_r{int(round(r * 100))}"] for k in keys} for r in rates}
    _save_sweep_paper(results, rates, rate_name, keys, subdir, xlabel, show_std=show_std)


def _save_R_vs_noise(all_R, noise_rates):
    from config import r_human_quantization, r_vel_quantization
    x = np.array([int(round(nr * 100)) for nr in noise_rates])
    const = {'Density': r_human_quantization, 'Velocity': r_vel_quantization}

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.0), sharex=True)
        for j, (ax, part) in enumerate(zip(axes, ['Density', 'Velocity'])):
            st_fixed = PAPER_STYLE['nlp_fixed']
            ax.axhline(const[part], color=st_fixed['color'], ls=st_fixed['ls'], label=st_fixed['label'])
            for key in ('distonly', 'nlp'):
                st = PAPER_STYLE[key]
                med = [np.median(all_R[key][nr][part]) for nr in noise_rates]
                q1  = [np.percentile(all_R[key][nr][part], 25) for nr in noise_rates]
                q3  = [np.percentile(all_R[key][nr][part], 75) for nr in noise_rates]
                ax.plot(x, med, color=st['color'], marker=st['marker'], ls=st['ls'],
                        lw=st.get('lw', 1.1), zorder=st.get('zorder', 3),
                        markeredgewidth=0, label=st['label'])
                ax.fill_between(x, q1, q3, color=st['color'], alpha=0.18, lw=0,
                                zorder=st.get('zorder', 3) - 1)
            ax.set_yscale('log')
            ax.set_ylabel(r'$R_k^H$')
            ax.set_title(f"({'ab'[j]}) {part}", loc='left', fontweight='bold', pad=4)
            ax.grid(True, lw=0.3, alpha=0.5)
        axes[0].legend()
        axes[-1].set_xticks(x)
        axes[-1].set_xlabel('Ratio of irrelevant texts (%)')
        for ext in ('pdf', 'png'):
            path = _fig_path('noise_rate', f'R_vs_noise.{ext}')
            fig.savefig(path, dpi=400, bbox_inches='tight')
            print(f"R_k^H 図保存: {path}")
        plt.close(fig)

    # 生データ (キー例: nlp_Density_r20 = Case3, 密度, noise 20%)
    raw_path = _fig_path('noise_rate', 'R_values.npz')
    np.savez(raw_path, **{f'{k}_{p}_r{int(round(nr * 100))}': np.array(all_R[k][nr][p])
                          for k in all_R for nr in noise_rates for p in ('Density', 'Velocity')})
    print(f"R_k^H 生データ保存: {raw_path}")


def _save_noiserate_comparison(results, noise_rates):
    x_ticks = [int(nr * 100) for nr in noise_rates]
    x_arr   = np.array(x_ticks)

    den_nf_mean  = np.array([results[nr]['avg_nlp_fixed'][:num_sections].mean() for nr in noise_rates])
    den_nf_std   = np.array([results[nr]['std_nlp_fixed'][:num_sections].mean() for nr in noise_rates])
    vel_nf_mean  = np.array([results[nr]['avg_nlp_fixed'][num_sections:].mean() for nr in noise_rates])
    vel_nf_std   = np.array([results[nr]['std_nlp_fixed'][num_sections:].mean() for nr in noise_rates])
    den_do_mean  = np.array([results[nr]['avg_distonly'][:num_sections].mean() for nr in noise_rates])
    den_do_std   = np.array([results[nr]['std_distonly'][:num_sections].mean() for nr in noise_rates])
    vel_do_mean  = np.array([results[nr]['avg_distonly'][num_sections:].mean() for nr in noise_rates])
    vel_do_std   = np.array([results[nr]['std_distonly'][num_sections:].mean() for nr in noise_rates])
    den_nlp_mean = np.array([results[nr]['avg_nlp'][:num_sections].mean() for nr in noise_rates])
    den_nlp_std  = np.array([results[nr]['std_nlp'][:num_sections].mean() for nr in noise_rates])
    vel_nlp_mean = np.array([results[nr]['avg_nlp'][num_sections:].mean() for nr in noise_rates])
    vel_nlp_std  = np.array([results[nr]['std_nlp'][num_sections:].mean() for nr in noise_rates])

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 5.0))
    for i, (ax, nf_mean, nf_std, do_mean, do_std, nd_mean, nd_std, title) in enumerate([
        (axes[0], den_nf_mean, den_nf_std, den_do_mean, den_do_std, den_nlp_mean, den_nlp_std, 'Density'),
        (axes[1], vel_nf_mean, vel_nf_std, vel_do_mean, vel_do_std, vel_nlp_mean, vel_nlp_std, 'Speed'),
    ]):
        panel = ['(a)', '(b)'][i]
        ax.plot(x_arr, nf_mean, marker='s', color=COLORS['nlp_fixed'], label='Case1')
        ax.fill_between(x_arr, nf_mean - nf_std, nf_mean + nf_std, color=COLORS['nlp_fixed'], alpha=0.15)
        ax.plot(x_arr, do_mean, marker='^', color=COLORS['nlp_distonly'], label='Case2')
        ax.fill_between(x_arr, do_mean - do_std, do_mean + do_std, color=COLORS['nlp_distonly'], alpha=0.15)
        ax.plot(x_arr, nd_mean, marker='o', color=COLORS['nlp_dyn'],   label='Case3(Proposed)')
        ax.fill_between(x_arr, nd_mean - nd_std, nd_mean + nd_std, color=COLORS['nlp_dyn'],   alpha=0.15)
        ax.set_title(f'{panel}  RMSE vs. Noise Rate ({title})', loc='left', fontweight='bold')
        ax.set_xlabel('Noise Rate (%)')
        ax.set_ylabel('Mean RMSE')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{v}%' for v in x_ticks])
        ax.legend()
        ax.grid(True)
    path = _fig_path("noise_rate", "noise_rate_comparison.png")
    plt.savefig(path)
    print(f"グラフ保存: {path}")
    plt.close()


def _save_errorrate_comparison(results, error_rates):
    x_arr = np.array([int(round(er * 100)) for er in error_rates])
    base = results[error_rates[0]]
    parts = [('Density', slice(None, num_sections)), ('Speed', slice(num_sections, None))]
    st, sp = PAPER_STYLE['nlp'], PAPER_STYLE['phy']

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.0), sharex=True, layout='constrained')
        for j, (ax, (part, sl)) in enumerate(zip(axes, parts)):
            m  = np.array([results[er]['avg_nlp'][sl].mean() for er in error_rates])
            sd = np.array([results[er]['std_nlp'][sl].mean() for er in error_rates])
            ax.plot(x_arr, m, color=st['color'], marker=st['marker'], ls=st['ls'],
                    lw=st['lw'], markeredgewidth=0, zorder=5, label=st['label'])
            ax.fill_between(x_arr, m - sd, m + sd, color=st['color'], alpha=0.18, lw=0)
            ax.axhline(base['avg_phy'][sl].mean(), color=sp['color'], ls=sp['ls'], label=sp['label'])
            ax.set_title(f"({'ab'[j]}) {part}", loc='left', fontweight='bold', pad=4)
            ax.set_ylabel('RMSE')
            ax.legend()
            ax.grid(True, lw=0.3, alpha=0.5)
        axes[-1].set_xticks(x_arr)
        axes[-1].set_xlabel('Misreport rate (%)')
        for ext in ('pdf', 'png'):
            path = _fig_path('error_rate', f'error_rate_comparison.{ext}')
            fig.savefig(path, dpi=400, bbox_inches='tight')
            print(f"グラフ保存: {path}")
        plt.close(fig)
