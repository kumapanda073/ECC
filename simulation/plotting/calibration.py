import numpy as np
import matplotlib.pyplot as plt

from config import CALIB_N_BINS
from plotting.style import FIGURE_DIR, _fig_path, PAPER_RC, PAPER_STYLE


# =========================================================
# キャリブレーション図: モデルが付けたRと、実際の観測誤差の二乗が合っているか
#   横軸 = 付けた R (ビン内平均) / 縦軸 = 実際の (y - x_true)^2 (ビン内平均)
#   対角線 y = x に乗っていれば R が正しく見積もれている。
#   対角線より上 = 自信過剰 (R が小さすぎ) / 下 = 自信過小 (R が大きすぎ)
# =========================================================
CALIB_SUBDIR = 'calibration'
CALIB_KEYS = ('nlp_fixed', 'distonly', 'nlp')


def _calib_bins(R, err2, n_bins=CALIB_N_BINS):
    R, err2 = np.asarray(R, float), np.asarray(err2, float)
    edges = np.unique(np.quantile(R, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        groups = [np.ones_like(R, dtype=bool)]
    else:
        idx = np.clip(np.searchsorted(edges, R, side='right') - 1, 0, len(edges) - 2)
        groups = [idx == b for b in range(len(edges) - 1)]
    xs, ys, cis, ns = [], [], [], []
    for m in groups:
        n = int(m.sum())
        if n == 0:
            continue
        xs.append(R[m].mean())
        ys.append(err2[m].mean())
        cis.append(1.96 * err2[m].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0)
        ns.append(n)
    return np.array(xs), np.array(ys), np.array(cis), np.array(ns)


def _plot_calibration(pooled, fname_stem, n_bins=CALIB_N_BINS, subdir=CALIB_SUBDIR):
    import csv
    parts = ['Density', 'Velocity']
    rows = []

    # 先に全パネルのビン・軸範囲を計算する
    panels = []
    for j, part in enumerate(parts):
        xlo, xhi, ylo, yhi = np.inf, -np.inf, np.inf, -np.inf
        curves = []
        for key in CALIB_KEYS:
            R, e = pooled[key][part]['R'], pooled[key][part]['err2']
            if len(R) == 0:
                continue
            x, y, ci, _ = _calib_bins(R, e, n_bins)
            curves.append((key, x, y, ci))
            xlo, xhi = min(xlo, x.min()), max(xhi, x.max())
            ylo, yhi = min(ylo, y.min()), max(yhi, y.max())
            rows.append({
                'case': PAPER_STYLE[key]['label'], 'state': part, 'n_obs': len(R),
                'mean_R': round(float(np.mean(R)), 6),
                'mean_err2': round(float(np.mean(e)), 6),
                'mean_err2_over_R': round(float(np.mean(np.asarray(e) / np.asarray(R))), 4),
            })
        xlo, xhi = xlo / 1.5, xhi * 1.5
        ylo, yhi = ylo / 1.5, yhi * 1.5
        if j == 0:
            yhi *= 6                     # (a)だけ上に凡例用の空きを作る
        panels.append(dict(part=part, curves=curves, xlim=(xlo, xhi), ylim=(ylo, yhi)))

    # 縦横比(縦の桁数 / 横の桁数)から枠の高さと図の高さを決める
    ratios = [np.log10(p['ylim'][1] / p['ylim'][0]) / np.log10(p['xlim'][1] / p['xlim'][0])
              for p in panels]
    AX_W  = 2.9    # 軸の横幅 [inch] (3.6 から y軸ラベル・目盛りの分を引いた目安)
    EXTRA = 1.2    # タイトル・x軸ラベル・目盛りなどの高さの合計 [inch]
    fig_h = AX_W * sum(ratios) + EXTRA

    # 描画
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 1, figsize=(3.6, fig_h), layout='constrained',
                                 gridspec_kw={'height_ratios': ratios})
        for j, (ax, p) in enumerate(zip(axes, panels)):
            for key, x, y, ci in p['curves']:
                st = PAPER_STYLE[key]
                lower = np.minimum(ci, y * 0.999)          # 対数軸で下ひげが 0 以下にならないように
                ax.errorbar(x, y, yerr=[lower, ci], color=st['color'], marker=st['marker'],
                            ls=st['ls'], lw=2.1, markersize=6, zorder=st.get('zorder', 3),
                            markeredgewidth=0, capsize=1.5, elinewidth=0.6, label=st['label'])
            xlo, xhi = p['xlim']
            ax.plot([xlo, xhi], [xlo, xhi], color='k', lw=0.6, ls=':', zorder=1,
                    label='Perfect calibration')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(*p['xlim'])
            ax.set_ylim(*p['ylim'])
            ax.set_aspect('equal')
            ax.set_xlabel(r'Assigned variance $R_{\mathrm{k}}^{\mathrm{H}}$')
            ax.set_ylabel(r'Empirical error $(y - x_{\mathrm{true}})^2$')
            disp = 'Speed' if p['part'] == 'Velocity' else p['part']
            ax.set_title(f"({'ab'[j]}) {disp}", loc='left',
                         fontweight='bold', pad=4)
            ax.grid(True, lw=0.3, alpha=0.5)
        axes[0].legend(loc='upper left')
        for ext in ('pdf', 'png'):
            path = _fig_path(subdir, f'{fname_stem}.{ext}')
            fig.savefig(path, dpi=400, bbox_inches='tight')
            print(f"キャリブレーション図保存: {path}")
        plt.close(fig)

    csv_path = _fig_path(subdir, f'{fname_stem}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"キャリブレーション数値保存: {csv_path}")
    for r in rows:
        print(f"  {r['state']:8s} {r['case']:36s} mean(err2/R) = {r['mean_err2_over_R']}")


def _save_calibration(all_obs, noise_rates, n_bins=CALIB_N_BINS):
    raw = {}
    for key in CALIB_KEYS:
        for nr in noise_rates:
            for part in ('Density', 'Velocity'):
                tag = f'{key}_{part}_r{int(round(nr * 100))}'
                raw[f'{tag}_R'] = np.asarray(all_obs[key][nr][part]['R'], float)
                raw[f'{tag}_err2'] = np.asarray(all_obs[key][nr][part]['err2'], float)
    raw_path = _fig_path(CALIB_SUBDIR, 'calibration_raw.npz')
    np.savez(raw_path, **raw)
    print(f"キャリブレーション生データ保存: {raw_path}")
    _plot_calibration_from_raw(raw, noise_rates, n_bins)


def _plot_calibration_from_raw(raw, noise_rates, n_bins=CALIB_N_BINS):
    # 全noise rateの観測をまとめて1枚の図にする
    pooled = {}
    for key in CALIB_KEYS:
        pooled[key] = {}
        for part in ('Density', 'Velocity'):
            tags = [f'{key}_{part}_r{int(round(nr * 100))}' for nr in noise_rates]
            pooled[key][part] = {
                'R':    np.concatenate([raw[f'{t}_R'] for t in tags]),
                'err2': np.concatenate([raw[f'{t}_err2'] for t in tags]),
            }
    _plot_calibration(pooled, 'calibration_all', n_bins=n_bins)


def replot_calibration_from_npz(n_bins=CALIB_N_BINS):
    # replot用
    data = np.load(FIGURE_DIR / CALIB_SUBDIR / 'calibration_raw.npz')
    raw = {k: data[k] for k in data.files}
    noise_rates = sorted({int(k.split('_r')[-1].split('_')[0]) / 100 for k in raw})
    _plot_calibration_from_raw(raw, noise_rates, n_bins)
