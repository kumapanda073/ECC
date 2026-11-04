from pathlib import Path
import matplotlib
matplotlib.use('Agg')

# 図の出力先
FIGURE_DIR = Path(__file__).resolve().parents[1] / 'figure'

def _fig_path(subdir, fname):
    d = FIGURE_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / fname


# 論文用図と全図共通の設定のベース
PAPER_RC = {
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'lines.linewidth': 1.1, 'lines.markersize': 3.5,
    'grid.linestyle': '-',
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
}

# 旧形式の図(区間ごとの棒グラフ・時系列など)の色
COLORS = {
    'phy':          '#1565C0',  # dark blue
    'nlp_fixed':    '#B71C1C',  # dark red
    'nlp_distonly': '#00ACC1',  # cyan-blue (分類ヘッドのみ = dist_var, Phyの濃紺とは別系統)
    'nlp_dyn':      '#F57C00',  # bright orange
    'true':         '#222222',  # near-black
}

# 論文用図
PAPER_STYLE = {
    'phy':       dict(color='#999999', marker='v', ls=(0, (1, 1)),  label='Physical only'),
    'nlp_fixed': dict(color='#656565', marker='s', ls='--',         label='Fixed var. (Case 1)'),
    'distonly':  dict(color='#0072B2', marker='^', ls='-.',         label='Cls Head only (Case 2)'),
    'nlp':       dict(color='#D55E00', marker='o', ls='-', lw=1.6,  label='Dual Head (Case 3)', zorder=5),
}


def _setup_paper_style():
    matplotlib.rcParams.update({
        **PAPER_RC,
        'font.family':        'Hiragino Sans',
        'legend.framealpha':  0.85,
        'grid.alpha':         0.35,
        'savefig.dpi':        300,
        'savefig.bbox':       'tight',
    })

_setup_paper_style()
