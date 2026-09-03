"""Per-species figure for every usable APOGEE [X/Fe]: V_tan-[Fe/H] densities with the
four labelled blocks (top), and BOTH spread measures below over -0.8<[Fe/H]<-0.5:
  - dispersion sigma (1.48*MAD), error-DECONVOLVED  (sqrt(MAD^2 - <err^2>))
  - B&K22 P95-P5
Standard global-quality sample, NO per-element flag cut. Aurora (high-a, -1.5<[Fe/H]<-1,
V_tan<100) shown as a reference line on both. One PNG per element into
figures_repro/species_dispersion/. Error bars are bootstrap (300x).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.stats import hist2d
from eos_figures.plotting import density_panel, label_axes

DATA = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_allspecies.fits.gz')
OUTD = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro/species_dispersion')
OUTD.mkdir(parents=True, exist_ok=True)

SPECIES = ['c_fe', 'ci_fe', 'n_fe', 'o_fe', 'na_fe', 'mg_fe', 'al_fe', 'si_fe', 's_fe',
           'k_fe', 'ca_fe', 'ti_fe', 'tiii_fe', 'v_fe', 'cr_fe', 'mn_fe', 'co_fe', 'ni_fe', 'ce_fe']
PRETTY = {'c_fe': '[C/Fe]', 'ci_fe': '[C I/Fe]', 'n_fe': '[N/Fe]', 'o_fe': '[O/Fe]',
          'na_fe': '[Na/Fe]', 'mg_fe': '[Mg/Fe]', 'al_fe': '[Al/Fe]', 'si_fe': '[Si/Fe]',
          's_fe': '[S/Fe]', 'k_fe': '[K/Fe]', 'ca_fe': '[Ca/Fe]', 'ti_fe': '[Ti/Fe]',
          'tiii_fe': '[Ti II/Fe]', 'v_fe': '[V/Fe]', 'cr_fe': '[Cr/Fe]', 'mn_fe': '[Mn/Fe]',
          'co_fe': '[Co/Fe]', 'ni_fe': '[Ni/Fe]', 'ce_fe': '[Ce/Fe]'}
SEL_NOTE = {'mg_fe': ' (SELECTION var)', 'al_fe': ' (SELECTION var)'}

rng = np.random.default_rng(0)
c = Cuts()
cat = load_catalog(DATA)
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); vphi = np.asarray(cat['galvt'], float)
BOX = (-0.8, -0.5); NBOOT = 300
series = [('thin_al',  (-75, 75),  'royalblue',  '-',  r'low-$\alpha$ Eos ($V_{tan}<75$)'),
          ('thin_al',  (150, 300), 'firebrick',  '-',  r'low-$\alpha$ disc ($V_{tan}>150$)'),
          ('thick_al', (-75, 75),  'darkorange', '--', r'high-$\alpha$ Splash ($V_{tan}<75$)'),
          ('thick_al', (150, 300), 'seagreen',   '--', r'high-$\alpha$ disc ($V_{tan}>150$)')]

# densities are element-independent -> precompute once
DENS = {}
for pop in ('thick_al', 'thin_al'):
    P = np.asarray(m[pop], bool) & np.isfinite(feh) & np.isfinite(vphi)
    DENS[pop] = hist2d(feh[P], vphi[P], (-1.5, 0.5), (-200, 350), 70, 70, normalize='y')

edges = np.arange(BOX[0], BOX[1] + 1e-9, 0.1); cen = 0.5 * (edges[:-1] + edges[1:])


def mad(x): x = x[np.isfinite(x)]; return 1.4826 * np.median(np.abs(x - np.median(x))) if x.size else np.nan
def sig_int(y, e):
    ok = np.isfinite(y) & np.isfinite(e); y, e = y[ok], e[ok]
    return np.sqrt(max(mad(y) ** 2 - np.mean(e ** 2), 0)) if y.size >= 10 else np.nan
def p95m5(y):
    y = y[np.isfinite(y)]; return (np.percentile(y, 95) - np.percentile(y, 5)) if y.size >= 25 else np.nan


def curves(band, y, e):
    si = np.full(len(cen), np.nan); sie = np.full(len(cen), np.nan)
    ps = np.full(len(cen), np.nan); pse = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        b = band & (feh >= edges[i]) & (feh < edges[i + 1]); yy = y[b]; ee = e[b]
        m1 = np.isfinite(yy)
        if m1.sum() >= 10:
            si[i] = sig_int(yy, ee)
            sie[i] = np.std([sig_int(*(lambda k: (yy[k], ee[k]))(rng.integers(0, yy.size, yy.size))) for _ in range(NBOOT)])
        if m1.sum() >= 25:
            ps[i] = p95m5(yy)
            pse[i] = np.std([p95m5(yy[rng.integers(0, yy.size, yy.size)]) for _ in range(NBOOT)])
    return si, sie, ps, pse


for col in SPECIES:
    y = np.asarray(cat[col], float); e = np.asarray(cat[col.replace('_fe', '_fe_err')], float)
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    # top row: densities + boxes
    for a, (pop, ptitle) in zip(ax[0], [('thick_al', r'high-$\alpha$'), ('thin_al', r'low-$\alpha$')]):
        h, xe, ye = DENS[pop]
        density_panel(a, h, xe, ye, percentiles=(2, 98))
        a.axhline(0, color='k', lw=0.6, ls=':')
        for spop, (vlo, vhi), color, ls, lab in series:
            if spop == pop:
                a.add_patch(Rectangle((BOX[0], vlo), BOX[1] - BOX[0], vhi - vlo, fill=False, edgecolor=color, lw=2.2, zorder=5))
        a.set_xlim(-1.5, 0.5); a.set_ylim(-200, 350)
        label_axes(a, '[Fe/H]', r'$V_{\rm tan}$ [km/s]', ptitle + ' sample')
    # bottom row: sigma (deconvolved) and P95-P5
    svals, pvals = [], []   # collect value +/- err to set data-driven y-limits (not forced to 0)
    for pop, (vlo, vhi), color, ls, lab in series:
        band = np.asarray(m[pop], bool) & (vphi > vlo) & (vphi < vhi)
        si, sie, ps, pse = curves(band, y, e)
        ax[1, 0].errorbar(cen, si, yerr=sie, color=color, ls=ls, marker='o', ms=5, lw=1.6, capsize=3, label=lab)
        ax[1, 1].errorbar(cen, ps, yerr=pse, color=color, ls=ls, marker='o', ms=5, lw=1.6, capsize=3, label=lab)
        f = np.isfinite(si); svals += list(si[f] - np.nan_to_num(sie)[f]) + list(si[f] + np.nan_to_num(sie)[f])
        f = np.isfinite(ps); pvals += list(ps[f] - np.nan_to_num(pse)[f]) + list(ps[f] + np.nan_to_num(pse)[f])
    # Aurora reference
    au = np.asarray(m['thick_al'], bool) & (feh > -1.5) & (feh < -1.0) & (vphi < 100)
    if np.isfinite(y[au]).sum() >= 20:
        asi = sig_int(y[au], e[au]); aps = p95m5(y[au])
        for a, val in [(ax[1, 0], asi), (ax[1, 1], aps)]:
            a.axhline(val, color='purple', ls=(0, (5, 2)), lw=1.3, zorder=1)
            a.text(-0.795, val, 'Aurora', fontsize=7, color='purple', va='bottom')
        svals.append(asi); pvals.append(aps)
    for a, vals, ttl in [(ax[1, 0], svals, 's'), (ax[1, 1], pvals, 'p')]:
        a.set_xlim(-0.82, -0.48)
        if vals:
            lo, hi = min(vals), max(vals); span = (hi - lo) or (hi or 1)
            a.set_ylim(max(0.0, lo - 0.15 * span), hi + 0.15 * span)   # data-driven, not forced from 0
        a.legend(frameon=False, fontsize=7.5, loc='upper right')
    label_axes(ax[1, 0], '[Fe/H]', rf'$\sigma_{{{PRETTY[col][1:-1]}}}$ [dex]', r'dispersion $\sigma$ (deconvolved), boxed range')
    label_axes(ax[1, 1], '[Fe/H]', f'{PRETTY[col]} P95$-$P5 [dex]', 'B&K22 spread P95$-$P5, boxed range')
    fig.suptitle(f'{PRETTY[col]}{SEL_NOTE.get(col, "")}', fontsize=13)
    fig.savefig(OUTD / f'{col}.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  {PRETTY[col]:12s} -> {col}.png')
print('done ->', OUTD)
