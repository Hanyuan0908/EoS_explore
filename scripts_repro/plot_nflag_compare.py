"""Compare the nitrogen dispersion (deconvolved sigma) and B&K22 spread (P95-P5) vs [Fe/H]
for the four V_tan blocks, WITHOUT vs WITH the per-element N_FE_FLAG==0 cut, to see how much
the result depends on the flag. 2x2: rows = statistic (sigma / P95-P5), cols = (standard / flag-clean).
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
from eos_figures.plotting import label_axes
rng = np.random.default_rng(0); c = Cuts()
FIG = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/figures_repro')
cat = load_catalog('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); vphi = np.asarray(cat['galvt'], float)
nraw = np.asarray(cat['n_fe'], float); nerr = np.asarray(cat['n_fe_err'], float)
aid = np.asarray(cat['apogee_id'])

# N_FE_FLAG from allStar (dedup highest SNR, like the cache build)
d = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_DR17_all.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
did = norm(d['APOGEE_ID']); o = np.argsort(-np.asarray(d['SNR'], float))
dids = did[o]; dfl = np.asarray(d['N_FE_FLAG'], np.int64)[o]
_, fi = np.unique(dids, return_index=True); uid = dids[fi]; ufl = dfl[fi]
sr = np.argsort(uid); uid = uid[sr]; ufl = ufl[sr]
p = np.clip(np.searchsorted(uid, aid), 0, len(uid)-1); ok = uid[p] == aid
nflag = np.where(ok, ufl[p], 1)
nclean = np.where(nflag == 0, nraw, np.nan)     # flag-cleaned N

series = [('thin_al', (-75, 75), 'royalblue', '-', r'low-$\alpha$ Eos'),
          ('thin_al', (150, 300), 'firebrick', '-', r'low-$\alpha$ disc'),
          ('thick_al', (-75, 75), 'darkorange', '--', r'high-$\alpha$ Splash'),
          ('thick_al', (150, 300), 'seagreen', '--', r'high-$\alpha$ disc')]
BOX = (-0.8, -0.5); edges = np.arange(*BOX, 0.1); edges = np.append(edges, BOX[1]); cen = 0.5*(edges[:-1]+edges[1:])

def mad(x): x = x[np.isfinite(x)]; return 1.4826*np.median(np.abs(x-np.median(x))) if x.size else np.nan
def sigint(y, e):
    ok2 = np.isfinite(y) & np.isfinite(e); y, e = y[ok2], e[ok2]
    return np.sqrt(max(mad(y)**2 - np.mean(e**2), 0)) if y.size >= 10 else np.nan
def p95m5(y): y = y[np.isfinite(y)]; return (np.percentile(y, 95)-np.percentile(y, 5)) if y.size >= 25 else np.nan

def curve(band, y, stat, nb=400):
    val = np.full(len(cen), np.nan); err = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        b = band & (feh >= edges[i]) & (feh < edges[i+1])
        yy = y[b]; ee = nerr[b]; nfin = np.isfinite(yy).sum()
        need = 10 if stat == 'sig' else 25
        if nfin >= need:
            val[i] = sigint(yy, ee) if stat == 'sig' else p95m5(yy)
            if stat == 'sig':
                err[i] = np.std([sigint(*(lambda k: (yy[k], ee[k]))(rng.integers(0, yy.size, yy.size))) for _ in range(nb)])
            else:
                err[i] = np.std([p95m5(yy[rng.integers(0, yy.size, yy.size)]) for _ in range(nb)])
    return val, err

def matched(a, b, y, stat, nb=3000):
    fa, ya, ea = feh[a], y[a], nerr[a]; fb, yb, eb = feh[b], y[b], nerr[b]
    mk = np.isfinite(ya); fa, ya, ea = fa[mk], ya[mk], ea[mk]; mk = np.isfinite(yb); fb, yb, eb = fb[mk], yb[mk], eb[mk]
    st = (lambda yy, ee: sigint(yy, ee)) if stat == 'sig' else (lambda yy, ee: p95m5(yy))
    bins = np.arange(-0.8, -0.5+1e-9, 0.05); da = np.digitize(fa, bins); db = np.digitize(fb, bins); out = []
    for _ in range(nb):
        ii = rng.integers(0, len(ya), len(ya)); sa = st(ya[ii], ea[ii]); ds = []; de = []
        for k in range(1, len(bins)):
            na = (da == k).sum(); pool = np.where(db == k)[0]
            if na > 0 and len(pool) > 0:
                jj = rng.choice(pool, na, replace=True); ds.append(yb[jj]); de.append(eb[jj])
        out.append(sa - st(np.concatenate(ds), np.concatenate(de)))
    out = np.array(out); return out.mean(), out.std()

fig, ax = plt.subplots(2, 2, figsize=(12.5, 9), constrained_layout=True)
COLS = [('standard (no flag cut)', nraw), (r'$N\_FE\_FLAG==0$', nclean)]
ROWS = [('sig', r'$\sigma_{\rm [N/Fe]}$ (deconvolved) [dex]'), ('p', r'[N/Fe] P95$-$P5 [dex]')]
for r, (stat, ylab) in enumerate(ROWS):
    for cc, (ctitle, yarr) in enumerate(COLS):
        a = ax[r, cc]; vals = []
        for pop, (vlo, vhi), col, ls, lab in series:
            band = np.asarray(m[pop], bool) & (vphi > vlo) & (vphi < vhi)
            v, e = curve(band, yarr, stat)
            a.errorbar(cen, v, yerr=e, color=col, ls=ls, marker='o', ms=5, lw=1.6, capsize=3, label=lab)
            f = np.isfinite(v); vals += list(v[f]-np.nan_to_num(e)[f]) + list(v[f]+np.nan_to_num(e)[f])
        # Aurora reference (same flag state as the column)
        au = np.asarray(m['thick_al'], bool) & (feh > -1.5) & (feh < -1.0) & (vphi < 100)
        aval = sigint(yarr[au], nerr[au]) if stat == 'sig' else p95m5(yarr[au])
        if np.isfinite(aval):
            a.axhline(aval, color='purple', ls=(0, (5, 2)), lw=1.3, zorder=1)
            a.text(-0.815, aval, 'Aurora', fontsize=7.5, color='purple', va='bottom')
            vals.append(aval)
        # matched Eos-disc annotation
        eos = np.asarray(m['thin_al'], bool) & (vphi > -75) & (vphi < 75) & (feh >= -0.8) & (feh < -0.5)
        disc = np.asarray(m['thin_al'], bool) & (vphi > 150) & (vphi < 300) & (feh >= -0.8) & (feh < -0.5)
        dm, dsd = matched(eos, disc, yarr, stat)
        a.set_xlim(-0.82, -0.48)
        if vals: lo, hi = min(vals), max(vals); sp = (hi-lo) or hi; a.set_ylim(max(0, lo-0.15*sp), hi+0.15*sp)
        a.text(0.5, 0.05, f'matched Eos$-$disc = {dm:+.3f}$\\pm${dsd:.3f} ({dm/dsd:+.1f}$\\sigma$)',
               transform=a.transAxes, ha='center', fontsize=8.5, color='0.25')
        label_axes(a, '[Fe/H]', ylab, ctitle)
        if r == 0 and cc == 1: a.legend(frameon=False, fontsize=8, loc='upper right')
fig.suptitle('Nitrogen dispersion vs [Fe/H]: effect of the per-element N_FE_FLAG cut (left=off, right=on)', fontsize=12)
fig.savefig(FIG / '01_eos_Nflag_compare.png', dpi=150, bbox_inches='tight')
print('wrote', FIG / '01_eos_Nflag_compare.png')
print(f'flagged fraction (N_FE_FLAG!=0) in Eos block: {np.mean(nflag[np.asarray(m["thin_al"],bool)&(vphi>-75)&(vphi<75)&(feh>=-0.8)&(feh<-0.5)]!=0)*100:.0f}%')
