"""Injection-recovery in the FULL age-[Fe/H] plane, overall sample (LAMOST truth).

Companion to plot_lamost_injrec_deconv.py (which shows the 1-D age marginals per
population).  Here we take the WHOLE LAMOST sample as ground truth, degrade every
star's age with noise drawn from the full AstroNN sigma_lnage distribution (per
star, not the mean), and deconvolve with XD -- to show how well the age-metallicity
STRUCTURE survives AstroNN-scale errors and is recovered.

Three panels of the same age-[Fe/H] plane, common colour scale (log density):
  (a) truth      LAMOST observed AMR
  (b) degraded   after per-star AstroNN-scale age noise  (before deconvolution)
  (c) recovered  XD deconvolution of (b)                 (after deconvolution)
The truth 50/90%-mass contours are overlaid on (b) and (c) so the smearing and the
recovery are visible against the same reference.

    python scripts_repro/plot_lamost_injrec_amr.py
Writes figures_repro/01_eos_amr_deconv_lamost_amr.png .
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.xd import XD
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 15, 'axes.labelsize': 17, 'axes.titlesize': 16,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

FIG = REPO + '/figures_repro'
AGER = (0.5, 14.0); FEHR = (-1.1, 0.5)
K_ALL = 10                 # generous, floor-regularised
WREG = np.array([0.05**2, 0.02**2])
FIT_CAP = 20000
SEED = 7
rng = np.random.default_rng(SEED)

c = Cuts()
L = load_catalog(REPO + '/data_repro/our_lamost_subgiant_ddpayne.fits.gz')
mL = make_masks(L, c)
feh = np.asarray(L['fe_h'], float)
feherr = np.asarray(L['fe_h_err'], float) if 'fe_h_err' in L.dtype.names else np.full(len(feh), 0.02)
age = np.asarray(L['age'], float); aerr = np.asarray(L['age_model_error'], float)
base = np.asarray(mL['base'], bool)
# strict age-quality cut for the LAMOST ground truth: absolute age error < 1 Gyr
AERR_ABS_MAX = 1.0
rel = (base & np.isfinite(age) & np.isfinite(aerr) & (age > 0) & (age < 14)
       & (aerr / age < 0.3) & (aerr < AERR_ABS_MAX) & np.isfinite(feh)
       & (feh > FEHR[0]) & (feh < FEHR[1]))
age = age[rel]; feh = feh[rel]; feherr = feherr[rel]
N = len(age)
print('overall N =', N)

# AstroNN sigma_lnage pool
A = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
aa = np.asarray(A['age'], float); ae = np.asarray(A['age_model_error'], float)
okA = np.isfinite(aa) & np.isfinite(ae) & (aa > 0) & (ae / aa < 0.3)
ASTRONN_SIG = (ae / aa)[okA]

# inject per-star AstroNN-scale age noise (whole sample) for the degraded map
sig_inj = rng.choice(ASTRONN_SIG, N)
age_deg = np.exp(np.log(age) + rng.normal(0, sig_inj))

# ---- density maps -----------------------------------------------------------
NB = 90
xe = np.linspace(*AGER, NB + 1); ye = np.linspace(*FEHR, NB + 1)
xc = 0.5 * (xe[:-1] + xe[1:]); yc = 0.5 * (ye[:-1] + ye[1:])
dx = xc[1] - xc[0]; dy = yc[1] - yc[0]


def hist_density(a, f):
    h, _, _ = np.histogram2d(a, f, bins=[xe, ye])
    h = gaussian_filter(h, 1.0)
    h /= h.sum() * dx * dy                       # normalise to a pdf
    return h


truth = hist_density(age, feh)
degr = hist_density(age_deg, feh)

# ---- XD deconvolution of the degraded plane ---------------------------------
idx = rng.choice(N, min(FIT_CAP, N), replace=False)
X = np.column_stack([np.log(age_deg[idx]), feh[idx]])
S = np.zeros((len(idx), 2, 2)); S[:, 0, 0] = sig_inj[idx] ** 2; S[:, 1, 1] = feherr[idx] ** 2
mdl = XD(K_ALL, w_reg=WREG, max_iter=500, random_state=SEED).fit(X, S)
AG, FG = np.meshgrid(xc, yc, indexing='ij')
pts = np.column_stack([np.log(AG).ravel(), FG.ravel()])
det = mdl.V[:, 0, 0] * mdl.V[:, 1, 1] - mdl.V[:, 0, 1] * mdl.V[:, 1, 0]
rec = np.zeros(len(pts))
for a_, mk, Vk, dk in zip(mdl.alpha, mdl.mu, mdl.V, det):
    inv = np.array([[Vk[1, 1], -Vk[0, 1]], [-Vk[1, 0], Vk[0, 0]]]) / dk
    d = pts - mk
    rec += a_ * np.exp(-0.5 * np.einsum('ni,ij,nj->n', d, inv, d)) / (2 * np.pi * np.sqrt(dk))
rec = (rec.reshape(AG.shape) / AG)             # ln->linear age Jacobian
rec /= rec.sum() * dx * dy


def mass_contours(Zc, fracs=(0.3, 0.6, 0.9)):
    z = Zc.ravel(); order = np.argsort(z)[::-1]; cs = np.cumsum(z[order]) * dx * dy
    cs /= cs[-1]
    return sorted({z[order][min(np.searchsorted(cs, fr), len(z) - 1)] for fr in fracs})


pool = np.concatenate([np.log10(m[m > 0]).ravel() for m in (truth, degr, rec)])
vmin, vmax = np.percentile(pool, [5, 99.7])

fig, ax = plt.subplots(1, 3, figsize=(15.8, 5.0), sharex=True, sharey=True,
                       constrained_layout=True)
maps = [('(a) truth  (LAMOST MSTO)', truth), ('(b) degraded  (+ AstroNN-scale age noise)', degr),
        ('(c) recovered  (XD deconvolution)', rec)]
tru_lev = mass_contours(truth)
im = None
for a, (title, Z) in zip(ax, maps):
    Zm = np.ma.masked_less_equal(Z, 0)
    im = a.pcolormesh(xe, ye, np.log10(Zm).T, cmap='magma', vmin=vmin, vmax=vmax,
                      shading='auto', rasterized=True)
    # truth reference (cyan) on the degraded / recovered panels
    if not title.startswith('(a)'):
        a.contour(xc, yc, truth.T, levels=tru_lev, colors='cyan', linewidths=1.3,
                  linestyles='--', zorder=5)
    # this panel's OWN 30/60/90% mass contours (white) so the shapes compare directly
    a.contour(xc, yc, Z.T, levels=mass_contours(Z), colors='white', linewidths=1.5, zorder=6)
    a.set_title(title, fontsize=14)
    a.set_xlabel('age [Gyr]'); a.set_xlim(*AGER); a.set_ylim(*FEHR)
ax[0].set_ylabel('[Fe/H]')
# contour legend on panel (a)
ax[0].plot([], [], color='white', lw=1.8, label='this panel, 30/60/90% mass')
ax[0].plot([], [], color='cyan', lw=1.6, ls='--', label='truth, 30/60/90% mass')
ax[0].legend(loc='lower left', fontsize=10.5, frameon=True, facecolor='black',
             edgecolor='none', framealpha=0.45, labelcolor='white')
cb = fig.colorbar(im, ax=list(ax), location='right', pad=0.012, aspect=32)
cb.set_label(r'$\log_{10}$ density')
fig.suptitle('Age-[Fe/H] plane, overall sample: before vs after XD deconvolution at AstroNN age errors',
             fontsize=14)

os.makedirs(FIG, exist_ok=True)
out = f'{FIG}/01_eos_amr_deconv_lamost_amr.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
print(f'injected sigma_lnage: median {np.median(sig_inj):.3f}  (AstroNN)')
