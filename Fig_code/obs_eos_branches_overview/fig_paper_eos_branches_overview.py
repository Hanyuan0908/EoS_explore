"""Publication figure (observational): five-panel overview of the two Eos branches
(metal-poor = alpha-rich upper; metal-rich = alpha-poor lower), split by the Davies
divider [Mg/Fe]=0.317*[Fe/H]+0.353.

Top row (large 2D maps):
  (a) halo [Mg/Fe]-[Fe/H] density with the accreted (dashed), high/low-a (dotted)
      and Eos divider (green) lines.
  (b) same plane coloured by mean radial action J_R (RdYlBu_r: red = high J_R).
Bottom row:
  (c) J_R distribution of the two branches.
  (d) deconvolved sigma_[N/Fe] per branch vs [Fe/H], with the low-a disc.
  (e) age distribution: low-a disc by [Fe/H]<0 (viridis, lines alpha=0.5) + the two Eos branches.

Branch->metallicity mapping is data-driven: alpha-rich (upper) median [Fe/H]=-0.71
(=> metal-poor), alpha-poor (lower) median [Fe/H]=-0.46 (=> metal-rich).
Consistent colours: red = Eos metal-poor, blue = Eos metal-rich.

Actions via AGAMA ActionFinder in the McMillan (2017) potential; 6D from the AstroNN
VAC (Mac-only). Cuts identical to all previous analysis (canonical Eos n=353).

Writes Fig_paper/obs_eos_branches_overview.pdf and .png.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy.stats import binned_statistic_2d, gaussian_kde
from astropy.io import fits
import agama

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 15, 'axes.labelsize': 18, 'axes.titlesize': 17,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 13,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 130, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

OUT = REPO + '/Fig_paper'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
c = Cuts()

CMP = '#E8112D'    # Eos metal-poor  (alpha-rich, upper)  -> red
CMR = '#1F6FB2'    # Eos metal-rich  (alpha-poor, lower)  -> blue
CDISC = '#2ca02c'  # low-alpha disc

agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(os.path.join(os.path.dirname(agama.__file__), 'data', 'McMillan17.ini'))
af = agama.ActionFinder(pot)

cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c); base = np.asarray(m['base'], bool)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); al = np.asarray(cat['al_fe'], float)
nfe = np.asarray(cat['n_fe'], float); nerr = np.asarray(cat['n_fe_err'], float)
lz = np.asarray(cat['lz'], float); rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
vphi = np.asarray(cat['galvt'], float)
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
with np.errstate(invalid='ignore'):
    ecc = (rap - rperi) / (rap + rperi)
aid = np.asarray(cat['apogee_id'])
rel_ok = np.isfinite(age) & np.isfinite(aerr) & (aerr / age < 0.3)
halo = base & ((ecc > 0.7) | (lz < 0))

def acc(f):     return c.slope_acc * f + c.inter_acc
def hl(f):      return c.slope_acc2 * f + c.inter_acc2
def divline(f): return 0.317 * f + 0.353
lowa   = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)
eos_mp = lowa & (mg > divline(feh))    # metal-poor  (alpha-rich, upper)
eos_mr = lowa & (mg <= divline(feh))   # metal-rich  (alpha-poor, lower)
disc   = np.asarray(m['thin_al'], bool) & (vphi > 150)
disc_bl = disc & (feh > -0.8) & (feh < -0.2) & np.isfinite(nfe)

# --- 6D phase space from the AstroNN VAC -> J_R for the halo ---
ann = fits.open('/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits')[1].data
def norm(a): return np.array([(s.decode() if isinstance(s, bytes) else str(s)).strip() for s in np.asarray(a)])
nid = norm(ann['APOGEE_ID']); o = np.argsort(nid); nid_s = nid[o]
pp = np.clip(np.searchsorted(nid_s, aid), 0, len(nid_s) - 1); ok = nid_s[pp] == aid; src = o[pp]
cc = lambda n: np.where(ok, np.asarray(ann[n], float)[src], np.nan)
R, phi, zz = cc('galr'), cc('galphi'), cc('galz'); vR, vt, vz = cc('galvr'), cc('galvt'), cc('galvz')
x = R * np.cos(phi); y = R * np.sin(phi); vx = vR * np.cos(phi) - vt * np.sin(phi); vy = vR * np.sin(phi) + vt * np.cos(phi)
fin = ok & np.isfinite(x) & np.isfinite(vx) & np.isfinite(vz)
sel = halo & fin
Jr = np.full(len(feh), np.nan)
Jr[sel] = af(np.column_stack([x[sel], y[sel], zz[sel], vx[sel], vy[sel], vz[sel]]))[:, 0]

# ----------------------------------------------------------------------------
# top row: two large 2D maps; bottom row: the three 1D-summary panels
fig = plt.figure(figsize=(19, 11), constrained_layout=True)
gs = fig.add_gridspec(2, 6)
ax_a = fig.add_subplot(gs[0, 0:3])   # density map
ax_b = fig.add_subplot(gs[0, 3:6])   # J_R map
ax_c = fig.add_subplot(gs[1, 0:2])   # J_R distribution
ax_d = fig.add_subplot(gs[1, 2:4])   # N dispersion
ax_e = fig.add_subplot(gs[1, 4:6])   # age distribution

def tag(a, t):
    a.text(0.03, 0.965, t, transform=a.transAxes, fontsize=18, fontweight='bold',
           va='top', ha='left', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))

XR = (-2.1, 0.6); YMG = (-0.1, 0.5)
def mg_lines(ax):
    xx = np.linspace(*XR, 50)
    ax.plot(xx, acc(xx), color='k', ls='--', lw=1.7, zorder=3, label='accreted / in-situ')
    ax.plot(xx, hl(xx),  color='k', ls=':',  lw=2.1, zorder=3, label=r'high-$\alpha$ / low-$\alpha$')
    xe = np.linspace(-0.9, -0.2, 30)
    ax.plot(xe, divline(xe), color='lime', ls='-', lw=2.6, zorder=4, label='Eos divider')

# (a) halo Mg-Fe density + lines
s = halo & np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[70, 55], range=[XR, YMG])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
im_a = ax_a.imshow(him.T, origin='lower', extent=[*XR, *YMG], aspect='auto', cmap='Greys',
                   vmin=np.nanpercentile(him, 2), vmax=np.nanpercentile(him, 99), zorder=0)
im_a.set_rasterized(True)
mg_lines(ax_a)
ax_a.text(-0.62, 0.155, 'Eos', color='k', fontsize=21, fontweight='bold', zorder=5)
ax_a.set_xlim(*XR); ax_a.set_ylim(*YMG); ax_a.set_xlabel('[Fe/H]'); ax_a.set_ylabel('[Mg/Fe]')
ax_a.text(0.03, 0.04, r'kinematic cuts: $e>0.7$ or $L_z<0$', transform=ax_a.transAxes,
          fontsize=20, va='bottom', ha='left')
tag(ax_a, '(a)')

# (b) same plane coloured by mean J_R (RdYlBu_r: red = high J_R)
VMIN, VMAX, NMIN = 300, 1000, 3
sJ = sel & np.isfinite(feh) & np.isfinite(mg)
med = binned_statistic_2d(feh[sJ], mg[sJ], Jr[sJ], statistic='mean', bins=(45, 35), range=[XR, YMG]).statistic
cnt = binned_statistic_2d(feh[sJ], mg[sJ], None, statistic='count', bins=(45, 35), range=[XR, YMG]).statistic
mj = np.where(cnt >= NMIN, med, np.nan)
im_b = ax_b.imshow(mj.T, origin='lower', extent=[*XR, *YMG], aspect='auto', cmap='RdYlBu_r',
                   vmin=VMIN, vmax=VMAX, zorder=0)
im_b.set_rasterized(True)
mg_lines(ax_b)
ax_b.text(-0.62, 0.155, 'Eos', color='k', fontsize=21, fontweight='bold', zorder=5)
ax_b.set_xlim(*XR); ax_b.set_ylim(*YMG); ax_b.set_xlabel('[Fe/H]'); ax_b.set_ylabel('[Mg/Fe]')
cb_b = fig.colorbar(im_b, ax=ax_b, pad=0.02, fraction=0.055)
cb_b.set_label(r'mean $J_R$ [kpc km/s]', fontsize=14); cb_b.ax.tick_params(labelsize=12)
tag(ax_b, '(b)')

# (c) J_R distribution of the two branches
xg = np.linspace(0, 2000, 300)
for br, col, lab in [(eos_mp, CMP, 'Eos metal-poor'), (eos_mr, CMR, 'Eos metal-rich')]:
    v = Jr[br & fin]; v = v[np.isfinite(v)]
    ax_c.hist(v, bins=np.linspace(0, 2000, 26), density=True, color=col, alpha=0.22)
    ax_c.plot(xg, gaussian_kde(v)(xg), color=col, lw=2.8, label=lab)
    ax_c.axvline(np.median(v), color=col, ls=':', lw=1.8)
ax_c.set_xlim(0, 2000); ax_c.set_xlabel(r'$J_R$ [kpc km/s]'); ax_c.set_ylabel('density')
ax_c.legend(loc='upper right', fontsize=16); tag(ax_c, '(c)')

# (d) deconvolved sigma_N per branch vs [Fe/H]
def mad(v): v = v[np.isfinite(v)]; return 1.4826 * np.median(np.abs(v - np.median(v)))
def sigint(v, e):
    o2 = np.isfinite(v) & np.isfinite(e); v, e = v[o2], e[o2]
    return np.sqrt(max(mad(v) ** 2 - np.mean(e ** 2), 0)) if v.size >= 8 else np.nan
edges = np.arange(-0.9, -0.2 + 1e-9, 0.1); cen = 0.5 * (edges[:-1] + edges[1:])
for br, col, ls, mk, lab in [(eos_mp, CMP, '-', 'o', 'Eos metal-poor'),
                             (eos_mr, CMR, '-', 'o', 'Eos metal-rich'),
                             (disc_bl, CDISC, '--', 's', r'low-$\alpha$ disc ($V_\phi>150$)')]:
    si = np.full(len(cen), np.nan); se = np.full(len(cen), np.nan)
    for i in range(len(cen)):
        b = br & (feh >= edges[i]) & (feh < edges[i+1]); v = nfe[b]; e = nerr[b]
        if np.isfinite(v).sum() >= 10:
            si[i] = sigint(v, e)
            se[i] = np.std([sigint(*(lambda k: (v[k], e[k]))(rng.integers(0, v.size, v.size))) for _ in range(400)])
    ax_d.errorbar(cen, si, yerr=se, color=col, ls=ls, marker=mk, ms=6, lw=1.9, capsize=3, label=lab)
ax_d.set_xlim(-0.92, -0.18); ax_d.set_ylim(0.03, None)
ax_d.set_xlabel('[Fe/H]'); ax_d.set_ylabel(r'$\sigma_{\rm [N/Fe]}$ (deconvolved) [dex]')
ax_d.legend(loc='upper right', fontsize=16); tag(ax_d, '(d)')

# (e) age distributions: low-alpha disc by [Fe/H] (viridis) + Eos branches
ag = np.linspace(0.5, 14, 300)
cmap = cm.viridis; nrm = colors.Normalize(-0.8, 0.0)   # cap at [Fe/H]=0
aedges = np.arange(-0.8, 0.0 + 1e-9, 0.1); EOS_LO, EOS_HI = -0.8, -0.3  # drop disc bins with [Fe/H]>0
ymax = 0.0
for i in range(len(aedges) - 1):
    lo, hi = aedges[i], aedges[i+1]; b = disc & rel_ok & (feh >= lo) & (feh < hi); v = age[b]
    if v.size >= 50: ymax = max(ymax, gaussian_kde(v)(ag).max())
for i in range(len(aedges) - 1):
    lo, hi = aedges[i], aedges[i+1]; fc = 0.5 * (lo + hi)
    b = disc & rel_ok & (feh >= lo) & (feh < hi); v = age[b]
    if v.size >= 50:
        match = (fc >= EOS_LO) & (fc <= EOS_HI)
        ax_e.plot(ag, gaussian_kde(v)(ag), color=cmap(nrm(fc)), lw=4.5 if match else 1.3,
                  alpha=0.5, zorder=4 if match else 2)
        ax_e.plot([np.median(v)] * 2, [1.05 * ymax, 1.12 * ymax], color=cmap(nrm(fc)), lw=3, solid_capstyle='butt')
for br, col in [(eos_mp, CMP), (eos_mr, CMR)]:
    v = age[br & rel_ok]
    ax_e.plot(ag, gaussian_kde(v)(ag), color=col, lw=3.8, zorder=6)
    ax_e.plot([np.median(v)] * 2, [1.05 * ymax, 1.12 * ymax], color=col, lw=4.5, solid_capstyle='butt')
ax_e.set_xlim(2, 11); ax_e.set_ylim(0, 1.18 * ymax)
ax_e.set_xlabel('age [Gyr]'); ax_e.set_ylabel('number density')
sm = cm.ScalarMappable(norm=nrm, cmap=cmap); sm.set_array([])
cb_e = fig.colorbar(sm, ax=ax_e, pad=0.02, fraction=0.055)
cb_e.set_label(r'[Fe/H] of low-$\alpha$ disc bin', fontsize=14); cb_e.ax.tick_params(labelsize=12)
tag(ax_e, '(e)')

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/obs_eos_branches_overview.{ext}', bbox_inches='tight')
print('wrote', OUT + '/obs_eos_branches_overview.{pdf,png}')
print('medians [Fe/H]: metal-poor(a-rich)=%.2f  metal-rich(a-poor)=%.2f' % (np.median(feh[eos_mp]), np.median(feh[eos_mr])))
