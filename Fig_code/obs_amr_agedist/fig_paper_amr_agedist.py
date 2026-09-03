"""Publication figure (observational): Eos metallicity-age structure vs the
high/low-alpha disc and Splash (AstroNN ages), 3 panels.

  (a) [Fe/H]-[Mg/Fe] with the high/low-alpha (Mg) split line (solid) and the
      accreted line (dashed).
  (b) age distributions (KDE) of Eos / Splash / low-a disc / high-a disc.
  (c) age-[Fe/H] plane: nested 90/60/30% KDE contours per population over the
      grey base-sample density.

Populations (same cuts as all previous analysis):
  Eos        = base & ((ecc>0.7)|(lz<0)) & -0.9<[Fe/H]<-0.2 & in-situ low-a Mg
               wedge & [Al/Fe]>-0.12   -> n=353 (191 a-rich / 162 a-poor)
  low-a disc = thin_al  & V_phi>150     high-a disc = thick_al & V_phi>150
  Splash     = thick_al & V_phi<80
AstroNN ages with the age-quality cut sigma_age/age<0.3 (Eos -> 318 in the age panels).

Data: data_repro/our_apogee_dr17_lite_ann.fits.gz (in-repo, portable).
Writes Fig_paper/obs_amr_agedist.pdf and .png.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
# imports first, then our own rcParams below so the publication style wins.
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 16, 'axes.labelsize': 20, 'axes.titlesize': 18,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

OUT = REPO + '/Fig_paper'
os.makedirs(OUT, exist_ok=True)
c = Cuts()
cat = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); vphi = np.asarray(cat['galvt'], float)
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float)
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
with np.errstate(invalid='ignore'):
    ecc = (rap - rperi) / (rap + rperi)
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool); thick_al = np.asarray(m['thick_al'], bool)
rel_ok = np.isfinite(age) & np.isfinite(aerr) & (aerr / age < 0.3)

# CANONICAL Eos cut (reproduces the anchor n=353 / 191 / 162)
halo = base & ((ecc > 0.7) | (lz < 0))
def acc(f): return c.slope_acc * f + c.inter_acc
def hl(f):  return c.slope_acc2 * f + c.inter_acc2
eos_sel = halo & (feh > -0.9) & (feh < -0.2) & (mg > acc(feh)) & (mg < hl(feh)) & (al > c.alfe_cut)

lowa_disc  = thin_al & (vphi > 150)
higha_disc = thick_al & (vphi > 150)
splash     = thick_al & (vphi < 80)

AGER = (0, 13.5); FEHR = (-1.15, 0.5)

fig, ax = plt.subplots(1, 3, figsize=(18, 5.4), gridspec_kw={'width_ratios': [1, 1, 1.35]},
                       constrained_layout=True)


def tag(a, t):
    a.text(0.035, 0.965, t, transform=a.transAxes, fontsize=17, fontweight='bold',
           va='top', ha='left', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))


# --- (a) [Fe/H]-[Mg/Fe] with the Mg cut ---
s = base & np.isfinite(feh) & np.isfinite(mg)
h, xe, ye = np.histogram2d(feh[s], mg[s], bins=[80, 60], range=[(-1.6, 0.5), (-0.05, 0.45)])
him = np.full_like(h, np.nan); him[h > 0] = np.log10(h[h > 0])
im0 = ax[0].imshow(him.T, origin='lower', extent=[-1.6, 0.5, -0.05, 0.45], aspect='auto', cmap='Greys',
                   vmin=np.nanpercentile(him, 3), vmax=np.nanpercentile(him, 99.5), zorder=0)
im0.set_rasterized(True)
xx = np.linspace(-1.6, 0.5, 60)
ax[0].plot(xx, c.slope_acc2 * xx + c.inter_acc2, 'r-', lw=2.2, zorder=3)
ax[0].plot(xx, c.slope_acc * xx + c.inter_acc, 'r--', lw=1.4, zorder=3)
ax[0].text(-0.62, 0.325, r'high-$\alpha$', color='seagreen', fontsize=20, fontweight='bold')
ax[0].text(-0.33, 0.075, r'low-$\alpha$', color='royalblue', fontsize=20, fontweight='bold', rotation=-15)
ax[0].text(-1.52, 0.005, 'accreted', color='0.3', fontsize=17)
ax[0].set_xlim(-1.6, 0.5); ax[0].set_ylim(-0.05, 0.45)
ax[0].set_xlabel('[Fe/H]'); ax[0].set_ylabel('[Mg/Fe]')
tag(ax[0], '(a)')

# --- (b) age KDE (Eos / Splash / low-alpha & high-alpha disc), consistent colours ---
kde_pops = [('Eos', eos_sel, 'red', '-'),
            ('Splash', splash, 'darkorange', '-'),
            (r'low-$\alpha$ disc', lowa_disc, 'royalblue', '--'),
            (r'high-$\alpha$ disc', higha_disc, 'seagreen', '--')]
xg = np.linspace(*AGER, 300)
for lab, sel, col, ls in kde_pops:
    p = sel & rel_ok & np.isfinite(age)
    kde = gaussian_kde(age[p])
    ax[1].plot(xg, kde(xg), color=col, ls=ls, lw=2.4, label=lab)
ax[1].set_xlim(*AGER); ax[1].set_ylim(0, 0.44)   # headroom so the legend clears the curves
ax[1].legend(loc='upper right')
ax[1].set_xlabel('age [Gyr]'); ax[1].set_ylabel('Density')
tag(ax[1], '(b)')

# --- (c) grey base density + nested contours per population ---
pops = [('high-$\\alpha$ disc', higha_disc, 'seagreen'),
        ('low-$\\alpha$ disc',  lowa_disc,  'royalblue'),
        ('Splash',              splash,     'darkorange'),
        ('Eos',                 eos_sel,    'red')]
sb = base & rel_ok & np.isfinite(feh) & np.isfinite(age)
hb, xb, yb = np.histogram2d(age[sb], feh[sb], bins=[70, 70], range=[AGER, FEHR])
hbi = np.full_like(hb, np.nan); hbi[hb > 0] = np.log10(hb[hb > 0])
im2 = ax[2].imshow(hbi.T, origin='lower', extent=[*AGER, *FEHR], aspect='auto', cmap='Greys',
                   alpha=0.6, vmin=np.nanpercentile(hbi, 3), vmax=np.nanpercentile(hbi, 99.5), zorder=0)
im2.set_rasterized(True)
AG, FG = np.meshgrid(np.linspace(*AGER, 120), np.linspace(*FEHR, 120))
grid = np.vstack([AG.ravel(), FG.ravel()])
ENC = [0.9, 0.6, 0.3]
for lab, sel, col in pops:
    p = sel & rel_ok & np.isfinite(feh) & np.isfinite(age)
    xy = np.vstack([age[p], feh[p]]); kde = gaussian_kde(xy); dens = kde(xy)
    levels = sorted(np.percentile(dens, [100 * (1 - f) for f in ENC]))
    Z = kde(grid).reshape(AG.shape)
    ax[2].contour(AG, FG, Z, levels=levels, colors=[col], linewidths=[2.0, 2.9, 3.8], zorder=3)
    ax[2].plot([], [], color=col, lw=2.4, label=lab)
ax[2].set_xlim(*AGER); ax[2].set_ylim(*FEHR)
ax[2].set_xlabel('age [Gyr]'); ax[2].set_ylabel('[Fe/H]')
ax[2].legend(loc='lower left')
tag(ax[2], '(c)')

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/obs_amr_agedist.{ext}', bbox_inches='tight')
print('wrote', OUT + '/obs_amr_agedist.{pdf,png}')
