"""Non-publish diagnostic: error-DECONVOLVED age-[Fe/H] structure of Eos vs the
high/low-alpha disc and Splash, using extreme deconvolution (XD).

Companion to Fig_paper/obs_amr_agedist (AstroNN) and obs_eos_age_dist (BINGO),
which show the raw, error-BROADENED KDE.  Here every population's (ln age, [Fe/H])
distribution is fit with an XD Gaussian mixture that folds in each star's age +
[Fe/H] error, so the maps/curves show the INTRINSIC distribution with the
measurement scatter removed.  Model params are built by build_amr_xd_cache.py.

  (a) age distribution: analytic age-marginal of each population's 2-D XD mixture
      (solid), with the raw observed KDE (thin dotted) for comparison -- the gap
      is the error broadening XD removes.
  (b) age-[Fe/H] plane: deconvolved density of ALL stars (grey) with nested
      30/60/90%-enclosed-mass contours of each population's deconvolved density.

Fits are in ln(age); densities are mapped to linear age with the 1/age Jacobian and
contour levels are enclosed-mass in the displayed (linear-age) space.

    python scripts_repro/plot_eos_amr_agedist_deconv.py [astronn|bingo]
Writes figures_repro/01_eos_amr_agedist_deconv_{mode}.png .
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
sys.path.insert(0, REPO + '/scripts_repro')
from amr_xd_data import load_amr_data

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

MODE = sys.argv[1] if len(sys.argv) > 1 else 'astronn'
FIG = REPO + '/figures_repro'
CACHE = REPO + f'/data_repro/amr_xd_cache_{MODE}.npz'
if not os.path.exists(CACHE):
    sys.exit(f'cache missing -- run build_amr_xd_cache.py {MODE} first')
xd = np.load(CACHE, allow_pickle=True)

AGER = tuple(xd['ager']); FEHR = (-1.15, 0.5)
LABEL = str(xd['label'])


def gmm_lnage_density(alpha, mu, V, T, F):
    pts = np.column_stack([T.ravel(), F.ravel()])
    dens = np.zeros(len(pts))
    for a, mk, Vk in zip(alpha, mu, V):
        det = Vk[0, 0] * Vk[1, 1] - Vk[0, 1] * Vk[1, 0]
        inv = np.array([[Vk[1, 1], -Vk[0, 1]], [-Vk[1, 0], Vk[0, 0]]]) / det
        d = pts - mk
        maha = np.einsum('ni,ij,nj->n', d, inv, d)
        dens += a * np.exp(-0.5 * maha) / (2 * np.pi * np.sqrt(det))
    return dens.reshape(T.shape)


def age_feh_density(name, AG, FG):
    alpha = xd[f'{name}_alpha']; mu = xd[f'{name}_mu']; V = xd[f'{name}_V']
    with np.errstate(divide='ignore'):
        T = np.log(AG)
    p = gmm_lnage_density(alpha, mu, V, T, FG) / AG
    p[~np.isfinite(p)] = 0.0
    return p


def mass_levels(Z, dx, dy, fracs=(0.9, 0.6, 0.3)):
    z = Z.ravel(); order = np.argsort(z)[::-1]
    csum = np.cumsum(z[order]) * dx * dy
    csum /= csum[-1]
    lev = [z[order][min(np.searchsorted(csum, f), len(z) - 1)] for f in fracs]
    return sorted(set(lev))


# observed samples (for the KDE overlay in panel a)
pops, _ = load_amr_data(MODE)

POP = {'eos': ('Eos', 'red'), 'splash': ('Splash', 'darkorange'),
       'lowa': (r'low-$\alpha$ disc', 'royalblue'),
       'higha': (r'high-$\alpha$ disc', 'seagreen')}

fig, ax = plt.subplots(1, 2, figsize=(12.63, 5.4),
                       gridspec_kw={'width_ratios': [1, 1.35]}, constrained_layout=True)


def tag(a, t):
    a.text(0.035, 0.965, t, transform=a.transAxes, fontsize=17, fontweight='bold',
           va='top', ha='left', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.5))


# ---- (a) deconvolved age marginal (solid) + observed KDE (thin dotted) --------
ag = np.linspace(0.15, AGER[1], 400)
for key in ['eos', 'splash', 'lowa', 'higha']:
    lab, col = POP[key]
    alpha = xd[f'{key}_alpha']; mu = xd[f'{key}_mu']; V = xd[f'{key}_V']
    dl = np.zeros_like(ag)
    for a_, mk, Vk in zip(alpha, mu, V):
        s = np.sqrt(Vk[0, 0])
        dl += a_ * np.exp(-0.5 * ((np.log(ag) - mk[0]) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    ax[0].plot(ag, dl / ag, color=col, lw=2.6, label=lab)
    obs = pops[key]['age']
    ax[0].plot(ag, gaussian_kde(obs)(ag), color=col, lw=1.1, ls=':', alpha=0.85)
ax[0].set_xlim(*AGER); ax[0].set_ylim(0, None)
ax[0].legend(loc='upper right')
ax[0].set_xlabel('age [Gyr]'); ax[0].set_ylabel('Density')
tag(ax[0], '(a)')
ax[0].text(0.035, 0.80, 'solid: XD deconvolved\ndotted: observed KDE',
           transform=ax[0].transAxes, fontsize=10, va='top', color='0.35')

# ---- (b) deconvolved all-star density (grey) + per-pop enclosed-mass contours -
NG = 300
AG, FG = np.meshgrid(np.linspace(0.15, AGER[1], NG), np.linspace(*FEHR, NG))
dx = (AGER[1] - 0.15) / (NG - 1); dy = (FEHR[1] - FEHR[0]) / (NG - 1)
Zall = age_feh_density('all', AG, FG)
Zall_m = np.ma.masked_less(Zall, Zall.max() * 1e-3)
ax[1].pcolormesh(AG, FG, np.log10(Zall_m), cmap='Greys', alpha=0.65, shading='auto',
                 vmin=np.log10(Zall.max()) - 2.8, vmax=np.log10(Zall.max()), zorder=0,
                 rasterized=True)
for key in ['higha', 'lowa', 'splash', 'eos']:
    lab, col = POP[key]
    Z = age_feh_density(key, AG, FG)
    lev = mass_levels(Z, dx, dy, fracs=(0.9, 0.6, 0.3))
    ax[1].contour(AG, FG, Z, levels=lev, colors=[col], linewidths=[2.0, 2.9, 3.8], zorder=3)
    ax[1].plot([], [], color=col, lw=2.4, label=lab)
ax[1].set_xlim(*AGER); ax[1].set_ylim(*FEHR)
ax[1].set_xlabel('age [Gyr]'); ax[1].set_ylabel('[Fe/H]')
ax[1].legend(loc='lower left')
tag(ax[1], '(b)')
ax[1].text(0.97, 0.96, LABEL, transform=ax[1].transAxes, fontsize=12, va='top',
           ha='right', color='0.3')

os.makedirs(FIG, exist_ok=True)
out = f'{FIG}/01_eos_amr_agedist_deconv_{MODE}.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
