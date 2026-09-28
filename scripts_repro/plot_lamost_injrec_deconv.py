"""Injection-recovery test of the XD age deconvolution, using LAMOST Xiang+24 MSTO
ages as GROUND TRUTH.

Question: AstroNN ages have ~15% (sigma_lnage~0.15) errors -- is that too large to
deconvolve back to the truth?  Test: take LAMOST MSTO ages (small errors, ~6-10%),
DEGRADE them with extra per-star noise drawn from the AstroNN error distribution,
then run the same XD pipeline and see whether it recovers the original LAMOST
signal.

Per population (Eos alpha-rich, Eos alpha-poor, low-alpha disc) we plot the age
distribution three ways, averaged over N_REAL noise realisations:
  truth      LAMOST observed (thick black)
  degraded   truth + injected AstroNN-level noise, no deconvolution (grey dotted)
  recovered  XD deconvolution of the degraded data (coloured solid + 1-sigma band)
If recovered ~ truth, AstroNN-level errors ARE invertible; if recovered stays near
degraded, they are not.  XD is fit in (ln age, [Fe/H]) as in the main pipeline;
the age marginal is shown.

    python scripts_repro/plot_lamost_injrec_deconv.py
Writes figures_repro/01_eos_amr_deconv_lamost_injrec.png .
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

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
N_REAL = 25
WREG = np.array([0.05**2, 0.02**2])
AGER = (0.5, 14.0)
DISC_CAP = 6000
SEED = 7
rng = np.random.default_rng(SEED)

c = Cuts()
L = load_catalog(REPO + '/data_repro/our_lamost_subgiant_ddpayne.fits.gz')
mL = make_masks(L, c)
feh = np.asarray(L['fe_h'], float); mg = np.asarray(L['mg_fe'], float)
vphi = np.asarray(L['galvt'], float); al = np.asarray(L['al_fe'], float); lz = np.asarray(L['lz'], float)
feherr = np.asarray(L['fe_h_err'], float) if 'fe_h_err' in L.dtype.names else np.full(len(feh), 0.02)
rap = np.asarray(L['rap'], float); rperi = np.asarray(L['rperi'], float)
with np.errstate(invalid='ignore'):
    ecc = (rap - rperi) / (rap + rperi)
age = np.asarray(L['age'], float); aerr = np.asarray(L['age_model_error'], float)
base = np.asarray(mL['base'], bool); thin_al = np.asarray(mL['thin_al'], bool)
# strict age-quality cut for the LAMOST ground truth: absolute age error < 1 Gyr
# (as well as age<14 and the relative cut).
AERR_ABS_MAX = 1.0
rel = (np.isfinite(age) & np.isfinite(aerr) & (age > 0) & (age < 14)
       & (aerr / age < 0.3) & (aerr < AERR_ABS_MAX) & np.isfinite(feh))
halo = base & ((ecc > 0.7) | (lz < 0))
divline = 0.317 * feh + 0.353
eos = (halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc * feh + c.inter_acc)
       & (mg < c.slope_acc2 * feh + c.inter_acc2) & (al > c.alfe_cut))
# (label, mask, colour, K, stats-box corner)
POPS = [('Eos ' + r'$\alpha$-rich', eos & (mg > divline) & rel, 'magenta', 2, 'left'),
        ('Eos ' + r'$\alpha$-poor', eos & (mg <= divline) & rel, 'cyan', 2, 'left'),
        (r'low-$\alpha$ disc', thin_al & (vphi > 150) & rel, 'royalblue', 5, 'right')]

# AstroNN error pool to inject
A = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
aa = np.asarray(A['age'], float); ae = np.asarray(A['age_model_error'], float)
okA = np.isfinite(aa) & np.isfinite(ae) & (aa > 0) & (ae / aa < 0.3)
ASTRONN_SIG = (ae / aa)[okA]           # sigma_lnage pool


def age_marginal(mdl, ag):
    dl = np.zeros_like(ag)
    for a_, mk, Vk in zip(mdl.alpha, mdl.mu, mdl.V):
        s = np.sqrt(Vk[0, 0])
        dl += a_ * np.exp(-0.5 * ((np.log(ag) - mk[0]) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return dl / ag


def marg_std_age(mdl):
    a = mdl.alpha; mln = mdl.mu[:, 0]; s2 = mdl.V[:, 0, 0]
    m1 = np.sum(a * np.exp(mln + 0.5 * s2)); m2 = np.sum(a * np.exp(2 * mln + 2 * s2))
    return np.sqrt(max(m2 - m1 ** 2, 0))


ag = np.linspace(*AGER, 300)
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), sharex=True, constrained_layout=True)

summary = []
handles = None
for axi, (lab, sel, col, K, boxloc) in zip(axes, POPS):
    lnt = np.log(age[sel]); ft = feh[sel]; sfeh = feherr[sel]
    slam = (aerr[sel] / age[sel])                 # LAMOST own error (small)
    ntot = len(lnt)
    truth_kde = gaussian_kde(age[sel])(ag)
    rec = np.zeros((N_REAL, len(ag))); deg = np.zeros((N_REAL, len(ag)))
    rec_std = np.zeros(N_REAL); deg_std = np.zeros(N_REAL)
    for r in range(N_REAL):
        # subsample large populations per realisation (density only needs a few k)
        if ntot > DISC_CAP:
            idx = rng.choice(ntot, DISC_CAP, replace=False)
        else:
            idx = np.arange(ntot)
        sig_inj = rng.choice(ASTRONN_SIG, len(idx))            # per-star AstroNN error
        noise = rng.normal(0, sig_inj)
        ln_noised = lnt[idx] + noise
        X = np.column_stack([ln_noised, ft[idx]])
        S = np.zeros((len(idx), 2, 2))
        S[:, 0, 0] = sig_inj ** 2                              # known (injected) error
        S[:, 1, 1] = sfeh[idx] ** 2
        mdl = XD(K, w_reg=WREG, random_state=SEED + r).fit(X, S)
        rec[r] = age_marginal(mdl, ag); rec_std[r] = marg_std_age(mdl)
        deg[r] = gaussian_kde(np.exp(ln_noised))(ag)
        deg_std[r] = np.std(np.exp(ln_noised))
    rmean = rec.mean(0); rlo, rhi = np.percentile(rec, [16, 84], axis=0)
    hT, = axi.plot(ag, truth_kde, color='k', lw=2.6, label='truth (LAMOST MSTO)', zorder=6)
    hD, = axi.plot(ag, deg.mean(0), color='0.45', lw=1.8, ls=':',
                   label='degraded (+ AstroNN-scale age noise)', zorder=4)
    band = axi.fill_between(ag, rlo, rhi, color=col, alpha=0.22, lw=0)
    hR, = axi.plot(ag, rmean, color=col, lw=2.8, label='recovered (XD deconvolution)', zorder=5)
    handles = [hT, hD, hR]
    axi.set_title(f'{lab}  (n={ntot})')
    axi.set_xlabel('age [Gyr]'); axi.set_xlim(*AGER); axi.set_ylim(0, None)
    axi.axvline(np.median(age[sel]), color='k', lw=0.8, ls='--', alpha=0.5)
    t_std = np.std(age[sel])
    xy = (0.03, 0.97) if boxloc == 'left' else (0.97, 0.97)
    axi.text(xy[0], xy[1],
             f'age $\\sigma$ [Gyr]:\n truth {t_std:.2f}\n degraded {deg_std.mean():.2f}\n recovered {rec_std.mean():.2f}$\\pm${rec_std.std():.2f}',
             transform=axi.transAxes, va='top', ha=boxloc, fontsize=11,
             bbox=dict(fc='white', ec='0.7', alpha=0.9, pad=3))
    summary.append((lab, ntot, t_std, deg_std.mean(), rec_std.mean(), rec_std.std()))

axes[0].set_ylabel('density')
fig.legend(handles, [h.get_label() for h in handles], loc='upper center',
           bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=13, handlelength=2.6)
fig.suptitle('Injection-recovery test: LAMOST MSTO ages degraded to AstroNN age-error scale, then XD-deconvolved',
             fontsize=14, y=1.10)
fig.text(0.5, 1.035, f'per-star noise drawn from the full AstroNN $\\sigma_{{\\ln age}}$ distribution '
         f'(median {np.median(ASTRONN_SIG):.2f}); {N_REAL} realisations',
         ha='center', va='bottom', fontsize=10.5, color='0.35')

os.makedirs(FIG, exist_ok=True)
out = f'{FIG}/01_eos_amr_deconv_lamost_injrec.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print('wrote', out)
print(f"{'pop':16s} {'n':>6s} {'true_sig':>9s} {'degr_sig':>9s} {'recov_sig':>10s}")
for lab, n, t, d, r, rs in summary:
    print(f"{lab:16s} {n:6d} {t:9.2f} {d:9.2f} {r:8.2f}±{rs:.2f}")
