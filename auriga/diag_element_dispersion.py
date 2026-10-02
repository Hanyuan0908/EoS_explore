"""Diagnostic: au18_nitrogen_dispersion repeated for other elements.

Same three panels, same sample, same apertures as fig_paper_nitrogen_dispersion;
only the element changes.  The snapshot is loaded once and every element is drawn
from it.

    usage:  diag_element_dispersion.py [EL ...]        (default N Mg O Si)

Two things that cannot be carried over from the nitrogen version and are measured
per element instead:

  * the colour range of panel (a), set to the 2-98 percentile of [X/Fe] among
    halo-born stars.  Auriga's zero points differ wildly between elements --
    [Mg/Fe] sits near -0.25 and [O/Fe] near +0.26 -- so a fixed range would put
    some panels entirely off scale.
  * the divider the diverging map is centred on, taken as the midpoint between
    the GS/E-analogue and MW-analogue medians.  That is exactly how the 0.09
    divider of the nitrogen figure was derived, so the construction is the same
    even though the number is not.

The y-limit of panels (b) and (c) is set from the data, so dispersions must be
compared BETWEEN elements by reading the axis, not by eye.

Also prints, per element, the median [X/Fe] in the lowest and highest [Fe/H] bin
of the MW analogue.  That trend is the empirical signature of the production
timescale: an element made promptly alongside Fe's slow SNIa contribution falls
with [Fe/H], one made by a delayed channel rises.

Writes figures/au18_<el>_dispersion.png -- figures/, not Fig_paper/.
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import auriga_public as ap
import config_au18 as C
import au18_frame as AF

OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)
ELS = sys.argv[1:] or ['N', 'Mg', 'O', 'Si']
SNAP = 72
CUT, ZCUT = 0.8, 1.5
FBINS = np.arange(-1.2, 0.4 + 1e-9, 0.2)
NMIN, NSITE, NBOOT = 150, 150, 500
R_SAT, R_HOST = 6.0, 8.0
W = 25.0
RNG = np.random.default_rng(0)
cDISC, cHALO = '#1F6FB2', '#FF6347'
cSAT, cBR, cHOST = '#8E24AA', '#00897B', '#2B2B2B'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 15.5, 'axes.labelsize': 17,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 200,
})
ctr = .5 * (FBINS[:-1] + FBINS[1:])
NB = len(ctr)


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def disp_profile(feh, y, m, nmin):
    s, lo, hi = (np.full(NB, np.nan) for _ in range(3))
    ok = m & np.isfinite(feh) & np.isfinite(y)
    for k in range(NB):
        b = ok & (feh >= FBINS[k]) & (feh < FBINS[k + 1])
        if b.sum() < nmin:
            continue
        v = y[b]
        s[k] = mad_sigma(v)
        bs = np.array([mad_sigma(v[RNG.integers(0, len(v), len(v))])
                       for _ in range(NBOOT)])
        lo[k], hi[k] = np.percentile(bs, (16, 84))
    return s, lo, hi


# ------------------------------- the sample ----------------------------------
a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
k = int(np.flatnonzero(st['snaps'] == SNAP)[0])
t_lo, t_hi = st['t_snap'][k - 1], st['t_snap'][k]

sub = ap.subhalos.subfind(SNAP, directory=C.SIM_DIR,
                          loadlist=['SubhaloPos', 'Group_R_Crit200'])
r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
ref = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
ref = ap.util.CentreOnHalo(ref, cen)
ref = ap.util.apply_mask(ref, stars=False, radialcut=.5 * r200)
ist, = np.where(ap.util.r(ref) < .1 * r200)
L = np.cross(ref.data['Coordinates'][ist],
             ref.data['Velocities'][ist] * ref.data['Masses'][ist, None])
Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
xd, yd, zd = ap.util.get_principal_axis(ref, ist, L=Ld)
del ref; gc.collect()

s = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'ParticleIDs', 'GFM_Metals'])
s = ap.util.CentreOnHalo(s, cen); ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
cc = s.data['Coordinates']
pos = np.column_stack([cc[:, 1], cc[:, 2], cc[:, 0]]) * 1e3
sids, met = s.data['ParticleIDs'], s.data['GFM_Metals']
del s; gc.collect()

gcen, _, _, _ = AF.gse_centroid(pos, sids, GSE)
ROT = AF.align_azimuth(gcen); pos = pos @ ROT.T; g2 = ROT @ gcen

sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
o = np.argsort(sids); ss = sids[o]; p = np.searchsorted(ss, ids_w)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
P, M = pos[o[p[ok]]], met[o[p[ok]]]
eps, zmx = eps_w[ok], zmx_w[ok]
disc = (eps > CUT) | (zmx < ZCUT)
del met, pos; gc.collect()

feh = C.bracket_abundance(M, 'Fe', 'H')
halo0 = ~disc
d_gse = np.linalg.norm(P - g2, axis=1); d_host = np.linalg.norm(P, axis=1)
sat = d_gse < R_SAT
host = (~sat) & (d_host < R_HOST)
bridge = ~(sat | host)
print(f'snapshot {SNAP}, t_form {t_lo:.2f}-{t_hi:.2f} Gyr; '
      f'disc-born {disc.sum():,}, halo-born {halo0.sum():,}')

for EL in ELS:
    xfe = C.bracket_abundance(M, EL, 'Fe')
    halo = halo0 & np.isfinite(feh) & np.isfinite(xfe)
    SITES = [('within GS/E-analogue', halo & sat, cSAT),
             ('along the bridge', halo & bridge, cBR),
             ('within MW-analogue', halo & host, cHOST)]
    msat, mhost = np.nanmedian(xfe[halo & sat]), np.nanmedian(xfe[halo & host])
    DIV = .5 * (msat + mhost)
    VLO, VHI = np.nanpercentile(xfe[halo], [2, 98])
    if not (VLO < DIV < VHI):                    # keep the diverging norm valid
        DIV = .5 * (VLO + VHI)
    print(f'\n[{EL}/Fe]: halo-born 2-98 per cent {VLO:+.3f} to {VHI:+.3f}; '
          f'GS/E median {msat:+.3f}, MW median {mhost:+.3f}, divider {DIV:+.3f}')

    prof = {}
    for lab, m, _ in (('disc-born', disc & np.isfinite(xfe), None),
                      ('halo-born', halo, None)):
        prof[lab] = disp_profile(feh, xfe, m, NMIN)
    for lab, m, _ in SITES:
        prof[lab] = disp_profile(feh, xfe, m, NSITE)
    top = np.nanmax([np.nanmax(v[2]) for v in prof.values()])
    YHI = float(np.ceil(top * 1.35 * 100) / 100)

    # the production-timescale signature: how [X/Fe] runs with [Fe/H]
    mw = halo & host
    bb = [((mw) & (feh >= FBINS[i]) & (feh < FBINS[i + 1])) for i in range(NB)]
    good = [i for i, b in enumerate(bb) if b.sum() >= NSITE]
    if len(good) >= 2:
        i0, i1 = good[0], good[-1]
        d0, d1 = np.median(xfe[bb[i0]]), np.median(xfe[bb[i1]])
        print(f'  MW analogue: median [{EL}/Fe] = {d0:+.3f} at [Fe/H] = {ctr[i0]:+.1f} '
              f'-> {d1:+.3f} at {ctr[i1]:+.1f}   (slope {(d1 - d0) / (ctr[i1] - ctr[i0]):+.3f})')

    fig = plt.figure(figsize=(15.2, 5.75))
    FW, FH, LEFT, BOT, AXH, GAP = 15.2, 5.75, .55, .80, 3.65, .85
    WA, WBC = AXH, AXH * 6. / 4.
    x0 = LEFT / FW
    rects = [(x0, WA), (x0 + (WA + GAP) / FW, WBC),
             (x0 + (WA + 2 * GAP + WBC) / FW, WBC)]
    AX = [fig.add_axes([x, BOT / FH, w / FW, AXH / FH]) for x, w in rects]

    ax = AX[0]
    ax.scatter(P[disc, 0], P[disc, 2], s=1.4, c='0.90', lw=0, rasterized=True, zorder=0)
    j = np.flatnonzero(halo); RNG.shuffle(j)
    sc = ax.scatter(P[j, 0], P[j, 2], s=7.5, c=xfe[j], cmap='RdYlBu_r',
                    norm=TwoSlopeNorm(vmin=VLO, vcenter=DIV, vmax=VHI),
                    lw=0, alpha=.35, rasterized=True, zorder=2)
    th = np.linspace(0, 2 * np.pi, 240)
    for r0, c0, xx, zz in ((R_SAT, cSAT, g2[0], g2[2]), (R_HOST, cHOST, 0., 0.)):
        ax.plot(r0 * np.cos(th) + xx, r0 * np.sin(th) + zz, color=c0, lw=2.0, zorder=4)
    bbx = dict(fc='white', ec='none', alpha=.85, pad=1.8)
    ax.text(-13.5, -22.3, 'GS/E analogue', color=cSAT, ha='center', va='center',
            fontsize=14.5, bbox=bbx, zorder=5)
    ax.text(0., R_HOST + 1.4, 'MW analogue', color=cHOST, ha='center', va='bottom',
            fontsize=14.5, bbox=bbx, zorder=5)
    ax.annotate('bridge', xy=(-5.5, -6.8), xytext=(13.5, -17.5), color=cBR,
                ha='center', va='center', fontsize=14.5, bbox=bbx, zorder=5,
                arrowprops=dict(arrowstyle='-', color=cBR, lw=1.3, shrinkA=2, shrinkB=2))
    ax.set(aspect='equal', xlim=(-W, W), ylim=(-W, W),
           xlabel='$x$ [kpc]', ylabel='$z$ [kpc]')
    cax = ax.inset_axes([.545, .885, .41, .032])
    cb = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cb.solids.set_alpha(1.)
    cb.ax.tick_params(labelsize=11.5, length=2.5, pad=1.5)
    cb.outline.set_linewidth(.7)
    cax.set_title(f'[{EL}/Fe]', fontsize=14, pad=3)
    cax.title.set_bbox(bbx)
    ax.text(.035, .965, '(a)', transform=ax.transAxes, va='top', fontsize=18,
            fontweight='bold')

    for n, (axi, items, tag) in enumerate((
            (AX[1], [('disc-born', cDISC), ('halo-born', cHALO)], '(b)'),
            (AX[2], [('halo-born', cHALO)] + [(l, c) for l, _, c in SITES], '(c)'))):
        for lab, col in items:
            sg, lo, hi = prof[lab]
            g = np.isfinite(sg)
            solid = (tag == '(b)') or lab == 'halo-born'
            axi.plot(ctr[g], sg[g], color=col, lw=2.8 if solid else 2.4,
                     ls='-' if solid else '--',
                     alpha=.45 if (tag == '(c)' and lab == 'halo-born') else 1.,
                     label='halo-born, all' if (tag == '(c)' and lab == 'halo-born') else lab)
            if not (tag == '(c)' and lab == 'halo-born'):
                axi.fill_between(ctr[g], lo[g], hi[g], color=col, alpha=.20, lw=0,
                                 rasterized=True)
        axi.set(xlabel='[Fe/H]', ylabel=rf'$\sigma_{{\rm [{EL}/Fe]}}$ [dex]',
                xlim=(-1.05, .45), ylim=(0., YHI))
        axi.legend(loc='upper right', handlelength=1.6,
                   fontsize=13.5 if tag == '(c)' else None,
                   labelspacing=.3 if tag == '(c)' else None)
        axi.text(.035, .965, tag, transform=axi.transAxes, va='top', fontsize=18,
                 fontweight='bold')
    f = f'{OUT}/au18_{EL.lower()}_dispersion.png'
    fig.savefig(f, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {f}')
