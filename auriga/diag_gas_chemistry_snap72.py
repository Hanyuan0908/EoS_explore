"""EXPERIMENT, not a paper figure: the chemistry of the gas at the GS/E pericentre.

Edge-on maps of the gas at snapshot 72 (t = 4.99 Gyr) in the disc frame.  First
panel is the surface density; the rest are per-pixel abundance statistics, one
species per panel, on the Asplund solar scale.

    usage:  diag_gas_chemistry_snap72.py [SNAP] [XH|XFe] [mean|std]

    XH    [X/H] for He, C, N, O, Ne, Mg, Si, Fe            -- overall enrichment
    XFe   [Fe/H], then [X/Fe] for He, C, N, O, Ne, Mg, Si  -- abundance pattern
    mean  mass-weighted mean in each pixel
    std   mass-weighted standard deviation in each pixel

Both statistics come from the same accumulated moments: sum(m), sum(m v) and
sum(m v^2) per pixel, so sigma^2 = <v^2> - <v>^2.

WHAT THE DISPERSION MEANS.  These are projected maps, so a pixel's spread mixes
two things: genuine cell-to-cell inhomogeneity, and the range of abundances along
the line of sight through the galaxy (here the y axis, ~tens of kpc).  A high
sigma therefore marks a sightline where chemically distinct gas overlaps, which is
what you want for spotting where host and satellite material meet -- but it is not
a local chemical spread.

[X/Fe] is computed per cell and then weighted, with a mask requiring BOTH X and Fe
finite in that cell; differencing two mass-weighted [X/H] maps would only be
equivalent if their finite masks matched, which they do not.

COLOUR SCALING (mean).  A percentile stretch over all pixels is useless: the frame
is mostly diffuse, very metal-poor gas spanning ~2 dex, while the disc-to-GS/E
transition is only ~0.4 dex, so everything of interest saturates.  Each panel is
scaled to its own measured transition -- the median in the disc (|z| < 2,
|x| < 8 kpc) and in the GS/E (an ellipse on its centroid) -- with a diverging map
centred midway.  Red is disc-like, blue GS/E-like.  The diffuse halo saturates
blue by construction.  Dispersion panels use a plain sequential 2-98 percentile
stretch.

Maps are cached in out/gas_chem_maps_snap<NN>_v3.npz.
"""
import gc, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
import config_au18 as C
import au18_frame as AF
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT

os.makedirs(C.FIG_DIR, exist_ok=True)
SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 72
MODE = sys.argv[2] if len(sys.argv) > 2 else 'XFe'
STAT = sys.argv[3] if len(sys.argv) > 3 else 'mean'
assert MODE in ('XH', 'XFe') and STAT in ('mean', 'std')
# The frame is sized to contain the GS/E, which moves outward between snapshots;
# fixed limits would crop it at snapshot 73, where it sits at r = 34 kpc.
XLIM, ZLIM, NB, MMIN = None, None, 240, 5e5
SPECIES = ['He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe']
PANELS = ([('XH', e) for e in SPECIES] if MODE == 'XH'
          else [('XH', 'Fe')] + [('XFe', e) for e in SPECIES if e != 'Fe'])
# The GS/E centroid moves between snapshots, so it is measured from the debris
# rather than hard-coded; GSE_AB is the ellipse used to define the "GS/E region".
GSE_AB = (8., 5.)
CACHE = C.OUT_DIR + f'/gas_chem_maps_snap{SNAP}_v4.npz'


def build():
    import auriga_public as ap
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
    s4 = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
        loadlist=['ParticleIDs', 'Coordinates', 'Masses', 'Velocities'])
    s4 = ap.util.CentreOnHalo(s4, cen); ap.util.rotateto(s4, xd, dir2=yd, dir3=zd)
    c4 = s4.data['Coordinates']
    sp = np.column_stack([c4[:, 1], c4[:, 2], c4[:, 0]]) * 1e3
    sid = s4.data['ParticleIDs']; del s4, c4; gc.collect()
    gid = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
    o = np.argsort(sid); ss = sid[o]; p = np.searchsorted(ss, gid)
    okg = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == gid)
    G = sp[o[p[okg]]]; del sp, sid; gc.collect()
    # Remove the arbitrary azimuth: rotate about the disc axis until the GS/E
    # centroid lies in the x-z plane, so the projection does not foreshorten it.
    ROT = AF.align_azimuth(np.median(G, axis=0))
    G = G @ ROT.T
    g = ap.snapshot.load_snapshot(SNAP, 0, snappath=C.SIM_DIR, verbose=False,
        loadlist=['Coordinates', 'Masses', 'Velocities', 'GFM_Metals'])
    g = ap.util.CentreOnHalo(g, cen); ap.util.rotateto(g, xd, dir2=yd, dir3=zd)
    cg = g.data['Coordinates']
    gp = (np.column_stack([cg[:, 1], cg[:, 2], cg[:, 0]]) * 1e3) @ ROT.T
    gm = g.data['Masses'] * C.MASS_TO_MSUN
    met = g.data['GFM_Metals'].astype(np.float64)
    del g, cg; gc.collect()
    global XLIM, ZLIM
    gc_ = np.median(G, axis=0)
    XLIM = float(max(30., abs(gc_[0]) + 9.)); ZLIM = float(max(24., abs(gc_[2]) + 7.))
    print(f'frame sized to the GS/E: XLIM {XLIM:.0f}, ZLIM {ZLIM:.0f} kpc', flush=True)
    ins = (np.abs(gp[:, 0]) < XLIM) & (np.abs(gp[:, 2]) < ZLIM)
    gp, gm, met = gp[ins], gm[ins], met[ins]
    rng = [[-XLIM, XLIM], [-ZLIM, ZLIM]]
    Wm, xe_, ze_ = np.histogram2d(gp[:, 0], gp[:, 2], bins=NB, range=rng, weights=gm)
    feh = C.bracket_abundance(met, 'Fe', 'H')
    out = {}
    for el in SPECIES:
        for kind in ('XH', 'XFe'):
            if kind == 'XFe' and el == 'Fe':
                continue
            if kind == 'XH':
                v = C.bracket_abundance(met, el, 'H'); ok = np.isfinite(v)
            else:
                v = C.bracket_abundance(met, el, 'Fe')
                ok = np.isfinite(v) & np.isfinite(feh)
            H = lambda w: np.histogram2d(gp[ok, 0], gp[ok, 2], bins=NB,
                                         range=rng, weights=w)[0]
            den, s1, s2 = H(gm[ok]), H(gm[ok] * v[ok]), H(gm[ok] * v[ok] ** 2)
            good = den > MMIN
            mean = np.where(good, s1 / np.where(den > 0, den, 1), np.nan)
            var = np.where(good, s2 / np.where(den > 0, den, 1) - mean ** 2, np.nan)
            out[f'mean_{kind}_{el}'] = mean
            out[f'std_{kind}_{el}'] = np.sqrt(np.clip(var, 0, None))
    np.savez(CACHE, W=Wm, xe=xe_, ze=ze_, GSE_x=G[:, 0], GSE_z=G[:, 2],
             XLIM=XLIM, ZLIM=ZLIM, **out)
    return Wm, xe_, ze_, G[:, 0], G[:, 2], out


if os.path.exists(CACHE):
    c = np.load(CACHE)
    W, xe, ze, gx, gz = c['W'], c['xe'], c['ze'], c['GSE_x'], c['GSE_z']
    MAPS = {k: c[k] for k in c.files if k.startswith(('mean_', 'std_'))}
    XLIM, ZLIM = float(c['XLIM']), float(c['ZLIM'])
    print(f'using cached maps: {CACHE}  (frame {XLIM:.0f} x {ZLIM:.0f} kpc)')
else:
    W, xe, ze, gx, gz, MAPS = build()

GSE_CEN = (float(np.median(gx)), float(np.median(gz)))
print(f'GS/E centroid at snapshot {SNAP}: (x, z) = ({GSE_CEN[0]:.1f}, {GSE_CEN[1]:.1f}) kpc')
xc, zc = .5 * (xe[:-1] + xe[1:]), .5 * (ze[:-1] + ze[1:])
X, Z = np.meshgrid(xc, zc, indexing='ij')
DISC = (np.abs(Z) < 2) & (np.abs(X) < 8)
GSEM = ((X - GSE_CEN[0]) / GSE_AB[0]) ** 2 + ((Z - GSE_CEN[1]) / GSE_AB[1]) ** 2 < 1
t = np.clip((X * GSE_CEN[0] + Z * GSE_CEN[1]) / (GSE_CEN[0] ** 2 + GSE_CEN[1] ** 2), 0, 1)
LANE = (np.hypot(X - t * GSE_CEN[0], Z - t * GSE_CEN[1]) < 3.5) & (t > .25) & (t < .85)
RNG = [[-XLIM, XLIM], [-ZLIM, ZLIM]]

fig, axes = plt.subplots(3, 3, figsize=(16.5, 13.2), sharex=True, sharey=True)
ax = axes.ravel()[0]
S = np.where(W < MMIN, np.nan, W / ((2 * XLIM / NB) * (2 * ZLIM / NB)))
im = ax.pcolormesh(xe, ze, S.T, cmap='Greys',
                   norm=LogNorm(vmin=np.nanpercentile(S, 40), vmax=np.nanpercentile(S, 99.9)))
fig.colorbar(im, ax=ax, pad=.02).set_label(r'$\Sigma_{\rm gas}$ [M$_\odot$ kpc$^{-2}$]')
ax.set_title('gas surface density', fontsize=13)

print(f'\n{STAT} maps\n{"sp":>9s} {"disc":>7s} {"lane":>7s} {"GS/E":>7s}')
for i, (kind, el) in enumerate(PANELS, start=1):
    a = axes.ravel()[i]
    base = f'[{el}/H]' if kind == 'XH' else f'[{el}/Fe]'
    lbl = base if STAT == 'mean' else r'$\sigma$(' + base + ')'
    M = MAPS[f'{STAT}_{kind}_{el}']; ok = np.isfinite(M)
    md = lambda m: float(np.nanmedian(M[m & ok])) if (m & ok).sum() else np.nan
    dm, lm, gg = md(DISC), md(LANE), md(GSEM)
    if STAT == 'std':
        lo, hi = np.nanpercentile(M, [2, 98])
        im = a.pcolormesh(xe, ze, M.T, cmap='magma_r', vmin=lo, vmax=hi)
    else:
        span = dm - gg
        if not np.isfinite(span) or abs(span) <= .05:
            lo, hi = np.nanpercentile(M, [2, 98]); ctr = float(np.nanmedian(M))
        else:
            lo, hi = sorted([gg - .8 * span, dm + .5 * span]); ctr = .5 * (dm + gg)
        im = a.pcolormesh(xe, ze, M.T, cmap='RdYlBu_r',
                          norm=TwoSlopeNorm(vmin=lo, vcenter=ctr, vmax=hi))
    cb = fig.colorbar(im, ax=a, pad=.02, extend='both'); cb.set_label(lbl)
    a.set_title(f'{lbl}   disc {dm:+.2f}, lane {lm:+.2f}, GS/E {gg:+.2f}', fontsize=12)
    print(f'{base:>9s} {dm:7.2f} {lm:7.2f} {gg:7.2f}')

for a in axes.ravel():
    OT.density_contours(a, gx, gz, RNG, '#00E5FF' if STAT == 'std' else '#4A148C',
                        levels=(0.9, 0.5), bins=70, smooth=1.6, lw=1.6)
    a.plot([0, GSE_CEN[0]], [0, GSE_CEN[1]],
           color='w' if STAT == 'std' else 'k', ls='--', lw=1.4, alpha=.8)
    a.set(aspect='equal', xlim=(-XLIM, XLIM), ylim=(-ZLIM, ZLIM))
for a in axes[2, :]:
    a.set_xlabel('$x$ [kpc]')
for a in axes[:, 0]:
    a.set_ylabel('$z$ [kpc]')
fig.suptitle(f'Au18 snapshot {SNAP}: gas abundance {STAT} at the GS/E pericentre '
             f'({MODE}).  Contour = GS/E debris, dashed = the lane', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, .97])
out = C.FIG_DIR + f'/diag_gas_chemistry_snap{SNAP}_{MODE}_{STAT}.png'
fig.savefig(out, dpi=130)
print('\nsaved', out)
