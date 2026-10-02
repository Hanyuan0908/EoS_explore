"""Where the nitrogen dispersion of the snapshot-72 birth classes comes from.

Isolates the N panel of au18_birth_class_abund_disp and then tests the obvious
reading of its shape: that sigma_[N/Fe] is low at both ends of [Fe/H] because
those stars form out of one chemically homogeneous reservoir, and peaks in
between because that is where two reservoirs are being stirred together.

Every star is assigned to the site it was born in,

  satellite   |r - r_GSE| <  6 kpc        (r_GSE = the clean-debris centroid)
  host        |r - r_GSE| >= 6 and |r| < 8 kpc
  bridge      everything else

and sigma_[N/Fe] is measured in the same fixed [Fe/H] bins globally, within each
site, and within each of the two branches the [N/Fe]-[Fe/H] plane splits into.
The branch divider is not assumed: it is the midpoint between the satellite and
host medians, fitted as a straight line over the bins where both sites have >= 30
stars.

Three separate questions, three separate tests:

 1. is the peak mixing?          -> compare global sigma with within-site sigma
 2. mixing of WHAT?              -> the two branches, and sigma within a branch
 3. is the metal-poor branch
    GS/E material?               -> compare it with the clean GS/E debris, which
                                    formed in the progenitor before infall

On (3): the parent sample is the in-situ catalogue, which by construction has
ZERO overlap with the clean GS/E ID list (FINDINGS section 9), so the stars in
the satellite aperture are NOT satellite stars.  They are stars the provenance
catalogue calls in-situ that happen to form at the satellite's position.  Whether
the gas they form from is host gas or stripped satellite gas is the open question
of FINDINGS section 10; the chemistry here is evidence about it, not a tracer of
the gas itself.

Writes figures/au18_nfe_dispersion.png        (the isolated panel)
       figures/au18_nfe_dispersion_origin.png (the tests)
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import auriga_public as ap
import config_au18 as C
import au18_frame as AF

SNAP = 72
CUT, ZCUT = 0.8, 1.5
FBIN_W = 0.2
FBINS = np.arange(-1.2, 0.6 + 1e-9, FBIN_W)
NMIN, NSITE, NBOOT = 50, 30, 400
R_SAT, R_HOST = 6.0, 8.0
RNG = np.random.default_rng(0)
cD, cH = '#1F6FB2', '#FF6347'
cSAT, cBR, cHOST = '#8E24AA', '#00897B', '#B8860B'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/figures'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14.5,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 11.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 140, 'savefig.dpi': 200,
})
ctr = .5 * (FBINS[:-1] + FBINS[1:])
NB = len(ctr)


def mad_sigma(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def disp_profile(feh, y, m, nmin=NMIN):
    s, lo, hi = (np.full(NB, np.nan) for _ in range(3))
    cnt = np.zeros(NB, int)
    ok = m & np.isfinite(feh) & np.isfinite(y)
    for k in range(NB):
        b = ok & (feh >= FBINS[k]) & (feh < FBINS[k + 1])
        cnt[k] = b.sum()
        if cnt[k] < nmin:
            continue
        v = y[b]
        s[k] = mad_sigma(v)
        bs = np.array([mad_sigma(v[RNG.integers(0, len(v), len(v))])
                       for _ in range(NBOOT)])
        lo[k], hi[k] = np.percentile(bs, (16, 84))
    return s, lo, hi, cnt


def med_profile(feh, y, m, nmin=NSITE):
    out = np.full(NB, np.nan)
    ok = m & np.isfinite(feh) & np.isfinite(y)
    for k in range(NB):
        b = ok & (feh >= FBINS[k]) & (feh < FBINS[k + 1])
        if b.sum() >= nmin:
            out[k] = np.median(y[b])
    return out


# ------------------------------- load the sample -----------------------------
a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
st = np.load(C.OUT_DIR + '/snapshot_times.npz')
GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
SN_ALL, T_ALL = st['snaps'], st['t_snap']
k = int(np.flatnonzero(SN_ALL == SNAP)[0])
t_lo, t_hi = T_ALL[k - 1], T_ALL[k]

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
s = ap.util.CentreOnHalo(s, cen)
ap.util.rotateto(s, xd, dir2=yd, dir3=zd)
c = s.data['Coordinates']
pos = np.column_stack([c[:, 1], c[:, 2], c[:, 0]]) * 1e3
sids, met = s.data['ParticleIDs'], s.data['GFM_Metals']
del s; gc.collect()

gcen, _, _, _ = AF.gse_centroid(pos, sids, GSE)
ROT = AF.align_azimuth(gcen)
pos = pos @ ROT.T
g2 = ROT @ gcen
print(f'GS/E centroid in the figure frame: ({g2[0]:.1f},{g2[1]:.1f},{g2[2]:.1f}) kpc, '
      f'|r| = {np.linalg.norm(g2):.1f}')

o = np.argsort(sids); ss = sids[o]

# the window stars
sel = (a['tform'] > t_lo) & (a['tform'] <= t_hi) & np.isfinite(a['eps_birth'])
ids_w, eps_w, zmx_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
p = np.searchsorted(ss, ids_w)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
idx = o[p[ok]]
P, M = pos[idx], met[idx]
eps, zmx = eps_w[ok], zmx_w[ok]
disc = (eps > CUT) | (zmx < ZCUT)

# the clean GS/E debris, for the provenance comparison
pg = np.searchsorted(ss, GSE)
okg = (pg < len(ss)) & (ss[np.minimum(pg, len(ss) - 1)] == GSE)
Mg = met[o[pg[okg]]]
del met, pos; gc.collect()

feh = C.bracket_abundance(M, 'Fe', 'H')
nfe = C.bracket_abundance(M, 'N', 'Fe')
feh_g = C.bracket_abundance(Mg, 'Fe', 'H')
nfe_g = C.bracket_abundance(Mg, 'N', 'Fe')

d_gse = np.linalg.norm(P - g2, axis=1)
d_host = np.linalg.norm(P, axis=1)
sat = d_gse < R_SAT
host = (~sat) & (d_host < R_HOST)
bridge = ~(sat | host)
halo = ~disc
# Every site mask used from here on is the HALO-BORN part of that site.  The
# host aperture is 87 per cent disc-born, so leaving it unrestricted would have
# described stars that never enter the halo-born curve this figure explains.
SITES = [('satellite', sat & halo, cSAT), ('bridge', bridge & halo, cBR),
         ('host', host & halo, cHOST)]
POPS = [('disc-born', disc, cD), ('halo-born', ~disc, cH)]
fin = np.isfinite(feh) & np.isfinite(nfe)

print(f'\nsnap {SNAP}, t_form {t_lo:.2f}-{t_hi:.2f} Gyr, N = {ok.sum():,}')
print(f'{"site":10s} {"N all":>8s} {"N halo":>8s} {"%halo":>6s} '
      f'{"med [Fe/H]":>11s} {"med [N/Fe]":>11s} {"sig[N/Fe]":>10s}   (halo-born only)')
for (lab, mh, _), mall in zip(SITES, (sat, bridge, host)):
    print(f'{lab:10s} {mall.sum():>8,} {mh.sum():>8,} '
          f'{100*mh.sum()/mall.sum():>5.1f}% '
          f'{np.nanmedian(feh[mh]):>+11.3f} {np.nanmedian(nfe[mh]):>+11.3f} '
          f'{mad_sigma(nfe[mh & fin]):>10.4f}')
print(f'{"GSE debris":10s} {okg.sum():>7,} {"--":>6s} '
      f'{np.nanmedian(feh_g):>+11.3f} {np.nanmedian(nfe_g):>+11.3f} '
      f'{mad_sigma(nfe_g[np.isfinite(nfe_g)]):>10.4f}')

# --------------------- the branch divider, measured not assumed --------------
m_sat = med_profile(feh, nfe, sat & halo)
m_host = med_profile(feh, nfe, host & halo)
mid = .5 * (m_sat + m_host)
gm = np.isfinite(mid)
cf = np.polyfit(ctr[gm], mid[gm], 1)
DIV = np.polyval(cf, feh)
print(f'\nbranch divider fitted on {gm.sum()} bins where both sites have >= {NSITE}: '
      f'[N/Fe]_div = {cf[0]:+.4f} x [Fe/H] {cf[1]:+.4f}')
low = fin & (nfe < DIV)      # the N-poor, satellite-like branch
high = fin & (nfe >= DIV)

sg_h = disp_profile(feh, nfe, halo)[0]
site_prof = {lab: disp_profile(feh, nfe, halo & m, nmin=NSITE) for lab, m, _ in SITES}
br_all = disp_profile(feh, nfe, bridge & halo, nmin=NSITE)
br_low = disp_profile(feh, nfe, bridge & halo & low, nmin=NSITE)
br_high = disp_profile(feh, nfe, bridge & halo & high, nmin=NSITE)

print('\nhalo-born, per [Fe/H] bin: global sigma, within-site sigma, site mix')
print(f'{"[Fe/H]":>7s} {"N":>6s} {"global":>8s} {"sat":>8s} {"bridge":>8s} '
      f'{"host":>8s}   sat/bri/host %')
for i in range(NB):
    b = halo & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
    if b.sum() < NMIN:
        continue
    fr = [100 * (b & m).sum() / b.sum() for _, m, _ in SITES]
    row = f'{ctr[i]:>+7.2f} {b.sum():>6,} {sg_h[i]:>8.4f} '
    row += ''.join(f'{site_prof[l][0][i]:>8.4f}' if np.isfinite(site_prof[l][0][i])
                   else f'{"--":>8s}' for l, _, _ in SITES)
    print(row + '   ' + '/'.join(f'{f:.0f}' for f in fr))

print('\nBRIDGE only: does its dispersion survive inside one branch?')
print(f'{"[Fe/H]":>7s} {"N":>6s} {"all":>8s} {"N-poor":>8s} {"N-rich":>8s} '
      f'{"f_Npoor":>9s}')
for i in range(NB):
    b = bridge & halo & fin & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
    if b.sum() < NSITE:
        continue
    f_lo = (b & low).sum() / b.sum()
    print(f'{ctr[i]:>+7.2f} {b.sum():>6,} {br_all[0][i]:>8.4f} '
          + ''.join(f'{pr[0][i]:>8.4f}' if np.isfinite(pr[0][i]) else f'{"--":>8s}'
                    for pr in (br_low, br_high))
          + f' {f_lo:>9.2f}')

print('\nprovenance of the N-poor branch: [N/Fe] at matched [Fe/H]')
print(f'{"[Fe/H]":>7s} {"satellite site":>16s} {"GSE debris":>13s} {"host site":>12s}')
m_g = np.full(NB, np.nan)
for i in range(NB):
    b = np.isfinite(feh_g) & np.isfinite(nfe_g) & (feh_g >= FBINS[i]) & (feh_g < FBINS[i + 1])
    if b.sum() >= NSITE:
        m_g[i] = np.median(nfe_g[b])
for i in range(NB):
    if not (np.isfinite(m_sat[i]) or np.isfinite(m_g[i])):
        continue
    f = lambda v: f'{v:>+.3f}' if np.isfinite(v) else '   --'
    print(f'{ctr[i]:>+7.2f} {f(m_sat[i]):>16s} {f(m_g[i]):>13s} {f(m_host[i]):>12s}')

# ------------------------- figure 1: the isolated panel ----------------------
f1, ax = plt.subplots(figsize=(7.2, 5.4))
for lab, m, col in POPS:
    sg, lo_, hi_, _ = disp_profile(feh, nfe, m)
    g = np.isfinite(sg)
    ax.plot(ctr[g], sg[g], color=col, lw=2.6, label=lab)
    ax.fill_between(ctr[g], lo_[g], hi_[g], color=col, alpha=.22, lw=0)
ax.set(xlabel='[Fe/H]', ylabel=r'$\sigma_{\rm [N/Fe]}$ [dex]',
       xlim=(FBINS[0], FBINS[-1]))
ax.legend(loc='upper right', handlelength=1.6)
ax.text(.03, .955, f'stars born in {t_lo:.2f}$-${t_hi:.2f} Gyr (snapshot {SNAP})\n'
        f'{FBIN_W:g} dex bins, $N \\geq {NMIN}$; band = 16$-$84 of {NBOOT} bootstraps',
        transform=ax.transAxes, va='top', fontsize=11)
f1.tight_layout(pad=.5)
f1.savefig(f'{OUT}/au18_nfe_dispersion.png', bbox_inches='tight')

# ---------------------------- figure 2: the tests ----------------------------
f2, AX = plt.subplots(3, 3, figsize=(16.5, 13.6))


def tag(ax, t, y=.96):
    ax.text(.03, y, t, transform=ax.transAxes, va='top', fontsize=12.5,
            fontweight='bold')


axm = AX[0, 0]
# grey background = every star born in the window, both classes
H, xe, ze = np.histogram2d(P[:, 0], P[:, 2], bins=160, range=[[-25, 25], [-25, 25]])
axm.pcolormesh(xe, ze, np.where(H > 0, H, np.nan).T, cmap='Greys',
               norm=LogNorm(vmin=1, vmax=H.max()), alpha=.45, rasterized=True)
for lab, m, col in SITES:
    axm.scatter(P[m, 0], P[m, 2], s=2.0, c=col, alpha=.25, lw=0, rasterized=True)
    axm.scatter([], [], s=26, c=col, label=f'{lab} ({m.sum():,})')
th = np.linspace(0, 2 * np.pi, 200)
axm.plot(R_SAT * np.cos(th) + g2[0], R_SAT * np.sin(th) + g2[2], color=cSAT, lw=1.6)
axm.plot(R_HOST * np.cos(th), R_HOST * np.sin(th), color=cHOST, lw=1.6)
axm.set(aspect='equal', xlim=(-25, 25), ylim=(-25, 25), xlabel='$x$ [kpc]',
        ylabel='$z$ [kpc]')
axm.legend(loc='upper right', markerscale=1.0, handletextpad=.4, fontsize=10.5)
tag(axm, '(a)  halo-born, by birth site')

axf = AX[0, 1]
for lab, m, col in SITES:
    axf.hist(feh[m], bins=np.arange(-1.6, .81, .05), density=True,
             histtype='step', color=col, lw=2.2, label=lab)
    axf.axvline(np.nanmedian(feh[m]), color=col, lw=1.1, ls='--')
axf.set(xlabel='[Fe/H]', ylabel='density', xlim=(-1.6, .8))
axf.legend(loc='upper left', bbox_to_anchor=(0, .90), fontsize=10.5)
tag(axf, '(b)  halo-born [Fe/H] per site')

axc = AX[0, 2]
h2, xe2, ye2 = np.histogram2d(feh[fin], nfe[fin], bins=(140, 120),
                              range=[[-1.6, .8], [-.35, .35]])
axc.pcolormesh(xe2, ye2, np.where(h2 > 0, h2, np.nan).T, cmap='Greys',
               norm=LogNorm(vmin=1, vmax=h2.max()), rasterized=True)
xx = np.linspace(-1.6, .8, 50)
axc.plot(xx, np.polyval(cf, xx), color='k', lw=1.6, ls=':', label='branch divider')
for lab, m, col in SITES:
    axc.plot(np.nanmedian(feh[m & fin]), np.nanmedian(nfe[m & fin]), '*', color=col,
             ms=19, mec='w', mew=1.0, zorder=5)
axc.set(xlabel='[Fe/H]', ylabel='[N/Fe]', xlim=(-1.6, .8), ylim=(-.35, .35))
axc.legend(loc='lower right', fontsize=10.5)
tag(axc, '(c)  two branches; halo-born site medians starred')

axs = AX[1, 0]
sg, lo_, hi_, _ = disp_profile(feh, nfe, halo)
g = np.isfinite(sg)
axs.plot(ctr[g], sg[g], color='k', lw=3.0, label='halo-born, all sites')
axs.fill_between(ctr[g], lo_[g], hi_[g], color='k', alpha=.15, lw=0)
for lab, m, col in SITES:
    s_, l_, h_, _ = site_prof[lab]
    gg = np.isfinite(s_)
    axs.plot(ctr[gg], s_[gg], color=col, lw=2.2, ls='--', label=f'within {lab}')
    axs.fill_between(ctr[gg], l_[gg], h_[gg], color=col, alpha=.18, lw=0)
axs.set(xlabel='[Fe/H]', ylabel=r'$\sigma_{\rm [N/Fe]}$ [dex]',
        xlim=(FBINS[0], FBINS[-1]))
axs.legend(loc='upper left', fontsize=10.5, handlelength=1.6)
tag(axs, '(d)  test 1: the peak is not site mixing', y=.70)

axr = AX[1, 1]
frac = np.full((NB, len(SITES)), np.nan)
for i in range(NB):
    b = halo & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
    if b.sum() >= NMIN:
        frac[i] = [(b & m).sum() / b.sum() for _, m, _ in SITES]
bot = np.zeros(NB)
for j, (lab, _, col) in enumerate(SITES):
    axr.bar(ctr, frac[:, j], bottom=bot, width=FBIN_W * .92, color=col, alpha=.8,
            label=lab)
    bot = bot + np.nan_to_num(frac[:, j])
axr.set(xlabel='[Fe/H]', ylabel='fraction of the halo-born bin',
        xlim=(FBINS[0], FBINS[-1]), ylim=(0, 1))
axr.legend(loc='lower center', ncol=3, fontsize=10.5)
tag(axr, '(e)  site mix per bin')

axb = AX[1, 2]
for nm, pr, col, ls in (('bridge, all', br_all, cBR, '-'),
                        ('bridge, N-poor branch', br_low, '#4C72B0', '--'),
                        ('bridge, N-rich branch', br_high, '#C44E52', '--')):
    g_ = np.isfinite(pr[0])
    axb.plot(ctr[g_], pr[0][g_], color=col, lw=2.6, ls=ls, label=nm)
    axb.fill_between(ctr[g_], pr[1][g_], pr[2][g_], color=col, alpha=.18, lw=0)
axb.set(xlabel='[Fe/H]', ylabel=r'$\sigma_{\rm [N/Fe]}$ [dex]',
        xlim=(FBINS[0], FBINS[-1]))
axb.legend(loc='upper left', fontsize=10.5, handlelength=1.8)
tag(axb, '(f)  test 2: branch mixing is ~2/3 of it', y=.70)

SHOW = [i for i in range(NB) if (bridge & halo & fin & (feh >= FBINS[i])
                                 & (feh < FBINS[i + 1])).sum() >= NMIN]
pick = [SHOW[0], int(np.nanargmax(np.where(np.isfinite(sg), sg, -1))), SHOW[-1]]
bins_n = np.arange(-.3, .301, .015)
for j, i in enumerate(pick):
    axh = AX[2, j]
    b = bridge & halo & fin & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
    axh.hist(nfe[b], bins=bins_n, density=True, histtype='stepfilled',
             color=cBR, alpha=.30, lw=0)
    axh.hist(nfe[b], bins=bins_n, density=True, histtype='step', color=cBR, lw=2.4,
             label=f'bridge ({b.sum():,})')
    for lab, m, col in (('satellite', sat & halo, cSAT), ('host', host & halo, cHOST)):
        bb = m & fin & (feh >= FBINS[i]) & (feh < FBINS[i + 1])
        if bb.sum() >= NSITE:
            axh.axvline(np.median(nfe[bb]), color=col, lw=2.0, ls='--',
                        label=f'{lab} median ({bb.sum():,})')
    axh.axvline(np.polyval(cf, ctr[i]), color='k', lw=1.4, ls=':', label='divider')
    axh.set(xlabel='[N/Fe]', ylabel='density' if j == 0 else '', xlim=(-.12, .30),
            title=f'[Fe/H] = {FBINS[i]:+.1f} to {FBINS[i+1]:+.1f}'
                  f'   ($\\sigma$ = {br_all[0][i]:.3f})')
    axh.set_ylim(top=axh.get_ylim()[1] * 1.32)
    axh.legend(loc='upper right', fontsize=10)
    tag(axh, f'({"ghi"[j]})')

f2.tight_layout(pad=.6, w_pad=1.1, h_pad=1.2)
f2.savefig(f'{OUT}/au18_nfe_dispersion_origin.png', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_nfe_dispersion.png and au18_nfe_dispersion_origin.png')
