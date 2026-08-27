"""Metallicity against GUIDING-CENTRE birth radius for the two Eos populations.

R_g is the radius of the circular orbit carrying the same angular momentum
(prep_rg_birth.py), obtained by inverting L_circ(R) = R v_c(R) against the
circular velocity curve of each birth snapshot.  Unlike the instantaneous
R_birth it does not depend on where in its orbit a star happened to be caught,
which matters here because the halo-born population is on eccentric orbits from
the moment it forms.  The snapshot is still the nearest one at or after
formation, so the ~0.15 Gyr sampling remains, but the phase noise is gone.

Retrograde stars (L_z < 0) are mapped through |L_z| and excluded from the
medians, since a guiding radius is not meaningful for them here.

Chemistry is frozen at birth, so this is the fair way to ask whether the
metallicity difference between the two populations is intrinsic or just reflects
where in the disc's radial gradient each one formed.

The comparison is made against the stars born in the SAME snapshot, not against
the whole merger window: the gradient's zero point rises as the galaxy enriches
over the 1.4 Gyr covered here, so an aggregate reference would conflate the
radial gradient with that enrichment, and the two populations do not form at the
same times.  Panel (c) removes both effects at once by subtracting, for every
star, the median [Fe/H] of stars sharing its birth snapshot and R_birth bin.

Reads cached arrays only.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import config_au18 as C
import eos_origins as EO

os.makedirs(C.FIG_DIR, exist_ok=True)
d = EO.load()
C_HALO, C_DISC, C_ALL = '#7b3294', 'crimson', '.55'
prograde = np.isfinite(d['Rg_birth']) & ~d['retrograde']
POPS = [('all merger-born', prograde, C_ALL),
        ('halo-born Eos (born hot)', d['halo_born'] & prograde, C_HALO),
        ('disc-born Eos (heated)', d['disc_born'] & prograde, C_DISC)]
print(f"retrograde at birth, dropped: halo-born "
      f"{(d['halo_born'] & d['retrograde']).sum():,} of {d['halo_born'].sum():,}, "
      f"disc-born {(d['disc_born'] & d['retrograde']).sum():,} of {d['disc_born'].sum():,}")

REDGE = np.concatenate([np.arange(0, 16, 1.5), [18, 22, 30]])
RCEN = .5 * (REDGE[:-1] + REDGE[1:])


def running_median(x, y, mask, edges, cen, nmin=15):
    out = np.full(len(cen), np.nan)
    good = mask & np.isfinite(x) & np.isfinite(y)
    for i in range(len(cen)):
        s = good & (x >= edges[i]) & (x < edges[i + 1])
        if s.sum() >= nmin:
            out[i] = np.median(y[s])
    return out


fig, axes = plt.subplots(1, 4, figsize=(25, 5.9))

# --- (a) where each population was born --------------------------------------
ax = axes[0]
bins = np.linspace(0, 25, 51)
for lab, m, c in POPS:
    ax.hist(d['Rg_birth'][m], bins=bins, density=True, histtype='step', lw=2.2, color=c, label=lab)
    ax.axvline(np.nanmedian(d['Rg_birth'][m]), color=c, ls=':', lw=1.4)
ax.set(xlabel=r'$R_{\rm g,birth}$ [kpc]', ylabel='normalised density',
       title='(a) Guiding radius at birth')
ax.legend(fontsize=9)

# --- (b) the requested plane --------------------------------------------------
ax = axes[1]
ok = np.isfinite(d['Rg_birth'])
ax.hist2d(d['Rg_birth'][ok], d['feh'][ok], bins=(90, 80), range=((0, 25), (-1.6, 0.5)),
          norm=LogNorm(), cmap='Greys')
for lab, m, c in POPS[1:]:
    ax.plot(RCEN, running_median(d['Rg_birth'], d['feh'], m, REDGE, RCEN), 'o-',
            color=c, lw=2.3, ms=5, label=lab)
# the gradient traced separately for every birth snapshot
snaps_b = np.unique(d['snap_birth'][np.isfinite(d['snap_birth'])])
for j, sn in enumerate(snaps_b):
    at = d['snap_birth'] == sn
    ax.plot(RCEN, running_median(d['Rg_birth'], d['feh'], at, REDGE, RCEN),
            '-', color=plt.cm.viridis(j / max(len(snaps_b) - 1, 1)), lw=1.1, alpha=.85,
            label='per-snapshot gradient' if j == 0 else None)
ax.plot(RCEN, running_median(d['Rg_birth'], d['feh'], prograde, REDGE, RCEN),
        's--', color='k', lw=1.8, ms=4, label='all merger-born, aggregate')
ax.set(xlim=(0, 25), xlabel=r'$R_{\rm g,birth}$ [kpc]', ylabel='[Fe/H]',
       title='(b) Metallicity vs guiding radius')
ax.legend(fontsize=9)

# --- (c) same-snapshot, same-radius reference --------------------------------
# For every star, subtract the median [Fe/H] of the stars born in its own
# snapshot within its own R_birth bin.  What survives is metallicity structure
# that is neither the radial gradient nor the enrichment of the epoch.
ax = axes[2]
snaps = np.unique(d['snap_birth'][np.isfinite(d['snap_birth'])])
dfeh = np.full(len(d['ids']), np.nan)
for sn in snaps:
    at = np.isfinite(d['Rg_birth']) & (d['snap_birth'] == sn)
    for i in range(len(RCEN)):
        cell = at & (d['Rg_birth'] >= REDGE[i]) & (d['Rg_birth'] < REDGE[i + 1])
        if cell.sum() >= 15:
            dfeh[cell] = d['feh'][cell] - np.median(d['feh'][cell])
bins = np.linspace(-0.6, 0.6, 49)
for lab, m, c in POPS:
    v = dfeh[m & np.isfinite(dfeh)]
    ax.hist(v, bins=bins, density=True, histtype='step', lw=2.2, color=c,
            label=f'{lab}  (median {np.median(v):+.3f})')
    ax.axvline(np.median(v), color=c, ls=':', lw=1.4)
ax.axvline(0, color='k', lw=1.0)
ax.set(xlabel=r'$\Delta$[Fe/H] vs stars born in the same snapshot and $R_{\rm g,birth}$ bin',
       ylabel='normalised density', title='(c) Gradient and epoch removed')
ax.legend(fontsize=8.5)

# --- (d) the birth plane ------------------------------------------------------
ax = axes[3]
ax.hist2d(d['Rg_birth'][ok], np.abs(d['z_birth'][ok]), bins=(90, 80),
          range=((0, 25), (0, 12)), norm=LogNorm(), cmap='Greys')
for lab, m, c in POPS[1:]:
    mm = m & ok
    step = max(1, mm.sum() // 2500)
    ax.scatter(d['Rg_birth'][mm][::step], np.abs(d['z_birth'][mm])[::step],
               s=5, color=c, alpha=.45, lw=0, label=lab)
ax.set(xlim=(0, 25), ylim=(0, 12), xlabel=r'$R_{\rm g,birth}$ [kpc]',
       ylabel=r'$|z_{\rm birth}|$ [kpc]', title='(d) Where they formed')
ax.legend(fontsize=9, markerscale=2.5)

fig.suptitle('Au18: the two Eos populations against guiding-centre birth radius '
             r'($R_{\rm g}$ from $L_z$ at the nearest snapshot after formation)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .94])
out = C.FIG_DIR + '/au18_eos_origins_rgbirth.png'
fig.savefig(out, dpi=145)

print(f'{"":28s} {"N":>7s} {"R_g":>9s} {"|z_birth|":>10s} {"[Fe/H]":>8s}')
for lab, m, c in POPS:
    print(f'{lab:28s} {m.sum():7,} {np.nanmedian(d["R_birth"][m]):9.2f} '
          f'{np.nanmedian(np.abs(d["z_birth"][m])):10.2f} {np.median(d["feh"][m]):+8.2f}')
a = running_median(d['Rg_birth'], d['feh'], d['halo_born'], REDGE, RCEN)
b = running_median(d['Rg_birth'], d['feh'], d['disc_born'], REDGE, RCEN)
both = np.isfinite(a) & np.isfinite(b)
print(f'\n[Fe/H] difference at matched R_birth (halo-born minus disc-born): '
      f'{np.mean((a - b)[both]):+.3f} dex over {both.sum()} shared bins')
allm = running_median(d['Rg_birth'], d['feh'], prograde, REDGE, RCEN)
g = np.isfinite(allm) & (RCEN < 16)
slope = np.polyfit(RCEN[g], allm[g], 1)[0]
print(f'radial gradient of all merger-born stars: {slope:+.4f} dex/kpc over R = 0-16 kpc')
print(f'raw [Fe/H] offset between the two populations: '
      f'{np.median(d["feh"][d["halo_born"]]) - np.median(d["feh"][d["disc_born"]]):+.3f} dex')
print()
print('per-snapshot gradient and zero point (all stars born in that snapshot):')
for sn in snaps:
    at = d['snap_birth'] == sn
    med = running_median(d['Rg_birth'], d['feh'], at, REDGE, RCEN)
    g2 = np.isfinite(med) & (RCEN < 16)
    if g2.sum() > 3:
        sl, ic = np.polyfit(RCEN[g2], med[g2], 1)
        print(f'  snap {int(sn)}: N={at.sum():6,}  gradient {sl:+.4f} dex/kpc, '
              f'[Fe/H] at R=0 {ic:+.3f}')
print()
print('after removing the same-snapshot, same-R_birth median:')
for lab, m, c in POPS:
    v = dfeh[m & np.isfinite(dfeh)]
    print(f'  {lab:28s} median {np.median(v):+.4f}  scatter {np.std(v):.3f}  (N={len(v):,})')
hb = dfeh[d['halo_born'] & np.isfinite(dfeh)]; db = dfeh[d['disc_born'] & np.isfinite(dfeh)]
print(f'  => halo-born minus disc-born: {np.median(hb) - np.median(db):+.4f} dex')
print('saved', out)
