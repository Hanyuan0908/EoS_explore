"""Age and chemistry of the two Eos populations in Au18: born hot vs heated.

The birth v_R-v_phi plane splits the Eos-like stars in two (see eos_origins.py):
one lobe born already hot, one born on the disc ridge and heated afterwards.
This asks whether the two are distinguishable by age and composition -- whether
an observer could tell them apart without access to birth kinematics.

Row 1: age distribution, the age-metallicity relation, [Fe/H], and the birth
v_phi histogram that defines the split.  Rows 2-3: [X/Fe] against [Fe/H] for the
six elements Auriga tracks besides Fe.

Reads cached arrays only; no snapshots are touched.
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
ref = np.load(C.OUT_DIR + '/z0_reference_pops.npz')
cat = d['cat']

C_HALO, C_DISC, C_GSE, C_ALL = '#7b3294', 'crimson', '#1f6fd0', '.55'
POPS = [('all merger-born', np.ones(len(d['ids']), bool), C_ALL),
        ('halo-born Eos (born hot)', d['halo_born'], C_HALO),
        ('disc-born Eos (heated)', d['disc_born'], C_DISC)]
ELS = [('cfe', 'C'), ('nfe', 'N'), ('ofe', 'O'),
       ('nefe', 'Ne'), ('mgfe', 'Mg'), ('sife', 'Si')]
EDGES = np.linspace(-1.6, 0.4, 17)
CEN = .5 * (EDGES[:-1] + EDGES[1:])


def running_median(x, y, mask, nmin=15):
    """Median of y in [Fe/H] bins, NaN where a bin is too sparse to trust."""
    out = np.full(len(CEN), np.nan)
    good = mask & np.isfinite(x) & np.isfinite(y)
    for i in range(len(CEN)):
        s = good & (x >= EDGES[i]) & (x < EDGES[i + 1])
        if s.sum() >= nmin:
            out[i] = np.median(y[s])
    return out


fig, axes = plt.subplots(3, 4, figsize=(22, 15))

# --- (a) age distribution -----------------------------------------------------
ax = axes[0, 0]
bins = np.linspace(6.5, 9.5, 45)
for lab, m, c in POPS:
    ax.hist(d['age'][m], bins=bins, density=True, histtype='step', lw=2.2, color=c, label=lab)
    ax.axvline(np.median(d['age'][m]), color=c, ls=':', lw=1.4)
ax.set(xlabel='age [Gyr]', ylabel='normalised density', title='(a) Age distribution')
ax.legend(fontsize=8.5)

# --- (b) age-metallicity ------------------------------------------------------
ax = axes[0, 1]
ax.hist2d(d['feh'], d['age'], bins=(90, 70), range=((-1.6, 0.5), (6.5, 9.5)),
          norm=LogNorm(), cmap='Greys')
for lab, m, c in POPS[1:]:
    ax.plot(CEN, running_median(d['feh'], d['age'], m), 'o-', color=c, lw=2.2, ms=4.5, label=lab)
ax.set(xlabel='[Fe/H]', ylabel='age [Gyr]', title='(b) Age-metallicity relation')
ax.legend(fontsize=8.5)

# --- (c) metallicity ----------------------------------------------------------
ax = axes[0, 2]
bins = np.linspace(-2.0, 0.6, 55)
for lab, m, c in POPS:
    ax.hist(d['feh'][m], bins=bins, density=True, histtype='step', lw=2.2, color=c, label=lab)
    ax.axvline(np.median(d['feh'][m]), color=c, ls=':', lw=1.4)
gf = np.isfinite(ref['gse_feh'])
ax.hist(ref['gse_feh'][gf], bins=bins, density=True, histtype='step', lw=1.8, ls='--',
        color=C_GSE, label='GS/E debris')
ax.set(xlabel='[Fe/H]', ylabel='normalised density', title='(c) Metallicity')
ax.legend(fontsize=8.5)

# --- (d) the definition -------------------------------------------------------
ax = axes[0, 3]
bins = np.linspace(-200, 400, 61)
ax.hist(d['bvphi'][d['eos']], bins=bins, histtype='stepfilled', color='.85',
        label=f"Eos-like, all ({d['eos'].sum():,})")
for lab, m, c in POPS[1:]:
    ax.hist(d['bvphi'][m], bins=bins, histtype='step', lw=2.2, color=c,
            label=f'{lab} ({m.sum():,})')
ax.axvline(EO.VPHI_SPLIT, color='k', lw=1.6, ls='--')
ax.set(xlabel=r'$v_\phi$ at birth [km s$^{-1}$]', ylabel='stars per bin',
       title=f'(d) The split, at $v_\\phi$ = {EO.VPHI_SPLIT:.0f} km/s')
ax.legend(fontsize=8.5, loc='upper left')

# --- [X/Fe] against [Fe/H] ----------------------------------------------------
gfeh = ref['gse_feh']
for k, (key, el) in enumerate(ELS):
    ax = axes[1 + k // 4, k % 4]
    ax.hist2d(d['feh'], d[key], bins=(100, 80), range=((-1.6, 0.5), (-0.7, 0.6)),
              norm=LogNorm(), cmap='Greys')
    for lab, m, c in POPS[1:]:
        ax.plot(CEN, running_median(d['feh'], d[key], m), 'o-', color=c, lw=2.2, ms=4.5,
                label=lab)
    ax.plot(CEN, running_median(gfeh, ref['gse_' + key], np.ones(len(gfeh), bool)),
            '--', color=C_GSE, lw=1.9, label='GS/E debris')
    ax.set(xlabel='[Fe/H]', ylabel=f'[{el}/Fe]', title=f'[{el}/Fe] vs [Fe/H]')
    if k == 0:
        ax.legend(fontsize=8)
for k in range(len(ELS), 8):
    axes[1 + k // 4, k % 4].axis('off')

fig.suptitle('Au18: the two Eos populations compared -- born hot (purple) vs born on the disc '
             'and heated (crimson).  Greyscale = all merger-born stars', fontsize=15)
fig.tight_layout(rect=[0, 0, 1, .96])
out = C.FIG_DIR + '/au18_eos_origins_chemistry.png'
fig.savefig(out, dpi=140)

hdr = f'{"":28s} {"N":>7s} {"age":>7s} {"[Fe/H]":>8s} ' + ' '.join(f'{e:>7s}' for _, e in ELS)
print(hdr)
for lab, m, c in POPS:
    print(f'{lab:28s} {m.sum():7,} {np.median(d["age"][m]):7.2f} {np.median(d["feh"][m]):+8.2f} '
          + ' '.join(f'{np.median(d[k][m]):+7.3f}' for k, _ in ELS))
g = np.isfinite(cat['gse_age'])
print(f'{"GS/E debris":28s} {gf.sum():7,} {np.median(cat["gse_age"][g]):7.2f} '
      f'{np.nanmedian(gfeh[gf]):+8.2f} '
      + ' '.join(f'{np.nanmedian(ref["gse_" + k]):+7.3f}' for k, _ in ELS))
print()
print('difference (halo-born minus disc-born), at matched [Fe/H]:')
for key, el in ELS:
    a = running_median(d['feh'], d[key], d['halo_born'])
    b = running_median(d['feh'], d[key], d['disc_born'])
    both = np.isfinite(a) & np.isfinite(b)
    print(f'  [{el}/Fe]: {np.mean((a - b)[both]):+.4f} dex over {both.sum()} shared bins')
a = running_median(d['feh'], d['age'], d['halo_born'])
b = running_median(d['feh'], d['age'], d['disc_born'])
both = np.isfinite(a) & np.isfinite(b)
print(f'  age    : {np.mean((a - b)[both]):+.3f} Gyr over {both.sum()} shared bins')
print('saved', out)
