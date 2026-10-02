"""Apocentre distribution of the Splash in Au18.

Selection, on the z = 0 in-situ catalogue (GS/E debris is held separately in that
file, so accreted stars cannot enter):

    age > 9 Gyr        formed before t = 4.82 Gyr, i.e. before the GS/E plunge
                       at t = 5.0 and coalescence at 5.4
    5 < R < 10 kpc     present-day cylindrical radius, a solar-ish annulus
    v_phi < 80 km/s    slow or counter-rotating

The v_phi cut is taken literally, as one-sided: it keeps the 4,891 stars with
v_phi < -80, a quarter of the sample, which the symmetric |v_phi| < 80 mask used
elsewhere in this project would throw away.  Both are drawn, because they are
different populations and the retrograde quarter is exactly where a merger origin
would show up.

r_apo comes from the roots of Phi_eff(r) = E in the spherically averaged potential
(orbit_tools.apo_peri), the same approximation as the APOGEE orbit integrations.
Its failure mode matters here: for a nearly unbound star the outer root runs away,
which is why the 99th percentile is 347 kpc while the 95th is 26.  The axis is
logarithmic so that tail is shown rather than hidden, and the fraction beyond
100 kpc is annotated -- treat those as "escaping", not as measurements.

Writes figures/au18_splash_rapo.png.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import config_au18 as C

AGE_MIN, RLO, RHI, VPHI_MAX = 9.0, 5.0, 10.0, 80.0
cMAIN, cSYM = '#E8112D', '#1F6FB2'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13, 'axes.labelsize': 14.5,
    'xtick.labelsize': 12.5, 'ytick.labelsize': 12.5, 'legend.fontsize': 11,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'figure.dpi': 140, 'savefig.dpi': 200,
})

cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
age, R, vphi, rapo = cat['age'], cat['R'], cat['vphi'], cat['rapo']
base = (age > AGE_MIN) & (R > RLO) & (R < RHI) & np.isfinite(rapo)
SELS = [(rf'$v_\phi < {VPHI_MAX:.0f}$ km s$^{{-1}}$', base & (vphi < VPHI_MAX), cMAIN),
        (rf'$|v_\phi| < {VPHI_MAX:.0f}$ km s$^{{-1}}$', base & (np.abs(vphi) < VPHI_MAX), cSYM)]

# Uniform bins in log10 r, so the plotted quantity is dN/dlog10(r_apo).  With
# density=True over log-spaced bins in LINEAR r the heights are dN/dr, which on a
# log axis makes the peak sit where the bins are narrow rather than where the
# stars are -- the shape is then an artefact of the binning.
LEDGE = np.linspace(np.log10(3.), np.log10(500.), 61)
bins = 10 ** LEDGE
fig, ax = plt.subplots(figsize=(8.2, 5.4))
ax.axvspan(RLO, RHI, color='.85', lw=0, zorder=0)
ax.text(np.sqrt(RLO * RHI), .97, 'selection\nannulus', transform=ax.get_xaxis_transform(),
        ha='center', va='top', fontsize=10.5, color='.35')

for lab, m, c in SELS:
    h = np.histogram(np.log10(rapo[m]), bins=LEDGE, density=True)[0]
    ax.step(bins, np.concatenate([h[:1], h]), color=c, lw=2.4,
            label=f'{lab}  ({m.sum():,})')
    med = np.nanmedian(rapo[m])
    ax.axvline(med, color=c, lw=1.2, ls='--')
    print(f'{lab}: N={m.sum():,}  median r_apo {med:.2f} kpc')

# The spike near 300 kpc is not a population: it is where apo_peri's outer root
# runs away for stars that are close to unbound, piling them against the r_max of
# its search grid.  Labelled rather than clipped.
frac100 = 100 * np.mean(rapo[SELS[0][1]] > 100)
ax.annotate(f'nearly unbound:\n{frac100:.1f}% at $R_\\mathrm{{apo}}>100$ kpc,\n'
            'piled at the search limit',
            xy=(330, .40), xytext=(75, 1.35), fontsize=10, color='.3', ha='center',
            arrowprops=dict(arrowstyle='->', color='.5', lw=1.1,
                            connectionstyle='arc3,rad=-0.2'))

ax.set(xscale='log', xlim=(3., 500.), ylim=(0, None), xlabel=r'$R_{\rm apo}$ [kpc]',
       ylabel=r'density per dex, $dN/d\log_{10} R_{\rm apo}$')
ax.set_xticks([3, 5, 10, 20, 50, 100, 300])
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_title(f'Au18 Splash candidates: age > {AGE_MIN:.0f} Gyr, '
             f'{RLO:.0f} < R < {RHI:.0f} kpc', fontsize=12.5)
ax.legend(loc='upper right', handlelength=1.8)

fig.tight_layout()
os.makedirs(C.FIG_DIR, exist_ok=True)
out = C.FIG_DIR + '/au18_splash_rapo.png'
fig.savefig(out, bbox_inches='tight')

# ------------------------------------------------------------------- numbers --
print(f'\nparent sample (age > {AGE_MIN}, {RLO} < R < {RHI}): {base.sum():,} in-situ stars')
for lab, m, _ in SELS:
    q = np.nanpercentile(rapo[m], [5, 16, 50, 84, 95, 99])
    print(f'\n{lab}  N = {m.sum():,}')
    print('  r_apo  5/16/50/84/95/99 = ' + ' '.join(f'{x:7.1f}' for x in q) + ' kpc')
    print(f'  inside the annulus (r_apo < {RHI:.0f}): {100 * np.mean(rapo[m] < RHI):.1f}%;  '
          f'> 15 kpc: {100 * np.mean(rapo[m] > 15):.1f}%;  '
          f'> 30 kpc: {100 * np.mean(rapo[m] > 30):.1f}%;  '
          f'> 100 kpc: {100 * np.mean(rapo[m] > 100):.2f}%')
    print(f'  ecc 16/50/84 = ' + ' '.join(f'{x:.2f}' for x in
          np.nanpercentile(cat['ecc'][m], [16, 50, 84])) +
          f'   r_peri 16/50/84 = ' + ' '.join(f'{x:.2f}' for x in
          np.nanpercentile(cat['rperi'][m], [16, 50, 84])) + ' kpc')
ret = base & (vphi < -VPHI_MAX)
print(f'\nretrograde tail (v_phi < -{VPHI_MAX:.0f}): {ret.sum():,} stars, '
      f'{100 * ret.sum() / (base & (vphi < VPHI_MAX)).sum():.1f}% of the one-sided sample; '
      f'median r_apo {np.nanmedian(rapo[ret]):.1f} kpc')
print('\nsaved', out)
