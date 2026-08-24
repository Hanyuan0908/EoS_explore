"""Does the Au18 gas disc respond to the GS/E merger?

The Auriga counterpart of ../gastro/ana_gas_disc.py.  Same four panels, same
question, so the two simulations can be compared directly.

Note the difference in what "the merger" means here.  In gastro the dwarf is put
on a controlled orbit with three known pericentric passages; in Au18 the epoch
was measured (PROGRESS.md): first apocentre t~3.0-3.5 Gyr, pericentre plunge
t~5.0, coalescence t~5.3-5.6.  Those are marked instead.

Reads out/gas_disc_evolution_au18.npz (built by prep_gas_disc_au18.py).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)
d = np.load(C.OUT_DIR + '/gas_disc_evolution_au18.npz')
t = d['time']
EVENTS = [(3.25, 'first apocentre'), (5.0, 'pericentre plunge'), (5.4, 'coalescence')]
CG, CS = '#2166ac', '#b2182b'


def events(ax, label=True):
    for x, lab in EVENTS:
        ax.axvline(x, color='k', lw=1.1, ls='--')
        if label:
            ax.text(x - .12, ax.get_ylim()[1], lab, rotation=90, ha='right', va='top', fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

ax = axes[0, 0]
# Primary = the whole gas disc (cold or star-forming).  Inside this aperture only
# ~5% of the gas is hot, so this is within 0.3% of taking every gas cell; the
# SF-only curve is shown too because it is a third of the mass and sits well
# inside, tracing the star-forming disc rather than the gas disc.
ax.plot(t, d['rhalf_coldsf'], 'o-', color=CG, lw=2.2, ms=4, label='gas disc (cold or star-forming)')
ax.plot(t, d['rhalf_sf'], 'o-', color='#66a0c8', lw=1.6, ms=3, label='star-forming gas only')
ax.plot(t, d['r90_sf'], 's--', color='#66a0c8', lw=1.1, ms=2.5, alpha=.6, label='star-forming gas, $R_{90}$')
ax.plot(t, d['rhalf_star'], 'o-', color=CS, lw=2.2, ms=4, label='stars')
ax.set(xlabel='cosmic time [Gyr]', ylabel='half-mass radius [kpc]', xlim=(0, 14))
ax.set_title('(a) Disc size against time')
events(ax)
ax.legend(fontsize=9, loc='lower right')

ax = axes[0, 1]
for k, lab, c in [('rhalf_coldsf', r'cold or SF gas, $R<30$, $|z|<5$ kpc', CG),
                  ('rhalf_sf', r'SF gas only, $R<30$, $|z|<5$ kpc', '#66a0c8'),
                  ('rhalf_sf_wide', r'SF gas only, $R<50$, $|z|<10$ kpc', '#1a9850')]:
    ax.plot(t, d[k], 'o-', lw=2, ms=3.5, color=c, label=lab)
ax.set(xlabel='cosmic time [Gyr]', ylabel=r'$R_{1/2}$ of the gas disc [kpc]', xlim=(0, 14))
ax.set_title('(b) Same trend under three gas selections')
events(ax, label=False)
ax.legend(fontsize=9)

ax = axes[1, 0]
ax.plot(t, d['m_coldsf'] / 1e9, 'o-', color=CG, lw=2.2, ms=4,
        label=r'cold or SF gas inside $R<30$, $|z|<5$ kpc')
ax.plot(t, d['m_sf'] / 1e9, 'o-', color='#66a0c8', lw=1.6, ms=3, label='star-forming gas only')
# m_sf_outside is NOT plotted: unlike the isolated gastro box, "outside the
# aperture" in a cosmological zoom is the entire high-resolution region and its
# other galaxies (5.2e12 Msol at z=0), so it tracks the zoom volume rather than
# the infalling satellite.  Panel (d) carries the independent merger tracer.
ax.set(xlabel='cosmic time [Gyr]', ylabel=r'gas mass [$10^9\,M_\odot$]', xlim=(0, 14))
ax.set_title('(c) Disc gas budget')
events(ax, label=False)
ax.legend(fontsize=9)

ax = axes[1, 1]
ax.plot(t, d['sfr'], 'o-', color='k', lw=1.8, ms=3.5)
ax.set(xlabel='cosmic time [Gyr]', ylabel=r'SFR [$M_\odot$ yr$^{-1}$]', xlim=(0, 14))
ax.set_title('(d) Star formation rate (independent merger tracer)')
events(ax, label=False)

fig.suptitle('Auriga halo 18: does the gas disc respond to the GS/E merger?  '
             '(dashed = first apocentre 3.25, plunge 5.0, coalescence 5.4 Gyr)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .95])
out = C.FIG_DIR + '/au18_gas_disc_evolution.png'
fig.savefig(out, dpi=150)

print('   t   R_half(gas) R_half(SF)  R_half(*)  M_gas[1e9] M_SF[1e9]  SFR')
for i in range(len(t)):
    print('  %5.2f %10.2f %10.2f %10.2f %11.3f %9.3f %7.2f' % (
        t[i], d['rhalf_coldsf'][i], d['rhalf_sf'][i], d['rhalf_star'][i],
        d['m_coldsf'][i] / 1e9, d['m_sf'][i] / 1e9, d['sfr'][i]))
for lab, m in [('before merger (3.5-4.8)', (t >= 3.5) & (t < 4.8)),
               ('merger        (4.8-5.8)', (t >= 4.8) & (t <= 5.8)),
               ('after         (5.8-7.5)', (t > 5.8) & (t <= 7.5)),
               ('late          (>9)     ', t > 9)]:
    if m.sum():
        print(f'{lab}: R_half(gas) = {np.nanmean(d["rhalf_coldsf"][m]):5.2f} kpc, '
              f'R_half(SF) = {np.nanmean(d["rhalf_sf"][m]):5.2f}, '
              f'M_gas = {np.nanmean(d["m_coldsf"][m])/1e9:5.2f}e9, '
              f'SFR = {np.nanmean(d["sfr"][m]):5.2f}')
print('saved', out)
