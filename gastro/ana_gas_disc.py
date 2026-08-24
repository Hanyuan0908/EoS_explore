"""Does the gas disc respond to the merger?

Diagnostic figure (not a publication cut).  Four panels:
  (a) half-mass radius of the cold gas disc against time, with the dwarf's
      pericentric passages marked, and the stellar half-mass radius for scale;
  (b) the same trend under three different gas selections, to show whether it
      survives the choice of cut;
  (c) cold gas mass inside and outside the aperture -- the latter tracks the
      satellite's own gas arriving;
  (d) star formation rate, an independent tracer of when the merger acts.

Reads out/gas_disc_evolution.npz (built by prep_gas_disc.py).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gastro_config as G

os.makedirs(G.FIG_DIR, exist_ok=True)
d = np.load(G.OUT_DIR + '/gas_disc_evolution.npz')
t = d['time']
PERI = [(1.6, '1st peri'), (2.5, '2nd peri'), (3.2, 'disrupted')]
CG, CS = '#2166ac', '#b2182b'


def peri(ax, label=True):
    for x, lab in PERI:
        ax.axvline(x, color='k', lw=1.1, ls='--')
        if label:
            ax.text(x - .1, ax.get_ylim()[1], lab, rotation=90, ha='right', va='top', fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

# (a) the headline measurement -------------------------------------------------
ax = axes[0, 0]
# Primary = star-forming gas on the VINTERGATAN criterion (Agertz et al. 2021),
# T < 1e4 K and n_H > 1 cm^-3.  This gas reaches n_H ~ 80 cm^-3 here so the cut
# is well resolved; the same cut is NOT transferable to Auriga, whose SF gas sits
# at n_H ~ 0.1-0.5 on the Springel-Hernquist effective EOS.
ax.plot(t, d['rhalf_agertz'], 'o-', color=CG, lw=2.2, ms=5,
        label=r'star-forming gas ($T<10^4$ K, $n_H>1$ cm$^{-3}$)')
ax.plot(t, d['r90_agertz'], 's--', color=CG, lw=1.4, ms=4, alpha=.6, label='star-forming gas, $R_{90}$')
ax.plot(t, d['rhalf_cold'], 'o-', color='#66a0c8', lw=1.6, ms=3.5,
        label=r'all cold gas ($T<3\times10^4$ K)')
ax.plot(t, d['rhalf_star'], 'o-', color=CS, lw=2.2, ms=5, label='stars')
ax.set(xlabel='time [Gyr]', ylabel='half-mass radius [kpc]', xlim=(0, 10))
ax.set_title('(a) Disc size against time')
peri(ax)
ax.legend(fontsize=9, loc='lower right')

# (b) robustness ---------------------------------------------------------------
ax = axes[0, 1]
for k, lab, c in [('rhalf_agertz', r'$T<10^4$ K, $n_H>1$ (Agertz)', CG),
                  ('rhalf_gasoline', r'$T<1.5\times10^4$ K, $n_H>0.1$ (GASOLINE std)', '#1a9850'),
                  ('rhalf_cold', r'$T<3\times10^4$ K (all cold)', '#66a0c8'),
                  ('rhalf_corot', r'$T<3\times10^4$ K, co-rotating', '#762a83')]:
    ax.plot(t, d[k], 'o-', lw=2, ms=4.5, color=c, label=lab)
ax.set(xlabel='time [Gyr]', ylabel=r'$R_{1/2}$ of the gas disc [kpc]', xlim=(0, 10))
ax.set_title('(b) Same trend under three gas selections')
peri(ax, label=False)
ax.legend(fontsize=9)

# (c) gas budget ---------------------------------------------------------------
ax = axes[1, 0]
ax.plot(t, d['m_agertz'] / 1e9, 'o-', color=CG, lw=2.2, ms=5,
        label=r'star-forming gas inside $R<30$, $|z|<3$ kpc')
ax.plot(t, d['m_cold'] / 1e9, 'o-', color='#66a0c8', lw=1.6, ms=3.5, label='all cold gas')
ax.plot(t, d['m_cold_outside'] / 1e9, 's--', color='#762a83', lw=1.8, ms=4,
        label='cold gas outside that aperture')
ax.set(xlabel='time [Gyr]', ylabel=r'cold gas mass [$10^9\,M_\odot$]', xlim=(0, 10))
ax.set_title('(c) Cold gas budget')
peri(ax, label=False)
ax.legend(fontsize=9)

# (d) star formation -----------------------------------------------------------
ax = axes[1, 1]
e = d['sfr_edges']
ax.step(.5 * (e[:-1] + e[1:]), d['sfr'], where='mid', color='k', lw=1.5)
ax.set(xlabel='time [Gyr]', ylabel=r'SFR [$M_\odot$ yr$^{-1}$]', xlim=(0, 10))
ax.set_title('(d) Star formation history (independent merger tracer)')
peri(ax, label=False)

fig.suptitle('Clumpy+merger: does the gas disc respond to the merger?  '
             '(dashed = pericentric passages at 1.6, 2.5, 3.2 Gyr)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .95])
out = G.FIG_DIR + '/gastro_gas_disc_evolution.png'
fig.savefig(out, dpi=150)

# ------------------------------------------------------------------- numbers --
print('  t   R_half(SF) R_half(cold) R_half(*)  M_SF[1e9] M_cold[1e9]')
for i in range(len(t)):
    print('  %4.1f %10.2f %11.2f %10.2f %10.3f %11.3f'%(
        t[i], d['rhalf_agertz'][i], d['rhalf_cold'][i], d['rhalf_star'][i],
        d['m_agertz'][i]/1e9, d['m_cold'][i]/1e9))
pre = (t >= 1.0) & (t < 1.6)
during = (t >= 1.6) & (t <= 3.2)
post = (t > 3.2) & (t <= 5.0)
late = t > 5.0
for lab, m in [('pre-merger  (1.0-1.6)', pre), ('during      (1.6-3.2)', during),
               ('just after  (3.2-5.0)', post), ('late        (>5)     ', late)]:
    print(f'{lab}: R_half(SF) = {np.nanmean(d["rhalf_agertz"][m]):5.2f} kpc, '
          f'R_half(cold) = {np.nanmean(d["rhalf_cold"][m]):5.2f}, '
          f'M_SF = {np.nanmean(d["m_agertz"][m])/1e9:5.2f}e9, '
          f'SFR = {np.nanmean(d["sfr"][(e[:-1] >= t[m].min() - .25) & (e[:-1] <= t[m].max())]):5.2f}')
print('saved', out)
