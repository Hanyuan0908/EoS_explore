"""Check the exponential fits behind rd_sf: Sigma(R) and the fitted line.

Not needed for any figure -- it is the evidence that the R_d in
../ana_gas_disc_scalelength.py describes the profile it is fitted to, at the
epochs that matter (pre-merger, the minimum, and late).

Writes figures/diag_sigma_exp_fit.png.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config_au18 as C

A = np.load(C.OUT_DIR + '/gas_disc_evolution_au18.npz')
edges, (lo, hi) = A['rbins'], A['fit_range']
ctr = .5 * (edges[:-1] + edges[1:])
EPOCHS = [3.2, 5.4, 9.4, 13.0]

fig, axes = plt.subplots(1, len(EPOCHS), figsize=(4 * len(EPOCHS), 3.6), sharey=True)
for ax, t in zip(axes, EPOCHS):
    k = int(np.nanargmin(np.abs(A['time'] - t)))
    sig, rh, rd = A['sigma_sf'][k], A['rhalf_sf'][k], A['rd_sf'][k]
    ax.plot(ctr, sig, 'o', ms=2.5, color='#b2182b')
    ok = (ctr > lo * rh) & (ctr < hi * rh) & (sig > 0)
    if ok.sum() >= 5 and np.isfinite(rd):
        c = np.polyfit(ctr[ok], np.log(sig[ok]), 1)
        ax.plot(ctr[ok], np.exp(np.polyval(c, ctr[ok])), 'k-', lw=2)
        ax.axvspan(lo * rh, hi * rh, color='.85', zorder=0)
    ax.set(yscale='log', xlim=(0, 25), xlabel='R [kpc]',
           title=f"t = {A['time'][k]:.2f} Gyr\n$R_d$={rd:.2f}, $R_{{1/2}}$={rh:.2f} kpc")
axes[0].set_ylabel(r'$\Sigma_{\rm SF\,gas}$ [M$_\odot$ kpc$^{-2}$]')
fig.tight_layout()
os.makedirs('figures', exist_ok=True)
fig.savefig('figures/diag_sigma_exp_fit.png', dpi=140)
print('saved figures/diag_sigma_exp_fit.png')
