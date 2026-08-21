"""Birth time versus present-day orbital circularity for all Au18 in-situ stars."""
import os
import numpy as np
import matplotlib.pyplot as plt

import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.FIG_DIR, exist_ok=True); os.makedirs(C.OUT_DIR, exist_ok=True)
m = np.load(C.OUT_DIR + "/matched_z0.npz")
s = snap_mod.load_snapshot(127, 4, snappath=C.SIM_DIR,
    loadlist=["Coordinates", "Velocities", "Masses", "Potential",
              "GFM_StellarFormationTime"])
real = s.data["GFM_StellarFormationTime"] > 0
old_to_new = np.full(len(real), -1, dtype=np.int64)
original_idx = np.flatnonzero(real)
old_to_new[original_idx] = np.arange(real.sum())
ii = old_to_new[m["ii"]]; ii = ii[ii >= 0]
for k in list(s.data): s.data[k] = s.data[k][real]
sf = sub_mod.subfind(127, directory=C.SIM_DIR, loadlist=["GroupFirstSub", "SubhaloPos"])
center = sf.data["SubhaloPos"][int(sf.data["GroupFirstSub"][0])]
util.CentreOnHalo(s, center)
rad0 = np.sqrt((s.data["Coordinates"]**2).sum(1))
inside = rad0 < .01
bulk = np.average(s.data["Velocities"][inside], axis=0, weights=s.data["Masses"][inside])
s.data["Velocities"] -= bulk
util.align_galaxy(s, radialcut=.01)

x = s.data["Coordinates"] * 1000.; v = s.data["Velocities"]
R = np.hypot(x[:,1], x[:,2]); r = np.sqrt((x*x).sum(1))
jz = x[:,1]*v[:,2] - x[:,2]*v[:,1]
if np.median(jz[(R>3)&(R<12)&(np.abs(x[:,0])<2)]) < 0: jz *= -1
energy = .5*(v*v).sum(1) + s.data["Potential"]

# Equal-count energy bins; the prograde 95th-percentile envelope estimates Jcirc(E).
valid = np.isfinite(energy) & np.isfinite(jz) & (r < 50)
edges = np.quantile(energy[valid], np.linspace(0, 1, 241))
ibin = np.clip(np.searchsorted(edges, energy, side="right")-1, 0, len(edges)-2)
jcirc_bin = np.full(len(edges)-1, np.nan)
for b in range(len(jcirc_bin)):
    q = valid & (ibin == b) & (jz > 0)
    if q.sum() > 30: jcirc_bin[b] = np.percentile(jz[q], 95)
ok = np.isfinite(jcirc_bin)
jcirc_bin = np.interp(np.arange(len(jcirc_bin)), np.flatnonzero(ok), jcirc_bin[ok])
eps = jz / jcirc_bin[ibin]

ins = np.zeros(len(r), bool); ins[ii] = True
sel = ins & (r > 5) & (r < 10) & np.isfinite(eps)
birth = C.a_to_age(s.data["GFM_StellarFormationTime"])

# Membership audit: the provenance in-situ selection must not overlap either
# the complete ex-situ catalogue or the clean GS/E subset.
selected_original = original_idx[np.flatnonzero(sel)]
ex_overlap = np.intersect1d(selected_original, m["ei"], assume_unique=False).size
print(f"membership audit: selected-index overlap with all ex-situ = {ex_overlap}")

time_lo = 0.0
time_hi = 11.5
tb = np.linspace(time_lo, time_hi, 59); tc=.5*(tb[:-1]+tb[1:])
med=np.full(len(tc),np.nan); frac=np.full(len(tc),np.nan)
for k in range(len(tc)):
    q=sel&(birth>=tb[k])&(birth<tb[k+1])
    if q.sum()>100: med[k]=np.median(eps[q]); frac[k]=np.mean(eps[q]>.7)
# Operational spin-up: first sustained crossing of median epsilon=0.7.
cross=np.flatnonzero((med>.7) & (np.convolve((med>.7).astype(int),np.ones(3,dtype=int),'same')>=2))
t_spin=float(tc[cross[0]]) if len(cross) else np.nan

fig,ax=plt.subplots(figsize=(9,6.2))
h=ax.hist2d(birth[sel],np.clip(eps[sel],-1.5,1.5),bins=[120,120],
            range=[[time_lo,time_hi],[-1.5,1.5]],cmap='Greys',cmin=1,norm='log')
ax.plot(tc,med,color='tab:red',lw=2,label='median circularity')
ax.plot(tc,frac,color='tab:blue',lw=2,label=r'fraction $\epsilon>0.7$')
ax.axhline(.7,color='0.5',ls='--',lw=1)
if np.isfinite(t_spin): ax.axvline(t_spin,color='tab:orange',ls='--',lw=2,label=f'spin-up ~{t_spin:.2f} Gyr')
ax.axvspan(4.99,6.54,color='gold',alpha=.16,label='merger window')
ax.set(xlabel='birth cosmic time [Gyr]',ylabel=r'present-day circularity $\epsilon=j_z/j_{circ}(E)$',
       xlim=(time_lo,time_hi),ylim=(-1.5,1.5))
ax.legend(loc='lower right'); ax.set_title('Au18 in-situ stars at z=0 (5 < r < 10 kpc)')
fig.colorbar(h[3],ax=ax,label='star-particle count'); fig.tight_layout()
out=C.FIG_DIR+'/au18_insitu_age_circularity_r5_10.png'; fig.savefig(out,dpi=150)

# Column-normalised view: P(circularity | birth-time bin). Each age column sums
# to unity, exposing changes in distribution shape independently of the SFH.
xedges = np.linspace(time_lo, time_hi, 121)
yedges = np.linspace(-1.5, 1.5, 121)
counts, _, _ = np.histogram2d(birth[sel], np.clip(eps[sel], -1.5, 1.5),
                              bins=[xedges, yedges])
denom = counts.sum(axis=1, keepdims=True)
column_fraction = np.divide(counts, denom, out=np.zeros_like(counts),
                            where=denom > 0)
fig2, ax2 = plt.subplots(figsize=(9, 6.2))
positive = column_fraction[column_fraction > 0]
vmin = max(np.percentile(positive, 2), 1e-4)
pcm = ax2.pcolormesh(xedges, yedges, column_fraction.T, cmap='Greys',
                     norm=plt.matplotlib.colors.LogNorm(vmin=vmin,
                                                        vmax=column_fraction.max()),
                     shading='auto')
ax2.plot(tc, med, color='tab:red', lw=2, label='median circularity')
ax2.plot(tc, frac, color='tab:blue', lw=2, label=r'fraction $\epsilon>0.7$')
ax2.axhline(.7, color='0.5', ls='--', lw=1)
if np.isfinite(t_spin):
    ax2.axvline(t_spin, color='tab:orange', ls='--', lw=2,
                label=f'spin-up ~{t_spin:.2f} Gyr')
ax2.axvspan(4.99, 6.54, color='gold', alpha=.16, label='merger window')
ax2.set(xlabel='birth cosmic time [Gyr]',
        ylabel=r'present-day circularity $\epsilon=j_z/j_{circ}(E)$',
        xlim=(time_lo, time_hi), ylim=(-1.5, 1.5))
ax2.legend(loc='lower right')
ax2.set_title('Au18 in-situ stars at z=0 (5 < r < 10 kpc) — column normalised')
fig2.colorbar(pcm, ax=ax2, label='fraction within each birth-time bin')
fig2.tight_layout()
out_norm = C.FIG_DIR+'/au18_insitu_age_circularity_r5_10_column_normalised.png'
fig2.savefig(out_norm, dpi=150)
np.savez(C.OUT_DIR+'/insitu_spinup_r5_10.npz',birth=birth[sel],circularity=eps[sel],
         time_bin=tc,median_circularity=med,disc_fraction=frac,t_spin=t_spin,
         column_time_edges=xedges, column_circularity_edges=yedges,
         column_fraction=column_fraction)
print(f'N={sel.sum():,}; operational spin-up t={t_spin:.2f} Gyr; saved {out} and {out_norm}')
