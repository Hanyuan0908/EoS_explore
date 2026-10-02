"""Orientation of the GS/E merger in Au18, and what the disc does about it.

Two directions per snapshot, both computed in the raw simulation frame -- NOT in
the disc-aligned frame the other scripts use.  That is the whole point: the
disc-aligned frame rotates with the disc, so in it the disc axis is fixed by
construction and any reorientation is invisible.  Here nothing is rotated, so the
two vectors can be compared with each other and across time.

  L_disc   angular momentum of the stars inside 10 kpc -- the disc spin axis
  L_gse    angular momentum of the clean GS/E debris (gse_clean_ids.npy) about
           the host centre, in two forms:
             orbital  M (R_com x V_com), the satellite's orbit as a whole.
                      Meaningful while the progenitor is a coherent object;
                      once it phase-mixes its centre of mass sits at the origin
                      and this becomes noise.
             total    sum of m r x v over the debris.  Meaningful throughout, and
                      after coalescence it is the net rotation of the debris.

Angles reported: theta(L_disc, L_gse) says whether the encounter is prograde
(0 deg), polar (90) or retrograde (180) with respect to the disc *at that time*;
theta(L_disc(t), L_disc(z=0)) says how far the disc has since turned; and
theta(L_disc(t), L_gse(T_INFALL)) is the one that answers whether they end up
aligned -- it compares the disc at every epoch against the ONE fixed direction
the satellite came in on, before the encounter scrambled it.

That last comparison has to use the infall value, not the running one.  Once the
progenitor phase-mixes, its centre of mass sits at the origin and the net
angular momentum of the debris is a small residual (|L_orb|/|L_tot| falls from
0.98 to 0.08) whose direction wanders; reading alignment off it says nothing
about the orbit that actually delivered the torque.

Bulk velocity is the mass-weighted mean of the inner stars, as in
prep_gas_disc_au18.py, so "angular momentum" means about the galaxy, not about a
drifting box.  Comoving-vs-physical and h factors are uniform scalings and cancel
out of a direction; the Hubble term is radial and drops out of r x v.

Writes out/gse_orientation.npz and figures/au18_gse_orientation.png.
Re-run with `plot` to redraw from the cache without touching the snapshots.
"""
import gc, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config_au18 as C

CACHE = C.OUT_DIR + '/gse_orientation.npz'
SNAPS = list(range(50, 101)) + [105, 110, 115, 120, 127]
RIN = 0.010          # 10 kpc in the snapshot's length unit, as elsewhere
T_PERI, T_COAL = 5.0, 5.4
T_INFALL = 4.0        # where L_gse is still the coherent infall orbit


def unit(v):
    n = np.sqrt((np.asarray(v, float) ** 2).sum(-1))
    return np.asarray(v, float) / np.where(n > 0, n, 1.)


def angle(u, v):
    """Angle between two vectors, in degrees."""
    c = float(np.clip(np.dot(unit(u), unit(v)), -1., 1.))
    return float(np.degrees(np.arccos(c)))


if 'plot' not in sys.argv[1:]:
    from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util
    GSE_IDS = np.load(C.OUT_DIR + '/gse_clean_ids.npy')
    print(f'GS/E clean debris: {len(GSE_IDS):,} stars')
    rec = {k: [] for k in ('snap', 'a', 'time', 'n_gse', 'r_com', 'f_bound')}
    vec = {k: [] for k in ('L_disc', 'L_orb', 'L_tot')}
    for sn in SNAPS:
        s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
            loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
                      'GFM_StellarFormationTime'])
        a = float(s.time)
        real = s.data['GFM_StellarFormationTime'] > 0
        for k in list(s.data):
            s.data[k] = s.data[k][real]
        sf = sub_mod.subfind(sn, directory=C.SIM_DIR,
                             loadlist=['GroupFirstSub', 'SubhaloPos'])
        cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
        util.CentreOnHalo(s, cen)
        x = np.asarray(s.data['Coordinates'], float)
        v = np.asarray(s.data['Velocities'], float)
        m = np.asarray(s.data['Masses'], float)
        r = np.sqrt((x ** 2).sum(1))
        inner = r < RIN
        if inner.sum() < 100:
            print(f'  snap {sn}: SKIP (too few central stars)', flush=True)
            del s; gc.collect(); continue
        v = v - np.average(v[inner], axis=0, weights=m[inner])

        Ld = np.cross(x[inner], v[inner] * m[inner, None]).sum(0)

        sid = np.asarray(s.data['ParticleIDs'])
        o = np.argsort(sid); ss = sid[o]
        p = np.searchsorted(ss, GSE_IDS)
        ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == GSE_IDS)
        ig = o[p[ok]]
        if len(ig) < 50:
            print(f'  snap {sn}: only {len(ig)} GS/E stars present, skipping', flush=True)
            del s; gc.collect(); continue
        mg, xg, vg = m[ig], x[ig], v[ig]
        M = mg.sum()
        xcom = (mg[:, None] * xg).sum(0) / M
        vcom = (mg[:, None] * vg).sum(0) / M
        Lorb = M * np.cross(xcom, vcom)
        Ltot = np.cross(xg, vg * mg[:, None]).sum(0)
        rcom = float(np.sqrt((xcom ** 2).sum()) * 1000.)
        # How coherent the progenitor still is: the fraction of its stars within
        # 15 kpc of its own centre of mass.  Once this collapses the "orbit" of
        # the centre of mass stops meaning anything.
        dr = np.sqrt(((xg - xcom) ** 2).sum(1)) * 1000.
        rec['f_bound'].append(float((dr < 15.).mean()))

        rec['snap'].append(sn); rec['a'].append(a)
        rec['time'].append(float(C.a_to_age(a)))
        rec['n_gse'].append(len(ig)); rec['r_com'].append(rcom)
        vec['L_disc'].append(Ld); vec['L_orb'].append(Lorb); vec['L_tot'].append(Ltot)
        print(f"  snap {sn:3d} t={rec['time'][-1]:5.2f}  N_gse={len(ig):6,d}  "
              f"r_com={rcom:6.1f} kpc  f_bound={rec['f_bound'][-1]:.2f}  "
              f"theta(disc,orb)={angle(Ld, Lorb):6.1f}  "
              f"theta(disc,tot)={angle(Ld, Ltot):6.1f}", flush=True)
        del s; gc.collect()

    np.savez(CACHE, **{k: np.array(v) for k, v in rec.items()},
             **{k: np.array(v) for k, v in vec.items()})
    print('\nsaved', CACHE)

d = np.load(CACHE)
t, Ld, Lo, Lt = d['time'], d['L_disc'], d['L_orb'], d['L_tot']
REF = Ld[-1]                                   # the z=0 disc axis
th_orb = np.array([angle(a_, b_) for a_, b_ in zip(Ld, Lo)])
th_tot = np.array([angle(a_, b_) for a_, b_ in zip(Ld, Lt)])
th_disc = np.array([angle(a_, REF) for a_ in Ld])
th_tot_ref = np.array([angle(b_, REF) for b_ in Lt])
INF = int(np.argmin(np.abs(t - T_INFALL)))
L_INF = Lt[INF]                                # the infall orbital axis
th_to_inf = np.array([angle(a_, L_INF) for a_ in Ld])

fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.9))
ax = axes[0]
coh = d['f_bound'] > .5                        # where the orbit is well defined
ax.plot(t[coh], th_orb[coh], 'o-', color='#8E24AA', lw=2.2, ms=4,
        label=r'$\theta$(disc, GS/E orbit)  [coherent progenitor]')
ax.plot(t, th_tot, 's--', color='#5E35B1', lw=1.6, ms=3.5, alpha=.8,
        label=r'$\theta$(disc, GS/E total $L$)')
for y, ls in [(90, ':'), (180, '-')]:
    ax.axhline(y, color='.6', lw=.9, ls=ls)
# On the right-hand edge: at early times the curve itself runs along 150-180 and
# the legend sits in the bottom-left corner.
for y, lab in [(93, 'polar'), (172, 'retrograde'), (4, 'prograde')]:
    ax.text(9.85, y, lab, fontsize=10, color='.4', ha='right')
for x, lab in [(T_PERI, 'plunge'), (T_COAL, 'coalescence')]:
    ax.axvline(x, color='k', lw=1.1, ls='--')
    ax.text(x - .08, 0.5, lab, rotation=90, ha='right', va='center', fontsize=9.5,
            transform=ax.get_xaxis_transform(),
            bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))
ax.set(xlim=(t.min(), 10.), ylim=(0, 185), xlabel='cosmic time [Gyr]',
       ylabel=r'angle to the disc spin axis [deg]', yticks=np.arange(0, 181, 30))
ax.set_title('Orientation of the GS/E merger', fontsize=12)
ax.legend(fontsize=9.5, loc='lower left')

ax = axes[1]
ax.plot(t, th_disc, 'o-', color='#1F6FB2', lw=2.4, ms=4,
        label=r'disc axis vs the $z=0$ disc axis')
ax.plot(t, th_to_inf, 'D-', color='#8E24AA', lw=2.4, ms=4,
        label=r'disc axis vs the GS/E infall axis ($t=%.1f$ Gyr)' % T_INFALL)
ax.plot(t, th_tot_ref, 's--', color='#5E35B1', lw=1.4, ms=3, alpha=.6,
        label=r'GS/E running $L$ vs the $z=0$ disc axis')
for x, lab in [(T_PERI, 'plunge'), (T_COAL, 'coalescence')]:
    ax.axvline(x, color='k', lw=1.1, ls='--')
    ax.text(x - .08, 0.5, lab, rotation=90, ha='right', va='center', fontsize=9.5,
            transform=ax.get_xaxis_transform(),
            bbox=dict(fc='white', ec='none', alpha=.85, pad=1.5))
ax.axhline(90, color='.6', lw=.9, ls=':')
ax.set(xlim=(t.min(), t.max()), ylim=(0, 185), xlabel='cosmic time [Gyr]',
       ylabel=r'angle to the $z=0$ disc axis [deg]', yticks=np.arange(0, 181, 30))
ax.set_title('The disc turns into the merger plane', fontsize=12)
ax.legend(fontsize=9.5, loc='upper right')

fig.suptitle('Au18: the GS/E orbital plane and the disc plane through the merger',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .93])
os.makedirs(C.FIG_DIR, exist_ok=True)
out = C.FIG_DIR + '/au18_gse_orientation.png'
fig.savefig(out, dpi=150)

# ------------------------------------------------------------------- numbers --
def at(tt):
    return int(np.argmin(np.abs(t - tt)))


print('\n   t[Gyr]  r_com  f_bound  th(disc,orb)  th(disc,Ltot)  th(disc,z=0)')
for i in range(len(t)):
    if t[i] > 10 and d['snap'][i] % 5:
        continue
    print(f'  {t[i]:6.2f} {d["r_com"][i]:6.1f} {d["f_bound"][i]:8.2f} '
          f'{th_orb[i]:13.1f} {th_tot[i]:14.1f} {th_disc[i]:13.1f}')
pre = (t > 3.5) & (t < 4.9)
post = (t > 6.5) & (t < 8.5)
print(f'\npre-merger  (3.5-4.9 Gyr): theta(disc, GS/E orbit) = '
      f'{np.nanmean(th_orb[pre & coh]):.1f} deg')
print(f'disc axis:  pre-merger vs z=0 {np.nanmean(th_disc[pre]):.1f} deg, '
      f'post-merger (6.5-8.5) {np.nanmean(th_disc[post]):.1f} deg')
print(f'GS/E infall axis (t={t[INF]:.2f}) vs z=0 disc axis: {angle(L_INF, REF):.1f} deg')
print(f'disc vs the GS/E infall axis: pre-merger {np.nanmean(th_to_inf[pre]):.1f} deg -> '
      f'post-merger {np.nanmean(th_to_inf[post]):.1f} deg -> z=0 {th_to_inf[-1]:.1f} deg')
print(f'GS/E RUNNING L vs z=0 disc axis, post-merger: {np.nanmean(th_tot_ref[post]):.1f} deg '
      f'(a small residual: |L_orb|/|L_tot| = '
      f'{np.nanmean([np.linalg.norm(a_) / np.linalg.norm(b_) for a_, b_ in zip(Lo[post], Lt[post])]):.2f})')
print(f'total disc turn across the merger: '
      f'{angle(Ld[at(4.0)], Ld[at(7.5)]):.1f} deg')
print('\nsaved', out)
