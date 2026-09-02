"""z=0 kinematic + chemical catalogue of every in-situ star in Au18.

Existing scripts select Eos channels by *birth* properties (A/B/C).  This one
does the complementary thing: it reproduces the **observational** selection in
the simulation, so the sim and the data are cut the same way.  For that we need,
for every in-situ star at z=0, the quantities the APOGEE/LAMOST analysis uses --
v_phi (~ v_tan), eccentricity, r_apo, L_z -- alongside age and chemistry.

Eccentricity and r_apo come from the spherically-averaged snapshot potential:
for each star, E = 0.5 v^2 + Phi(r) and L = |r x v| define an effective potential
Phi_eff(r) = Phi(r) + L^2 / 2r^2, whose two roots around the present radius are
r_peri and r_apo.  Au18 has a disc, so this is an approximation, but it is the
same approximation the observational orbit integration effectively makes for
halo-orbit stars, and it is what makes ecc/r_apo comparable at all.

The GS/E debris is measured the same way and stored alongside, so the accreted
reference population sits on exactly the same eccentricity/apocentre scale.

Output: out/z0_insitu_catalog.npz
"""
import gc, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.OUT_DIR, exist_ok=True)
ELS = ['C', 'N', 'O', 'Ne', 'Mg', 'Si']

print('loading snapshot 127 ...', flush=True)
s = snap_mod.load_snapshot(127, 4, snappath=C.SIM_DIR,
    loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses', 'Potential',
              'GFM_StellarFormationTime', 'GFM_Metals'])
real = s.data['GFM_StellarFormationTime'] > 0

# matched_z0['ii'] indexes the *untrimmed* snapshot arrays; remap onto the trimmed ones.
old_to_real = np.full(len(real), -1, np.int64)
old_to_real[np.flatnonzero(real)] = np.arange(int(real.sum()))
insitu_idx = old_to_real[np.load(C.OUT_DIR + '/matched_z0.npz')['ii']]
insitu_idx = insitu_idx[insitu_idx >= 0]
for k in list(s.data):
    s.data[k] = s.data[k][real]
print(f'stars (winds removed): {int(real.sum()):,};  in-situ: {len(insitu_idx):,}', flush=True)

sf = sub_mod.subfind(127, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
util.CentreOnHalo(s, cen)
rr = np.sqrt((s.data['Coordinates'] ** 2).sum(1))
inner = rr < .01
bulk = np.average(s.data['Velocities'][inner], axis=0, weights=s.data['Masses'][inner])
s.data['Velocities'] -= bulk
util.align_galaxy(s, radialcut=.01)

# align_galaxy puts the disc angular momentum on component 0 -> disc plane = (1,2).
x = s.data['Coordinates'] * 1000.
v = s.data['Velocities']
r_all = np.sqrt((x * x).sum(1))
R_all = np.hypot(x[:, 1], x[:, 2])
jz_all = x[:, 1] * v[:, 2] - x[:, 2] * v[:, 1]
disc_ref = (R_all > 3) & (R_all < 12) & (np.abs(x[:, 0]) < 2)
sign = -1. if np.median(jz_all[disc_ref]) < 0 else 1.
jz_all *= sign

phi_all = s.data['Potential'].astype(np.float64)
E_all = .5 * (v * v).sum(1) + phi_all

# Circularity on the same 240-bin energy ladder used by the channel scripts, so
# eps here is on the same scale as eps_z0 in eos_two_channels / premerger_splash.
valid = np.isfinite(E_all) & np.isfinite(jz_all) & (r_all < 50)
edges = np.quantile(E_all[valid], np.linspace(0, 1, 241))
ib = np.clip(np.searchsorted(edges, E_all, 'right') - 1, 0, 239)
jc = np.full(240, np.nan)
for b in range(240):
    q = valid & (ib == b) & (jz_all > 0)
    if q.sum() > 30:
        jc[b] = np.percentile(jz_all[q], 95)
ok = np.isfinite(jc)
jc = np.interp(np.arange(240), np.flatnonzero(ok), jc[ok])
eps_all = jz_all / jc[ib]

# ------------------------------------------------------------------ profile --
# Spherically-averaged potential from the star particles' own Potential field
# (which is the total potential of all species at their positions).
rc, phi_prof, k_out = OT.potential_profile(r_all, phi_all)
print(f'potential profile: {len(rc)} bins, r = {rc[0]:.2f}-{rc[-1]:.1f} kpc', flush=True)


# ---------------------------------------------------------------- in-situ ----
i = insitu_idx
xi, vi = x[i], v[i]
R = R_all[i]
z = xi[:, 0]
r = r_all[i]
vR = (xi[:, 1] * vi[:, 1] + xi[:, 2] * vi[:, 2]) / np.where(R > .1, R, 1.)
vphi = sign * (xi[:, 1] * vi[:, 2] - xi[:, 2] * vi[:, 1]) / np.where(R > .1, R, 1.)
vz = vi[:, 0]
Lz = jz_all[i]
Ltot = np.sqrt((np.cross(xi, vi) ** 2).sum(1))
E = E_all[i]
# Turning points are solved in the spherical potential, so the energy fed to the
# solver has to be the spherical one too (E above is kept for E-Lz diagrams).
E_sph = OT.spherical_energy((vi * vi).sum(1), r, rc, phi_prof, k_out)

print('solving apo/peri ...', flush=True)
rapo, rperi = OT.apo_peri(E_sph, Ltot, r, rc, phi_prof, k_out)
ecc = OT.eccentricity(rapo, rperi)

metals = s.data['GFM_Metals'][i]
feh = C.bracket_abundance(metals, 'Fe', 'H')
ratios = {e: C.bracket_abundance(metals, e, 'Fe') for e in ELS}
aform = s.data['GFM_StellarFormationTime'][i]
tform = C.a_to_age(aform)
age = C.T0_GYR - tform
ids = s.data['ParticleIDs'][i]
mass = s.data['Masses'][i] * C.MASS_TO_MSUN

out = dict(ids=ids, r=r, R=R, z=z, vR=vR, vphi=vphi, vz=vz, Lz=Lz, E=E, E_sph=E_sph,
           eps=eps_all[i], rapo=rapo, rperi=rperi, ecc=ecc,
           tform=tform, age=age, feh=feh, mass=mass,
           **{e.lower() + 'fe': ratios[e] for e in ELS})
# --------------------------------------------------------------- GS/E debris --
# Same measurement for the clean debris, so Eos analogues can be compared with
# the accreted population on an identical ecc / r_apo scale.
sid = s.data['ParticleIDs']
gse_ids = np.load(C.OUT_DIR + '/gse_clean_ids.npy')
o = np.argsort(sid); ss = sid[o]; pp = np.searchsorted(ss, gse_ids)
mm = (pp < len(ss)) & (ss[np.minimum(pp, len(ss) - 1)] == gse_ids)
gi = o[pp[mm]]
xg, vg = x[gi], v[gi]
Rg = R_all[gi]
gsafe = np.where(Rg > .1, Rg, 1.)
g_out = dict(
    gse_ids=sid[gi], gse_r=r_all[gi], gse_R=Rg, gse_z=xg[:, 0],
    gse_vphi=sign * (xg[:, 1] * vg[:, 2] - xg[:, 2] * vg[:, 1]) / gsafe,
    gse_vR=(xg[:, 1] * vg[:, 1] + xg[:, 2] * vg[:, 2]) / gsafe,
    gse_vz=vg[:, 0], gse_Lz=jz_all[gi], gse_eps=eps_all[gi], gse_E=E_all[gi],
    gse_feh=C.bracket_abundance(s.data['GFM_Metals'][gi], 'Fe', 'H'),
    gse_age=C.T0_GYR - C.a_to_age(s.data['GFM_StellarFormationTime'][gi]),
)
E_sph_g = OT.spherical_energy((vg * vg).sum(1), r_all[gi], rc, phi_prof, k_out)
Lg = np.sqrt((np.cross(xg, vg) ** 2).sum(1))
g_out['gse_rapo'], g_out['gse_rperi'] = OT.apo_peri(E_sph_g, Lg, r_all[gi], rc, phi_prof, k_out)
g_out['gse_ecc'] = OT.eccentricity(g_out['gse_rapo'], g_out['gse_rperi'])
out.update(g_out)
out['phi_prof_r'], out['phi_prof'], out['phi_k_out'] = rc, phi_prof, k_out
np.savez(C.OUT_DIR + '/z0_insitu_catalog.npz', **out)
print(f"GS/E debris: N={len(gi):,}  ecc med={np.nanmedian(g_out['gse_ecc']):.2f}  "
      f"r_apo med={np.nanmedian(g_out['gse_rapo']):.1f} kpc")

fin = np.isfinite(ecc)
solar = (R > 5) & (R < 11) & (np.abs(z) < 3)
print(f'\nsaved {C.OUT_DIR}/z0_insitu_catalog.npz  (N={len(ids):,})')
print(f'ecc finite: {fin.sum():,} ({100*fin.mean():.1f}%)')
print(f'solar annulus (5<R<11, |z|<3): N={solar.sum():,}  <vphi>={vphi[solar].mean():.1f} km/s')
print(f'median [Fe/H]={np.nanmedian(feh):+.2f}, age={np.nanmedian(age):.2f} Gyr, ecc={np.nanmedian(ecc[fin]):.2f}')
