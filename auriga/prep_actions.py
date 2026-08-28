"""Actions for the Au18 z=0 stars, via AGAMA.

Builds an axisymmetric potential from the z=0 particles and runs AGAMA's
ActionFinder on every in-situ star plus the GS/E debris, so the two Eos
populations can be compared in J_R and J_R/|L_z| as well as v_phi and
eccentricity.

The potential is two components, which is the usual decomposition: a Multipole
fitted to the dark matter (which is round enough for a spherical-harmonic
expansion) and a CylSpline fitted to the baryons (which are not).  Both are
forced axisymmetric, since the action finder assumes it.

Frame: the same one used everywhere else in this project.  align_galaxy puts the
disc angular momentum on component 0, so the disc plane is components (1, 2);
here that is remapped to AGAMA's convention with z as the symmetry axis.

Writes out/z0_actions.npz.
"""
import gc, os
import numpy as np
import agama
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

agama.setUnits(mass=1, length=1, velocity=1)          # Msun, kpc, km/s
os.makedirs(C.OUT_DIR, exist_ok=True)
RMAX_DM, RMAX_BAR = 400., 100.
NSUB_DM, NSUB_BAR = 2_000_000, 2_000_000

sf = sub_mod.subfind(127, directory=C.SIM_DIR, loadlist=['GroupFirstSub', 'SubhaloPos'])
cen = sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]

# Rotation matrix from the stars, exactly as align_galaxy derives it.
st = snap_mod.load_snapshot(127, 4, snappath=C.SIM_DIR,
    loadlist=['Coordinates', 'Velocities', 'Masses', 'GFM_StellarFormationTime'])
real = st.data['GFM_StellarFormationTime'] > 0
for k in list(st.data):
    st.data[k] = st.data[k][real]
util.CentreOnHalo(st, cen)
rr = np.sqrt((st.data['Coordinates'] ** 2).sum(1))
idx = np.flatnonzero(rr < .01)
st.data['Velocities'] -= np.average(st.data['Velocities'][idx], axis=0,
                                    weights=st.data['Masses'][idx])
L = np.cross(st.data['Coordinates'][idx, :],
             st.data['Velocities'][idx, :] * st.data['Masses'][idx, None]).sum(axis=0)
xdir, ydir, zdir = util.get_principal_axis(st, idx, L=L / np.sqrt((L ** 2).sum()))
MAT = np.array([xdir, ydir, zdir])
del st; gc.collect()


def component(ptypes, rmax, nsub):
    """Centred, disc-aligned positions and masses, remapped to z as symmetry axis."""
    P, M = [], []
    for pt in ptypes:
        try:
            p = snap_mod.load_snapshot(127, pt, snappath=C.SIM_DIR,
                                       loadlist=['Coordinates', 'Masses'])
        except Exception:
            continue
        if 'Masses' not in p.data or not len(np.atleast_1d(p.data['Masses'])):
            continue
        xyz = np.dot(p.data['Coordinates'] - cen, MAT.T) * 1000.
        m = np.asarray(p.data['Masses'], float) * C.MASS_TO_MSUN
        keep = np.sqrt((xyz ** 2).sum(1)) < rmax
        P.append(xyz[keep]); M.append(m[keep])
        del p, xyz, m; gc.collect()
    P = np.concatenate(P); M = np.concatenate(M)
    if len(P) > nsub:                                  # subsample, conserving total mass
        sel = np.random.default_rng(3).choice(len(P), nsub, replace=False)
        P, M = P[sel], M[sel] * (len(P) / nsub)
    # align_galaxy puts the symmetry axis on component 0; AGAMA wants it on 2
    return np.column_stack([P[:, 1], P[:, 2], P[:, 0]]), M


print('building the dark matter Multipole ...', flush=True)
Pdm, Mdm = component((1, 2, 3), RMAX_DM, NSUB_DM)
pot_dm = agama.Potential(type='Multipole', particles=(Pdm, Mdm),
                         symmetry='axisymmetric', gridsizeR=25, lmax=6)
print(f'  {len(Pdm):,} particles, M = {Mdm.sum():.3e} Msol', flush=True)
del Pdm, Mdm; gc.collect()

print('building the baryonic CylSpline ...', flush=True)
Pb, Mb = component((0, 4), RMAX_BAR, NSUB_BAR)
pot_bar = agama.Potential(type='CylSpline', particles=(Pb, Mb), symmetry='axisymmetric',
                          gridsizer=25, gridsizez=25, mmax=0, Rmin=0.2, Rmax=80, Zmin=0.05, Zmax=40)
print(f'  {len(Pb):,} particles, M = {Mb.sum():.3e} Msol', flush=True)
del Pb, Mb; gc.collect()

pot = agama.Potential(pot_dm, pot_bar)
for R in (5., 8., 15., 30.):
    print(f'  v_c({R:4.0f} kpc) = {(-R * pot.force(R, 0, 0)[0]) ** .5:6.1f} km/s')

af = agama.ActionFinder(pot)
cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')


def actions_for(R, z, vR, vphi, vz):
    """Cylindrical -> cartesian at phi=0; valid because the potential is axisymmetric."""
    xv = np.column_stack([R, np.zeros_like(R), z, vR, vphi, vz])
    J = af(xv)
    return J[:, 0], J[:, 1], J[:, 2]          # Jr, Jz, Jphi

out = {}
Jr, Jz, Jphi = actions_for(cat['R'], cat['z'], cat['vR'], cat['vphi'], cat['vz'])
out.update(ids=cat['ids'], Jr=Jr, Jz=Jz, Jphi=Jphi)
print(f'in-situ: {len(Jr):,} stars, median J_r = {np.nanmedian(Jr):.1f} kpc km/s')

gJr, gJz, gJphi = actions_for(cat['gse_R'], cat['gse_z'], cat['gse_vR'],
                             cat['gse_vphi'], cat['gse_vz'])
out.update(gse_ids=cat['gse_ids'], gse_Jr=gJr, gse_Jz=gJz, gse_Jphi=gJphi)
print(f'GS/E   : {len(gJr):,} stars, median J_r = {np.nanmedian(gJr):.1f} kpc km/s')

np.savez(C.OUT_DIR + '/z0_actions.npz', **out)
print('saved', C.OUT_DIR + '/z0_actions.npz')
