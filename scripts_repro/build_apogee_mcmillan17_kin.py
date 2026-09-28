"""From-scratch APOGEE kinematics with AGAMA + McMillan (2017), replacing the
AstroNN-VAC (galpy/MWPotential2014) orbital parameters used until now.

Inputs (ALL from APOGEE_DR17_all.fits = APOGEE DR17 x Gaia DR3 TOPCAT crossmatch):
  position   RAdeg, DEdeg                      (Gaia DR3 ICRS)
  distance   GAIAEDR3_R_MED_PHOTOGEO           (Bailer-Jones+2021 photogeometric)
  proper mot pmRA, pmDE                        (Gaia DR3)
  RV         RV                                (Gaia DR3)
  abundances FE_H, MG_FE, AL_FE, ... + *_ERR   (ASPCAP)
Solar frame: R0=8.21 kpc, z_sun=20.8 pc, v_sun=(11.1, 245.3, 7.25) km/s (McMillan17
+ Schoenrich10 peculiar). Lz sign flipped so the prograde disc is POSITIVE
(Lz = -(x*vy - y*vx)), matching the XP cache and the paper convention (Lz<0 = retrograde).

Orbits: AGAMA McMillan17, time=4 (~3.9 Gyr), trajsize=500 -> rap, rperi, zmax.
Actions: AGAMA ActionFinder (Staeckel) -> Jr. Energy from the potential.
GC-membership / satellite masks are copied from the lite cache by APOGEE_ID
(they are spatial, potential-independent).

Output: data_repro/apogee_mcmillan17_kin.fits
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import numpy as np
import agama
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactocentric

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
ALL = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_DR17_all.fits'
POT = '/Users/hanyuan/Desktop/PhD_projects/Agama-master/data/McMillan17.ini'
OUT = REPO + '/data_repro/apogee_mcmillan17_kin.fits'
CHUNK = 20000; TINT = 4.0; TRAJ = 500


def norm(a):
    return np.char.strip(np.array([(s.decode() if isinstance(s, bytes) else str(s)) for s in np.asarray(a)]))


agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(POT)
f = fits.open(ALL)[1].data
apid = norm(f['APOGEE_ID'])
ra = np.asarray(f['RAdeg'], float); de = np.asarray(f['DEdeg'], float)
rmed = np.asarray(f['GAIAEDR3_R_MED_PHOTOGEO'], float)
rlo = np.asarray(f['GAIAEDR3_R_LO_PHOTOGEO'], float); rhi = np.asarray(f['GAIAEDR3_R_HI_PHOTOGEO'], float)
dist = rmed / 1e3
pmra = np.asarray(f['pmRA'], float); pmde = np.asarray(f['pmDE'], float); rv = np.asarray(f['RV'], float)
ruwe = np.asarray(f['RUWE'], float)
with np.errstate(invalid='ignore'):
    dfrac = (rhi - rlo) / (2 * rmed)

sixd = (np.isfinite(ra) & np.isfinite(de) & np.isfinite(dist) & (dist > 0)
        & np.isfinite(pmra) & np.isfinite(pmde) & np.isfinite(rv))
idx = np.where(sixd)[0]; n = len(idx)
print('full DR3 6D:', n, flush=True)

gcf = Galactocentric(galcen_distance=8.21 * u.kpc, z_sun=20.8 * u.pc,
                     galcen_v_sun=(11.1, 245.3, 7.25) * u.km / u.s)
sc = SkyCoord(ra=ra[idx] * u.deg, dec=de[idx] * u.deg, distance=dist[idx] * u.kpc,
              pm_ra_cosdec=pmra[idx] * u.mas / u.yr, pm_dec=pmde[idx] * u.mas / u.yr,
              radial_velocity=rv[idx] * u.km / u.s).transform_to(gcf)
x = sc.x.to(u.kpc).value; y = sc.y.to(u.kpc).value; z = sc.z.to(u.kpc).value
vx = sc.v_x.to(u.km / u.s).value; vy = sc.v_y.to(u.km / u.s).value; vz = sc.v_z.to(u.km / u.s).value
posvel = np.column_stack([x, y, z, vx, vy, vz])
R = np.hypot(x, y)
lz = -(x * vy - y * vx)                 # prograde disc positive
galvt = lz / R                          # V_phi (prograde positive)

# actions (Jr) + energy
af = agama.ActionFinder(pot)
acts = af(posvel)                       # columns: Jr, Jz, Jphi
jr = acts[:, 0]; jz = acts[:, 1]
energy = pot.potential(posvel[:, :3]) + 0.5 * (vx**2 + vy**2 + vz**2)

# orbits -> rap, rperi, zmax
rap = np.empty(n); rperi = np.empty(n); zmax = np.empty(n)
for s in range(0, n, CHUNK):
    e = min(s + CHUNK, n)
    orb = agama.orbit(potential=pot, ic=posvel[s:e], time=TINT, trajsize=TRAJ)
    T = np.stack([o[1] for o in orb])
    r = np.sqrt(T[:, :, 0]**2 + T[:, :, 1]**2 + T[:, :, 2]**2)
    rap[s:e] = r.max(axis=1); rperi[s:e] = r.min(axis=1); zmax[s:e] = np.abs(T[:, :, 2]).max(axis=1)
    if (s // CHUNK) % 5 == 0:
        print(f'  orbits {e}/{n}', flush=True)
with np.errstate(invalid='ignore'):
    ecc = (rap - rperi) / (rap + rperi)

# GC / satellite cleaning flags from the lite cache (spatial, potential-independent)
lite = Table.read(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
lap = np.char.strip(np.asarray(lite['apogee_id']).astype(str))
lgc = {a: g for a, g in zip(lap, np.asarray(lite['gc_member'], bool))}
lsat = {a: s for a, s in zip(lap, np.asarray(lite['satellite_out'], bool))}
aid = apid[idx]
gc_member = np.array([lgc.get(a, False) for a in aid], bool)
sat_out = np.array([lsat.get(a, True) for a in aid], bool)

col = lambda name: np.asarray(f[name], float)[idx]
out = {'apogee_id': aid, 'source_id': np.asarray(f['GAIAEDR3_SOURCE_ID'], np.int64)[idx],
       'programname': norm(f['PROGRAMNAME'])[idx], 'logg': col('LOGG'),
       'dist': dist[idx], 'dist_frac_err': dfrac[idx], 'ruwe': ruwe[idx],
       'galr': R, 'galz': z, 'galvr': (x * vx + y * vy) / R, 'galvt': galvt,
       'galvz': vz, 'lz': lz, 'energy': energy, 'jr': jr, 'jz': jz,
       'rap': rap, 'rperi': rperi, 'zmax': zmax, 'ecc': ecc,
       'gc_member': gc_member, 'satellite_out': sat_out}
for c in ['FE_H', 'MG_FE', 'AL_FE', 'C_FE', 'N_FE', 'O_FE', 'SI_FE', 'NI_FE', 'CR_FE', 'MN_FE', 'TI_FE', 'CI_FE']:
    out[c.lower()] = col(c)
    if c + '_ERR' in f.columns.names:
        out[c.lower() + '_err'] = col(c + '_ERR')
Table(out).write(OUT, overwrite=True)
print('wrote', OUT, '| N =', n, flush=True)
