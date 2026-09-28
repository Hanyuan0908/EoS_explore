"""Compute & cache orbital kinematics (ecc, r_apo, r_peri, z_max, Lz) for the Li+2024
Gaia-XP sample using AGAMA + the McMillan (2017) Milky Way potential.

Quality sample: finite [M/H],[a/M]; e_moh_xp<0.1; e_aom_xp<0.05; parallax/parallax_error>5;
finite 6D (galactocentric x,y,z [pc], vx,vy,vz [km/s] in the file).

Lz sign: this frame puts the Sun at x~-8.2 kpc with median vy~+205, so the prograde disc
has PHYSICAL Lz=x*vy-y*vx<0. We flip (lz=-(x*vy-y*vx)) so prograde disc is POSITIVE and
"Lz<0" selects retrograde, matching the APOGEE convention.

ROBUST / RESUMABLE: rapo/rperi/zmax are written to disk memmaps and a done-counter after
every chunk, so if the job is killed (memory pressure etc.) just relaunch and it resumes
from where it stopped. agama.orbit parallelises internally; run single-process. Progress
output is silenced. When all stars are done it assembles the final npz + fits.

Integration time=4 (~3.9 Gyr), trajsize=500 (validated vs time10/traj800 to <0.01).
Outputs: data_repro/xp_kinematics.npz and .fits (source_id, ra, dec, moh, aom, emoh,
eaom, lz, ecc, rapo, rperi, zmax). Chunk memmaps live in data_repro/_xpkin_ckpt/.
"""
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import sys
import contextlib
import numpy as np
import h5py
import agama
from astropy.table import Table

XP = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/Gaia/Li_2024_XP.hdf5'
POT = '/Users/hanyuan/Desktop/PhD_projects/Agama-master/data/McMillan17.ini'
REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
CK = REPO + '/data_repro/_xpkin_ckpt'
OUT_NPZ = REPO + '/data_repro/xp_kinematics.npz'
OUT_FITS = REPO + '/data_repro/xp_kinematics.fits'
CHUNK = 10000
TINT = 4.0
TRAJ = 500

agama.setUnits(mass=1, length=1, velocity=1)
pot = agama.Potential(POT)
os.makedirs(CK, exist_ok=True)

# --- load quality sample (deterministic order) ---
f = h5py.File(XP, 'r'); C = f['table/columns']
col = lambda n: C[n + '/data'][:]
moh = col('moh_xp'); aom = col('aom_xp'); emoh = col('e_moh_xp'); eaom = col('e_aom_xp')
plx = col('parallax'); eplx = col('parallax_error'); rv = col('radial_velocity')
x = col('x') / 1e3; y = col('y') / 1e3; z = col('z') / 1e3
vx = col('vx'); vy = col('vy'); vz = col('vz')
sid = col('source_id'); ra = col('ra'); dec = col('dec')
good = (np.isfinite(moh) & np.isfinite(aom) & (emoh < 0.1) & (eaom < 0.05)
        & np.isfinite(rv) & (eplx > 0) & (plx / eplx > 5)
        & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz))
idx = np.where(good)[0]; n = len(idx)
print('quality sample:', n, flush=True)
xk, yk, zk = x[idx], y[idx], z[idx]; vxk, vyk, vzk = vx[idx], vy[idx], vz[idx]
lz = -(xk * vyk - yk * vxk)
ic = np.column_stack([xk, yk, zk, vxk, vyk, vzk])

# --- open/create checkpoint memmaps + done counter ---
def mm(name):
    p = f'{CK}/{name}.npy'
    if os.path.exists(p):
        a = np.lib.format.open_memmap(p, mode='r+')
        if a.shape[0] == n:
            return a
    return np.lib.format.open_memmap(p, mode='w+', dtype=np.float32, shape=(n,))
rapo = mm('rapo'); rperi = mm('rperi'); zmax = mm('zmax')
donef = f'{CK}/done.txt'
done = 0
if os.path.exists(donef):
    try: done = int(open(donef).read().strip())
    except Exception: done = 0
    done = max(0, min(done, n))
print(f'resuming from {done}/{n}', flush=True)

# --- integrate, checkpointing after each chunk ---
for s in range(done, n, CHUNK):
    e = min(s + CHUNK, n)
    with open(os.devnull, 'w') as dn, contextlib.redirect_stderr(dn):
        orb = agama.orbit(potential=pot, ic=ic[s:e], time=TINT, trajsize=TRAJ)
    T = np.stack([o[1] for o in orb])                      # (m, TRAJ, 6)
    r = np.sqrt(T[:, :, 0]**2 + T[:, :, 1]**2 + T[:, :, 2]**2)
    rapo[s:e] = r.max(axis=1); rperi[s:e] = r.min(axis=1)
    zmax[s:e] = np.abs(T[:, :, 2]).max(axis=1)
    rapo.flush(); rperi.flush(); zmax.flush()
    with open(donef, 'w') as fh:
        fh.write(str(e))
    if (s // CHUNK) % 20 == 0:
        print(f'  {e}/{n}', flush=True)

# --- assemble final outputs ---
rapo = np.asarray(rapo); rperi = np.asarray(rperi); zmax = np.asarray(zmax)
with np.errstate(invalid='ignore'):
    ecc = (rapo - rperi) / (rapo + rperi)
cols = {'source_id': sid[idx].astype(np.int64), 'ra': ra[idx], 'dec': dec[idx],
        'moh': moh[idx].astype(np.float32), 'aom': aom[idx].astype(np.float32),
        'emoh': emoh[idx].astype(np.float32), 'eaom': eaom[idx].astype(np.float32),
        'lz': lz.astype(np.float32), 'ecc': ecc.astype(np.float32),
        'rapo': rapo.astype(np.float32), 'rperi': rperi.astype(np.float32),
        'zmax': zmax.astype(np.float32)}
np.savez(OUT_NPZ, **cols)
Table(cols).write(OUT_FITS, overwrite=True)
print('wrote', OUT_NPZ, 'and', OUT_FITS,
      '| halo (e>0.7|lz<0):', int(np.sum((ecc > 0.7) | (lz < 0))), flush=True)
