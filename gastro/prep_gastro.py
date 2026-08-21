"""Measure each gastro model once and cache the per-star arrays.

Loading a snapshot, aligning it and solving the turning points takes minutes, so
the measurement is separated from the plotting: this writes out/<model>_stars.npz
and the figure scripts read that.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import orbit_tools as OT
import gastro_config as G

os.makedirs(G.OUT_DIR, exist_ok=True)
KEEP = ['x', 'y', 'z', 'r', 'R', 'vR', 'vphi', 'vz', 'tform', 'age', 'mass',
        'mh', 'Lz', 'Ltot', 'E', 'rapo', 'rperi', 'ecc']


def measure(model):
    f = G.load(model)
    d = G.star_frame(f)
    rc, prof, k_out = OT.potential_profile(d['r'], d['phi'], rmin=0.05, rmax=400.)
    v2 = d['vR'] ** 2 + d['vphi'] ** 2 + d['vz'] ** 2
    E_sph = OT.spherical_energy(v2, d['r'], rc, prof, k_out)
    d['rapo'], d['rperi'] = OT.apo_peri(E_sph, d['Ltot'], d['r'], rc, prof, k_out, rmax=400.)
    d['ecc'] = OT.eccentricity(d['rapo'], d['rperi'])
    out = {k: np.asarray(d[k], np.float32) for k in KEEP}
    out['t_now'] = np.float32(d['t_now'])
    out['phi_prof_r'], out['phi_prof'] = rc, prof
    return out


for model in ('clumpy', 'notclumpy'):
    if not os.path.exists(G.MODELS[model]):
        print(f'{model}: snapshot missing, skipped'); continue
    dest = f'{G.OUT_DIR}/{model}_stars.npz'
    try:
        out = measure(model)
    except Exception as exc:
        print(f'{model}: UNREADABLE ({type(exc).__name__}: {exc}) -- skipped'); continue
    np.savez(dest, **out)
    ecc, R = out['ecc'], out['R']
    ins = (R > 4) & (R < 30) & np.isfinite(ecc)
    print(f'{model}: N*={len(R):,}  4<R<30 kpc: {ins.sum():,}  '
          f'median ecc={np.nanmedian(ecc[ins]):.2f}  t_now={float(out["t_now"]):.2f} Gyr')
    print('  saved', dest)
