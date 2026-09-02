"""The two Eos origins, for an ARBITRARY birth-time window.

eos_origins.py is fixed to t_form = 4.99-6.54 Gyr because it reads
merger_birth_vs_z0_kinematics.npz, which only exists for snapshots 73-82.  This
module rebuilds the same quantities from birth_orbits.npz, which carries birth
and z=0 kinematics for every in-situ star at every snapshot, so the window can be
moved.  Everything else -- the Eos cut, the birth-v_phi channel split, the z=0
aperture -- is identical, so a figure made through here differs from its
eos_origins.py counterpart only by the window.

  WINDOW      birth-time interval, default 4.7-5.7 Gyr
  APERTURE    3 < r < 30 kpc at z=0, the same aperture ana_merger_epoch_z0.py
              applied when it built the original parent sample
  Eos cut     |v_phi| < 80 km/s and ecc > 0.6 at z=0
  channels    halo-born  v_phi,birth <  150 km/s
              disc-born  v_phi,birth >= 150 km/s

One deliberate difference: R_g at birth comes from birth_orbits_zmax.npz, i.e.
AGAMA's Rcirc(L=|J_phi|) in the star's own birth potential, rather than from
merger_rg_birth.npz.  It is the better estimate and it exists for every star.
"""
import numpy as np
import config_au18 as C

WINDOW = (4.7, 5.7)
APERTURE = (3., 30.)
VPHI_SPLIT = 150.
VPHI_EOS, ECC_EOS = 80., 0.6


def load(window=None, aperture=APERTURE):
    lo, hi = window or WINDOW
    b = np.load(C.OUT_DIR + '/birth_orbits.npz')
    zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
    ac = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
    cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
    assert np.array_equal(b['ids'], zx['ids']) and np.array_equal(b['ids'], ac['ids'])

    o = np.argsort(cat['ids']); sid = cat['ids'][o]
    p = np.searchsorted(sid, b['ids'])
    ok = (p < len(sid)) & (sid[np.minimum(p, len(sid) - 1)] == b['ids'])
    ix = o[p[ok]]

    sel = ok & (b['tform'] >= lo) & (b['tform'] <= hi) & np.isfinite(b['vphi_birth'])
    ixs = o[p[sel]]
    if aperture is not None:
        r0 = cat['r'][ixs]
        keep = (r0 >= aperture[0]) & (r0 < aperture[1])
        sel = np.flatnonzero(sel)[keep]
        ixs = ixs[keep]
    else:
        sel = np.flatnonzero(sel)

    d = dict(ids=b['ids'][sel], tform=b['tform'][sel],
             bvR=b['vR_birth'][sel], bvphi=b['vphi_birth'][sel],
             zvR=b['vR_z0'][sel], zvphi=b['vphi_z0'][sel],
             R_birth=b['R_birth'][sel], z_birth=b['z_birth'][sel],
             r_birth=b['r_birth'][sel], snap_birth=b['snap_birth'][sel],
             Rg_birth=zx['Rg_birth'][sel], Lz_birth=ac['Lz_birth'][sel])
    d['retrograde'] = d['Lz_birth'] < 0
    for key in ('ecc', 'age', 'feh', 'r', 'R', 'z', 'rapo', 'eps', 'mass',
                'cfe', 'nfe', 'ofe', 'nefe', 'mgfe', 'sife'):
        d[key] = cat[key][ixs]
    d['eos'] = (np.abs(d['zvphi']) < VPHI_EOS) & (d['ecc'] > ECC_EOS)
    d['halo_born'] = d['eos'] & (d['bvphi'] < VPHI_SPLIT)
    d['disc_born'] = d['eos'] & (d['bvphi'] >= VPHI_SPLIT)
    d['cat'] = cat
    d['window'] = (lo, hi)
    return d


if __name__ == '__main__':
    import sys
    w = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else WINDOW
    for win in ({(4.99, 6.54), w} if w != (4.99, 6.54) else {w}):
        d = load(win)
        print(f'\nwindow {win[0]}-{win[1]} Gyr, aperture {APERTURE[0]}-{APERTURE[1]} kpc')
        for lab, m in [('all in window  ', np.ones(len(d['ids']), bool)),
                       ('Eos-like       ', d['eos']),
                       ('  halo-born Eos', d['halo_born']),
                       ('  disc-born Eos', d['disc_born'])]:
            print(f'  {lab}: N={m.sum():>7,}  vphi_birth {np.median(d["bvphi"][m]):+6.1f}  '
                  f'z0 {np.median(d["zvphi"][m]):+6.1f}  age {np.median(d["age"][m]):5.2f}  '
                  f'[Fe/H] {np.median(d["feh"][m]):+.2f}')
