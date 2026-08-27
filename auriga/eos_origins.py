"""The two origins of the Eos-like population in Au18, defined once and shared.

The v_R-v_phi plane at birth (ana_merger_vr_vphi_maps.py) shows that the stars
which are Eos-like at z=0 are bimodal in their BIRTH azimuthal velocity: one lobe
peaks at v_phi ~ +70 km/s, already inside the Eos band when it formed, the other
at ~ +220 km/s, still on the disc ridge.  Smoothing that histogram puts the dip
between them at v_phi,birth = +150 km/s, which is the split used here.

  halo-born Eos   v_phi,birth <  150 km/s   born hot
  disc-born Eos   v_phi,birth >= 150 km/s   born on the disc, heated later

Both are drawn from the in-situ stars formed during the merger window
(t_form = 4.99-6.54 Gyr) that satisfy the observational Eos cut at z=0,
-80 < v_phi < +80 km/s and ecc > 0.6.
"""
import os
import numpy as np
import config_au18 as C

VPHI_SPLIT = 150.
VPHI_EOS, ECC_EOS = 80., 0.6


def load():
    k = np.load(C.OUT_DIR + '/merger_birth_vs_z0_kinematics.npz')
    cat = np.load(C.OUT_DIR + '/z0_insitu_catalog.npz')
    o = np.argsort(cat['ids']); sid = cat['ids'][o]
    p = np.searchsorted(sid, k['ids'])
    ok = (p < len(sid)) & (sid[np.minimum(p, len(sid) - 1)] == k['ids'])
    ix = o[p[ok]]

    d = dict(ids=k['ids'][ok], bvR=k['birth_vR'][ok], bvphi=k['birth_vphi'][ok],
             zvR=k['z0_vR'][ok], zvphi=k['z0_vphi'][ok])

    # Birth radii, from ana_birth_radii.py: each star is measured in the first
    # stored snapshot at or after it formed, i.e. R_birth is approximated by the
    # radius at the nearest snapshot rather than interpolated to the exact
    # formation time.  Snapshot spacing here is ~0.15 Gyr.
    br = np.load(C.OUT_DIR + '/merger_birth_radii.npz')
    ob = np.argsort(br['ids']); bids = br['ids'][ob]
    pb = np.searchsorted(bids, d['ids'])
    okb = (pb < len(bids)) & (bids[np.minimum(pb, len(bids) - 1)] == d['ids'])
    ixb = ob[pb[okb]]
    for key in ('R_birth', 'z_birth', 'r_birth', 'snap_birth'):
        col = np.full(len(d['ids']), np.nan)
        col[okb] = br[key][ixb]
        d[key] = col

    # Guiding-centre radius at birth, if it has been computed (prep_rg_birth.py).
    # R_g is the radius of the circular orbit carrying the same L_z, so unlike
    # R_birth it does not depend on where in its orbit a star was caught.
    rgp = C.OUT_DIR + '/merger_rg_birth.npz'
    if os.path.exists(rgp):
        rg = np.load(rgp)
        og = np.argsort(rg['ids']); gids = rg['ids'][og]
        pg = np.searchsorted(gids, d['ids'])
        okg = (pg < len(gids)) & (gids[np.minimum(pg, len(gids) - 1)] == d['ids'])
        ixg = og[pg[okg]]
        for key in ('Rg_birth', 'Lz_birth', 'retrograde'):
            col = np.full(len(d['ids']), np.nan)
            col[okg] = rg[key][ixg]
            d[key] = col
        d['retrograde'] = d['retrograde'] > 0.5
    for key in ('ecc', 'age', 'tform', 'feh', 'r', 'R', 'z', 'rapo', 'eps',
                'cfe', 'nfe', 'ofe', 'nefe', 'mgfe', 'sife'):
        d[key] = cat[key][ix]
    d['eos'] = (np.abs(d['zvphi']) < VPHI_EOS) & (d['ecc'] > ECC_EOS)
    d['halo_born'] = d['eos'] & (d['bvphi'] < VPHI_SPLIT)
    d['disc_born'] = d['eos'] & (d['bvphi'] >= VPHI_SPLIT)
    d['cat'] = cat
    return d


if __name__ == '__main__':
    d = load()
    np.savez(C.OUT_DIR + '/eos_two_origins.npz',
             halo_born_ids=d['ids'][d['halo_born']],
             disc_born_ids=d['ids'][d['disc_born']],
             all_merger_ids=d['ids'], vphi_split=VPHI_SPLIT)
    for lab, m in [('all merger-born', np.ones(len(d['ids']), bool)),
                   ('Eos-like total ', d['eos']),
                   ('  halo-born Eos', d['halo_born']),
                   ('  disc-born Eos', d['disc_born'])]:
        print(f'{lab}: N={m.sum():>7,}  '
              f'v_phi birth {np.median(d["bvphi"][m]):+6.1f}  z=0 {np.median(d["zvphi"][m]):+6.1f}  '
              f'age {np.median(d["age"][m]):5.2f}  [Fe/H] {np.median(d["feh"][m]):+.2f}')
    print('\nsaved', C.OUT_DIR + '/eos_two_origins.npz')
