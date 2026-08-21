"""Single definition of the Au18 Eos channels, shared by every downstream script.

Keeping this in one place stops the A/B cuts drifting between analyses.

  A heated disc : eps_birth > 0.7 -> eps_z0 < 0.3, and |z_birth| < 1 kpc
  B born radial : eps_birth < 0.3 -> eps_z0 < 0.3, and |z_birth| > 3 kpc

The |z_birth| conditions are the cleaning step added after the birth-height
analysis; pass zcut=False to recover the earlier kinematics-only channels.
No z=0 radial cut is applied: the channels differ strongly in r_z0, so cutting on
it biases the comparison (see ana_channel_radial_gradient).
"""
import numpy as np
import config_au18 as C

ELS = ['C', 'N', 'O', 'Ne', 'Mg', 'Si']
Z_A_MAX, Z_B_MIN = 1.0, 3.0


def load(zcut=True):
    """Birth and z=0 properties of the merger-born sample, plus the A/B masks."""
    ch = np.load(C.OUT_DIR + '/eos_two_channels.npz')
    br = np.load(C.OUT_DIR + '/merger_birth_radii.npz')
    o = np.argsort(br['ids']); bids = br['ids'][o]
    p = np.searchsorted(bids, ch['ids'])
    ok = (p < len(bids)) & (bids[np.minimum(p, len(bids) - 1)] == ch['ids'])
    ix = o[p[ok]]

    d = dict(ids=ch['ids'][ok], eps_birth=ch['eps_birth'][ok], eps_z0=ch['eps_z0'][ok],
             r_z0=ch['r_z0'][ok], feh=ch['feh'][ok],
             R_birth=br['R_birth'][ix], z_birth=br['z_birth'][ix], tform=br['tform'][ix])
    d['ratios'] = {e: ch[e.lower() + 'fe'][ok] for e in ELS}

    base = (np.isfinite(d['eps_birth']) & np.isfinite(d['eps_z0'])
            & np.isfinite(d['z_birth']) & np.isfinite(d['feh']))
    A = base & (d['eps_birth'] > .7) & (d['eps_z0'] < .3)
    B = base & (d['eps_birth'] < .3) & (d['eps_z0'] < .3)
    if zcut:
        A = A & (d['z_birth'] < Z_A_MAX)
        B = B & (d['z_birth'] > Z_B_MIN)
    d['base'], d['A'], d['B'] = base, A, B
    d['label_A'] = f'A: heated disc (N={A.sum():,})'
    d['label_B'] = f'B: born radial (N={B.sum():,})'
    return d
