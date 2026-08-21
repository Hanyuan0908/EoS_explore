"""Shared setup for the joaorun003 (Amarante) GASOLINE models.

Only the final snapshot (0.1000, t = 10 Gyr code units) of each model is held
locally, and only the main snapshot file -- the .FeMassFrac / .OxMassFrac
auxiliary arrays and the accreted-particle id lists live in the original run
directory, which is not accessible here.  Consequences:

  * metallicity is total Z only, expressed as [M/H] = log10(Z / Z_sun);
    there is no [Fe/H] and no alpha ratio, so the low-alpha / high-alpha split
    that defines Eos observationally cannot be reproduced -- the Eos analogue
    here is selected on kinematics + age alone;
  * accreted stars cannot be flagged from the provenance list;
  * with a single snapshot there are no birth kinematics, so the born-hot vs
    heated test lives on the Auriga side only.

Units: see jrun003.param -- reconstructed, not the original run file.
"""
import os
import numpy as np
import pynbody

Z_SUN = 0.0142                      # Asplund et al. (2009)
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = {
    'clumpy':    HERE + '/jrun003.dwarfM06XY138Z37Vxy20FB20.01000',
    'notclumpy': HERE + '/jrun003.dwarfM06XY138Z37Vxy20.01000',
}
LABELS = {'clumpy': 'clumpy + merger (FB20)', 'notclumpy': 'not clumpy + merger'}
FIG_DIR = HERE + '/figures'
OUT_DIR = HERE + '/out'

# Eos-like operationalisation, mirroring src/eos/config.py where possible.
EOS_VPHI_MAX = 100.0     # km/s, |v_phi| < this  (EOS_VTAN_MAX)
EOS_ECC_MIN = 0.6        # EOS_ECC_MIN
DISC_VPHI_MIN = 150.0    # DISC_VTAN_MIN
DISC_ECC_MAX = 0.35      # DISC_ECC_MAX


def load(model):
    """Load a model, put it in physical units, and align the stellar disc face-on."""
    f = pynbody.load(MODELS[model])
    f.physical_units()
    pynbody.analysis.angmom.faceon(f.stars)
    return f


def star_frame(f):
    """Cylindrical kinematics, age and [M/H] for every star particle."""
    s = f.s
    pos = np.asarray(s['pos'], float)
    vel = np.asarray(s['vel'], float)
    R = np.hypot(pos[:, 0], pos[:, 1])
    safe = np.where(R > .1, R, 1.)
    d = dict(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        r=np.sqrt((pos ** 2).sum(1)), R=R,
        vR=(pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1]) / safe,
        vphi=(pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / safe,
        vz=vel[:, 2],
        tform=np.asarray(s['tform'], float),
        mass=np.asarray(s['mass'], float),
        phi=np.asarray(s['phi'], float),
    )
    d['t_now'] = d['tform'].max()
    d['age'] = d['t_now'] - d['tform']
    d['Lz'] = pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]
    d['Ltot'] = np.sqrt((np.cross(pos, vel) ** 2).sum(1))
    d['E'] = .5 * (vel ** 2).sum(1) + d['phi']
    Z = np.asarray(s['metals'], float)
    d['mh'] = np.log10(np.clip(Z, 1e-8, None) / Z_SUN)
    if np.median(d['vphi'][(R > 3) & (R < 12) & (np.abs(pos[:, 2]) < 2)]) < 0:
        for k in ('vphi', 'Lz'):
            d[k] = -d[k]
    return d
