"""Fit extreme-deconvolution (XD) Gaussian mixtures to the age-[Fe/H] plane and
cache the model parameters, for the error-deconvolved AMR figures.

We fit XD (Bovy, Hogg & Roweis 2011; eos_figures.xd.XD, a fast vectorised
reimplementation validated against astroML) to the (ln age, [Fe/H]) plane, folding
in each star's OWN Gaussian error (diagonal -- the catalogue carries no age/[Fe/H]
error covariance).  Fitting in log age keeps the Gaussians well-behaved and stops
probability leaking below age 0.  Data prep (age source, errors, quality cuts,
population masks) lives in amr_xd_data.load_amr_data so the two age-sources share
one definition.

A COVARIANCE FLOOR (w_reg) is essential: without it XD components collapse to delta
spikes (the deconvolved likelihood is unbounded).  With the floor the result is
insensitive to the exact number of components, so K is just chosen generously per
population rather than by fragile BIC selection (see _xd_kscan.py for the BIC scan
that motivated these values).

  mode astronn -> data_repro/amr_xd_cache_astronn.npz   (ages: lite['age'])
  mode bingo   -> data_repro/amr_xd_cache_bingo.npz      (ages: 10^pred_logAge, raw)

Run once per mode:
    python scripts_repro/build_amr_xd_cache.py [astronn|bingo]
"""
import os
import sys
import time
import numpy as np

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
sys.path.insert(0, REPO + '/eos-figures')
sys.path.insert(0, REPO + '/scripts_repro')
from eos_figures.xd import XD
from amr_xd_data import load_amr_data, POP_NAMES

MODE = sys.argv[1] if len(sys.argv) > 1 else 'astronn'
SEED = 1
# Covariance floor (regularisation): minimum resolvable intrinsic dispersion --
# 0.05 in ln(age) (~5%, ~0.35 Gyr at 7 Gyr, below the age errors) and 0.02 dex in
# [Fe/H].  Prevents delta-spike collapse and makes the fit insensitive to K.
WREG = np.array([0.05**2, 0.02**2])

# generous, floor-regularised component counts + restarts + fit-sample cap per pop.
# (name: K, n_restarts, cap)  cap=None -> use every star in the population.
CFG = {'all':    (6, 3, 40000),
       'eos':    (2, 8, None),
       'splash': (3, 8, None),
       'lowa':   (5, 6, 40000),
       'higha':  (4, 6, 40000)}


def fit_best(X, S, K, n_restarts):
    best = None
    for r in range(n_restarts):
        mdl = XD(K, max_iter=500, tol=1e-6, w_reg=WREG, random_state=SEED + r).fit(X, S)
        if best is None or mdl.loglike_ > best.loglike_:
            best = mdl
    return best


pops, meta = load_amr_data(MODE)
rng = np.random.default_rng(SEED)
out = {'mode': MODE, 'log_age': 'ln', 'w_reg': WREG, 'ager': np.array(meta['ager']),
       'label': meta['label'], 'pops': np.array(POP_NAMES)}
for name in POP_NAMES:
    d = pops[name]
    K, nrst, cap = CFG[name]
    idx = np.arange(len(d['lnage']))
    ntot = len(idx)
    if cap and ntot > cap:
        idx = rng.choice(idx, cap, replace=False)
    X = np.column_stack([d['lnage'][idx], d['feh'][idx]])
    S = np.zeros((len(idx), 2, 2))
    S[:, 0, 0] = d['sig_lnage'][idx] ** 2
    S[:, 1, 1] = d['sig_feh'][idx] ** 2
    t = time.time()
    mdl = fit_best(X, S, K, nrst)
    print(f"[{MODE}] {name:8s} N={len(idx):6d} (of {ntot})  K={K}  logL={mdl.loglike_:.1f}  "
          f"BIC={mdl.bic(X, S):.1f}  ({time.time()-t:.1f}s)", flush=True)
    out[f'{name}_alpha'] = mdl.alpha
    out[f'{name}_mu'] = mdl.mu
    out[f'{name}_V'] = mdl.V
    out[f'{name}_K'] = K
    out[f'{name}_N'] = len(idx)
    out[f'{name}_Ntot'] = ntot

os.makedirs(REPO + '/data_repro', exist_ok=True)
path = REPO + f'/data_repro/amr_xd_cache_{MODE}.npz'
np.savez(path, **out)
print('wrote', path)
