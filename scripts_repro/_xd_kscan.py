import sys, numpy as np, warnings, time
warnings.filterwarnings('ignore'); np.random.seed(1)
sys.path.insert(0, 'eos-figures')
from eos_figures.xd import XD
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
c = Cuts(); cat = load_catalog('data_repro/our_apogee_dr17_lite_ann.fits.gz'); m = make_masks(cat, c)
feh = np.asarray(cat['fe_h'], float); mg = np.asarray(cat['mg_fe'], float); vphi = np.asarray(cat['galvt'], float)
age = np.asarray(cat['age'], float); aerr = np.asarray(cat['age_model_error'], float); feherr = np.asarray(cat['fe_h_err'], float)
al = np.asarray(cat['al_fe'], float); lz = np.asarray(cat['lz'], float)
rap = np.asarray(cat['rap'], float); rperi = np.asarray(cat['rperi'], float)
with np.errstate(invalid='ignore'): ecc = (rap - rperi) / (rap + rperi)
base = np.asarray(m['base'], bool); thin_al = np.asarray(m['thin_al'], bool); thick_al = np.asarray(m['thick_al'], bool)
halo = base & ((ecc > 0.7) | (lz < 0))
eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc * feh + c.inter_acc) & (mg < c.slope_acc2 * feh + c.inter_acc2) & (al > c.alfe_cut)
lowa = thin_al & (vphi > 150); higha = thick_al & (vphi > 150); splash = thick_al & (vphi < 80)
rel_ok = np.isfinite(age) & np.isfinite(aerr) & (aerr / age < 0.3)

WREG = np.array([0.05**2, 0.02**2])   # floor: 0.05 in log-age, 0.02 dex in [Fe/H]


def scan(name, sel, Ks, cap=4000, nrst=3):
    idx = np.where(sel & rel_ok & np.isfinite(feh) & np.isfinite(age))[0]; ntot = len(idx)
    if len(idx) > cap: idx = np.random.choice(idx, cap, replace=False)
    a = age[idx]; f = feh[idx]; ae = aerr[idx]; fe = feherr[idx]
    X = np.column_stack([np.log(a), f]); S = np.zeros((len(idx), 2, 2)); S[:, 0, 0] = (ae / a)**2; S[:, 1, 1] = fe**2
    bics = []; t = time.time()
    for K in Ks:
        best = None
        for r in range(nrst):
            mdl = XD(K, w_reg=WREG, random_state=1 + r).fit(X, S)
            if best is None or mdl.loglike_ > best.loglike_: best = mdl
        bics.append(best.bic(X, S))
    best = Ks[int(np.argmin(bics))]
    print(f"{name:8s} Ntot={ntot:6d} fit={len(idx):5d} K*={best:2d} ({time.time()-t:.1f}s)  " +
          "  ".join(f"{k}:{b:.0f}" for k, b in zip(Ks, bics)), flush=True)


scan("eos", eos, [1, 2, 3, 4], cap=4000, nrst=5)
scan("splash", splash, [2, 3, 4, 5, 6], cap=4000)
scan("lowa", lowa, [2, 3, 4, 5, 6, 8], cap=4000)
scan("higha", higha, [2, 3, 4, 5, 6, 8], cap=4000)
scan("all", base, [4, 6, 8, 10, 12], cap=5000)
