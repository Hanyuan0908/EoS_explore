import sys, numpy as np, warnings, time
warnings.filterwarnings('ignore'); np.random.seed(1)
sys.path.insert(0, 'eos-figures')
from astroML.density_estimation import XDGMM
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


def data(sel, cap=None):
    idx = np.where(sel & rel_ok & np.isfinite(feh) & np.isfinite(age))[0]
    if cap and len(idx) > cap: idx = np.random.choice(idx, cap, replace=False)
    a = age[idx]; f = feh[idx]; ae = aerr[idx]; fe = feherr[idx]
    X = np.column_stack([np.log(a), f]); S = np.zeros((len(idx), 2, 2))
    S[:, 0, 0] = (ae / a) ** 2; S[:, 1, 1] = fe ** 2
    return X, S


print("=== validate: Eos, K=1..5, mine vs astroML (BIC) ===")
X, S = data(eos)
for K in range(1, 6):
    t = time.time(); mine = XD(K, random_state=1).fit(X, S); tm = time.time() - t
    t = time.time(); am = XDGMM(K, max_iter=300, random_state=1).fit(X, S); ta = time.time() - t
    nparam = (K - 1) + K * 2 + K * 2 * 3 // 2
    bam = -2 * am.logL(X, S) + nparam * np.log(len(X))
    print(f"  K={K}  mine BIC={mine.bic(X,S):9.1f} ({tm:.2f}s)   astroML BIC={bam:9.1f} ({ta:.2f}s)")

print("\n=== speed: my XD high-K on capped samples ===")
for nm, sel, Ks, cap in [("base", base, [4, 6, 8, 10, 12, 14], 4000),
                          ("low-a", lowa, [2, 4, 6, 8, 10], 3000),
                          ("high-a", higha, [2, 4, 6, 8, 10], 3000),
                          ("Splash", splash, [2, 3, 4, 5, 6, 7], 3000)]:
    X, S = data(sel, cap=cap)
    bics = []; t = time.time()
    for K in Ks:
        bics.append(XD(K, random_state=1).fit(X, S).bic(X, S))
    best = Ks[int(np.argmin(bics))]
    print(f"  {nm:8s} n={len(X):5d}  K*={best:2d}  ({time.time()-t:.1f}s)  " +
          " ".join(f"{k}:{b:.0f}" for k, b in zip(Ks, bics)))
