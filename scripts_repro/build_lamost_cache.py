"""Build a LAMOST cache whose columns match what eos_figures.data.make_masks reads,
so the SAME reference selection (chemistry + kinematics, same coefficients) applies.

Survey-appropriate adaptations (LAMOST subgiants, not APOGEE giants):
  * chemistry: [Mg/Fe],[Al/Fe],[Fe/H] from DD-Payne (join on SPECID); FLAG_MG_FE==0
    & FLAG_AL_FE==0 required (the LAMOST analogue of ASPCAP flag cleaning).
  * the giant logg<3.0 cut is a no-op here (subgiants have logg~3.8); the sample is
    already the Xiang MSTO/subgiant selection.  logg is set to 0 so make_masks keeps all.
  * element-error cuts: real fe/mg/al errors kept (<0.2); the other 7 element errors
    (mn,c,cr,o,n,ni,si) and the 3 velocity errors are unavailable -> set 0 (pass).
  * distance: Bailer-Jones rpgeo (pc).  satellite_out=True, gc_member=False (keep all).
Kinematics (VT, LZ, ENERGY, R_APO, R_PERI) are in the same units as APOGEE AstroNN.
Ages: AGE + E_AGE (MSTO) -> age / age_model_error.
"""
import sys
from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.table import Table

REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
from eos_figures.data import _fits_format  # reuse

D = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/LAMOST/'
OUT = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_lamost_subgiant_ddpayne.fits.gz')
OUT.parent.mkdir(parents=True, exist_ok=True)


def norm(col):
    def one(s):
        if isinstance(s, bytes):
            s = s.decode('ascii', 'ignore')
        return str(s).strip()
    return np.array([one(s) for s in np.asarray(col)])


# ---- subgiant + Gaia (ages, orbits) ----
g = fits.open(D + 'LAMOST_Gaia_subgiants_Xiangetal2024_withGaia.fits')[1].data
sg_specid = norm(g['LAMOST_SPECID'])
AGE_MAX = 14.0          # physical ceiling: reject ages older than the Universe (~13.8 Gyr)
RUWE_LO, RUWE_HI = 0.6, 1.4   # Gaia astrometric quality window
age = np.asarray(g['AGE'], float)
ruwe = np.asarray(g['RUWE'], float)
finite = (np.isfinite(age) & np.isfinite(g['VT']) & np.isfinite(g['LZ'])
          & np.isfinite(g['ENERGY']) & np.isfinite(g['R_APO']) & np.isfinite(g['rpgeo'])
          & (np.asarray(g['rpgeo'], float) / 1000.0 < 15.0))
age_ok = age < AGE_MAX
ruwe_ok = np.isfinite(ruwe) & (ruwe > RUWE_LO) & (ruwe < RUWE_HI)
keep = finite & age_ok & ruwe_ok
print(f'subgiant rows: {len(g):,}  after finite+dist: {int(finite.sum()):,}')
print(f'  age<{AGE_MAX}: removes {int((finite & ~age_ok).sum()):,}')
print(f'  {RUWE_LO}<RUWE<{RUWE_HI}: removes {int((finite & age_ok & ~ruwe_ok).sum()):,}')
print(f'  after age+RUWE quality: {int(keep.sum()):,}')

# ---- DD-Payne abundances (join on SPECID) ----
dp = fits.open(D + 'LMDR9_DDPAYNE_recommend_202505.fits')[1].data
dp_specid = norm(dp['SPECID'])
order = np.argsort(dp_specid, kind='stable')
dp_s = dp_specid[order]
pos = np.clip(np.searchsorted(dp_s, sg_specid), 0, len(dp_s) - 1)
match = dp_s[pos] == sg_specid
src = order[pos]                      # row in dp for each subgiant
print(f'SPECID matched to DD-Payne: {int(match.sum()):,}')

def dpcol(name):
    return np.asarray(dp[name], float)[src]

flag_ok = (np.asarray(dp['FLAG_MG_FE'], float)[src] == 0) & (np.asarray(dp['FLAG_AL_FE'], float)[src] == 0)
mg = dpcol('MG_FE'); al = dpcol('AL_FE'); feh = dpcol('FEH')
chem_finite = np.isfinite(mg) & np.isfinite(al) & np.isfinite(feh)

CHEM_ERR_MAX = 0.15     # moderate chemistry cut: err<0.15 dex on [Fe/H],[Mg/Fe],[Al/Fe]
fe_err = dpcol('FEH_ERR'); mg_err = dpcol('MG_FE_ERR'); al_err = dpcol('AL_FE_ERR')
chem_err_ok = (fe_err < CHEM_ERR_MAX) & (mg_err < CHEM_ERR_MAX) & (al_err < CHEM_ERR_MAX)

sel = keep & match & flag_ok & chem_finite & chem_err_ok
print(f'final LAMOST sample (good flags + finite + err<{CHEM_ERR_MAX} on Fe/Mg/Al): {int(sel.sum()):,}')
print(f'  (chem err<{CHEM_ERR_MAX} removes {int((keep & match & flag_ok & chem_finite & ~chem_err_ok).sum()):,} of the flag-clean matches)')

n = int(sel.sum())
z = np.zeros(n)
t = Table()
# chemistry (DD-Payne, [X/Fe])
t['fe_h'] = feh[sel]; t['mg_fe'] = mg[sel]; t['al_fe'] = al[sel]
t['fe_h_err'] = dpcol('FEH_ERR')[sel]
t['mg_fe_err'] = dpcol('MG_FE_ERR')[sel]
t['al_fe_err'] = dpcol('AL_FE_ERR')[sel]
# element/velocity errors we don't have -> 0 so the make_masks cuts pass
for cerr in ('mn_fe_err', 'c_fe_err', 'cr_fe_err', 'o_fe_err', 'n_fe_err', 'ni_fe_err', 'si_fe_err',
             'galvr_err', 'galvt_err', 'galvz_err'):
    t[cerr] = z
# evolutionary: subgiants -> make giant logg<3.0 a no-op
t['logg'] = z
t['programname'] = np.array(['subgiant'] * n)
t['weighted_dist'] = np.asarray(g['rpgeo'], float)[sel]      # pc
t['ra'] = np.asarray(g['RApmRAcor'] if 'RA' not in g.names else g['RA'], float)[sel] if False else z  # unused (satellite_out preset)
# kinematics / orbits (same units as AstroNN)
t['energy'] = np.asarray(g['ENERGY'], float)[sel]
t['lz'] = np.asarray(g['LZ'], float)[sel]
t['galvt'] = np.asarray(g['VT'], float)[sel]
t['rap'] = np.asarray(g['R_APO'], float)[sel]
t['rperi'] = np.asarray(g['R_PERI'], float)[sel]
t['zmax'] = np.full(n, np.nan)          # not in this catalogue
# ages (MSTO)
t['age'] = np.asarray(g['AGE'], float)[sel]
t['age_model_error'] = np.asarray(g['E_AGE'], float)[sel]
# keep-all masks
t['satellite_out'] = np.ones(n, bool)
t['gc_member'] = np.zeros(n, bool)
t['specid'] = sg_specid[sel]
t['source_id'] = np.asarray(g['GAIAEDR3_SOURCE_ID'], np.int64)[sel]

t.write(OUT, overwrite=True)
print('wrote', OUT)
print(f'  median age err/age = {np.nanmedian(t["age_model_error"]/t["age"]):.3f}')
