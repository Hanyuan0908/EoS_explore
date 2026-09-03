"""Build a WIDE APOGEE cache with all usable [X/Fe] species (value+err), for the
per-species dispersion sweep. Same reference build_cache/matching as the main cache,
just more columns. Git-ignored (derived; rebuild with this script).

For each species: keep X_FE and X_FE_ERR, dropping only FILL values (X_FE<-100) and
undefined errors. NO per-element X_FE_FLAG cut -- B&K22 don't use it, and it removes
stars one-sidedly (e.g. GRIDEDGE_WARN clips the low-N tail), biasing the spread. The
standard global STARFLAG/ASPCAPFLAG quality is already applied by the base sample.
Cast abundances to float32 to keep the file smaller.
"""
import sys
from pathlib import Path
import numpy as np
from astropy.table import Table

REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))
import eos_figures.data as efd
from eos_figures.data import build_cache, satellite_out_mask

APO = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_DR17_all.fits'
ANN = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits'
OUT = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_allspecies.fits.gz')
OUT.parent.mkdir(parents=True, exist_ok=True)

# usable species (Cu, P are all-fill in DR17 here -> excluded)
SPECIES = ['C', 'CI', 'N', 'O', 'NA', 'MG', 'AL', 'SI', 'S', 'K', 'CA',
           'TI', 'TIII', 'V', 'CR', 'MN', 'CO', 'NI', 'CE']

for sp in SPECIES:
    for suffix in ('_FE', '_FE_ERR'):   # no _FE_FLAG: we do NOT flag-cut per element
        col = sp + suffix
        if col not in efd.APOGEE_COLUMNS:
            efd.APOGEE_COLUMNS.append(col)

print('building wide cache (RA/DEC match)...')
build_cache(apogee_path=APO, astronn_path=ANN, cache_path=OUT, overwrite=True)

t = Table.read(OUT)
print(f'rows: {len(t):,}')

# drop only FILL / undefined values (NO per-element flag cut)
for sp in SPECIES:
    v, e = f'{sp.lower()}_fe', f'{sp.lower()}_fe_err'
    x = np.asarray(t[v], float); er = np.asarray(t[e], float)
    bad = (x < -100) | ~np.isfinite(er) | (er <= 0)
    x[bad] = np.nan; er[bad] = np.nan
    t[v] = x.astype(np.float32); t[e] = er.astype(np.float32)

# keep-all selection columns so make_masks runs
t['satellite_out'] = satellite_out_mask(np.asarray(t['ra']), np.asarray(t['dec']))
t['gc_member'] = np.zeros(len(t), bool)
t.write(OUT, overwrite=True)
print('wrote', OUT, f'({OUT.stat().st_size/1e6:.0f} MB)')
