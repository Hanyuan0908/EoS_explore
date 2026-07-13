"""Build the compact matched cache from OUR APOGEE catalogues using the reference
repo's own build_cache logic, then attach the satellite_out mask (from the repo CSV).
gc_member is set all-False (we lack the private Vasiliev member file); make_masks
treats that as 'keep all', so the only difference vs. the reference cache is the
small GC-member removal.
"""
import sys
from pathlib import Path
import numpy as np
from astropy.table import Table

REPO = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/eos-figures')
sys.path.insert(0, str(REPO))

from eos_figures.data import build_cache, satellite_out_mask

APO = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/APOGEE_DR17_all.fits'
ANN = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE/apogee_astroNN-DR17.fits'
OUT = Path('/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/our_apogee_dr17_lite_ann.fits.gz')
OUT.parent.mkdir(parents=True, exist_ok=True)

print('building cache from OUR files (RA/DEC match, tol=0.001 arcsec)...')
build_cache(apogee_path=APO, astronn_path=ANN, cache_path=OUT, overwrite=True)

t = Table.read(OUT)
print(f'matched rows: {len(t):,}')

print('computing satellite_out from CompiledSatCatalogv2_gabriel.csv ...')
sat = satellite_out_mask(np.asarray(t['ra']), np.asarray(t['dec']))
t['satellite_out'] = sat
t['gc_member'] = np.zeros(len(t), bool)   # no private Vasiliev file -> keep all
t.write(OUT, overwrite=True)
print(f'satellite_out=False (removed): {(~sat).sum():,}')
print('wrote', OUT)
