"""Shared data preparation for the error-deconvolved age-[Fe/H] figures.

Provides load_amr_data(mode) -> (pops, meta) for mode in {'astronn', 'bingo'},
so the XD cache-builder and the plotter use ONE definition of the populations,
the log-age transform, and the per-star errors.

Everything is returned in NATURAL log age (the space we fit XD in):
    astronn : age = lite['age'],  sigma_lnage = age_model_error / age
              quality cut sigma_age/age < 0.3
    bingo   : age = 10^pred_logAge (raw BINGO scale, to ~18 Gyr),
              sigma_lnage = pred_logAge_std * ln(10)   (log10 -> ln)
              Ciuca+24 quality cuts + pred_logAge_std <= 0.1
[Fe/H] and its error always come from the lite catalogue (fe_h, fe_h_err) so the
two age-sources share an identical metallicity axis; BINGO ages are matched to the
lite stars by APOGEE_ID.

Populations (same masks as obs_amr_agedist / obs_eos_age_dist):
    all (base) | eos | splash | lowa (low-a disc) | higha (high-a disc)
"""
import sys
import numpy as np

REPO = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore'
CATDIR = '/Users/hanyuan/Desktop/PhD_projects/spectroscopic_catalogues/APOGEE'
sys.path.insert(0, REPO + '/eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts

POP_NAMES = ['all', 'eos', 'splash', 'lowa', 'higha']


def _base_masks():
    c = Cuts()
    lite = load_catalog(REPO + '/data_repro/our_apogee_dr17_lite_ann.fits.gz')
    m = make_masks(lite, c)
    feh = np.asarray(lite['fe_h'], float); mg = np.asarray(lite['mg_fe'], float)
    al = np.asarray(lite['al_fe'], float); lz = np.asarray(lite['lz'], float)
    vphi = np.asarray(lite['galvt'], float); feherr = np.asarray(lite['fe_h_err'], float)
    rap = np.asarray(lite['rap'], float); rperi = np.asarray(lite['rperi'], float)
    with np.errstate(invalid='ignore'):
        ecc = (rap - rperi) / (rap + rperi)
    base = np.asarray(m['base'], bool)
    thick = np.asarray(m['thick_al'], bool); thin = np.asarray(m['thin_al'], bool)
    halo = base & ((ecc > 0.7) | (lz < 0))
    eos = (halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc * feh + c.inter_acc)
           & (mg < c.slope_acc2 * feh + c.inter_acc2) & (al > c.alfe_cut))
    masks = {'all': base, 'eos': eos, 'splash': thick & (vphi < 80) & (feh > -0.9),
             'lowa': thin & (vphi > 150), 'higha': thick & (vphi > 150)}
    return lite, masks, feh, feherr


def _astronn_arrays(lite):
    age = np.asarray(lite['age'], float)
    aerr = np.asarray(lite['age_model_error'], float)
    ok = np.isfinite(age) & (age > 0) & (aerr / age < 0.3)
    sig_lnage = np.where(age > 0, aerr / age, np.nan)
    return age, sig_lnage, ok


def _bingo_arrays(lite):
    from astropy.table import Table
    b = Table.read(CATDIR + '/APOGEE_DR17_bingoages.fits')
    apb = np.char.strip(np.asarray(b['APOGEE_ID']).astype(str))
    bage = np.asarray(b['age'], float); std = np.asarray(b['pred_logAge_std'], float)
    qb = ((np.asarray(b['MG_FE_FLAG']) == 0) & (np.asarray(b['FE_H_FLAG']) == 0)
          & (np.asarray(b['SNR'], float) > 100)
          & (np.asarray(b['TEFF_1'], float) > 4000) & (np.asarray(b['TEFF_1'], float) < 5500)
          & (np.asarray(b['LOGG_1'], float) > 1) & (np.asarray(b['LOGG_1'], float) < 3.5)
          & (std <= 0.1) & (np.asarray(b['FE_H_1'], float) > -1)
          & ~((bage < 8) & (np.asarray(b['MG_FE_1'], float) > 0.2)))
    bidx = {a: i for i, a in enumerate(apb)}
    apid = np.char.strip(np.asarray(lite['apogee_id']).astype(str))
    N = len(apid)
    age = np.full(N, np.nan); sig_lnage = np.full(N, np.nan); ok = np.zeros(N, bool)
    LN10 = np.log(10.0)
    for i in range(N):
        j = bidx.get(apid[i])
        if j is None or not qb[j]:
            continue
        age[i] = bage[j]; sig_lnage[i] = std[j] * LN10; ok[i] = np.isfinite(bage[j]) & (bage[j] > 0)
    return age, sig_lnage, ok


def load_amr_data(mode):
    """Return (pops, meta).

    pops[name] = dict(lnage, feh, sig_lnage, sig_feh, age)  arrays for the stars in
                 that population passing the mode's quality cut.
    meta       = dict(mode, ager, label) for axis range / titles.
    """
    lite, masks, feh, feherr = _base_masks()
    if mode == 'astronn':
        age, sig_lnage, ok = _astronn_arrays(lite)
        meta = {'mode': 'astronn', 'ager': (0.0, 13.5), 'label': 'AstroNN'}
    elif mode == 'bingo':
        age, sig_lnage, ok = _bingo_arrays(lite)
        meta = {'mode': 'bingo', 'ager': (0.0, 18.0), 'label': 'BINGO (raw)'}
    else:
        raise ValueError(mode)
    good = ok & np.isfinite(feh) & np.isfinite(feherr) & np.isfinite(sig_lnage)
    pops = {}
    for name in POP_NAMES:
        sel = np.asarray(masks[name], bool) & good
        pops[name] = dict(lnage=np.log(age[sel]), feh=feh[sel],
                          sig_lnage=sig_lnage[sel], sig_feh=feherr[sel], age=age[sel])
    return pops, meta
