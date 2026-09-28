"""Shared loader for the AGAMA + McMillan17 (from-scratch, BJ21 dist + Gaia DR3)
APOGEE kinematics. Reproduces the reference `make_masks` mask set on the new cache
(`data_repro/apogee_mcmillan17_kin.fits`), so the Fig_paper_mcmillan17 scripts get
the SAME populations as the Fig_paper ones but with our own orbits.

Difference from the reference base cut: the AstroNN galactocentric velocity errors
(`galv*_err < 50`) are not available from scratch, so the astrometric/distance quality
is enforced with **RUWE < 1.4 & fractional BJ21 distance error < 0.2** instead. Distance
cut is `dist < 15 kpc`. Everything else (chem errors, logg<3, no MagClouds, GC and
satellite removal, all chemical dividers) is identical to the reference.
"""
import numpy as np
from astropy.table import Table
from eos_figures.config import Cuts

CACHE = '/Users/hanyuan/Library/CloudStorage/Dropbox/python_script/EoS_explore/data_repro/apogee_mcmillan17_kin.fits'


def load_mcm(path=CACHE):
    return Table.read(path)


def make_masks_mcm(cat, c: Cuts = Cuts(), age=None, age_err=None):
    g = lambda n: np.asarray(cat[n], float)
    feh = g('fe_h'); mg = g('mg_fe'); al = g('al_fe')
    finite_core = np.isfinite(feh) & np.isfinite(mg) & np.isfinite(al)
    noclouds = np.char.lower(np.asarray(cat['programname']).astype(str)) != 'magclouds'
    # quality proxy for the missing galactocentric velocity errors
    velerr = (g('ruwe') < 1.4) & np.isfinite(g('dist_frac_err')) & (g('dist_frac_err') < 0.2)
    species = ['fe_h', 'mg_fe', 'mn_fe', 'al_fe', 'c_fe', 'cr_fe', 'o_fe', 'n_fe', 'ni_fe', 'si_fe']
    chemerr = np.ones(len(cat), bool)
    for s in species:
        lim = c.feh_err_lim if s == 'fe_h' else c.chem_err_lim
        chemerr &= (g(s + '_err') < lim)
    logg = g('logg') < c.logg_lim
    dist = g('dist') < 15.0
    # bound-orbit sanity: drops the handful of bad-distance outliers (unbound /
    # huge r_apo) that slip past RUWE/dist quality (McMillan17 E<0 = bound).
    sane = (g('energy') < 0) & np.isfinite(g('rap')) & (g('rap') < 50)
    out = np.asarray(cat['satellite_out'], bool)
    no_gc = ~np.asarray(cat['gc_member'], bool)
    base = finite_core & velerr & chemerr & logg & out & no_gc & dist & noclouds & sane

    energy = g('energy'); lz = g('lz'); galvt = g('galvt'); rap = g('rap')
    # McMillan17 energy bands (deeper zero-point). EN_LIM_MCM reproduces the stars
    # the reference en_lim=(-0.75e5,-0.4e5) 'base_en' selected (mapped via shared stars);
    # EN_LIM_ACC_MCM is the McMillan17 analogue of en_lim_acc=-2e5.
    EN_LIM_MCM = (-1.92e5, -1.56e5)
    EN_LIM_ACC_MCM = -2.3e5
    encut = (energy > EN_LIM_MCM[0]) & (energy < EN_LIM_MCM[1])
    encut_acc = energy > EN_LIM_ACC_MCM
    lz_acc = np.abs(lz) < c.lz_lim_acc
    mg_in = mg > c.slope_acc * feh + c.inter_acc
    al_thin = al < c.kalfe * feh + c.offalfe
    al_insitu = al > c.alfe_cut
    al_acc = al < c.alfe_cut
    feh_acc = feh < c.feh_acc_cut
    mg_acc = mg < c.slope_acc * feh + c.inter_acc
    mg_thick = mg > c.slope_acc2 * feh + c.inter_acc2
    mg_thin = mg < c.slope_acc2 * feh + c.inter_acc2

    m = {
        'base': base,
        'base_en': base & encut,
        'acc': base & mg_acc & mg_thin & encut_acc & lz_acc,
        'acc_al': base & mg_acc & mg_thin & al_acc & feh_acc,
        'thin': base & mg_in & mg_thin & al_thin,
        'thick': base & mg_thick,
    }
    m['thin_en'] = m['thin'] & encut
    m['thick_en'] = m['thick'] & encut
    m['acc_en'] = m['acc'] & encut
    m['thin_al'] = base & mg_in & mg_thin & al_thin & al_insitu
    m['thick_al'] = base & mg_thick & al_insitu
    m['tri'] = m['base_en'] & (feh > c.feh_tri_cut[0]) & (feh < c.feh_tri_cut[1])
    vt = galvt < c.vt_sep
    m['ecc'] = base & vt & (rap > c.rap_min)

    if age is not None:
        age = np.asarray(age, float)
        aerr = (np.asarray(age_err, float) / age) < c.age_err_frac if age_err is not None else np.ones(len(cat), bool)
        ok = np.isfinite(age) & (age > 0) & aerr
        m['base_en_age'] = m['base_en'] & ok
        for k in ['thin', 'thick', 'thin_al', 'thick_al']:
            m[k + '_age'] = m[k] & ok
        m['thin_al_vt_age'] = m['thin_al_age'] & vt
        m['thin_al_vt_rap_age'] = m['thin_al_vt_age'] & (rap > c.rap_min_age)
        m['thick_al_splash_age'] = m['thick_al_age'] & vt & (feh > c.feh_splash_min)
    return m


def eos_split(cat, m, c: Cuts = Cuts()):
    """Return (eos, eos_metal_poor, eos_metal_rich) using the canonical wedge + Davies
    divider. metal-poor = alpha-rich/upper (mg>=divider); metal-rich = alpha-poor/lower."""
    g = lambda n: np.asarray(cat[n], float)
    feh = g('fe_h'); mg = g('mg_fe'); al = g('al_fe'); lz = g('lz'); ecc = g('ecc')
    halo = np.asarray(m['base'], bool) & ((ecc > 0.7) | (lz < 0))
    eos = halo & (feh > -0.9) & (feh < -0.2) & (mg > c.slope_acc * feh + c.inter_acc) \
        & (mg < c.slope_acc2 * feh + c.inter_acc2) & (al > c.alfe_cut)
    divider = 0.317 * feh + 0.353
    return eos, eos & (mg >= divider), eos & (mg < divider)
