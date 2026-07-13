"""Generate docs/Eos_selection_methods.pdf documenting samples, cuts, and orbits.

Rendered with matplotlib PdfPages (no LaTeX/pandoc dependency). Numbers are pulled
live from the catalogues / selection code so the document stays in sync.
"""
import _bootstrap  # noqa
import numpy as np
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from eos import loaders, selections as S, config as C

# ---------------------------------------------------------------- gather numbers
t = loaders.load_apogee()
ap_base = len(t)
ap_eos = int(S.eos_mask(t).sum())
ap_splash = int(S.splash_mask(t).sum())
ap_disc = int(S.lowalpha_disc_mask(t).sum())
e = t[S.eos_mask(t)]
ap_eos_age = int((np.isfinite(e["age"]) &
                  (np.asarray(e["age_err"]) / np.asarray(e["age"]) < C.AGE_REL_ERR_MAX)).sum())

m = Table.read(loaders.repo_path(C.RESULTS_DIR, "lamost_eos_sample.fits"))
lm_join = len(m)
lm_eos = int(S.eos_mask(m).sum())
lm_splash = int(S.splash_mask(m).sum())
lm_disc = int(S.lowalpha_disc_mask(m).sum())
age = np.asarray(m["age"]); rel = np.asarray(m["age_err"]) / age
lm_eos_age = int((S.eos_mask(m) & (rel < C.AGE_REL_ERR_MAX)).sum())

# ---------------------------------------------------------------- page helper
TITLE_FS, H_FS, BODY_FS = 16, 12.5, 10
LEFT = 0.07


def new_page(pdf, title):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.text(LEFT, 0.95, title, fontsize=TITLE_FS, fontweight="bold")
    fig.text(LEFT, 0.925, "Eos in APOGEE + LAMOST  -  selection & orbits  (generated 2026-06-19)",
             fontsize=8, color="0.4")
    return fig


def block(fig, y, lines, fs=BODY_FS, mono=True, color="black"):
    fig.text(LEFT, y, "\n".join(lines), fontsize=fs, va="top",
             family="monospace" if mono else "sans-serif", color=color)


pdf_path = loaders.repo_path("docs", "Eos_selection_methods.pdf")
with PdfPages(pdf_path) as pdf:
    # ===================================================== PAGE 1: samples
    fig = new_page(pdf, "1.  Sample sizes")
    block(fig, 0.88, [
        "APOGEE  (ASPCAP allStar chemistry + AstroNN orbits, row-matched)",
        "-" * 64,
        f"  base sample (clean red giants, SNR>70, no dups, dist<15 kpc)     : {ap_base:,}",
        f"  Eos                                                             : {ap_eos:,}",
        f"     ...of which with reliable age (rel.err < 20%)                : {ap_eos_age:,}",
        f"  Splash                                                          : {ap_splash:,}",
        f"  low-alpha disc (reference)                                      : {ap_disc:,}",
        "",
        "LAMOST  (DD-Payne abundances  x  subgiant MSTO ages, joined on SPECID)",
        "-" * 64,
        f"  joined base sample (DR9 DD-Payne  x  subgiant_fullparam_update) : {lm_join:,}",
        f"  Eos                                                             : {lm_eos:,}",
        f"     ...of which with reliable MSTO age (rel.err < 20%)           : {lm_eos_age:,}",
        f"  Splash                                                          : {lm_splash:,}",
        f"  low-alpha disc (reference)                                      : {lm_disc:,}",
    ])
    block(fig, 0.50, [
        "Notes",
        "-----",
        "* APOGEE giants give clean chemistry incl. [Al/Fe] but coarse ages that",
        f"  saturate above ~10 Gyr, so only {ap_eos_age} Eos stars have reliable APOGEE ages.",
        "* LAMOST subgiants give precise MSTO ages; [Al/Fe] is supplied by the",
        "  DR9 DD-Payne catalogue (median AL_FE_ERR ~ 0.15 dex).",
        "* The two surveys overlap in only ~6 stars by Gaia source_id (giants vs",
        "  subgiants are largely disjoint), so each survey is selected internally;",
        "  there is no star-by-star cross-survey transfer.",
    ], mono=False)
    pdf.savefig(fig); plt.close(fig)

    # ===================================================== PAGE 2: APOGEE cuts
    fig = new_page(pdf, "2.  Exact cuts  -  APOGEE")
    block(fig, 0.89, [
        "Data sources (the paper's choice):",
        "   chemistry  -> ASPCAP allStar (APOGEE_DR17_all.fits); MG_FE, AL_FE, FE_H",
        "                 already [X/Fe] / [Fe/H].  (AstroNN neural-net abundances",
        "                 are NOT used: they show a spurious gap at [Fe/H]~-0.55.)",
        "   orbits/age -> AstroNN VAC, row-matched by APOGEE_ID.",
        "   Kinematics:  Vtan = galvt   ecc = e   rap = rap   Lz = Lz   E = Energy",
        "",
        "(a) Base sample (clean of duplicates / unreliable measurements; red giants)",
        "      weighted_dist < 15 kpc",
        "      FE_H, MG_FE, AL_FE not fill (> -100) and finite",
        "      FE_H_FLAG = MG_FE_FLAG = AL_FE_FLAG = 0   (reliable measurements)",
        "      ASPCAPFLAG STAR_BAD (bit 23) not set;  SNR > 70",
        "      red giants: 1.0 < LOGG < 3.5",
        "      duplicates removed: one row per APOGEE_ID, highest SNR",
        "",
        "(b) Chemical sequence definitions",
        "      in-situ      :  [Al/Fe] > -0.12",
        "      accreted     :  [Al/Fe] <= -0.12",
        "      Mg split     :  S(feh) = 0.16 - 0.08*[Fe/H]",
        "      high-alpha   :  in-situ AND [Mg/Fe] >  S(feh)",
        "      low-alpha    :  in-situ AND [Mg/Fe] <= S(feh) AND [Fe/H] > -1.1",
        "                      AND ( [Fe/H] >= -0.6  OR  [Al/Fe] > 0.9*[Fe/H] + 0.9 )",
        "                      [the diagonal cleans the metal-poor extension only]",
        "",
        "(c) Kinematic operationalisation",
        "      halo orbit   :  |Vtan| < 100 km/s  AND  ecc > 0.6",
        "      disc orbit   :  Vtan > 150 km/s    AND  ecc < 0.35",
        "",
        "(d) Final samples",
        "      Eos          :  low-alpha AND halo-orbit AND  -1.1 < [Fe/H] < -0.5",
        "      Splash       :  high-alpha AND [Fe/H] > -0.9 AND |Vtan| < 80 km/s",
        "      low-a disc   :  low-alpha AND disc-orbit",
    ])
    block(fig, 0.20, [
        "The high/low-alpha split slope/intercept and the halo-orbit thresholds are",
        "tuned to reproduce Fig. 1 of the draft; all live in src/eos/config.py and",
        "are applied in src/eos/selections.py. The [Al/Fe]>0.9[Fe/H]+0.9 diagonal is",
        "taken verbatim from the paper.",
    ], mono=False, color="0.25")
    pdf.savefig(fig); plt.close(fig)

    # ===================================================== PAGE 3: LAMOST cuts
    fig = new_page(pdf, "3.  Exact cuts  -  LAMOST")
    block(fig, 0.89, [
        "Source catalogues (joined on LAMOST SPECID):",
        "   abundances : LMDR9_DDPAYNE_recommend_202505.fits  (MG_FE, AL_FE, FEH)",
        "                -> already [X/Fe]; AL_FE used directly",
        "   ages+orbits: subgiant_fullparam_update.fits",
        "                AGE, AGE_ERR, ECC, R_APO, R_PERI, LZ, E, VT (=Vtan)",
        "",
        "Identical selection logic to APOGEE (same constants), now with LAMOST cols:",
        "      in-situ      :  AL_FE > -0.12",
        "      Mg split     :  S(feh) = 0.16 - 0.08*FEH",
        "      low-alpha    :  in-situ AND MG_FE <= S(feh) AND FEH > -1.1",
        "                      AND ( FEH >= -0.6 OR AL_FE > 0.9*FEH + 0.9 )",
        "      halo orbit   :  |VT| < 100 km/s AND ECC > 0.6",
        "      disc orbit   :  VT > 150 km/s   AND ECC < 0.35",
        "",
        "      Eos          :  low-alpha AND halo-orbit AND -1.1 < FEH < -0.5",
        "      Splash       :  high-alpha AND FEH > -0.9 AND |VT| < 80 km/s",
        "",
        "Age figures additionally require relative age error < 20%.",
    ])
    block(fig, 0.42, [
        "Caveat specific to LAMOST: the DD-Payne aluminium error (~0.15 dex) is ~3x",
        "the APOGEE value, so the in-situ/accreted ([Al/Fe] > -0.12) boundary is",
        "noisier here; the metal-poor low-alpha selection leans mainly on [Mg/Fe]",
        "and [Fe/H], with [Al/Fe] as a secondary discriminant.",
    ], mono=False, color="0.25")
    pdf.savefig(fig); plt.close(fig)

    # ===================================================== PAGE 4: orbits
    fig = new_page(pdf, "4.  How r_apo (and other orbital quantities) were computed")
    block(fig, 0.89, [
        "No orbital integration was run in this project. All orbital quantities",
        "(r_apo, r_peri, eccentricity, Lz, Energy, zmax, actions) were read from",
        "PRE-COMPUTED columns in the source catalogues.",
        "",
        "APOGEE  (used for the orbit-vs-Splash comparison, Fig. ana3):",
        "  column            : rap   (also rperi, e, Lz, Energy, ...)",
        "  source            : apogee_astroNN-DR17.fits (AstroNN VAC)",
        "  computed by       : Mackereth & Bovy (2018); Leung & Bovy (2019)",
        "  integration code  : galpy  (Bovy 2015)",
        "  Galactic potential: MWPotential2014  (Bovy 2015)",
        "  6D input          : Gaia EDR3 astrometry + AstroNN spectro-photometric",
        "                      distances + APOGEE line-of-sight velocities",
        "",
        "LAMOST  (used for the age analyses):",
        "  columns           : R_APO, R_PERI, ECC, LZ, E  (subgiant catalogue)",
        "  source            : subgiant_fullparam_update.fits",
        "  computed by       : the subgiant-catalogue authors (pre-computed),",
        "                      from Gaia astrometry + LAMOST velocities + subgiant",
        "                      spectro-photometric distances.",
        "",
        "Consistency note:",
        "  The quantitative r_apo / ecc / Lz comparison between Eos and the Splash",
        "  (KS tests in results/README.md) was done WITHIN APOGEE only, so both",
        "  samples share the same MWPotential2014 / galpy orbit pipeline and the",
        "  comparison is internally consistent.",
    ])
    block(fig, 0.30, [
        "Unit conventions used in figures:",
        "   E x 10^-5  (Energy in km^2/s^2)      Lz x 10^-3  (Lz in kpc km/s)",
        "   r_apo, r_peri in kpc;  Vtan in km/s.",
    ], mono=False, color="0.25")
    pdf.savefig(fig); plt.close(fig)

print(f"wrote {pdf_path}")
