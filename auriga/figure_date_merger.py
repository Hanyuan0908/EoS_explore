"""Summary figure dating the Au18 GS/E merger (clean single-progenitor sample):
  (a) GS/E orbital decay (galactocentric distance & clump dispersion),
  (b) star-formation histories: in-situ starburst vs GS/E SF truncation,
  (c) chemistry: GS/E metal-poor vs in-situ disc.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import z_at_value
import astropy.units as u
import config_au18 as C

os.makedirs(C.FIG_DIR, exist_ok=True)

tr = np.load(C.OUT_DIR + "/gse_track_clean_55_127_2.npz")
tt, rmed, rp25, rp75, disp = (tr["times"], tr["r_med"], tr["r_p25"],
                              tr["r_p75"], tr["disp"])
d = np.load(C.OUT_DIR + "/matched_z0.npz")           # in-situ SFH + chemistry
ia, iff, im = d["ia"], d["iff"], d["im"]
g = np.load(C.OUT_DIR + "/gse_clean_z0.npz")          # clean GS/E
gage, gfeh, gim = g["age"], g["feh"], g["im"]

# timing markers (from the clean-sample track)
T_PERI = 5.0        # first/last pericentre plunge
T_COAL = 5.4        # disruption / phase-mixing complete
T_BURST = 5.25      # in-situ SFR peak

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) orbital decay
a0 = ax[0]
a0.fill_between(tt, rp25, rp75, color="tab:blue", alpha=0.15, label="25-75 pct")
a0.plot(tt, rmed, "-o", color="tab:blue", ms=3, lw=1.8, label="median galactocentric $r$")
a0.plot(tt, disp, "--", color="gray", lw=1.5, label="clump dispersion")
a0.axvline(T_PERI, color="orange", ls=":", lw=1.5)
a0.axvline(T_COAL, color="crimson", ls="-", lw=1.5)
a0.text(T_PERI, 205, " peri", color="orange", fontsize=8, va="top")
a0.text(T_COAL, 205, " coalescence", color="crimson", fontsize=8, va="top")
a0.set_xlabel("cosmic time [Gyr]"); a0.set_ylabel("distance from centre [kpc]")
a0.set_title("(a) GS/E progenitor orbital decay"); a0.set_ylim(0, 230)
a0.set_xlim(2, 14); a0.legend(fontsize=8)

# (b) star-formation histories
a1 = ax[1]
tb = np.arange(0, 14.01, 0.4); tc = 0.5 * (tb[:-1] + tb[1:])
hi, _ = np.histogram(ia, bins=tb, weights=im)
hg, _ = np.histogram(gage, bins=tb, weights=gim)
a1.plot(tc, hi / hi.max(), "-", color="tab:red", lw=2, label="in-situ SFH (main galaxy)")
a1.plot(tc, hg / hg.max(), "-", color="k", lw=2, label="GS/E stars (birth)")
a1.axvline(T_COAL, color="crimson", ls="-", lw=1.5)
a1.axvline(T_BURST, color="tab:red", ls=":", lw=1.3)
a1.text(T_BURST, 1.02, "burst peak", color="tab:red", fontsize=8, ha="center")
a1.set_xlabel("cosmic time (of birth) [Gyr]"); a1.set_ylabel("normalised SFR")
a1.set_title("(b) star-formation histories"); a1.set_xlim(0, 14); a1.legend(fontsize=8)

# (c) chemistry
a2 = ax[2]
bins = np.linspace(-2.5, 0.6, 60)
a2.hist(iff[np.isfinite(iff)], bins=bins, density=True, histtype="step",
        color="tab:red", lw=2, label="in-situ")
a2.hist(gfeh[np.isfinite(gfeh)], bins=bins, density=True, histtype="step",
        color="k", lw=2, label="GS/E debris")
a2.set_xlabel("[Fe/H]"); a2.set_ylabel("normalised")
a2.set_title("(c) chemistry: GS/E is metal-poor"); a2.legend(fontsize=8)

z_coal = float(z_at_value(C.COSMO.age, T_COAL * u.Gyr).value)
fig.suptitle(f"Au18 GS/E merger dating: pericentre $t\\approx${T_PERI} Gyr, "
             f"coalescence $t\\approx${T_COAL} Gyr "
             f"($z\\approx${z_coal:.2f}, lookback $\\approx${C.T0_GYR-T_COAL:.1f} Gyr); "
             f"$M_\\star\\approx1.6\\times10^9\\,M_\\odot$", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = C.FIG_DIR + "/au18_gse_merger_dating.png"
fig.savefig(out, dpi=140); print("saved", out)
for label, t in [("pericentre", T_PERI), ("coalescence", T_COAL), ("burst peak", T_BURST)]:
    z = float(z_at_value(C.COSMO.age, t * u.Gyr).value)
    print(f"  {label:12s}: t={t:.2f} Gyr, z={z:.2f}, lookback={C.T0_GYR-t:.2f} Gyr")
