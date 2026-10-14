"""Publication figure: what the lane's stars and gas look like ALONG the lane.

Two panels against arc length s, measured from the GS/E end of the traced gas
lane (diag_lane_ridge.py, lane_path.py):

  (a)  [Fe/H] -- the halo-born stars formed in this snapshot, with the
       star-forming gas of the same tube over them
  (b)  z_max TODAY of those same stars

THE SELECTION.  In-situ stars with t_form in the snapshot interval
(4.83-4.99 Gyr), halo-born by the paper classification (eps_birth <= 0.8 AND
z_max_birth >= 1.5 kpc), inside a TUBE of radius 3 kpc about the centreline.
The tube is 3D -- sqrt(d_xz^2 + y^2) -- unlike the lane mask on the gas maps,
which can only be projected: in projection the same cut admits 2,126 more stars,
65 per cent more, all of them fore- and background.  3 kpc rather than the
3.5 kpc the gas maps use: shrinking the tube from 3.5 to 2.0 kpc narrows the gas
spread from 0.45 to 0.37 dex and leaves the gradient at 0.49 dex either way, so
the extra width was surroundings, not lane.

THE TWO LEVELS ARE THE NEXT STEP ALONG THE LANE, not apertures on the two
galaxies.  Each is a sphere of the tube radius placed 1.5 kpc beyond the
corresponding end of the centreline, along that end's own tangent, with the tube
excluded -- the same gas, radius and mass weighting as the curve.  Apertures on
the objects were tried first and are not usable here: the host level came out at
-0.08, -0.05, +0.04 or +0.12 depending on whether it was R < 8, |R-R_land| < 1,
|R-R_land| < 2 or R < 4, because each is a different volume with a different
selection from the curve.

WHETHER THE CURVE MEETS THE LEVELS IS THEREFORE A RESULT.  It does at the GS/E
end (-0.470 against the curve's -0.485) and it does NOT at the host end (+0.239
against +0.009).  That 0.23 dex gap is the figure's point -- the lane arrives at
the disc still poorer than the gas it joins -- and it is robust to where the
sphere sits: 0, 1.5, 3.0 and 4.5 kpc beyond the end give +0.21, +0.24, +0.23,
+0.17, because every step on that side lands inside R = 3 kpc.

READ PANEL (b) WITH ITS DEGENERACY IN MIND.  The lane runs from z = -12.1 to
z = +0.2, so s and birth height are nearly the same variable: r = -0.98.  A star
born high on a halo orbit reaches a large z_max almost by construction, so (b) is
mostly "stars born high stay high" -- a statement that the lane's vertical
structure survives to z = 0, not an independent trend along the lane.  Panel (a)
carries no such degeneracy, because its gradient is in the gas.

z_max today is integrated for 30 circular periods in the snapshot-127 AGAMA
potential (prep_zmax_z0.py), NOT the action approximation, which underestimates
it by up to ~40 per cent for eccentric orbits.  2.5 per cent of the sample
exceeds the 30 kpc clip of panel (b), from orbits close to unbound.

The shaded ends are where the centreline is extrapolated 1.5 kpc along its own
tangent, towards each object; they are named rather than measured.

Reads the cache written by diag_lane_born_profile.py; run that first.
Writes Fig_paper/au18_lane_profile.pdf and .png.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import config_au18 as C

OUT = '/data/hz420-2/EoS_explore/Fig_paper'
os.makedirs(OUT, exist_ok=True)
SNAP, R_LANE = 72, 3.0
CACHE = C.OUT_DIR + f'/lane_profile_snap{SNAP}_R{R_LANE:g}.npz'
if not os.path.exists(CACHE):
    raise SystemExit(f'missing {CACHE}; run diag_lane_born_profile.py first')
YLIM = (-0.8, 0.3)
ZLIM = (0.8, 30.)
cH, cSF, cGSE, cMW = '#FF6347', '#1F6FB2', '#8E24AA', '#00897B'

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    # house sizes x 1.15, as in the other recent paper figures
    'font.size': 15.5, 'axes.labelsize': 17,
    'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 14,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})

d = np.load(CACHE)
S_LO, S_HI = float(d['s_lo']), float(d['s_hi'])
S0, S1 = float(d['s0_fit']), float(d['s1_fit'])
ctr, PG, PZ = d['ctr'], d['prof_gas'], d['prof_zmax']
RG_, RM_ = d['ref_gse'], d['ref_mw']
print(f"snapshot {int(d['snap'])}, {float(d['t_lo']):.2f}-{float(d['t_hi']):.2f} Gyr, "
      f"tube {float(d['r_lane']):g} kpc, N = {int(d['n_lane']):,}")
print(f'GS/E level {RG_[1]:+.3f}, curve at the GS/E end '
      f'{PG[np.isfinite(PG[:, 0])][0, 2]:+.3f}')
print(f'MW   level {RM_[1]:+.3f}, curve at the host end '
      f'{PG[np.isfinite(PG[:, 0])][-1, 2]:+.3f}')

# Wide and short on purpose: the figure is a pair of profiles against one
# variable, so it carries its information horizontally and only costs the
# manuscript vertical space.  The margins are figure fractions, so they have to
# be retuned whenever FH changes.
FW, FH = 16.0, 4.6
fig, axes = plt.subplots(1, 2, figsize=(FW, FH))
fig.subplots_adjust(left=.060, right=.988, bottom=.165, top=.970, wspace=.21)

for ax, tag in zip(axes, ('(a)', '(b)')):
    ax.axvspan(S_LO, S0, color='.88', lw=0, zorder=0)
    ax.axvspan(S1, S_HI, color='.88', lw=0, zorder=0)
    for v in (S0, S1):
        ax.axvline(v, color='.55', lw=.9, ls=(0, (4, 3)), zorder=0)
    ax.set_xlabel('$s$ along the lane [kpc]')
    ax.set_xlim(S_LO, S_HI)
    ax.text(.035, .955, tag, transform=ax.transAxes, va='top', fontsize=19,
            fontweight='bold')

# --- (a) [Fe/H] --------------------------------------------------------------
ax = axes[0]
for lev, col in ((RG_[1], cGSE), (RM_[1], cMW)):
    ax.axhline(lev, color=col, lw=1.8, ls=':', zorder=1)
ax.scatter(d['s_star'], d['feh_star'], s=5, c=cH, alpha=.25, lw=0,
           rasterized=True)
g = np.isfinite(PG[:, 0])
ax.fill_between(ctr[g], PG[g, 1], PG[g, 3], color=cSF, alpha=.22, lw=0,
                rasterized=True)
ax.plot(ctr[g], PG[g, 2], color=cSF, lw=2.8, zorder=4)
ax.plot([], [], color=cSF, lw=2.8, label='star-forming gas')
ax.plot([], [], 'o', color=cH, ms=6, label='Stars at this snapshot')
# shifted left of the default lower-right corner, which runs into the
# extrapolated margin and the dashed line that bounds it
ax.legend(loc='lower left', bbox_to_anchor=(.30, .015), handlelength=1.5,
          borderpad=.3, labelspacing=.35)
ax.set(ylabel='[Fe/H]', ylim=YLIM)
for x, nm, col in ((.5 * (S_LO + S0), 'GS/E', cGSE),
                   (.5 * (S1 + S_HI), 'MW', cMW)):
    ax.text(x, .5, nm, transform=ax.get_xaxis_transform(), rotation=90,
            ha='center', va='center', fontsize=15, color=col)

# --- (b) z_max today ---------------------------------------------------------
ax = axes[1]
ax.scatter(d['s_zmax'], d['zmax_star'], s=5, c=cH, alpha=.25, lw=0,
           rasterized=True)
g = np.isfinite(PZ[:, 0])
ax.fill_between(ctr[g], PZ[g, 1], PZ[g, 3], color='k', alpha=.16, lw=0,
                rasterized=True)
ax.plot(ctr[g], PZ[g, 2], color='k', lw=2.8, zorder=4)
ax.set(yscale='log', ylabel=r'$z_{\rm max,\,today}$ [kpc]', ylim=ZLIM)
for x, nm in ((.5 * (S_LO + S0), 'GS/E'), (.5 * (S1 + S_HI), 'MW')):
    ax.text(x, .5, nm, transform=ax.get_xaxis_transform(), rotation=90,
            ha='center', va='center', fontsize=15, color='.35')

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/au18_lane_profile.{ext}', bbox_inches='tight')
print(f'\nsaved {OUT}/au18_lane_profile.pdf and .png')
