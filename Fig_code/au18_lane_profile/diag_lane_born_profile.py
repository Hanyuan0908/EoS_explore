"""Diagnostic: what the stars born along the lane look like as a function of
distance ALONG it.

au18_birth_positions_gas4 shows that the halo-born stars of the merger snapshot
lie along the gas lane rather than filling the halo.  This asks the next
question: does anything vary from the satellite end of the lane to the host end?
Two things are plotted against arc length s measured from the GS/E end:

  (a)  [Fe/H] at birth -- the composition of the gas each star formed out of,
       with the [Fe/H] OF THE GAS ITSELF along the same tube drawn over it
  (b)  z_max TODAY     -- how far off the plane the star's orbit still reaches

THE GAS CURVE IS MEASURED THE SAME WAY AS THE STARS, deliberately: the same 3D
tube, the same s bins, the same snapshot.  It is NOT read off the cached maps of
fig_paper_gas_metallicity_sf -- those project through y, so a pixel's [Fe/H] is a
line-of-sight average that mixes in gas nowhere near the lane, and comparing it
with a 3D stellar selection would be comparing two different volumes.  The gas is
mass-weighted (cells differ in mass by a factor of a few), the stars are not;
both are stated as 16/50/84 percentiles so the comparison is of distributions,
not of means.

    usage:  diag_lane_born_profile.py [SNAP] [R_LANE] [EXT]   (default 72 3.0 1.5)

THE CENTRELINE IS EXTENDED BY EXT kpc AT EACH END, along its end tangents, so the
profile does not stop short of the two objects the lane joins.  s = 0 and s = L
mark the ends of the FITTED trace and are drawn on the figure: outside them the
line is an extrapolation, not a measurement, which is why EXT is kept small.
Even extended it does not reach either object -- the GS/E centroid is 5.5 kpc
beyond the lower end -- so the two endpoints are put on the figure directly, as
horizontal bands:

  GS/E gas   a sphere of radius R_LANE, STEP kpc beyond the GS/E end of the
             centreline along that end's own tangent
  MW gas     the same, beyond the host end

i.e. the NEXT STEP ALONG THE LANE in each direction, measured with the same
radius, the same gas and the same weighting as the curve itself, with the lane
tube excluded so the three volumes are disjoint.  Earlier versions used apertures
on the two objects -- a sphere on the satellite, an annulus or a disc box on the
host -- and the host answer moved between -0.08 and +0.12 depending on which one,
because those are different volumes with different selections from the curve.  A
curve and a level measured differently were never going to meet.

WHETHER THEY MEET IS THEN A RESULT, not a construction.  They do at the GS/E end
(-0.47 against the curve's -0.49) and they do NOT at the host end (+0.24 against
the curve's +0.01).  That 0.2 dex gap is the point: the lane arrives at the disc
still poorer than the disc it is joining, so mixing is not complete where the two
meet.  The gap is robust to where the sphere is placed -- stepping 0, 1.5, 3 or
4.5 kpc beyond the end gives +0.21, +0.24, +0.23, +0.17 -- because every step on
that side lands inside R = 3 kpc, where the host's gas is rich.

Each is drawn as a SINGLE line, not a band.  The band was the 16-84 spread of
cells inside the aperture, and it is wide (0.55 dex) for both: that width is the
gas's own cell-to-cell inhomogeneity and it is the same at every aperture size
tried, so showing it twice -- once per endpoint, over the whole panel -- said
nothing the lane's own shaded band does not already say, and invited the eye to
read two fuzzy levels rather than two endpoints.  What the figure needs from
these is the LEVEL the lane runs between.

The MW endpoint is the disc at the lane's own landing radius rather than the disc
as a whole because the host's star-forming gas has a 0.75 dex radial gradient
(+0.25 at R < 2, -0.43 at R = 8-12): "the metallicity of the disc" is not a
number, so the one to compare against is the gas the lane actually joins.

Both EXCLUDE the lane tube itself, so the three samples are disjoint and "the
lane is intermediate" is a statement about three separate volumes rather than
about overlapping ones.

SELECTION.  Stars with t_form in the snapshot interval (4.83-4.99 Gyr for snap
72), in-situ, halo-born by the paper classification (eps_birth <= 0.8 AND
z_max_birth >= 1.5 kpc), and inside a TUBE of radius R_LANE about the traced lane
centreline (diag_lane_ridge.py, via lane_path.py).

THE TUBE IS 3D, the mask on the gas maps is not.  Those maps project through y,
so a pixel "on the lane" there integrates the whole line of sight; for stars we
have the third coordinate, so the selection uses the true distance to the
centreline, sqrt(d_xz^2 + y^2) < R_LANE.  The script prints how many stars the
projected mask would have added -- they are the foreground and background that a
map cannot separate.

FRAME.  Identical to au18_birth_positions_gas4 and to the gas maps: the snapshot's
own disc frame, then au18_frame.align_azimuth on the GS/E centroid so the
satellite and the lane lie in the x-z plane at their true lengths.  The lane
centreline is only meaningful in that frame.

z_max TODAY comes from out/zmax_z0_4.5_6.5.npz -- orbits integrated for 30
circular periods in the snapshot-127 AGAMA potential (prep_zmax_z0.py).  NOT the
zmax_z0 column of birth_orbits_zmax.npz, which is the action approximation and
underestimates z_max by up to ~40 per cent for eccentric orbits
(METHOD_zmax_from_Jz.md).  The integrated values have a runaway tail -- 0.61 per
cent of that file exceeds 30 kpc and the maximum is 1.9e5 kpc, from orbits close
to unbound -- so the panel is clipped at ZMAX_HI and the number dropped is
printed.

Writes figures/au18_lane_born_profile.png
"""
import gc, os, sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import auriga_public as ap
import config_au18 as C
import au18_frame as AF
import lane_path as LP

OUT = C.FIG_DIR
os.makedirs(OUT, exist_ok=True)
SNAP = int(sys.argv[1]) if len(sys.argv) > 1 else 72
# 3.0 kpc, not the 3.5 kpc the gas-map mask inherited from the straight line.
# The lane's transverse FWHM is 6.1 kpc in all gas, but much of that width is the
# diffuse envelope: shrinking the tube from 3.5 to 2.0 kpc narrows the gas spread
# from 0.45 to 0.37 dex while leaving the gradient untouched (0.49 dex end to
# end either way), so the extra width was surroundings, not lane.  3.0 kpc keeps
# most of that gain with ~2.5x the stars of a 2 kpc tube.  R_LANE is the first
# thing to vary if a trend looks marginal.
R_LANE = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
EXT = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
R_GSE = 5.0                     # kpc, 3D, about the GS/E stellar centroid
# THE HOST APERTURE IS TIGHT ON PURPOSE.  R < 8, |z| < 2 was too generous: the
# host's star-forming gas has a 0.75 dex radial gradient (+0.28 at R < 2, -0.46
# at R = 8-12), so that box averaged the disc together with its own metal-poor
# outskirts and returned [Fe/H] = -0.17, well below the ~0.0 the gas map shows
# for the disc.  The lane reaches the disc at R = 2.9 kpc, so the reference is
# the star-forming disc inside that radius: R < 4 kpc, |z| < 1 kpc, median +0.02.
# Tightening does NOT narrow the band -- 0.55 dex at R < 4, |z| < 0.5 against
# 0.55 at R < 8, |z| < 2.  That width is not the radial gradient leaking in, it
# is the disc's own cell-to-cell inhomogeneity, the same scatter the lane shows
# and the same scatter the projected map averages away.
DR_MW, Z_MW = 1.0, 1.0
CUT, ZCUT = 0.8, 1.5
ZMAX_HI = 30.
NMIN = 20
cH = '#FF6347'          # halo-born, the house colour
cSF = '#1F6FB2'         # star-forming gas
cALL = '#2B2B2B'        # all gas
cGSE = '#8E24AA'        # the GS/E, the house colour
cMW = '#00897B'         # the host disc

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13.5, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 140, 'savefig.dpi': 200,
})

def wpct(x, w, q):
    """Mass-weighted percentiles."""
    i = np.argsort(x)
    x, w = x[i], w[i]
    cw = np.cumsum(w)
    return np.interp(np.asarray(q) / 100. * cw[-1], cw, x)


st = np.load(C.OUT_DIR + '/snapshot_times.npz')
k = int(np.flatnonzero(st['snaps'] == SNAP)[0])
T_LO, T_HI = float(st['t_snap'][k - 1]), float(st['t_snap'][k])
print(f'snapshot {SNAP}: stars with {T_LO:.3f} < t_form <= {T_HI:.3f} Gyr')

# ------------------------------- frame ---------------------------------------
sub = ap.subhalos.subfind(SNAP, directory=C.SIM_DIR,
                          loadlist=['SubhaloPos', 'Group_R_Crit200'])
r200 = float(sub.data['Group_R_Crit200'][0]); cen = sub.data['SubhaloPos'][0]
ref = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'Potential', 'Velocities'])
ref = ap.util.CentreOnHalo(ref, cen)
ref = ap.util.apply_mask(ref, stars=False, radialcut=.5 * r200)
ist, = np.where(ap.util.r(ref) < .1 * r200)
L = np.cross(ref.data['Coordinates'][ist],
             ref.data['Velocities'][ist] * ref.data['Masses'][ist, None])
Ld = L.sum(0); Ld /= np.sqrt((Ld ** 2).sum())
xd, yd, zd = ap.util.get_principal_axis(ref, ist, L=Ld)
del ref; gc.collect()

s4 = ap.snapshot.load_snapshot(SNAP, 4, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs', 'GFM_Metals'])
s4 = ap.util.CentreOnHalo(s4, cen); ap.util.rotateto(s4, xd, dir2=yd, dir3=zd)
c4 = s4.data['Coordinates']
pos = np.column_stack([c4[:, 1], c4[:, 2], c4[:, 0]]) * 1e3
sid = s4.data['ParticleIDs']
met = s4.data['GFM_Metals']
del s4, c4; gc.collect()

GSE = np.sort(np.load(C.OUT_DIR + '/gse_clean_ids.npy'))
gcen, _, _, _ = AF.gse_centroid(pos, sid, GSE)
ROT = AF.align_azimuth(gcen)
pos = pos @ ROT.T
print(f'GS/E centroid -> ({(ROT @ gcen)[0]:+.2f}, {(ROT @ gcen)[1]:+.2f}, '
      f'{(ROT @ gcen)[2]:+.2f}) kpc after the azimuth alignment')

# ------------------------- the halo-born cohort ------------------------------
a = np.load(C.OUT_DIR + '/birth_orbits_actions.npz')
zx = np.load(C.OUT_DIR + '/birth_orbits_zmax.npz')
sel = (a['tform'] > T_LO) & (a['tform'] <= T_HI) & np.isfinite(a['eps_birth'])
ids_w, eps_w, zmb_w = a['ids'][sel], a['eps_birth'][sel], zx['zmax_birth'][sel]
o = np.argsort(sid); ss = sid[o]
p = np.searchsorted(ss, ids_w)
ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == ids_w)
idx = o[p[ok]]
P, IDS = pos[idx], ids_w[ok]
eps, zmb = eps_w[ok], zmb_w[ok]
feh = C.bracket_abundance(met[idx], 'Fe', 'H')
halo = ~((eps > CUT) | (zmb < ZCUT))
print(f'born in the interval and matched: {len(IDS):,}; halo-born {halo.sum():,} '
      f'({100 * halo.mean():.1f} %)')
del a, zx, met, pos, sid; gc.collect()

# ------------------------------ the lane -------------------------------------
LX, LZ = LP.load(C.OUT_DIR, SNAP)
S_FIT = float(np.hypot(np.diff(LX), np.diff(LZ)).sum())
TAN = {}
if EXT > 0:
    # extend along the end tangents, taken over the last 1 kpc of the trace so a
    # single node's noise cannot set the direction
    n_ = max(int(round(1.0 / np.hypot(np.diff(LX), np.diff(LZ)).mean())), 2)
    for end in (0, -1):
        i0, i1 = (n_, 0) if end == 0 else (-n_ - 1, -1)
        tx, tz = LX[i1] - LX[i0], LZ[i1] - LZ[i0]
        tn = np.hypot(tx, tz); tx, tz = tx / tn, tz / tn
        TAN[end] = (LX[i1], LZ[i1], tx, tz)       # outward from this end
        k_ = int(round(EXT / (S_FIT / (len(LX) - 1))))
        d_ = np.arange(1, k_ + 1) * (S_FIT / (len(LX) - 1))
        ex, ez = LX[i1] + tx * d_, LZ[i1] + tz * d_
        LX = np.concatenate([ex[::-1], LX]) if end == 0 else np.concatenate([LX, ex])
        LZ = np.concatenate([ez[::-1], LZ]) if end == 0 else np.concatenate([LZ, ez])
    print(f'centreline extended by {EXT:g} kpc at each end along its tangents')
seg = np.hypot(np.diff(LX), np.diff(LZ))
S = np.concatenate([[0.], np.cumsum(seg)])
S = S - (S[-1] - S_FIT) / 2. if EXT > 0 else S   # s = 0 at the start of the FIT
S0_FIT, S1_FIT = 0., S_FIT
print(f'lane centreline: {len(LX)} nodes, s = {S[0]:+.2f} to {S[-1]:+.2f} kpc '
      f'(fitted part 0 to {S_FIT:.2f}), '
      f'from (x, z) = ({LX[0]:+.2f}, {LZ[0]:+.2f}) to ({LX[-1]:+.2f}, {LZ[-1]:+.2f})')

# Project onto the SEGMENTS, not onto the nodes.  Snapping each star to its
# nearest node quantises s to the 0.2 kpc node spacing, which shows up as
# vertical striping in the scatter and biases nothing but looks like structure.
ax_, az_ = LX[:-1], LZ[:-1]
dx_, dz_ = np.diff(LX), np.diff(LZ)
L2 = dx_ ** 2 + dz_ ** 2
qx = P[:, 0][:, None] - ax_
qz = P[:, 2][:, None] - az_
tt = np.clip((qx * dx_ + qz * dz_) / L2, 0., 1.)
d_seg = np.hypot(qx - tt * dx_, qz - tt * dz_)
j = np.argmin(d_seg, axis=1)
r_ = np.arange(len(P))
d_min = d_seg[r_, j]
s_of = S[j] + tt[r_, j] * np.sqrt(L2)[j]
d_3d = np.hypot(d_min, P[:, 1])

lane = halo & (d_3d < R_LANE)
lane_proj = halo & (d_min < R_LANE)
print(f'halo-born within {R_LANE:g} kpc of the centreline:')
print(f'  in 3D (the selection) : {lane.sum():,}')
print(f'  in projection only    : {lane_proj.sum():,}'
      f'  -- {lane_proj.sum() - lane.sum():,} more, the fore/background a map '
      f'cannot separate')
print(f'  |y| of the 3D sample  : 16/50/84 = '
      + ' / '.join(f'{v:+.2f}' for v in np.percentile(P[lane, 1], [16, 50, 84])) + ' kpc')

# --------------------------- z_max today -------------------------------------
zf = C.OUT_DIR + '/zmax_z0_4.5_6.5.npz'
if not os.path.exists(zf):
    raise SystemExit(f'missing {zf}; run "prep_zmax_z0.py 4.5 6.5" first')
zc = np.load(zf)
oz = np.argsort(zc['ids']); zi = zc['ids'][oz]
pz = np.searchsorted(zi, IDS)
hz = (pz < len(zi)) & (zi[np.minimum(pz, len(zi) - 1)] == IDS)
zmax0 = np.full(len(IDS), np.nan)
zmax0[hz] = zc['zmax'][oz[pz[hz]]]
m = lane & np.isfinite(zmax0)
print(f'z_max today matched for {int((lane & hz).sum()):,} of {lane.sum():,} '
      f'lane stars ({100 * (lane & hz).sum() / lane.sum():.1f} %)')
runaway = m & (zmax0 > ZMAX_HI)
print(f'  above the {ZMAX_HI:g} kpc clip: {runaway.sum():,} '
      f'({100 * runaway.sum() / m.sum():.2f} %), max {np.nanmax(zmax0[m]):.3g} kpc')

# ------------------------------- the gas -------------------------------------
# Same frame, same tube, same bins as the stars.
g = ap.snapshot.load_snapshot(SNAP, 0, snappath=C.SIM_DIR, verbose=False,
    loadlist=['Coordinates', 'Masses', 'GFM_Metals', 'StarFormationRate'])
g = ap.util.CentreOnHalo(g, cen); ap.util.rotateto(g, xd, dir2=yd, dir3=zd)
cg = g.data['Coordinates']
GP = (np.column_stack([cg[:, 1], cg[:, 2], cg[:, 0]]) * 1e3) @ ROT.T
GM = g.data['Masses'] * C.MASS_TO_MSUN
GSF = np.asarray(g.data['StarFormationRate'], float) > 0
GFE = C.bracket_abundance(g.data['GFM_Metals'], 'Fe', 'H')
del g, cg; gc.collect()

# The segment-distance array is (cells x segments) and the snapshot holds 20
# million cells, so the distances are computed ONLY inside a bounding box about
# the lane -- every cell outside it is further than R_LANE from the centreline by
# inspection.  The box is applied to the distance calculation, NOT to the cell
# arrays: the GS/E reference aperture lies outside it (the satellite centroid is
# 5.5 kpc beyond the lower end of the lane), and cutting the arrays here silently
# emptied it.
box = ((GP[:, 0] > LX.min() - R_LANE) & (GP[:, 0] < LX.max() + R_LANE)
       & (GP[:, 2] > LZ.min() - R_LANE) & (GP[:, 2] < LZ.max() + R_LANE)
       & (np.abs(GP[:, 1]) < R_LANE))
ib = np.flatnonzero(box)
print(f'gas cells in the lane bounding box: {len(ib):,} of {len(GP):,}')
qx = GP[ib, 0][:, None] - ax_
qz = GP[ib, 2][:, None] - az_
tt = np.clip((qx * dx_ + qz * dz_) / L2, 0., 1.)
d_seg = np.hypot(qx - tt * dx_, qz - tt * dz_)
jg = np.argmin(d_seg, axis=1)
rg_ = np.arange(len(ib))
sg = np.full(len(GP), np.nan)
dg = np.full(len(GP), np.inf)
sg[ib] = S[jg] + tt[rg_, jg] * np.sqrt(L2)[jg]
dg[ib] = np.hypot(d_seg[rg_, jg], GP[ib, 1])
gtube = (dg < R_LANE) & np.isfinite(GFE)
print(f'gas cells in the same tube: {gtube.sum():,} '
      f'({(gtube & GSF).sum():,} star-forming, '
      f'{100 * GM[gtube & GSF].sum() / GM[gtube].sum():.1f} % of the mass)')
del qx, qz, tt, d_seg, box; gc.collect()

# The two endpoints the lane is supposed to join, measured on the same gas, in
# 3D, and with the lane tube removed so the three samples are disjoint.
GCEN = ROT @ gcen
dgse = np.linalg.norm(GP - GCEN, axis=1)
Rg_ = np.hypot(GP[:, 0], GP[:, 1])
out_lane = dg >= R_LANE
R_LAND = float(np.hypot(LX[-1], 0.))
# THE TWO REFERENCE VOLUMES ARE THE NEXT STEP ALONG THE LANE, not apertures on
# the two objects.  Earlier versions measured "the GS/E" and "the MW disc" in
# spheres and annuli, and the answer moved every time the aperture moved: the
# host endpoint came out at +0.12, +0.04, -0.05 or -0.08 depending on whether it
# was R < 4, |R - R_land| < 2, |R - R_land| < 1 or R < 8.  Those are different
# volumes with different selections from the curve, so the curve was never going
# to meet them.
#
# Instead: step STEP kpc beyond each end of the extended centreline, along that
# end's own tangent, and take a sphere of radius R_LANE there -- the same radius
# as the tube, the same gas, the same weighting, just one step further towards
# the object.  The lane tube is excluded so the volumes are disjoint.  If the
# lane really is continuous with the satellite at one end and the disc at the
# other, these levels should sit close to the curve's ends.  That they do is a
# result; it is not built in.
STEP = 1.5
print('\nwhere to anchor the reference spheres: stepping along each end tangent')
for st_ in (0., 1.5, 3.0, 4.5):
    row = f'  step {st_:3.1f} kpc:'
    for end, nm_ in ((0, 'GS/E'), (-1, 'MW')):
        ex_, ez_, tx_, tz_ = TAN[end]
        cc = np.array([ex_ + tx_ * (EXT + st_), 0., ez_ + tz_ * (EXT + st_)])
        q = ((np.linalg.norm(GP - cc, axis=1) < R_LANE) & out_lane & GSF
             & np.isfinite(GFE))
        row += (f'   {nm_} at ({cc[0]:+5.2f},{cc[2]:+5.2f}) R={np.hypot(cc[0], cc[1]):4.1f} '
                + (f'{wpct(GFE[q], GM[q], (50,))[0]:+.3f} ({q.sum():>6,})'
                   if q.sum() >= NMIN else '   too few'))
    print(row)
REFC = {}
for end, nm_ in ((0, 'GS/E'), (-1, 'MW disc')):
    ex_, ez_, tx_, tz_ = TAN[end]
    REFC[nm_] = np.array([ex_ + tx_ * (EXT + STEP), 0., ez_ + tz_ * (EXT + STEP)])
A_GSE = (np.linalg.norm(GP - REFC['GS/E'], axis=1) < R_LANE) & out_lane & np.isfinite(GFE)
A_MW = (np.linalg.norm(GP - REFC['MW disc'], axis=1) < R_LANE) & out_lane & np.isfinite(GFE)
# WHY THE MW APERTURE MATTERS, with the numbers.  R < 8, |z| < 2 is a generous
# box: it reaches out to where the star-forming disc is already 0.3 dex poorer
# than its centre, and a 2 kpc slab admits the diffuse layer above and below the
# thin gas disc.  The map reads ~0.0 for the disc because the eye follows the
# bright dense core, which is a small part of that box by volume.  This prints
# the profile the choice is made from.
print('\nstar-forming gas of the host disc, mass-weighted [Fe/H] '
      '(lane tube excluded):')
print(f'{"R [kpc]":>9s} {"|z|<0.5":>22s} {"|z|<1":>22s} {"|z|<2":>22s}')
Rg_ = np.hypot(GP[:, 0], GP[:, 1])
for lo, hi in ((0, 2), (2, 4), (4, 6), (6, 8), (8, 12)):
    row = f'{f"{lo}-{hi}":>9s}'
    for zc_ in (0.5, 1.0, 2.0):
        q = (Rg_ >= lo) & (Rg_ < hi) & (np.abs(GP[:, 2]) < zc_) & GSF \
            & (dg >= R_LANE) & np.isfinite(GFE)
        row += (' ' + ' / '.join(f'{v:+.2f}' for v in wpct(GFE[q], GM[q], (16, 50, 84)))
                if q.sum() >= NMIN else f'{"--":>22s}')
    print(row)

print(f'\nthe lane reaches the disc at R = {R_LAND:.1f} kpc.  Disc apertures, '
      f'for context only -- the reference spheres above are what the figure draws:')
for nm_, q_ in (('R<8,  |z|<2', (Rg_ < 8) & (np.abs(GP[:, 2]) < 2)),
                ('R<6,  |z|<1', (Rg_ < 6) & (np.abs(GP[:, 2]) < 1)),
                ('R<4,  |z|<1', (Rg_ < 4) & (np.abs(GP[:, 2]) < 1)),
                (f'|R-{R_LAND:.1f}|<2,|z|<1',
                 (np.abs(Rg_ - R_LAND) < 2) & (np.abs(GP[:, 2]) < 1)),
                (f'|R-{R_LAND:.1f}|<1,|z|<1',
                 (np.abs(Rg_ - R_LAND) < 1) & (np.abs(GP[:, 2]) < 1))):
    q = q_ & GSF & (dg >= R_LANE) & np.isfinite(GFE)
    v = wpct(GFE[q], GM[q], (16, 50, 84))
    print(f'  {nm_:>14s}: {q.sum():>7,} cells, {GM[q].sum():.2e} Msun, '
          + ' / '.join(f'{x:+.3f}' for x in v)
          + f'   (16-84 width {v[2] - v[0]:.3f} dex)')

print(f'\nreference volumes: spheres of radius {R_LANE:g} kpc, '
      f'{STEP:g} kpc beyond each end of the lane along its own tangent, '
      f'lane tube excluded, SF gas:')
REF = {}
for nm, aa in (('GS/E', A_GSE), ('MW disc', A_MW)):
    q = aa & GSF
    if q.sum() < NMIN:
        print(f'  {nm:>8s}: {q.sum():,} cells -- too few')
        continue
    REF[nm] = wpct(GFE[q], GM[q], (16, 50, 84))
    print(f'  {nm:>8s} at (x, z) = ({REFC[nm][0]:+.2f}, {REFC[nm][2]:+.2f}): '
          f'{q.sum():>7,} cells, {GM[q].sum():.3e} Msun, '
          f'[Fe/H] 16/50/84 = ' + ' / '.join(f'{v:+.3f}' for v in REF[nm]))

# ------------------------------ the trends -----------------------------------
EB = np.linspace(S[0], S[-1], 13)
ctr = .5 * (EB[:-1] + EB[1:])


def prof(v, mm):
    out = np.full((len(ctr), 4), np.nan)
    for i, (lo, hi) in enumerate(zip(EB[:-1], EB[1:])):
        q = mm & (s_of >= lo) & (s_of < hi) & np.isfinite(v)
        if q.sum() >= NMIN:
            out[i] = [q.sum(), *np.percentile(v[q], [16, 50, 84])]
    return out


def gprof(mm):
    out = np.full((len(ctr), 4), np.nan)
    for i, (lo, hi) in enumerate(zip(EB[:-1], EB[1:])):
        q = mm & (sg >= lo) & (sg < hi)
        if q.sum() >= NMIN:
            out[i] = [q.sum(), *wpct(GFE[q], GM[q], (16, 50, 84))]
    return out


pf, pz_ = prof(feh, lane), prof(zmax0, m)
gsf_, gall_ = gprof(gtube & GSF), gprof(gtube)
# THE MIXING FRACTION.  Mixing is linear in metal mass fraction, not in dex, so
# f is formed from Z = 10^[Fe/H] and not from [Fe/H] itself:
#     Z_lane = f * Z_MW + (1 - f) * Z_GSE
# f = 0 is pure satellite gas, f = 1 is pure host gas.  Values above 1 are not a
# failure of the mixing picture but of the reference: the MW aperture is a median
# over R < 8 kpc, which includes metal-poor outer gas, while the lane lands on the
# disc near R = 5 kpc where the gas is richer than that median.
def mixfrac(x):
    if not {'GS/E', 'MW disc'} <= set(REF):
        return np.nan
    zg, zm = 10 ** REF['GS/E'][1], 10 ** REF['MW disc'][1]
    return (10 ** np.asarray(x) - zg) / (zm - zg)


print(f'\n{"s [kpc]":>8s} {"N*":>6s} {"stars [Fe/H] 16/50/84":>26s}'
      f' | {"SF gas [Fe/H] 16/50/84":>26s} | {"z_max,0 16/50/84":>24s}')
for i in range(len(ctr)):
    if np.isnan(pf[i, 0]):
        continue
    gg = (f'{gsf_[i,1]:>+8.3f} {gsf_[i,2]:>+8.3f} {gsf_[i,3]:>+8.3f}'
          if np.isfinite(gsf_[i, 0]) else f'{"":>26s}')
    print(f'{ctr[i]:>8.2f} {pf[i,0]:>6.0f} '
          f'{pf[i,1]:>+8.3f} {pf[i,2]:>+8.3f} {pf[i,3]:>+8.3f} | {gg} | '
          f'{pz_[i,1]:>7.2f} {pz_[i,2]:>7.2f} {pz_[i,3]:>7.2f}'
          + (f' | f = {mixfrac(gsf_[i,2]):+.2f}' if np.isfinite(gsf_[i, 0]) else ''))

# the quantities the comparison is actually about
def spread(p):
    ok_ = np.isfinite(p[:, 0])
    return np.median(p[ok_, 3] - p[ok_, 1]), p[ok_, 2].max() - p[ok_, 2].min()


for nm, p in (('stars', pf), ('SF gas', gsf_), ('all gas', gall_)):
    w, rng_ = spread(p)
    print(f'  {nm:>8s}: median 16-84 width {w:.3f} dex, '
          f'median rises {rng_:.3f} dex end to end')
gq = gtube & GSF
print(f'  SF gas [Fe/H] against s: slope '
      f'{np.polyfit(sg[gq], GFE[gq], 1)[0]:+.4f} per kpc (mass-unweighted fit), '
      f'Pearson r {np.corrcoef(sg[gq], GFE[gq])[0,1]:+.3f}')

# WHY THE MAP LOOKS SO MUCH CLEANER THAN EITHER OF THESE.
# fig_paper_gas_metallicity_sf draws one number per pixel, the mass-weighted mean
# [Fe/H] through the whole line of sight.  Averaging is what removes the scatter:
# the cells behind a single pixel span almost as much [Fe/H] as the whole lane
# does.  This compares the two directly, on the same pixels.
mf = C.OUT_DIR + f'/gas_sf_maps_snap{SNAP}.npz'
if os.path.exists(mf):
    mc = np.load(mf)
    mxe, mze, MM = mc['xe'], mc['ze'], mc['M']
    ix = np.clip(np.searchsorted(mxe, GP[gq, 0]) - 1, 0, len(mxe) - 2)
    iz = np.clip(np.searchsorted(mze, GP[gq, 2]) - 1, 0, len(mze) - 2)
    pix = MM[ix, iz]
    resid = GFE[gq] - pix
    okp = np.isfinite(resid)
    # spread of the map's own pixel values along the lane, vs cell-level spread
    key = ix.astype(np.int64) * len(mze) + iz
    uq, inv = np.unique(key[okp], return_inverse=True)
    pv = np.array([np.nanmedian(pix[okp][inv == i]) for i in range(len(uq))])
    print('\nprojection is what makes the map look clean:')
    print(f'  SF cells along the lane, 16-84 spread     {np.diff(np.percentile(GFE[gq], [16, 84]))[0]:.3f} dex')
    print(f'  the map pixels they fall in, 16-84 spread {np.diff(np.percentile(pv[np.isfinite(pv)], [16, 84]))[0]:.3f} dex')
    print(f'  cell minus its own pixel, 16-84 spread    {np.diff(np.percentile(resid[okp], [16, 84]))[0]:.3f} dex'
          f'  -- the scatter a projected map averages away')
# s is NOT independent of where the star was born: the lane runs from z = -10.7
# to z = -0.7, so s and the birth height are nearly the same quantity, and a star
# born high on a halo orbit will reach a large z_max whatever else is true of it.
# Panel (b) is therefore mostly "stars born high stay high" -- which is a result
# about the orbits surviving, not an independent trend along the lane.  These
# correlations say how degenerate it is; panel (a) is not degenerate in the same
# way, because the gradient it shows is in the gas, not in the geometry.
zb = np.abs(P[:, 2])
rb = np.linalg.norm(P, axis=1)
print('\ncorrelations within the lane sample (Pearson r):')
qq = m & np.isfinite(feh) & (zmax0 < ZMAX_HI)
for nm, u in (('s', s_of), ('|z| at birth', zb), ('r at birth', rb)):
    print(f'  {nm:>13s}  vs [Fe/H] {np.corrcoef(u[qq], feh[qq])[0,1]:+.3f}   '
          f'vs log z_max,0 {np.corrcoef(u[qq], np.log10(zmax0[qq]))[0,1]:+.3f}')
print(f'  {"s":>13s}  vs |z| at birth {np.corrcoef(s_of[qq], zb[qq])[0,1]:+.3f}')

for nm, v, mm in (('[Fe/H]', feh, lane), ('log10 z_max,0', np.log10(zmax0), m)):
    q = mm & np.isfinite(v) & (np.abs(v) < 1e3)
    sl = np.polyfit(s_of[q], v[q], 1)[0]
    r = np.corrcoef(s_of[q], v[q])[0, 1]
    print(f'{nm:>14s} against s: slope {sl:+.4f} per kpc, Pearson r {r:+.3f}, '
          f'N {q.sum():,}')

# Cache everything the paper figure needs, so fig_paper_lane_profile.py does not
# have to reload the snapshot to change a font size.
CACHE = C.OUT_DIR + f'/lane_profile_snap{SNAP}_R{R_LANE:g}.npz'
np.savez(CACHE, s_star=s_of[lane], feh_star=feh[lane],
         s_zmax=s_of[m], zmax_star=zmax0[m],
         ctr=ctr, prof_gas=gsf_, prof_zmax=pz_,
         ref_gse=REF.get('GS/E', np.full(3, np.nan)),
         ref_mw=REF.get('MW disc', np.full(3, np.nan)),
         s_lo=S[0], s_hi=S[-1], s0_fit=S0_FIT, s1_fit=S1_FIT,
         r_lane=R_LANE, ext=EXT, step=STEP, snap=SNAP, t_lo=T_LO, t_hi=T_HI,
         n_lane=int(lane.sum()), zmax_hi=ZMAX_HI)
print(f'cached the profile to {CACHE}')

# ---------------------------------- figure -----------------------------------
fig, AX = plt.subplots(1, 2, figsize=(13.0, 5.4))
# (-0.8, 0.3), not the (-0.8, 0.2) first chosen: the host reference sits at
# +0.24 and a 0.2 cap silently clipped the line out of the panel.
PAN = [(feh, lane, '[Fe/H]', None, (-0.8, 0.3)),
       (zmax0, m, r'$z_{\rm max}$ today [kpc]', 'log', (0.3, ZMAX_HI))]
for ax, (v, mm, lab, scale, ylim), tag in zip(AX, PAN, ('(a)', '(b)')):
    q = mm & np.isfinite(v)
    ax.scatter(s_of[q], v[q], s=5, c=cH, alpha=.25, lw=0, rasterized=True)
    if tag == '(b)':
        pp = prof(v, mm)
        good = np.isfinite(pp[:, 0])
        ax.fill_between(ctr[good], pp[good, 1], pp[good, 3], color='k',
                        alpha=.16, lw=0)
        ax.plot(ctr[good], pp[good, 2], color='k', lw=2.4)
    if scale:
        ax.set_yscale(scale)
    if ylim[0] is not None:
        ax.set_ylim(*ylim)
    ax.set(xlabel='$s$ along the lane [kpc]   (0 = GS/E end)', ylabel=lab,
           xlim=(S[0], S[-1]))
    if tag == '(a)':
        # ONE curve, not three.  The all-gas and the stellar running medians sat
        # on top of the star-forming one -- that agreement is the result, and it
        # is in the text and the printout; drawn, three coincident lines only
        # made the panel harder to read.  The stars are still here, as the
        # points the curve is to be compared with.
        for nm, col in (('GS/E', cGSE), ('MW disc', cMW)):
            if nm in REF:
                ax.axhline(REF[nm][1], color=col, lw=1.8, ls=':', zorder=1)
        gd = np.isfinite(gsf_[:, 0])
        ax.fill_between(ctr[gd], gsf_[gd, 1], gsf_[gd, 3], color=cSF,
                        alpha=.22, lw=0)
        ax.plot(ctr[gd], gsf_[gd, 2], color=cSF, lw=2.6,
                label='star-forming gas')
        ax.plot([], [], 'o', color=cH, ms=5, label='Stars at this snapshot')
        ax.legend(loc='lower right', handlelength=1.4, borderpad=.3,
                  labelspacing=.3, fontsize=11.5)
    # The extrapolated ends of the centreline, shaded and named for the object
    # each one runs towards.  They are where the lane stops being measured and
    # starts being an extension along its own tangent -- not a measurement of
    # either object, which is why they carry a name rather than a level.
    ax.axvspan(S[0], S0_FIT, color='.88', lw=0, zorder=0)
    ax.axvspan(S1_FIT, S[-1], color='.88', lw=0, zorder=0)
    for v_ in (S0_FIT, S1_FIT):
        ax.axvline(v_, color='.55', lw=.9, ls=(0, (4, 3)), zorder=0)
    for x_, nm_, col_ in ((.5 * (S[0] + S0_FIT), 'GS/E', cGSE),
                          (.5 * (S1_FIT + S[-1]), 'MW', cMW)):
        ax.text(x_, .5, nm_, transform=ax.get_xaxis_transform(), rotation=90,
                ha='center', va='center', fontsize=13,
                color=col_ if tag == '(a)' else '.35')
    ax.text(.035, .955, tag, transform=ax.transAxes, va='top', fontsize=16,
            fontweight='bold')
fig.suptitle(f'Halo-born stars formed within {R_LANE:g} kpc of the lane, '
             f'{T_LO:.2f} < $t$ < {T_HI:.2f} Gyr (snapshot {SNAP}); '
             f'N = {lane.sum():,}', fontsize=14, y=.98)
fig.tight_layout(rect=[0, 0, 1, .96])
f = f'{OUT}/au18_lane_born_profile.png'
fig.savefig(f, bbox_inches='tight')
print(f'\nsaved {f}')
