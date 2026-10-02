"""Animation: the Au18 box through the GS/E merger, with both angular momenta drawn.

The visual check on diag_gse_orientation.py.  That script reduces the geometry to
angles; this one shows the geometry itself, so the angles can be believed or not.

A 150 kpc box, centred on the main halo, seen from a FIXED arbitrary direction in
the simulation frame -- fixed so that everything that moves on screen is the
galaxy moving, not the camera.  Nothing is rotated into the disc frame.

  grey            all stars of the box (surface density, log scale)
  violet points   the clean GS/E debris
  blue arrow      L_disc, the angular momentum of the stars inside 10 kpc
  violet arrow    L_gse, the running angular momentum of the debris
  dashed arrow    the GS/E infall axis, fixed at its t = T_INFALL value -- the
                  direction the satellite actually came in on, before the
                  encounter scrambled the running value

Arrows are unit vectors scaled to a fixed length and then projected, so one
pointing at the camera looks short.  That foreshortening is why the printed
angles matter more than the picture: the panel lists each vector's angles to the
box axes (x, y, z), as a full description of where it points, and the single
angle between the two vectors, which is the frame-independent answer to whether
they are aligned.

Pass 1 loads the snapshots and caches the projected frames; `render` re-renders
from that cache without touching the snapshots, so the camera can be changed
cheaply.

Writes out/gse_movie_frames.npz, figures/au18_gse_orientation_movie.gif and the
same animation as .mp4.  There is no system ffmpeg on this machine; the mp4 comes
from the static binary bundled with the `imageio-ffmpeg` package, which
matplotlib is pointed at below.  If that package is absent the mp4 is skipped
with a message and the gif is still written.
"""
import gc, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import config_au18 as C

CACHE = C.OUT_DIR + '/gse_movie_frames.npz'
SNAPS = list(range(58, 96))
W = 75.0                      # half-width [kpc]; the box is 150 kpc across
NBIN = 300
RIN = 0.010                   # 10 kpc, the disc aperture, in snapshot units
T_INFALL = 4.0
CGSE, CDISC = '#8E24AA', '#1F6FB2'

# A fixed, arbitrary camera.  n is the viewing direction; e1, e2 span the screen.
VIEW = np.array([0.42, -0.66, 0.62])
n = VIEW / np.linalg.norm(VIEW)
e1 = np.cross(n, [0., 0., 1.]); e1 /= np.linalg.norm(e1)
e2 = np.cross(e1, n)
CAM = np.array([e1, e2, n])                 # rows: screen-x, screen-y, depth


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-30)


def angle(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(u), unit(v)), -1., 1.))))


if 'render' not in sys.argv[1:]:
    from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util
    GSE_IDS = np.load(C.OUT_DIR + '/gse_clean_ids.npy')
    H, GX, GY, TT, LD, LG = [], [], [], [], [], []
    ng = []
    for sn in SNAPS:
        s = snap_mod.load_snapshot(sn, 4, snappath=C.SIM_DIR,
            loadlist=['ParticleIDs', 'Coordinates', 'Velocities', 'Masses',
                      'GFM_StellarFormationTime'])
        a = float(s.time)
        real = s.data['GFM_StellarFormationTime'] > 0
        for k in list(s.data):
            s.data[k] = s.data[k][real]
        sf = sub_mod.subfind(sn, directory=C.SIM_DIR,
                             loadlist=['GroupFirstSub', 'SubhaloPos'])
        util.CentreOnHalo(s, sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])])
        x = np.asarray(s.data['Coordinates'], float) * 1000.       # kpc
        v = np.asarray(s.data['Velocities'], float)
        m = np.asarray(s.data['Masses'], float)
        r = np.sqrt((x ** 2).sum(1))
        inner = r < RIN * 1000.
        v = v - np.average(v[inner], axis=0, weights=m[inner])
        Ld = np.cross(x[inner], v[inner] * m[inner, None]).sum(0)

        sid = np.asarray(s.data['ParticleIDs'])
        o = np.argsort(sid); ss = sid[o]
        p = np.searchsorted(ss, GSE_IDS)
        ok = (p < len(ss)) & (ss[np.minimum(p, len(ss) - 1)] == GSE_IDS)
        ig = o[p[ok]]
        Lg = np.cross(x[ig], v[ig] * m[ig, None]).sum(0)

        # Project into the fixed camera frame and keep only the box.
        sc = x @ CAM.T
        box = (np.abs(sc[:, 0]) < W) & (np.abs(sc[:, 1]) < W) & (np.abs(sc[:, 2]) < W)
        h = np.histogram2d(sc[box, 0], sc[box, 1], bins=NBIN,
                           range=[[-W, W], [-W, W]], weights=m[box])[0]
        gsc = sc[ig]
        gbox = (np.abs(gsc) < W).all(1)
        H.append(h.astype(np.float32))
        GX.append(gsc[gbox, 0].astype(np.float32))
        GY.append(gsc[gbox, 1].astype(np.float32))
        TT.append(float(C.a_to_age(a))); LD.append(Ld); LG.append(Lg)
        ng.append(int(gbox.sum()))
        print(f'  snap {sn:3d}  t={TT[-1]:5.2f} Gyr  GS/E in box {ng[-1]:6,d}  '
              f'theta(disc,gse)={angle(Ld, Lg):6.1f}', flush=True)
        del s, x, v, m, sc; gc.collect()

    # Ragged GS/E arrays: stored flat with offsets.
    np.savez(CACHE, hist=np.array(H), time=np.array(TT), L_disc=np.array(LD),
             L_gse=np.array(LG), n_gse=np.array(ng),
             gx=np.concatenate(GX), gy=np.concatenate(GY),
             off=np.concatenate([[0], np.cumsum(ng)]), cam=CAM, W=W)
    print('\nsaved', CACHE)

d = np.load(CACHE)
hist, t, LD, LG, off = d['hist'], d['time'], d['L_disc'], d['L_gse'], d['off']
gx, gy, CAM, W = d['gx'], d['gy'], d['cam'], float(d['W'])
# The satellite's distance, for the frames where it is outside the box: taken
# from the diag_gse_orientation.py cache rather than recomputed, since that run
# covers the same snapshots.  Without it the early frames look like the debris is
# missing, when in fact it is 200 kpc away and a 150 kpc box cannot hold it.
try:
    _o = np.load(C.OUT_DIR + '/gse_orientation.npz')
    RCOM = np.interp(t, _o['time'], _o['r_com'])
except FileNotFoundError:
    RCOM = np.full(len(t), np.nan)

INF = int(np.argmin(np.abs(t - T_INFALL)))
L_INF = LG[INF]
ARR = 0.62 * W                                   # arrow length on screen [kpc]
VMAX = float(np.nanpercentile(hist[hist > 0], 99.9))
VMIN = VMAX / 3e3

fig, ax = plt.subplots(figsize=(7.4, 7.4))
fig.subplots_adjust(left=.09, right=.98, top=.93, bottom=.08)


def arrow(vec, color, ls='-', lw=2.4, label=None):
    s = unit(vec) @ CAM.T                        # screen x, y and depth
    return ax.annotate('', xy=(ARR * s[0], ARR * s[1]), xytext=(0, 0),
                       arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                       linestyle=ls, shrinkA=0, shrinkB=0,
                                       mutation_scale=18))


def draw(k):
    ax.clear()
    ax.pcolormesh(np.linspace(-W, W, hist.shape[1] + 1),
                  np.linspace(-W, W, hist.shape[2] + 1),
                  np.where(hist[k] > 0, hist[k], np.nan).T, cmap='Greys',
                  norm=LogNorm(vmin=VMIN, vmax=VMAX), rasterized=True)
    ax.scatter(gx[off[k]:off[k + 1]], gy[off[k]:off[k + 1]], s=1.6, c=CGSE,
               alpha=.30, lw=0, rasterized=True)
    arrow(L_INF, CGSE, ls='--', lw=2.0)
    arrow(LG[k], CGSE, lw=2.8)
    arrow(LD[k], CDISC, lw=2.8)
    ax.plot(0, 0, '+', color='k', ms=10, mew=1.5)

    ax_ = lambda v, i: angle(v, np.eye(3)[i])
    txt = (f'$t$ = {t[k]:5.2f} Gyr\n'
           f'angles to the box axes (x, y, z):\n'
           f'  disc  L : {ax_(LD[k],0):5.1f}, {ax_(LD[k],1):5.1f}, {ax_(LD[k],2):5.1f}\n'
           f'  GS/E  L : {ax_(LG[k],0):5.1f}, {ax_(LG[k],1):5.1f}, {ax_(LG[k],2):5.1f}\n'
           f'  GS/E infall: {ax_(L_INF,0):5.1f}, {ax_(L_INF,1):5.1f}, {ax_(L_INF,2):5.1f}\n'
           r'$\bf{\theta}$'
           f'(disc, GS/E infall) = {angle(LD[k], L_INF):5.1f}'
           '°\n'
           f'  (disc, GS/E running) = {angle(LD[k], LG[k]):5.1f}°')
    ax.text(.02, .98, txt, transform=ax.transAxes, va='top', ha='left', fontsize=10.5,
            family='monospace',
            bbox=dict(fc='white', ec='.7', alpha=.88, pad=4))
    nin = off[k + 1] - off[k]
    note = (f'GS/E centre at $r$ = {RCOM[k]:.0f} kpc'
            + ('  (outside the box)' if nin < 50 else f';  {nin:,} stars in frame'))
    # Bottom LEFT: the legend has the bottom-right corner.
    ax.text(.02, .015, note, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10.5, color=CGSE,
            bbox=dict(fc='white', ec='none', alpha=.85, pad=2))
    for lab, c, ls, y in [('disc $L$', CDISC, '-', .12),
                          ('GS/E $L$ (running)', CGSE, '-', .08),
                          ('GS/E infall axis', CGSE, '--', .04)]:
        ax.plot([], [], color=c, ls=ls, lw=2.4, label=lab)
    ax.legend(loc='lower right', fontsize=10, framealpha=.9)
    ax.set(xlim=(-W, W), ylim=(-W, W), aspect='equal',
           xlabel='screen x [kpc]', ylabel='screen y [kpc]')
    ax.set_title('Au18, 150 kpc box, fixed arbitrary view: '
                 'do the two angular momenta align?', fontsize=11.5)


anim = FuncAnimation(fig, draw, frames=len(t), interval=320)
os.makedirs(C.FIG_DIR, exist_ok=True)
out = C.FIG_DIR + '/au18_gse_orientation_movie.gif'
anim.save(out, writer=PillowWriter(fps=3))
print('saved', out)

# The mp4 runs faster than the gif -- 6 fps rather than 3 -- because a video
# player can be paused and stepped, which a gif in a browser cannot.
try:
    import imageio_ffmpeg
    matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = out.replace('.gif', '.mp4')
    anim.save(mp4, writer=FFMpegWriter(fps=6, bitrate=3200,
                                       extra_args=['-pix_fmt', 'yuv420p']))
    print('saved', mp4)
except ImportError:
    print('imageio-ffmpeg not installed: mp4 skipped, gif written')
print(f'\ncamera n = {CAM[2].round(3)}, screen x = {CAM[0].round(3)}, '
      f'screen y = {CAM[1].round(3)}')
print(f'GS/E infall axis fixed at t = {t[INF]:.2f} Gyr')
for x in (3.5, 4.5, 5.0, 5.5, 6.5, 7.5):
    k = int(np.argmin(np.abs(t - x)))
    print(f'  t={t[k]:5.2f}  theta(disc, infall)={angle(LD[k], L_INF):6.1f}  '
          f'theta(disc, running)={angle(LD[k], LG[k]):6.1f}')

