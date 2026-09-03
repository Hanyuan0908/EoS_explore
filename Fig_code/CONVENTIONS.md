# Publication figure conventions

Read this before making or editing any figure in `../Fig_paper/`.  It exists so a
fresh session can match the existing figures without re-deriving the choices, and
without repeating the mistakes listed at the bottom -- several of which produced
figures that looked fine and were wrong.

Run everything with `/data/hz420-2/astro312/bin/python` -- **except the `gastro/`
figures**, whose `gastro_config.py` imports pynbody, which that environment does
not have.  Those need
`/data/ioasoft/software/miniforge3/envs/python-3.11-2026-01a/bin/python3`.

---

## Workflow

1. Write the figure script in `auriga/` (or `gastro/`), named `fig_paper_*.py`.
2. Output **both** PDF and PNG to `../Fig_paper/`, same basename.
3. Copy the script into `Fig_code/<figure_name>/` and add a row to
   `Fig_code/README.md`.  Symlink shared prep scripts rather than copying them,
   so two figures on one chain cannot drift apart.
4. Re-sync the archived copy after every edit -- `Fig_code` is a frozen copy, not
   a live link.

**Clear the context after finishing a figure.**  Every turn re-reads the whole
conversation, and figure PNGs read back during iteration stay in it permanently
(~5k tokens each, re-read on every later turn).  Finishing a figure is the
natural point to reset.  Do not clear mid-iteration on one figure -- the
in-flight decisions are not written down anywhere.

---

## Style

```python
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman', 'Liberation Serif',
                   'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 13.5, 'axes.labelsize': 15,
    'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 12.5,
    'axes.linewidth': 1.0, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True, 'legend.frameon': False,
    'xtick.major.size': 5, 'ytick.major.size': 5,
    'figure.dpi': 150, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
```

- **Times New Roman is not installed here.**  Nimbus Roman and Liberation Serif
  are metric-compatible clones, so the output is Times either way; Times New
  Roman is listed first in case the real font ever appears.
- **No panel titles and no suptitle** unless the figure specifically calls for
  them.  Use bold `(a)`, `(b)`, ... tags at fontsize 16, placed in a corner that
  is actually empty -- check the rendered figure, do not assume.
### Always save both formats

Every figure writes **PDF and PNG**, same basename, in one loop -- the PDF for the
manuscript, the PNG for looking at and for pasting into messages:

```python
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}/name.{ext}', bbox_inches='tight')
```

### Rasterize everything data-bearing

Pass `rasterized=True` to every artist that draws data.  Axes, ticks, text,
legends and thin annotation lines stay vector, so the PDF stays sharp and
selectable while the heavy content stops it ballooning.  A 2D histogram of a
million points or a 30,000-point scatter left vector produces a PDF that is tens
of MB and that some viewers will not open.

| Artist | Rasterize |
|---|---|
| `pcolormesh`, `imshow`, `hist2d` | **yes, always** -- the biggest offender |
| `scatter` | **yes, always**, even at a few thousand points |
| `fill_between`, `contourf` | yes |
| `plot` of a KDE / a few hundred points | no -- keep vector |
| `contour` line sets | no -- keep vector, they are few and want to be crisp |
| axes, text, legends, median lines | no |

`hist2d` returns the mesh as its 4th element, so rasterize via
`ax.hist2d(...)[3].set_rasterized(True)` or use `pcolormesh` directly.
- `pdf.fonttype: 42` embeds TrueType so preflight tools do not complain.
- Annotations that sit over data need `bbox=dict(fc='white', ec='none',
  alpha=.8, pad=2)` or they become unreadable.

---

## Colours -- keep these consistent across figures

| Meaning | Colour | Notes |
|---|---|---|
| born-hot / halo-born | `#FF6347` tomato | density maps use the custom `TOMATO` sequential map defined in `fig_paper_birth_positions.py` -- **not** `Oranges` or `Reds` |
| born-cold / disc-born | `#1F6FB2` blue | also `Blues` |
| total / all stars | `#2B2B2B` near-black, or `Greys` | |
| GS/E marker (pericentre line, debris contour) | `#8E24AA` violet | must not be any warm red or orange -- clashes with born-hot |
| disc spin-up marker | `#00897B` teal | |
| kinematic cut lines (Eos band, birth split) | `#E8112D` red | warm, so a cool map cannot swallow it |
| gas surface density | `Greys`, capped ~0.7-0.82 | so overplotted points stay visible |
| gas metallicity | `viridis` | see the diverging-map warning below |

Terminology: **born-cold** and **born-hot**, hyphenated, in figures and text.

---

## Data provenance

The default sample is the **original merger window, t_form = 4.99-6.54 Gyr**,
via `eos_origins.py`.  A figure using it must reproduce these counts exactly --
check them, they are the fastest way to catch a wrong sample:

| | N |
|---|---|
| merger-born (parent) | 171,826 |
| Eos-like (\|v_phi\|<80, ecc>0.6) | 7,583 |
| born-hot (v_phi,birth < 150) | 4,283 |
| born-cold (v_phi,birth >= 150) | 3,300 |

`eos_origins_window.py` gives the same quantities for any other window; it
reproduces the numbers above to <1 per cent when handed the original window.
The retimed 4.7-5.7 Gyr exploration lives in the `*_window.py` scripts and is a
separate line of work -- do not mix them.

Birth orbits (eps, J_r, J_z, z_max) come from `prep_birth_actions.py` /
`prep_zmax.py` against the AGAMA CylSpline potentials.  See
`au18_birth_orbits/METHOD_zmax_from_Jz.md` before touching z_max.

**The scientific results these figures show, with their caveats and the open
questions, are in `../auriga/FINDINGS.md`.**  Read it before writing a caption.

---

## Mistakes already made -- do not repeat

**KDE bandwidth is a sigma, not a bin width.**  Setting `sigma = 0.15 Gyr` to
"match" a 0.15 Gyr histogram smooths over 2.4x too much (FWHM = 0.35 Gyr) and
silently halved a peak ratio, 0.54 -> 0.35.  The equivalent sigma for a top-hat
of width h is h/sqrt(12).  Use sigma = 0.05 Gyr against a 0.15 Gyr histogram, and
verify by comparing peak heights against the unsmoothed version.  A KDE can
integrate to exactly the right mass and still be wrong about every peak.

**Frame handedness.**  `ap.util.rotateto` puts the disc axis on component 0.
Mapping it to z with `(c[:,2], c[:,1], c[:,0])` -- which
`compute_auriga_potential.py` does -- is a *transposition*, determinant -1, a
reflection.  Harmless for fitting a density, fatal for kinematics: it negates
L_z and puts the whole disc at eps = -1.  Use the cyclic `(c[:,1], c[:,2],
c[:,0])`, determinant +1.  (Fixing an already-computed file is exact: negate
eps, L_z, J_phi; J_r, J_z, E, R, z are invariant.)

**Edge-on views need the azimuth pinned.**  The disc axis is well defined, but
the rotation about it comes from near-degenerate in-plane principal axes and
points somewhere different at every snapshot.  At snapshot 73 that left the GS/E
at |y|/r = 0.59, so an x-z view showed a satellite at r = 34 kpc only 28 kpc out.
Use `au18_frame.align_azimuth` to rotate until the feature of interest lies in
the projection plane -- and skip it once the debris is phase-mixed, where its
centroid is at the origin and the implied azimuth is noise.

**Do not percentile-stretch a colour scale over a field dominated by
background.**  The gas frame is ~96 per cent diffuse metal-poor material spanning
2 dex, while the disc-to-GS/E transition is 0.4 dex -- a 2-98 percentile stretch
put everything of interest in the top 20 per cent of the bar.  Scale to the
feature being shown, and say so in the caption.  Likewise `vmin` far below the
median renders the whole frame as speckle.

**A sequential map needs its nodes placed, not just its colours picked.**
Building the tomato map with six evenly-spaced stops put tomato at the midpoint,
so the dense merger panel of `au18_birth_positions` sat almost entirely in the
saturated half and read as blood-red -- the look that was rejected once already.
Placing tomato at 0.72 and giving the pale end most of the range keeps the same
hue while the bulk of the panel stays light.  Judge a sequential map on the
figure that uses it most heavily, not on the colourbar.

**Diverging colour maps imply a midpoint.**  `RdYlBu_r` on gas [Fe/H] made the
pale band near -0.45 read as a boundary between two regimes when it is just a
point on a continuous gradient.  Use `viridis` for quantities with no natural
centre.

**A number annotated on a map will be read as describing what is under it.**
Region medians drawn on the gas metallicity map said -0.29 for the disc while the
visible yellow core was ~0.0 -- both correct, because the label was an unweighted
median over a mask containing far more faint pixels than bright ones.  Either
state the weighting or leave the numbers to the text.

**Per-panel normalisation must be disclosed.**  Where panels differ in N by more
than ~20x a shared colour scale blanks the sparse ones, so each is scaled to its
own peak -- which means colour shows *shape*, not abundance.  Say it in the
caption and annotate N per panel.  Expressing the bar as "fraction of panel peak"
makes one bar honest for a whole column.

**Equal aspect needs square bins.**  With `aspect='equal'` and a uniform bin
count over unequal ranges the pixels come out stretched.  Match the bin counts to
the ranges (e.g. 120 x 105 over 800 x 700 km/s = 6.67 km/s both ways).

**Blank the empty cells.**  Below a stated surface-density threshold, set NaN
rather than plotting near-zero.  Quote the threshold and what fraction of pixels
and of mass survive it; check the region statistics do not shift when you apply
it (at 1.5e7 Msun/kpc^2 they do not, at 3e7 they do).

**Check the rendered figure, not just the code.**  White text on a white
background, a `:.0e` format printing "2e+07" for a 1.5e7 threshold, a stale
hard-coded "t = 4.99 Gyr" on a snapshot-73 figure, panel labels buried under the
data -- none of these raise an error.
