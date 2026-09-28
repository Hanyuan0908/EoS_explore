import sys, numpy as np
sys.path.insert(0,'eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
c=Cuts(); cat=load_catalog('data_repro/our_apogee_dr17_lite_ann.fits.gz'); m=make_masks(cat,c)
base=np.asarray(m['base'],bool); thick_al=np.asarray(m['thick_al'],bool)
feh=np.asarray(cat['fe_h'],float); mg=np.asarray(cat['mg_fe'],float); al=np.asarray(cat['al_fe'],float)
nfe=np.asarray(cat['n_fe'],float); nerr=np.asarray(cat['n_fe_err'],float); vphi=np.asarray(cat['galvt'],float)
al_insitu = al > c.alfe_cut
mg_thick  = mg > c.slope_acc2*feh + c.inter_acc2   # high-alpha (above the diagonal divider)
boxcut = (feh>-1.5)&(feh<-1.0)&(vphi<100)
def s(sel):
    v=nfe[sel]; e=nerr[sel]; o=np.isfinite(v)&np.isfinite(e); v,e=v[o],e[o]
    if v.size<8: return v.size,np.nan,np.nan
    mad=1.4826*np.median(np.abs(v-np.median(v))); err2=np.mean(e**2)
    return v.size, mad, np.sqrt(max(mad**2-err2,0))
box_all   = base & al_insitu & boxcut                    # overview's "Aurora"
box_high  = base & al_insitu & mg_thick & boxcut         # high-alpha part (true Aurora)
box_low   = base & al_insitu & (~mg_thick) & boxcut      # low-alpha in-situ part (NOT Aurora)
box_thick = thick_al & boxcut                            # ndispersion's Aurora mask
for nm,sel in [('base&Al (overview 0.178)',box_all),
               ('  -> high-a part (Aurora)',box_high),
               ('  -> low-a  part (not Aurora)',box_low),
               ('thick_al (ndisp 0.149)',box_thick)]:
    n,mr,md=s(sel); print(f"{nm:34s} n={n:5d}  MADraw={mr:.3f}  MADdec={md:.3f}")
print(f"\nfraction of overview box that is LOW-alpha: {box_low.sum()}/{box_all.sum()} = {box_low.sum()/box_all.sum():.0%}")
