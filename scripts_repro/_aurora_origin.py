import sys, numpy as np
sys.path.insert(0,'eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
c=Cuts(); cat=load_catalog('data_repro/our_apogee_dr17_lite_ann.fits.gz'); m=make_masks(cat,c)
base=np.asarray(m['base'],bool); thick=np.asarray(m['thick_al'],bool)
feh=np.asarray(cat['fe_h'],float); al=np.asarray(cat['al_fe'],float)
nfe=np.asarray(cat['n_fe'],float); nerr=np.asarray(cat['n_fe_err'],float); vphi=np.asarray(cat['galvt'],float)
def stats(sel):
    v=nfe[sel]; e=nerr[sel]; o=np.isfinite(v)&np.isfinite(e); v,e=v[o],e[o]
    if v.size<8: return v.size,np.nan,np.nan,np.nan,np.nan,np.nan
    mad=1.4826*np.median(np.abs(v-np.median(v))); std=np.std(v); err2=np.mean(e**2)
    p95,p5=np.percentile(v,[95,5])
    return v.size, mad, np.sqrt(max(mad**2-err2,0)), std, np.sqrt(max(std**2-err2,0)), p95-p5
cands={
 'base & al>cut & vphi<100 (overview)': base&(feh>-1.5)&(feh<-1)&(vphi<100)&(al>c.alfe_cut),
 'base NO al   & vphi<100'           : base&(feh>-1.5)&(feh<-1)&(vphi<100),
 'thick_al     & vphi<100'           : thick&(feh>-1.5)&(feh<-1)&(vphi<100),
 'thick_al     & vphi<75'            : thick&(feh>-1.5)&(feh<-1)&(vphi<75),
 'base & al>cut& vphi<75'            : base&(feh>-1.5)&(feh<-1)&(vphi<75)&(al>c.alfe_cut),
 'base & al>cut& vphi<100 feh<-1.3'  : base&(feh>-1.5)&(feh<-1.3)&(vphi<100)&(al>c.alfe_cut),
 'thick_al & vphi<100 feh -1.2..-1'  : thick&(feh>-1.2)&(feh<-1)&(vphi<100),
}
print(f"{'selection':40s} {'n':>5s} {'MADraw':>7s} {'MADdec':>7s} {'STDraw':>7s} {'STDdec':>7s} {'P95-P5':>7s}")
for k,s in cands.items():
    n,mr,md,sr,sd,p=stats(s)
    print(f"{k:40s} {n:5d} {mr:7.3f} {md:7.3f} {sr:7.3f} {sd:7.3f} {p:7.3f}")
