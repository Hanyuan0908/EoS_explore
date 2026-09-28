import sys, numpy as np
sys.path.insert(0,'eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
c=Cuts(); cat=load_catalog('data_repro/our_apogee_dr17_lite_ann.fits.gz'); m=make_masks(cat,c)
base=np.asarray(m['base'],bool)
feh=np.asarray(cat['fe_h'],float); mg=np.asarray(cat['mg_fe'],float); al=np.asarray(cat['al_fe'],float)
nfe=np.asarray(cat['n_fe'],float); nerr=np.asarray(cat['n_fe_err'],float)
lz=np.asarray(cat['lz'],float); rap=np.asarray(cat['rap'],float); rperi=np.asarray(cat['rperi'],float)
vphi=np.asarray(cat['galvt'],float)
with np.errstate(invalid='ignore'): ecc=(rap-rperi)/(rap+rperi)
halo=base&((ecc>0.7)|(lz<0))
def acc(f): return c.slope_acc*f+c.inter_acc
def hl(f): return c.slope_acc2*f+c.inter_acc2
def divline(f): return 0.317*f+0.353
lowa=halo&(feh>-0.9)&(feh<-0.2)&(mg>acc(feh))&(mg<hl(feh))&(al>c.alfe_cut)
eos_mp=lowa&(mg>divline(feh)); eos_mr=lowa&(mg<=divline(feh))
disc=np.asarray(m['thin_al'],bool)&(vphi>150); disc_bl=disc&(feh>-0.8)&(feh<-0.2)

def both(v,e):
    o=np.isfinite(v)&np.isfinite(e); v,e=v[o],e[o]
    if v.size<10: return v.size,np.nan,np.nan,np.nan,np.nan
    raw_mad=1.4826*np.median(np.abs(v-np.median(v)))
    raw_std=np.std(v)
    err2=np.mean(e**2)
    dec_mad=np.sqrt(max(raw_mad**2-err2,0))
    dec_std=np.sqrt(max(raw_std**2-err2,0))
    return v.size,raw_mad,raw_std,dec_mad,dec_std

def run(name,br,edges):
    print(f"\n=== {name} ===")
    print(f"  {'bin':>14s} {'n':>5s} {'MADraw':>7s} {'STDraw':>7s} | {'MADdec':>7s} {'STDdec':>7s}")
    for i in range(len(edges)-1):
        b=br&(feh>=edges[i])&(feh<edges[i+1])
        n,rm,rs,dm,ds=both(nfe[b],nerr[b])
        lbl=f"{edges[i]:.2f}..{edges[i+1]:.2f}"
        if n>=10:
            print(f"  {lbl:>14s} {n:5d} {rm:7.3f} {rs:7.3f} | {dm:7.3f} {ds:7.3f}")
        else:
            print(f"  {lbl:>14s} {n:5d}   (skip)")

run("Eos metal-poor",eos_mp,np.linspace(-0.9,-0.5,4))
run("Eos metal-rich",eos_mr,np.linspace(-0.8,-0.2,4))
run("low-a disc",disc_bl,np.linspace(-0.8,-0.2,4))

# Aurora
aur=base&(feh>-1.5)&(feh<-1.0)&(vphi<100)&(al>c.alfe_cut)
n,rm,rs,dm,ds=both(nfe[aur],nerr[aur])
print(f"\n=== Aurora (base & -1.5<feh<-1 & vphi<100 & al>alfe_cut) ===")
print(f"  n={n}  MADraw={rm:.3f} STDraw={rs:.3f} | MADdec={dm:.3f} STDdec={ds:.3f}")
print(f"  (overview uses MADdec -> ~{dm:.3f};  ndispersion hard-codes 0.149)")
