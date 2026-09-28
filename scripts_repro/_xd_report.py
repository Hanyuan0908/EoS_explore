import sys, numpy as np
sys.path.insert(0,'eos-figures')
from eos_figures.data import load_catalog, make_masks
from eos_figures.config import Cuts
xd=np.load('data_repro/amr_xd_cache.npz',allow_pickle=True)
c=Cuts(); cat=load_catalog('data_repro/our_apogee_dr17_lite_ann.fits.gz'); m=make_masks(cat,c)
feh=np.asarray(cat['fe_h'],float); mg=np.asarray(cat['mg_fe'],float); vphi=np.asarray(cat['galvt'],float)
age=np.asarray(cat['age'],float); aerr=np.asarray(cat['age_model_error'],float)
al=np.asarray(cat['al_fe'],float); lz=np.asarray(cat['lz'],float)
rap=np.asarray(cat['rap'],float); rperi=np.asarray(cat['rperi'],float)
with np.errstate(invalid='ignore'): ecc=(rap-rperi)/(rap+rperi)
base=np.asarray(m['base'],bool); thin_al=np.asarray(m['thin_al'],bool); thick_al=np.asarray(m['thick_al'],bool)
rel_ok=np.isfinite(age)&np.isfinite(aerr)&(aerr/age<0.3)
halo=base&((ecc>0.7)|(lz<0))
eos=halo&(feh>-0.9)&(feh<-0.2)&(mg>c.slope_acc*feh+c.inter_acc)&(mg<c.slope_acc2*feh+c.inter_acc2)&(al>c.alfe_cut)
lowa=thin_al&(vphi>150); higha=thick_al&(vphi>150); splash=thick_al&(vphi<80)
sel={'eos':eos,'splash':splash,'lowa':lowa,'higha':higha}

def moments(name):
    a=xd[f'{name}_alpha']; mu=xd[f'{name}_mu']; V=xd[f'{name}_V']
    # marginal age (linear) mean & std from log-age GMM via sampling the mixture analytically
    # mean_age = sum a_k exp(mu_k+0.5 s_k^2); use lognormal moments per comp
    s2=V[:,0,0]; mln=mu[:,0]
    m1=np.sum(a*np.exp(mln+0.5*s2))
    m2=np.sum(a*np.exp(2*mln+2*s2))
    std_age=np.sqrt(max(m2-m1**2,0))
    return m1,std_age

print(f"{'pop':8s} {'obs_age_mean':>12s} {'obs_age_std':>11s} {'XD_age_mean':>11s} {'XD_age_std':>10s}")
for k,s in sel.items():
    p=s&rel_ok&np.isfinite(age)
    om=np.mean(age[p]); osd=np.std(age[p])
    xm,xsd=moments(k)
    print(f"{k:8s} {om:12.2f} {osd:11.2f} {xm:11.2f} {xsd:10.2f}")
