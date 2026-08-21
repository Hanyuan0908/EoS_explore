"""Chemistry of heated-disc and born-radial Eos channels in Au18."""
import gc, os
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.FIG_DIR,exist_ok=True); os.makedirs(C.OUT_DIR,exist_ok=True)
kin=np.load(C.OUT_DIR+'/merger_birth_vs_z0_kinematics.npz'); ids=kin['ids']

def aligned_snapshot(sn, chemistry=False):
    fields=['ParticleIDs','Coordinates','Velocities','Masses','Potential','GFM_StellarFormationTime']
    if chemistry: fields+=['GFM_Metals']
    s=snap_mod.load_snapshot(sn,4,snappath=C.SIM_DIR,loadlist=fields)
    real=s.data['GFM_StellarFormationTime']>0
    for key in list(s.data): s.data[key]=s.data[key][real]
    sf=sub_mod.subfind(sn,directory=C.SIM_DIR,loadlist=['GroupFirstSub','SubhaloPos'])
    cen=sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
    util.CentreOnHalo(s,cen); r0=np.sqrt((s.data['Coordinates']**2).sum(1)); inner=r0<.01
    bulk=np.average(s.data['Velocities'][inner],axis=0,weights=s.data['Masses'][inner])
    s.data['Velocities']-=bulk; util.align_galaxy(s,radialcut=.01)
    return s

def circularity(s):
    x=s.data['Coordinates']*1000.; v=s.data['Velocities']; r=np.sqrt((x*x).sum(1))
    R=np.hypot(x[:,1],x[:,2]); jz=x[:,1]*v[:,2]-x[:,2]*v[:,1]
    disc=(R>3)&(R<12)&(np.abs(x[:,0])<2)
    if np.median(jz[disc])<0: jz*=-1
    E=.5*(v*v).sum(1)+s.data['Potential']; valid=np.isfinite(E)&np.isfinite(jz)&(r<50)
    edges=np.quantile(E[valid],np.linspace(0,1,241)); ib=np.clip(np.searchsorted(edges,E,'right')-1,0,239)
    jc=np.full(240,np.nan)
    for b in range(240):
        q=valid&(ib==b)&(jz>0)
        if q.sum()>30: jc[b]=np.percentile(jz[q],95)
    ok=np.isfinite(jc); jc=np.interp(np.arange(240),np.flatnonzero(ok),jc[ok])
    return jz/jc[ib],r

def match(snapshot_ids,wanted):
    o=np.argsort(snapshot_ids); ss=snapshot_ids[o]; p=np.searchsorted(ss,wanted)
    ok=(p<len(ss))&(ss[np.minimum(p,len(ss)-1)]==wanted)
    return o[p[ok]],ok

# z=0 circularity, radius, chemistry, and formation time.
s0=aligned_snapshot(127,chemistry=True); ix,ok=match(s0.data['ParticleIDs'],ids)
ids=ids[ok]; eps0,r0=circularity(s0); eps_z0=eps0[ix]; rz0=r0[ix]
aform=s0.data['GFM_StellarFormationTime'][ix]; metals=s0.data['GFM_Metals'][ix]

# Assign each particle to the first available snapshot after it formed.
snaps=np.arange(73,83); snap_a=[]
for sn in snaps:
    tmp=snap_mod.load_snapshot(int(sn),4,snappath=C.SIM_DIR,loadlist=['GFM_StellarFormationTime'])
    snap_a.append(float(tmp.time)); del tmp
assigned=np.clip(np.searchsorted(np.asarray(snap_a),aform),0,len(snaps)-1)
eps_birth=np.full(len(ids),np.nan)
for k,sn in enumerate(snaps):
    rows=np.flatnonzero(assigned==k); s=aligned_snapshot(int(sn)); ep,_=circularity(s)
    ix,ok=match(s.data['ParticleIDs'],ids[rows]); eps_birth[rows[ok]]=ep[ix]
    print(f'snap {sn}: circularity recovered {ok.sum():,}/{len(rows):,}',flush=True)
    del s,ep; gc.collect()

base=(rz0>5)&(rz0<10)&np.isfinite(eps_birth)&np.isfinite(eps_z0)
heated=base&(eps_birth>.7)&(eps_z0<.5)
bornrad=base&(eps_birth<.5)&(eps_z0<.5)
print(f'base={base.sum():,}; A heated={heated.sum():,}; B born-radial={bornrad.sum():,}')
for name,m in [('A heated disc',heated),('B born radial',bornrad)]:
    print(f'{name}: med eps birth={np.median(eps_birth[m]):.2f}, z0={np.median(eps_z0[m]):.2f}, '
          f'med FeH={np.median(C.bracket_abundance(metals[m],"Fe","H")):+.2f}')

feh=C.bracket_abundance(metals,'Fe','H'); els=['C','N','O','Ne','Mg','Si']
ratios=[C.bracket_abundance(metals,e,'Fe') for e in els]
fig,axes=plt.subplots(2,6,figsize=(19,6.6),sharex=True)
groups=[('A: heated disc',heated,'#2166ac'),('B: born radial',bornrad,'#b2182b')]
union=heated|bornrad
for j,(el,y) in enumerate(zip(els,ratios)):
    good=union&np.isfinite(feh)&np.isfinite(y); lo,hi=np.percentile(y[good],[.05,99.95]); pad=.12*(hi-lo); yr=(lo-pad,hi+pad)
    for i,(label,m,color) in enumerate(groups):
        q=m&np.isfinite(feh)&np.isfinite(y)
        axes[i,j].hexbin(feh[q],y[q],gridsize=48,extent=(-4,1,*yr),bins='log',mincnt=1,cmap='magma')
        axes[i,j].set(xlim=(-4,1),ylim=yr)
        if i==0: axes[i,j].set_title(f'[{el}/Fe]')
        if i==1: axes[i,j].set_xlabel('[Fe/H]')
        if j==0: axes[i,j].set_ylabel(label+'\n[X/Fe]',color=color)
fig.suptitle(r'Au18 merger-born Eos channels at $z=0$, $5<r<10$ kpc: A $epsilon_b>0.7\rightarrow\epsilon_0<0.5$; B $epsilon_b<0.5\rightarrow\epsilon_0<0.5$')
fig.tight_layout(rect=[0,0,1,.94]); out=C.FIG_DIR+'/au18_eos_two_channels_chemistry.png'; fig.savefig(out,dpi=150)
np.savez(C.OUT_DIR+'/eos_two_channels.npz',ids=ids,eps_birth=eps_birth,eps_z0=eps_z0,r_z0=rz0,
         heated_ids=ids[heated],born_radial_ids=ids[bornrad],feh=feh,
         **{e.lower()+'fe':y for e,y in zip(els,ratios)})
print('saved',out)
