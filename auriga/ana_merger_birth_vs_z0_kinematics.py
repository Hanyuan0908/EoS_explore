"""Birth and z=0 kinematics of post-spin-up host stars formed during the merger."""
import gc, os
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod, util

os.makedirs(C.FIG_DIR,exist_ok=True); os.makedirs(C.OUT_DIR,exist_ok=True)
ids=np.load(C.OUT_DIR+'/merger_epoch_z0_samples.npz')['host_during']

def align_and_cyl(sn):
    s=snap_mod.load_snapshot(sn,4,snappath=C.SIM_DIR,
        loadlist=['ParticleIDs','Coordinates','Velocities','Masses','GFM_StellarFormationTime'])
    real=s.data['GFM_StellarFormationTime']>0
    for k in list(s.data): s.data[k]=s.data[k][real]
    sf=sub_mod.subfind(sn,directory=C.SIM_DIR,loadlist=['GroupFirstSub','SubhaloPos'])
    cen=sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
    util.CentreOnHalo(s,cen); rr=np.sqrt((s.data['Coordinates']**2).sum(1)); q=rr<.01
    bulk=np.average(s.data['Velocities'][q],axis=0,weights=s.data['Masses'][q])
    s.data['Velocities']-=bulk; util.align_galaxy(s,radialcut=.01)
    x=s.data['Coordinates']*1000.; v=s.data['Velocities']; R=np.hypot(x[:,1],x[:,2])
    vR=(x[:,1]*v[:,1]+x[:,2]*v[:,2])/np.where(R>.1,R,1)
    vp=(x[:,1]*v[:,2]-x[:,2]*v[:,1])/np.where(R>.1,R,1)
    if np.median(vp[(R>3)&(R<12)&(np.abs(x[:,0])<2)])<0: vp*=-1
    return s,s.data['ParticleIDs'],vR,vp,R,x[:,0]

# Formation interval represented by each snapshot 73..82. Assign every star to
# the first stored snapshot at or after its formation scale factor.
snapshots=np.arange(73,83)
ascale=[]
for sn in snapshots:
    tmp=snap_mod.load_snapshot(sn,4,snappath=C.SIM_DIR,loadlist=['GFM_StellarFormationTime'])
    ascale.append(float(tmp.time)); del tmp
ascale=np.asarray(ascale)

# Obtain exact formation factors at z=0 and assign IDs to birth snapshots.
s0=snap_mod.load_snapshot(127,4,snappath=C.SIM_DIR,
    loadlist=['ParticleIDs','GFM_StellarFormationTime'])
o=np.argsort(s0.data['ParticleIDs']); ss=s0.data['ParticleIDs'][o]
p=np.searchsorted(ss,ids); ok=(p<len(ss))&(ss[np.minimum(p,len(ss)-1)]==ids)
ids=ids[ok]; aform=s0.data['GFM_StellarFormationTime'][o[p[ok]]]; del s0
assigned=np.clip(np.searchsorted(ascale,aform),0,len(snapshots)-1)

bid=[]; bvr=[]; bvp=[]
for k,sn in enumerate(snapshots):
    want=ids[assigned==k]
    if not len(want): continue
    s,sid,vr,vp,R,z=align_and_cyl(int(sn)); o=np.argsort(sid); ss=sid[o]
    p=np.searchsorted(ss,want); ok=(p<len(ss))&(ss[np.minimum(p,len(ss)-1)]==want)
    ix=o[p[ok]]; bid.append(want[ok]); bvr.append(vr[ix]); bvp.append(vp[ix])
    print(f'snap {sn}: recovered {ok.sum():,}/{len(want):,}',flush=True)
    del s,sid,vr,vp,R,z; gc.collect()
bid=np.concatenate(bid); bvr=np.concatenate(bvr); bvp=np.concatenate(bvp)

s,sid,vr,vp,R,z=align_and_cyl(127); o=np.argsort(sid); ss=sid[o]
p=np.searchsorted(ss,bid); ok=(p<len(ss))&(ss[np.minimum(p,len(ss)-1)]==bid)
ix=o[p[ok]]; bid=bid[ok]; bvr=bvr[ok]; bvp=bvp[ok]
zvr=vr[ix]; zvp=vp[ix]

fig,ax=plt.subplots(1,2,figsize=(12.5,5.4),sharex=True,sharey=True)
for a,xv,yv,title in [(ax[0],bvr,bvp,'Near birth'),(ax[1],zvr,zvp,'Present day (z=0)')]:
    h=a.hist2d(xv,yv,bins=130,range=[[-350,350],[-350,400]],cmap='viridis',cmin=1,norm='log')
    a.axhline(0,color='.7',lw=.6); a.axvline(0,color='.7',lw=.6)
    a.set_xlabel(r'$v_R$ [km s$^{-1}$]'); a.set_title(title)
    a.text(.03,.97,f'N={len(xv):,}\n<vphi>={np.mean(yv):.0f}\nsigmaR={np.std(xv):.0f}',
           transform=a.transAxes,va='top',bbox=dict(fc='white',alpha=.8,ec='none'))
ax[0].set_ylabel(r'$v_\phi$ [km s$^{-1}$]')
fig.suptitle('Au18 host-born stars formed during the merger (t=4.99–6.54 Gyr): birth vs present kinematics')
fig.tight_layout(rect=[0,0,1,.94]); out=C.FIG_DIR+'/au18_merger_birth_vs_z0_kinematics.png'; fig.savefig(out,dpi=150)
np.savez(C.OUT_DIR+'/merger_birth_vs_z0_kinematics.npz',ids=bid,birth_vR=bvr,birth_vphi=bvp,z0_vR=zvr,z0_vphi=zvp)
print(f'birth <vphi>={np.mean(bvp):.1f}, sigmaR={np.std(bvr):.1f}; z0 <vphi>={np.mean(zvp):.1f}, sigmaR={np.std(zvr):.1f}; saved {out}')
