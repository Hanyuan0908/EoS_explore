"""Select and display low-vphi Eos candidates among merger-born Au18 host stars."""
import os
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C
from auriga_public import snapshot as snap_mod, subhalos as sub_mod

os.makedirs(C.FIG_DIR,exist_ok=True); os.makedirs(C.OUT_DIR,exist_ok=True)
k=np.load(C.OUT_DIR+'/merger_birth_vs_z0_kinematics.npz')
ids=k['ids']; bvr=k['birth_vR']; bvp=k['birth_vphi']; zvr=k['z0_vR']; zvp=k['z0_vphi']
s=snap_mod.load_snapshot(127,4,snappath=C.SIM_DIR,
    loadlist=['ParticleIDs','Coordinates','GFM_StellarFormationTime','GFM_Metals'])
real=s.data['GFM_StellarFormationTime']>0
sid=s.data['ParticleIDs'][real]; co=s.data['Coordinates'][real]; met=s.data['GFM_Metals'][real]
sf=sub_mod.subfind(127,directory=C.SIM_DIR,loadlist=['GroupFirstSub','SubhaloPos'])
cen=sf.data['SubhaloPos'][int(sf.data['GroupFirstSub'][0])]
rad=np.sqrt((((co-cen)*1000.)**2).sum(axis=1))
o=np.argsort(sid); ss=sid[o]; p=np.searchsorted(ss,ids)
ok=(p<len(ss))&(ss[np.minimum(p,len(ss)-1)]==ids)
if not np.all(ok):
    ids=ids[ok]; bvr=bvr[ok]; bvp=bvp[ok]; zvr=zvr[ok]; zvp=zvp[ok]; p=p[ok]
ix=o[p]
feh=C.bracket_abundance(met[ix],'Fe','H')
els=['C','N','O','Ne','Mg','Si']
ratios=[C.bracket_abundance(met[ix],el,'Fe') for el in els]
r=rad[ix]
select_epoch=os.environ.get('EOS_SELECT_EPOCH','z0').lower()
if select_epoch not in {'z0','birth'}:
    raise ValueError("EOS_SELECT_EPOCH must be 'z0' or 'birth'")
selection_vphi=bvp if select_epoch=='birth' else zvp
epoch_label='near-birth' if select_epoch=='birth' else 'present-day'
suffix='_birthlow' if select_epoch=='birth' else ''
candidate=(r>5)&(r<10)&(np.abs(selection_vphi)<100)
reference=(r>5)&(r<10)
print(f'merger-born in 5<r<10: {reference.sum():,}; {epoch_label} |vphi|<100 candidates: {candidate.sum():,} ({candidate.sum()/reference.sum():.2%})')
print(f'candidate birth <vphi>={np.mean(bvp[candidate]):.1f}, sigmaR={np.std(bvr[candidate]):.1f}; '
      f'z0 <vphi>={np.mean(zvp[candidate]):.1f}, sigmaR={np.std(zvr[candidate]):.1f}')

# Chemistry: full merger-born reference as grey density, candidates as contours/scatter.
fig,axes=plt.subplots(2,3,figsize=(14,8.2),sharex=True)
for ax,el,y in zip(axes.flat,els,ratios):
    good=reference&np.isfinite(feh)&np.isfinite(y)
    yc=y[good]; lo,hi=np.percentile(yc,[.05,99.95]); pad=.12*(hi-lo); yr=(lo-pad,hi+pad)
    ax.hexbin(feh[good],y[good],gridsize=65,extent=(-4,1,*yr),bins='log',mincnt=1,cmap='Greys')
    q=candidate&np.isfinite(feh)&np.isfinite(y)
    # Candidate contours show their locus without hiding the parent density.
    H,xe,ye=np.histogram2d(feh[q],y[q],bins=[55,45],range=[[-4,1],list(yr)])
    xc=.5*(xe[:-1]+xe[1:]); yc=.5*(ye[:-1]+ye[1:])
    levels=np.unique(np.percentile(H[H>0],[55,75,90])) if np.any(H>0) else []
    if len(levels): ax.contour(xc,yc,H.T,levels=levels,colors='crimson',linewidths=1.2)
    ax.set(xlim=(-4,1),ylim=yr,xlabel='[Fe/H]',ylabel=f'[{el}/Fe]')
    ax.set_title(f'[{el}/Fe]')
fig.suptitle(f'Au18 merger-born host stars at z=0, 5<r<10 kpc: Eos candidates ({epoch_label} |vphi|<100 km/s) in red')
fig.tight_layout(rect=[0,0,1,.95]); chemout=C.FIG_DIR+f'/au18_eos_candidates{suffix}_chemistry.png'; fig.savefig(chemout,dpi=150)

# Kinematics, always vR on x and vphi on y.
fig2,ax=plt.subplots(1,2,figsize=(12.5,5.4),sharex=True,sharey=True)
for a,x,y,xc,yc,title in [(ax[0],bvr,bvp,bvr[candidate],bvp[candidate],'Near birth'),
                          (ax[1],zvr,zvp,zvr[candidate],zvp[candidate],'Present day (z=0)')]:
    a.hist2d(x[reference],y[reference],bins=120,range=[[-350,350],[-350,400]],cmap='Greys',cmin=1,norm='log')
    a.scatter(xc,yc,s=2,c='crimson',alpha=.18,lw=0,rasterized=True,label='Eos candidates')
    a.axhline(0,color='.6',lw=.7); a.axvline(0,color='.6',lw=.7)
    a.axhline(100,color='crimson',ls='--',lw=.8); a.axhline(-100,color='crimson',ls='--',lw=.8)
    a.set(xlabel=r'$v_R$ [km s$^{-1}$]',title=title)
ax[0].set_ylabel(r'$v_\phi$ [km s$^{-1}$]'); ax[1].legend(loc='lower right')
fig2.suptitle(f'Au18 Eos candidates: merger-born in-situ stars, 5<r<10 kpc and {epoch_label} |vphi|<100 km/s')
fig2.tight_layout(rect=[0,0,1,.94]); kout=C.FIG_DIR+f'/au18_eos_candidates{suffix}_birth_z0_kinematics.png'; fig2.savefig(kout,dpi=150)

np.savez(C.OUT_DIR+f'/eos_candidates{suffix}.npz',ids=ids[candidate],r_z0=r[candidate],
         birth_vR=bvr[candidate],birth_vphi=bvp[candidate],z0_vR=zvr[candidate],z0_vphi=zvp[candidate],
         feh=feh[candidate],**{el.lower()+'fe':y[candidate] for el,y in zip(els,ratios)})
print('saved',chemout,'and',kout)
