"""Wide-range multi-element chemistry of clean GS/E and post-spin-up host cohorts."""
import os
import numpy as np
import matplotlib.pyplot as plt
import config_au18 as C
from auriga_public import snapshot as snap_mod

os.makedirs(C.FIG_DIR,exist_ok=True)
samples=np.load(C.OUT_DIR+'/merger_epoch_z0_samples.npz')
groups=[('Clean GS/E','clean_gs/e' if False else 'clean_gs/e')]
# np.savez keys cannot contain '/', inspect/use the actual normalized names.
keys=['clean_gs/e','host_before','host_during','host_after']
labels=['Clean GS/E','Host: before','Host: during','Host: after']
colors=['crimson','#7b3294','#e66101','#018571']

s=snap_mod.load_snapshot(127,4,snappath=C.SIM_DIR,
    loadlist=['ParticleIDs','GFM_StellarFormationTime','GFM_Metals'])
real=s.data['GFM_StellarFormationTime']>0
sid=s.data['ParticleIDs'][real]; met=s.data['GFM_Metals'][real]
order=np.argsort(sid); ss=sid[order]
def mask_ids(ids):
    p=np.searchsorted(ss,ids); good=p<len(ss); out=np.zeros(len(sid),bool)
    ii=np.flatnonzero(good); hit=ss[p[good]]==ids[good]; out[order[p[ii[hit]]]]=True
    return out
masks=[mask_ids(samples[k]) for k in keys]
feh=C.bracket_abundance(met,'Fe','H')
els=['C','N','O','Ne','Mg','Si']
ratios=[C.bracket_abundance(met,e,'Fe') for e in els]

fig,axes=plt.subplots(4,6,figsize=(19,11),sharex=True)
for j,(el,yall) in enumerate(zip(els,ratios)):
    union=np.logical_or.reduce(masks)&np.isfinite(feh)&np.isfinite(yall)
    lo,hi=np.nanpercentile(yall[union],[.05,99.95]); pad=.12*(hi-lo)
    yr=(lo-pad,hi+pad)
    for i,(lab,col,m) in enumerate(zip(labels,colors,masks)):
        q=m&np.isfinite(feh)&np.isfinite(yall)
        axes[i,j].hexbin(feh[q],yall[q],gridsize=55,extent=(-4,1,*yr),
                         bins='log',mincnt=1,cmap='magma')
        axes[i,j].set_ylim(*yr); axes[i,j].set_xlim(-4,1)
        if i==0: axes[i,j].set_title(f'[{el}/Fe]',fontsize=11)
        if j==0:
            axes[i,j].set_ylabel(lab+'\n[X/Fe]',color=col,fontsize=10)
        if i==3: axes[i,j].set_xlabel('[Fe/H]')
fig.suptitle('Au18 z=0 chemistry: clean GS/E and post-spin-up host cohorts (wide abundance ranges)',fontsize=14)
fig.tight_layout(rect=[0,0,1,.97])
out=C.FIG_DIR+'/au18_merger_chemistry_atlas.png'; fig.savefig(out,dpi=140)
print('saved',out)
