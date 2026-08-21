"""Strict-tail chemistry comparison for the two Au18 Eos channels."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import config_au18 as C

os.makedirs(C.FIG_DIR,exist_ok=True); os.makedirs(C.OUT_DIR,exist_ok=True)
x=np.load(C.OUT_DIR+'/eos_two_channels.npz')
ids=x['ids']; eb=x['eps_birth']; e0=x['eps_z0']; r=x['r_z0']; feh=x['feh']
els=['C','N','O','Ne','Mg','Si']; ratios=[x[e.lower()+'fe'] for e in els]
base=(r>4)&(r<15)&np.isfinite(eb)&np.isfinite(e0)
A=base&(eb>.7)&(e0<.3)
B=base&(eb<.3)&(e0<.3)
print(f'strict A heated: {A.sum():,}; strict B born-radial: {B.sum():,}')
print('quantity       A median   B median   B-A')
for name,y in [('Fe/H',feh)]+list(zip([e+'/Fe' for e in els],ratios)):
    aa=np.nanmedian(y[A]); bb=np.nanmedian(y[B]); print(f'{name:8s} {aa:+9.3f} {bb:+9.3f} {bb-aa:+8.3f}')

fig,axes=plt.subplots(3,6,figsize=(19,9.5),sharex=True)
groups=[(r'A strict: $\epsilon_b>0.7\rightarrow\epsilon_0<0.3$',A,'#2166ac'),
        (r'B strict: $\epsilon_b<0.3\rightarrow\epsilon_0<0.3$',B,'#b2182b')]
union=A|B
for j,(el,y) in enumerate(zip(els,ratios)):
    good=union&np.isfinite(feh)&np.isfinite(y); lo,hi=np.percentile(y[good],[.05,99.95]); pad=.12*(hi-lo); yr=(lo-pad,hi+pad)
    for i,(label,m,color) in enumerate(groups):
        q=m&np.isfinite(feh)&np.isfinite(y)
        axes[i,j].hexbin(feh[q],y[q],gridsize=38,extent=(-2,0.8,*yr),bins='log',mincnt=1,cmap='magma')
        axes[i,j].set(xlim=(-2,.8),ylim=yr)
        if i==0: axes[i,j].set_title(f'[{el}/Fe]')
        if j==0: axes[i,j].set_ylabel(f'{label}\nN={m.sum():,}\n[X/Fe]',color=color)
    # Direct comparison: smoothed number-density contours on a common grid.
    ax=axes[2,j]
    for label,m,color in groups:
        q=m&np.isfinite(feh)&np.isfinite(y)
        H,xe,ye=np.histogram2d(feh[q],y[q],bins=[75,60],range=[[-2,.8],list(yr)])
        H=gaussian_filter(H,1.2)
        xc=.5*(xe[:-1]+xe[1:]); yc=.5*(ye[:-1]+ye[1:])
        levels=np.asarray([.12,.3,.55,.8])*H.max()
        ax.contour(xc,yc,H.T,levels=levels,colors=color,linewidths=1.35)
    ax.set(xlim=(-2,.8),ylim=yr,xlabel='[Fe/H]')
    if j==0: ax.set_ylabel('A + B contours\n[X/Fe]')
    if j==5:
        ax.plot([],[],color='#2166ac',label='A: heated disc')
        ax.plot([],[],color='#b2182b',label='B: born radial')
        ax.legend(loc='best',fontsize=8)
fig.suptitle(r'Au18 strict Eos-channel chemistry, merger-born in situ, $4<r_{z=0}<15$ kpc')
fig.tight_layout(rect=[0,0,1,.94]); out=C.FIG_DIR+'/au18_eos_two_channels_chemistry_strict_r4_15.png'; fig.savefig(out,dpi=160)
np.savez(C.OUT_DIR+'/eos_two_channels_strict_r4_15.npz',heated_ids=ids[A],born_radial_ids=ids[B],
         eps_birth_A=eb[A],eps_z0_A=e0[A],eps_birth_B=eb[B],eps_z0_B=e0[B])
print('saved',out)
