# Use this script to generate plots from figure 2 in manuscript
# from previously generated data from exploration scripts
# (eqs 14 , 18 , and 21 for orange, blue, and green, respectively).

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Functions used in plotting analytical predictions

def fluc_pred_new(x, sig_z2, B, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2)/(2*r)

def nofluc_pred_new(x, sig_z2, r):
    return (4*x + 5*sig_z2)/(2*r)

def plast_pred_new(x, sig_z2, B, rho, sig_eps2, r):
    return (4*x + 5*sig_z2 + (B**2)*sig_eps2*(1-rho**2))/(2*r)

# List of data files and folder
fnl = ["NoFluc_main.csv" , "Fluc_main.csv", "EvolvingPlasticity_main.csv"]
subfold = "figure2"

# Create 'images' folder if it doesn't exist
if not os.path.exists(os.path.join('images', subfold)):
    os.makedirs(os.path.join('images', subfold))

datas = []
for i, fn in enumerate(fnl):
    datas.append(pd.read_csv(filepath_or_buffer=os.path.join("hdf5", subfold, fn)))
    datas[i]['sig_s2'] = datas[i]['sig_s']**2
    datas[i]['sig_u2'] = datas[i]['sig_u']**2
    datas[i]['sig_z2'] = datas[i]['sp1.Gaa'] + datas[i]['sp1.Gbb']*datas[i].sig_eps**2 + datas[i].sig_e**2

sig_z2 = np.unique(datas[0].sig_z2)[0]
rho = np.unique(datas[0].rho)[0]
r = np.unique(datas[0].r)[0]
sig_eps2 = np.unique(datas[0].sig_eps**2)[0]
B = np.unique(datas[0]['sp1.B'])

x = np.arange(np.min(datas[0].sig_u)**2, np.max(datas[0].sig_u)**2, (np.max(datas[0].sig_u)**2 - np.min(datas[0].sig_u)**2)/100)
nfl = nofluc_pred_new(x, sig_z2, r)
fl = fluc_pred_new(x, sig_z2, B, sig_eps2, r)
pls = plast_pred_new(x, sig_z2, B, rho, sig_eps2, r)

leg_size = 23

## Plotting character displacement (Fig 2A) for condition with no fluctuations
fig, ax = plt.subplots(figsize=(10,8))
plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
plt.ylim((np.min(datas[0].sig_s2), np.max(datas[0].sig_s2)))
plt.xlim((np.min(datas[0].sig_u2), np.max(datas[0].sig_u2)))
cm0 = plt.cm.get_cmap('gist_yarg')
im0 = plt.scatter(datas[0].sig_u2, datas[0].sig_s2, c=datas[0].delz, s=40, cmap=cm0, alpha=1.0)
plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)

# Add red open circle at (10, 550)
plt.scatter(10, 550, s=230, facecolors='none', edgecolors='red', linewidths=3, marker='o')

clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
clb0.ax.set_ylabel('|$z̅_1$ - $z̅_2$|', fontsize=25)
clb0.ax.tick_params(labelsize=18)
ax.tick_params(axis='both', which='major', labelsize=23)
plt.legend(loc='lower right', fontsize=leg_size)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fnl[0][:-4] + '.jpg'
fig.savefig(dumsav, dpi=550)

## Plotting character displacement (Fig 2B) for condition with fluctuations
fig, ax = plt.subplots(figsize=(10,8))
plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
plt.ylim((np.min(datas[1].sig_s2), np.max(datas[1].sig_s2)))
plt.xlim((np.min(datas[1].sig_u2), np.max(datas[1].sig_u2)))
cm0 = plt.cm.get_cmap('gist_yarg')
im0 = plt.scatter(datas[1].sig_u2, datas[1].sig_s2, c=datas[1].delz, s=40, cmap=cm0, alpha=1.0)
plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating env.', linewidth=3, alpha=1.0)

# Add red open circle at (10, 550)
plt.scatter(10, 550, s=230, facecolors='none', edgecolors='red', linewidths=3, marker='o')

clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
clb0.ax.set_ylabel('|$z̅_1$ - $z̅_2$|', fontsize=25)
clb0.ax.tick_params(labelsize=18)
ax.tick_params(axis='both', which='major', labelsize=23)
plt.legend(loc='lower right', fontsize=leg_size)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fnl[1][:-4] + '.jpg'
fig.savefig(dumsav, dpi=550)

## Plotting character displacement (Fig 2C) for condition with evolving plasticity
fig, ax = plt.subplots(figsize=(10,8))
plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
plt.ylim((np.min(datas[2].sig_s2), np.max(datas[2].sig_s2)))
plt.xlim((np.min(datas[2].sig_u2), np.max(datas[2].sig_u2)))
cm0 = plt.cm.get_cmap('gist_yarg')
im0 = plt.scatter(datas[2].sig_u2, datas[2].sig_s2, c=datas[2].delz, s=40, cmap=cm0, alpha=1.0)
plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating env.', linewidth=3, alpha=1.0)
plt.plot(x, pls, c = 'tab:green', label = 'Evolving plasticity', linewidth=3, alpha=1.0)

# Add red open circle at (10, 550)
plt.scatter(10, 550, s=230, facecolors='none', edgecolors='red', linewidths=3, marker='o')

clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 2), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
clb0.ax.set_ylabel('|$z̅_1$ - $z̅_2$|', fontsize=25)
clb0.ax.tick_params(labelsize=18)
ax.tick_params(axis='both', which='major', labelsize=23)
plt.legend(loc='lower right', fontsize=leg_size)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fnl[2][:-4] + '.jpg'
fig.savefig(dumsav, dpi=550)

## Plotting divergence in plasticity (Fig 2D) for condition with evolving plasticity
fig, ax = plt.subplots(figsize=(10,8))
plt.xlabel("Resource utilization breadth ($\sigma_u^2$)", fontsize=28) 
plt.ylabel("Stabilizing selection breadth ($\sigma_s^2$)", fontsize=28)  
plt.ylim((np.min(datas[2].sig_s2), np.max(datas[2].sig_s2)))
plt.xlim((np.min(datas[2].sig_u2), np.max(datas[2].sig_u2)))
cm0 = plt.cm.get_cmap('gist_yarg')
im0 = plt.scatter(datas[2].sig_u2, datas[2].sig_s2, c=datas[2].delb, s=40, cmap=cm0, alpha=1.0)
plt.plot(x, nfl, c = 'tab:orange', label = 'No fluctuations', linewidth=3, alpha=1.0)
plt.plot(x, fl, c = 'tab:blue', label = 'Fluctuating env.', linewidth=3, alpha=1.0)
plt.plot(x, pls, c = 'tab:green', label = 'Evolving plasticity', linewidth=3, alpha=1.0)

# Add red open circle at (10, 550)
plt.scatter(10, 550, s=230, facecolors='none', edgecolors='red', linewidths=3, marker='o')

clb0 = fig.colorbar(im0, ax=ax, ticks=np.arange(0, 13, 0.01), pad=0.02, fraction=0.1, shrink=0.8) # Adjust shrink value
clb0.ax.set_ylabel('|$b̅_1$ - $b̅_2$|', fontsize=25)
clb0.ax.tick_params(labelsize=18)
ax.tick_params(axis='both', which='major', labelsize=23)
plt.legend(loc='lower right', fontsize=leg_size)
plt.tight_layout()
plt.show()
dumsav = 'images/' + subfold + '/' + fnl[2][:-4] + "_delb" + '.jpg'
fig.savefig(dumsav, dpi=550)