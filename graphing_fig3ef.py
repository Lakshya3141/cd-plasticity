# Use this script to generate plots from figure 3 (EF) in manuscript
# from previously generated data from exploration scripts

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Graphing Exploration of Assymetric conditions
# file, folder and image folder name here
# Data file allexp.csv is a comprehensive parameter exploration,
#  will need to be filtered according to plotting requirements
fn = 'allexp.csv'
subfold = "figure3"   
isubfold = "figure3"

# Create 'images' folder if it doesn't exist
if not os.path.exists(os.path.join('images', isubfold)):
    os.makedirs(os.path.join('images', isubfold))
    

def psi_cheatexpected(B, rho, tau1, tau2):
    b1 = B*rho
    b2 = B*(rho**(tau2/tau1))
    return (b1**2 + b2**2 - 2*b1*b2*(rho**(abs(tau2 - tau1)/tau1)))/2

def CD_pred_avg(x, sig_z2, B, rho, sig_eps2, r, psi, tau1, tau2):
    rho1 = rho
    rho2 = (rho**(tau2/tau1))
    return (4*(x+sig_eps2*psi) + 5*sig_z2 + (B**2)*sig_eps2*(1-(rho1**2 + rho2**2)/2))/(2*r)

def sig_tcalc(sig_u2, sig_z2, sig_eps2, psi):
    return np.sqrt(sig_u2 + sig_z2 + sig_eps2*psi)


# Get terms easier to use in analytical predictions from data
datas = pd.read_csv(filepath_or_buffer=os.path.join("hdf5", subfold, fn))
datas['sig_s2'] = datas['sig_s']**2
datas['sig_u2'] = datas['sig_u']**2
datas['sig_eps2'] = datas['sig_eps']**2
datas['sig_z2'] = datas['sp1.Gaa'] + datas['sp1.Gbb']*datas.sig_eps**2 + datas.sig_e**2
datas['delbExpected'] = datas['sp1.B']*datas['rho'] - datas['sp2.B']*datas['rho']**(datas['tau2']/datas['tau1'])

# FILTER data according to parameter set needed here
desired_tau2 = 0.4 # 0.4 or 0.8 or 0.2
dumsigeps = np.sort(np.unique(datas['sig_eps'])) # array([1.41421356, 3.16227766])
desired_sig_eps = dumsigeps[2] # 0 or 1 or 2 (increasing index means higher fluctuations)
ifn = "unstableCD_overlay"
desired_a0 = 4.99 # 4.99 or 5.99
cmap = 'turbo' #'cividis' # cmap

# Subset the DataFrame and create a copy
data = datas[(datas['tau2'] == desired_tau2) & 
             (datas['sig_eps'] == desired_sig_eps) & 
             (datas['sp2.a0'] == desired_a0)].copy()

# Find 
data['psiexpected'] = psi_cheatexpected(data['sp1.B'], data['rho'], data['tau1'], data['tau2'])
data['sigtexp'] = sig_tcalc(data['sig_u2'], data['sig_z2'], data['sig_eps2'], data['psiexpected'])
data['sign_cdexp'] = np.sign(1 - ((data.delzprime**2))/(2*(data.sigtexp**2)))
# Find sign of 
data['sign_conv'] = np.sign(abs(data['delb']) / abs(data['delbExpected']) - 1)

# Get single values from data columns for analytical functions
sig_z2 = np.unique(data.sig_z2)[0]
sig_u2 = np.unique(data.sig_u2)[0]
rho = np.unique(data.rho)[0]
r = np.unique(data.r)[0]
sig_eps2 = np.unique(data.sig_eps**2)[0]
B = np.unique(data['sp1.B'])[0]
car = np.unique(data['sp1.kar'])[0]

x = np.arange(np.min(data.sig_u)**2, np.max(data.sig_u)**2, (np.max(data.sig_u)**2 - np.min(data.sig_u)**2)/100)
sig_u2_unique = np.unique(data.sig_u2)
cdexp = CD_pred_avg(sig_u2_unique, sig_z2, B, rho, sig_eps2, r, np.unique(data.psiexpected), np.unique(data.tau1), np.unique(data.tau2))

# Define extinction point
carrying_capacity = 60000
threshold = 0.02 * carrying_capacity  # 2% of carrying capacity, which is 1200

# Filter the data to only include rows where both sp1.n and sp2.n are greater than the threshold
data = data[(data['sp1.n'] >= threshold) & (data['sp2.n'] >= threshold)]

leg_size = 23

fig, ax = plt.subplots(figsize=(10, 8))

# Define axis limits
xlim_sigu = (1.0, 80.0)
ylim_sigs = (445.0, 1570.0)

# Set the colormap for the main scatter plot
cm0 = plt.cm.get_cmap(cmap)

# Plot the main scatter plot
im = ax.scatter(data.sig_u2, data.sig_s2, c=abs(data['delb'])/abs(data['delbExpected']), 
                s=200, cmap=cm0, alpha=1.0)

# Add colorbar
clb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.1, shrink=0.8)
formula = r"$\left|\frac{b̅_1 - b̅_2}{B(\rho - \rho^{\tau_2/\tau_1})}\right|$"
clb.ax.set_ylabel(formula, fontsize=28)
clb.ax.tick_params(labelsize=18)

# Plot the condition where CD in reaction norm intercept is lower than threshold
for i in range(len(data)):
    if data.iloc[i]['sign_cdexp'] == 1:
        ax.scatter(data.iloc[i]['sig_u2'], data.iloc[i]['sig_s2'], color='black', s=10) #, edgecolor='black', linewidth=0.5

# Plot the condition where convergent CD in plasticity evolves
for i in range(len(data)):
    if data.iloc[i]['sign_conv'] == -1:
        ax.scatter(data.iloc[i]['sig_u2'], data.iloc[i]['sig_s2'], color='white', s=10)

# Plot the common union of the above two equations
for i in range(len(data)):
    if data.iloc[i]['sign_conv'] == -1 and data.iloc[i]['sign_cdexp'] == 1:
        ax.scatter(data.iloc[i]['sig_u2'], data.iloc[i]['sig_s2'], marker='^', color='#A9A9A9', s=130, edgecolor='none')

# Plot the complement of the union of the above two equations
for i in range(len(data)):
    if data.iloc[i]['sign_conv'] == 1 and data.iloc[i]['sign_cdexp'] == -1:
        ax.scatter(data.iloc[i]['sig_u2'], data.iloc[i]['sig_s2'], marker='s', color='#A9A9A9', s=100, edgecolor='none')

        
ax.scatter(10, 1000, s=230, facecolors='none', edgecolors='red', linewidths=3, marker='o')
# Add prediction line for CD
ax.plot(sig_u2_unique, cdexp, c='black', label='No fluctuations', linewidth=4, alpha=1.0)

# Set axis labels and limits
ax.set_xlabel("($\sigma_u^2$)", fontsize=28)
ax.set_ylabel("($\sigma_s^2$)", fontsize=28)
ax.set_xlim(xlim_sigu)
ax.set_ylim(ylim_sigs)
ax.tick_params(axis='both', which='major', labelsize=23)

# Add title
ax.set_title("Scaled Divergence in Plasticity", fontsize=30, pad=10)

# Adjust layout and display
plt.tight_layout()
plt.savefig(os.path.join('images', isubfold, ifn + '.jpg'), dpi=550)
plt.show()

