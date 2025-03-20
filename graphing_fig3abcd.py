# Use this script to generate plots from figure 3 (abcd) in manuscript
# from previously generated data from exploration scripts 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


colors = ['blue', 'green', 'red', 'purple', 'orange']  # Add more colors if needed
# DATA FOLDER AND IMAGE FOLDER HERE
subfold = "figure3"
isubfold = "figure3"
if not os.path.exists(os.path.join('images', isubfold)):
    os.makedirs(os.path.join('images', isubfold))
    
# DATA FILE NAME HERE
fn = "unstableCD.csv" # stableCD.csv or unstableCD.csv
data = pd.read_csv(filepath_or_buffer = os.path.join("hdf5", subfold, fn))
data['tau1'] = round(data['tau1'], 1)
# Subselect data
data = data[~data['tau1'].isin([0.3, 0.7])]


taus = np.sort(np.unique(data['tau1']))
data['delb'] = data['sp1.b'] - data['sp2.b']
data['sig_s2'] = data['sig_s']**2
data['sig_z2'] = data['sp1.Gaa'] + data['sp1.Gbb']*data.sig_eps**2 + data.sig_e**2
colnu = 3 # number of columns in legend

# Define a threshold to remove populations close to extinction from the simulations
carrying_capacity = 60000
threshold = 0.02 * carrying_capacity  # 2% of carrying capacity, which is 1200

# Filter the data to only include rows where both sp1.n and sp2.n are greater than the threshold
data = data[(data['sp1.n'] >= threshold) & (data['sp2.n'] >= threshold)]


#%% functions
def psi_usingplast(b1, b2, rho, tau1, tau2):
    return (b1**2 + b2**2 - 2*b1*b2*(rho**(abs(tau2 - tau1)/tau1)))/2



def db_species(B, rho, tau1, tau2, r, Nother, Kother, phi, sig_s):
    num = B*(rho - rho ** (tau2/tau1))
    denom = 1 + (r * Nother * phi * (1 + rho**(abs(tau2 - tau1)/tau1)) * sig_s**2)/Kother
    return num/denom

def db_total(B, rho, tau1, tau2, r, N1, N2, K, phi, sig_s):
    db1 = db_species(B, rho, tau1, tau2, r, N2, K, phi, sig_s)
    db2 = db_species(B, rho, tau1, tau2, r, N1, K, phi, sig_s)
    return abs(db1) + abs(db2)

def dCD_species(sig_u, sig_s, sig_z2, sig_eps2, r, Nother, K, psi):
    sig_t2 = sig_u**2 + sig_z2 + sig_eps2*psi
    d2 = sig_t2*np.log( (r * Nother * sig_u * sig_s**2) / K / (sig_t2)**(3/2))
    return np.sqrt(d2)


#%% Solving analytical predicitions using simulation data

data['psidata'] = psi_usingplast(data['sp1.b'], data['sp2.b'], data['rho'], data['tau1'], data['tau2'])
data['d1_bdata'] = dCD_species(data['sig_u'], data['sig_s'], data['sig_z2'], data['sig_eps']**2, data['r'], data['sp2.n'], data['sp1.kar'], data['psidata'])
data['d2_bdata'] = dCD_species(data['sig_u'], data['sig_s'], data['sig_z2'], data['sig_eps']**2, data['r'], data['sp1.n'], data['sp1.kar'], data['psidata'])
data['d_bdata'] = data['d1_bdata'] + data['d2_bdata']

# Calculating expected value of divergence in plasticity under no competition
data['delbExpected_base'] = data['sp1.B']*data['rho'] - data['sp2.B']*data['rho']**(data['tau2']/data['tau1'])
data['delzExpected'] = data['d_bdata']

#%% images divergence plasticity (Fig 3C / 3D)

## Absolute expression with +B
xlab = r"\tau_2"
ylab = "Absolute divergence in \nmean plasticity (|$b̅_1$ - $b̅_2$|)"
tit = "CD in plasticity with and without competition"
fig, ax = plt.subplots(figsize=(10, 8))
# plt.hlines(np.unique(data['sp1.B']), xmin=0.01, xmax = 0.99, color = "black", linewidth = 2.0, linestyle = "--")
for i, n in enumerate(taus):
    color = colors[i % len(colors)]  # Use modulus to cycle through colors if more taus than colors
    dumdat = data[data['tau1'] == n]
    plt.plot(dumdat['tau2'], abs(dumdat['delbExpected_base']), linestyle="--", label=round(n,1), color=color, linewidth = 3)
    

for i, n in enumerate(taus):
    color = colors[i % len(colors)]  # Use modulus to cycle through colors if more taus than colors
    dumdat = data[data['tau1'] == n]    
    plt.plot(dumdat['tau2'], abs(dumdat['delb']), color=color, linewidth = 3)
# plt.hlines(0, xmin=0.01, xmax = 0.99, color = "black")
plt.legend(title=fr'$\tau_1$', fontsize=26, title_fontsize=26, ncol=colnu, columnspacing=0.5)
plt.xlabel(fr"${xlab}$", fontsize=36, labelpad=15)
plt.ylabel(f"{ylab}", fontsize=36, labelpad=15)
plt.ylim(0,2.2)
# plt.ylim(0, 11)
ax.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.show()

ifn = "_delbAbsolute"
dumsav = 'images/' + isubfold + '/' + fn[:-4] + ifn + '.jpg'
fig.savefig(dumsav, dpi=550)

#%% images divergence phenotype

## No STDs
xlab = r"\tau_2"
ylab = "CD in phenotype"
tit = "CD in plasticity with and without competition"
fig, ax = plt.subplots(figsize=(10, 8))
for i, n in enumerate(taus):
    color = colors[i % len(colors)]  # Use modulus to cycle through colors if more taus than colors
    dumdat = data[data['tau1'] == n]
    plt.plot(dumdat['tau2'], abs(dumdat['delzprime']), label=round(n,1), color=color, linewidth = 3)
for i, n in enumerate(taus):
    color = colors[i % len(colors)]  # Use modulus to cycle through colors if more taus than colors
    dumdat = data[data['tau1'] == n]    
    plt.plot(dumdat['tau2'], abs(dumdat['delzExpected']), color=color, linewidth = 3, linestyle="--")
# plt.hlines(0, xmin=0.01, xmax = 0.99, color = "black")
# plt.legend(title=fr'$\tau_1$', fontsize=26, title_fontsize=26, ncol=colnu, columnspacing=0.5)
plt.xlabel(fr"${xlab}$", fontsize=36, labelpad=15)
plt.ylabel(f"{ylab}", fontsize=36, labelpad=15)
ax.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.show()
ifn = "_delz"
dumsav = 'images/' + isubfold + '/' + fn[:-4] + ifn + '.jpg'
fig.savefig(dumsav, dpi=550)

# Fig 3A / 3B
# Standard deviations included
xlab = r"\tau_2"
ylab = "Phenotype in reference \n environment ($a̅$)"
tit = "CD in plasticity with and without competition"

fig, ax = plt.subplots(figsize=(10, 8))

for i, n in enumerate(taus):
    color = colors[i % len(colors)]  # Use modulus to cycle through colors if more taus than colors
    dumdat = data[data['tau1'] == n]
    plt.plot(dumdat['tau2'], abs(dumdat['delaprime']), label=round(n,1), color=color, linewidth=3)
    # Add shaded regions representing standard deviations
    plt.fill_between(dumdat['tau2'], 
                      abs(dumdat['delaprime'] - dumdat['delastdprime']), 
                      abs(dumdat['delaprime'] + dumdat['delastdprime']),
                      color=color, alpha=0.2)

plt.legend(title=fr'$\tau_1$', fontsize=26, title_fontsize=26, ncol=colnu, columnspacing=0.5)
plt.xlabel(fr"${xlab}$", fontsize=36, labelpad=15)
plt.ylabel(f"{ylab}", fontsize=36, labelpad=15)
plt.ylim(0, 11)
ax.tick_params(axis='both', which='major', labelsize=26)
plt.tight_layout()
plt.show()
ifn = "_delastd"
dumsav = 'images/' + isubfold + '/' + fn[:-4] + ifn + '.jpg'
fig.savefig(dumsav, dpi=550)