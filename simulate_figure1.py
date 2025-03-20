# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:52:16 2023
@author: laksh

Use this code to run a single simulation and plot dynamics, niche, 
and final tolerance curves. Codes to plot figure 1 and figure 4 panels also within.
"""

import numpy as np
from helper_funcs import *
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Base params that don't change
# Define constants and parameters
A = np.array([5, 5])        # optimal value of trait
B = np.array([3, 3])        # optimal value of plasticity under constant environment
b0 = np.array([0.01, 0.0])  # initial value for plasticity
n0 = np.array([10000.0, 10000.0])  # initial populations

Gaa = np.array([0.5, 0.5])   # additive genetic variance for intercept a
Gbb = np.array([0.045, 0.045]) # additive genetic variance for plasticity b
sig_e = np.sqrt(0.5)  # noise in phenotype
sig_s = np.sqrt(550)  # width of stabilising selection curve
sig_eps = np.sqrt(0)  # variance in environmental fluctuations
sig_u = np.sqrt(10)   # width of resource utilization curve
rho = 0.5             # Correlation between development and selection for focal species
tau = [0.3, 0.3]      # time between maturation and development

r = 0.1               # intrinsic growth rate
kar = np.array([60000.0, 60000.0])  # Carrying capacity
grow = np.array([1, 1])  # growth flags
seed = 0              # random seed
lw = 3
avg_fin = 1000
subfold = "SingleRuns"

# Create 'images' folder if it doesn't exist
if not os.path.exists(os.path.join('images', subfold)):
    os.makedirs(os.path.join('images', subfold))


#%% Condition 1: No fluctuations, plasticity OFF
# Figure 1A and Fig 4A
plast = np.array([-1, -1])
tot = 8000
CD_point = 7000
CDnp_point = 7500
a0 = np.array([5.01, 4.99])
sig_eps = np.sqrt(0)
fn = "NoFluctuations_NoPlasticity"

a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main(
    a0, b0, n0, plast, grow, A, B, Gaa, Gbb, kar, rho, tau, r, 
    sig_s, sig_u, sig_e, sig_eps, tot, seed
)
alpha = organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2)

n1 = np.mean(n[-avg_fin:,0])
n2 = np.mean(n[-avg_fin:,1])
sig_z1 = np.mean(np.sqrt(Gaa[0] + Gbb[0] * epss[-avg_fin:] ** 2 + sig_e ** 2))
sig_z2 = np.mean(np.sqrt(Gaa[1] + Gbb[1] * epss[-avg_fin:] ** 2 + sig_e ** 2))

CD1 = no_fluc_CD(sig_u, sig_z1, sig_s, r, n1, kar[0])
CD2 = no_fluc_CD(sig_u, sig_z2, sig_s, r, n2, kar[1])
CDnp1 = NoFluc_noPop(sig_u, sig_z1, sig_s, r)
CDnp2 = NoFluc_noPop(sig_u, sig_z2, sig_s, r)

figure, axis = plt.subplots(figsize=(10,8))
plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', linewidth=lw)

axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,0], label="species1", color='blue', linewidth=lw)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,1], label="species2", color='red',  linewidth=lw)

# Add arrows
axis.annotate('', xy=(CD_point/1000, A[0]), 
              xytext=(CD_point/1000, A[0]+CD1), 
              arrowprops=dict(arrowstyle='<-', linestyle = "-", linewidth=2, 
                              mutation_scale=30, color = "blue"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), 
              xytext=(CDnp_point/1000, A[0]+CDnp1), 
              arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, 
                              mutation_scale=30, color = "blue"))

axis.annotate('', xy=(CD_point/1000, A[0]), 
              xytext=(CD_point/1000, A[0]-CD2), 
              arrowprops=dict(arrowstyle='<-', linestyle = "-", linewidth=2, 
                              mutation_scale=30, color = "red"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), 
              xytext=(CDnp_point/1000, A[0]-CDnp2), 
              arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, 
                              mutation_scale=30, color = "red"))

plt.ylim(1.0,9.0)
plt.title('No fluctuations, plast OFF', fontsize =36)
axis.legend(fontsize=26)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36, labelpad=15)
axis.set_ylabel("Mean Phenotype $z̅$", fontsize=36, labelpad=15)
plt.tight_layout()
plt.show()

dumsav = 'images/' + subfold + '/' + fn + '.jpg'
figure.savefig(dumsav, dpi=550)

plot_tolerance_curves(alpha, Gaa, Gbb, sig_e, sig_s, A, B, r, rho, tau,
                      avg_fin=1000, 
                      fig_title="No fluctuations, plast OFF",
                      save_path='images/{}/{}_fitness_curves.jpg'.format(subfold, fn))


#%% Condition 2: No evolvable plasticity (constant plast), low sig_eps
# Fig 1B

plast = np.array([-1, -1])
tot = 20000
CD_point = 17500
CDnp_point = 18500
a0 = np.array([5.01, 4.99])
sig_eps = np.sqrt(4)
fn = "LowFluctuations_NonEvolveablePlasticity"

a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main(
    a0, b0, n0, plast, grow, A, B, Gaa, Gbb, kar, rho, tau, r, 
    sig_s, sig_u, sig_e, sig_eps, tot, seed
)
alpha = organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2)

n1 = np.mean(n[-avg_fin:,0])
n2 = np.mean(n[-avg_fin:,1])
sig_z1 = np.mean(np.sqrt(Gaa[0] + Gbb[0] * epss[-avg_fin:] ** 2 + sig_e ** 2))
sig_z2 = np.mean(np.sqrt(Gaa[1] + Gbb[1] * epss[-avg_fin:] ** 2 + sig_e ** 2))

CD1 = no_fluc_CD(sig_u, sig_z1, sig_s, r, n1, kar[0])
CD2 = no_fluc_CD(sig_u, sig_z2, sig_s, r, n2, kar[1])
CDnp1 = Fluc_noPop(sig_u, sig_z1, sig_s, r, B[0], sig_eps)
CDnp2 = Fluc_noPop(sig_u, sig_z2, sig_s, r, B[1], sig_eps)

figure, axis = plt.subplots(figsize=(10,8))
plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', linewidth=lw, label='Optima')

axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,0], label="species1", color='blue', linewidth=lw)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,1], label="species2", color='red',  linewidth=lw)

axis.annotate('', xy=(CD_point/1000, A[0]), 
              xytext=(CD_point/1000, A[0]+CD1), 
              arrowprops=dict(arrowstyle='<-', linestyle = "-", linewidth=2, 
                              mutation_scale=30, color = "blue"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), 
              xytext=(CDnp_point/1000, A[0]+CDnp1), 
              arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, 
                              mutation_scale=30, color = "blue"))

axis.annotate('', xy=(CD_point/1000, A[0]), 
              xytext=(CD_point/1000, A[0]-CD2), 
              arrowprops=dict(arrowstyle='<-', linestyle = "-", linewidth=2, 
                              mutation_scale=30, color = "red"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), 
              xytext=(CDnp_point/1000, A[0]-CDnp2), 
              arrowprops=dict(arrowstyle='<-', linestyle = (0, (5, 4)), linewidth=2, 
                              mutation_scale=30, color = "red"))
plt.title("No evolvable plast, low sig_eps", fontsize = 36)
plt.ylim(1.0, 9.0)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36, labelpad=15)
axis.set_ylabel("Mean Phenotype $z̅$", fontsize=36, labelpad=15)
plt.tight_layout()
plt.show()

dumsav = 'images/' + subfold + '/' + fn + '.jpg'
figure.savefig(dumsav, dpi=550)

plot_tolerance_curves(alpha, Gaa, Gbb, sig_e, sig_s, A, B, r, rho, tau,
                      avg_fin=1000, 
                      fig_title="No evolvable plast, low sig_eps",
                      save_path='images/{}/{}_fitness_curves.jpg'.format(subfold, fn))


#%% Condition 3: No evolvable plasticity (constant plast), high sig_eps
# Figure 1C and 4B

plast = np.array([-1, -1])
tot = 30000
CD_point = 17500
a0 = np.array([7.8, 2.2])
sig_eps = np.sqrt(8)
fn = "HighFluctuations_NonEvolveablePlasticity"

a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main(
    a0, b0, n0, plast, grow, A, B, Gaa, Gbb, kar, rho, tau, r, 
    sig_s, sig_u, sig_e, sig_eps, tot, seed
)
alpha = organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2)

n1 = np.mean(n[-avg_fin:,0])
n2 = np.mean(n[-avg_fin:,1])
sig_z1 = np.mean(np.sqrt(Gaa[0] + Gbb[0] * epss[-avg_fin:] ** 2 + sig_e ** 2))
sig_z2 = np.mean(np.sqrt(Gaa[1] + Gbb[1] * epss[-avg_fin:] ** 2 + sig_e ** 2))

figure, axis = plt.subplots(figsize=(10,8))
axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,0], label="species1", color='blue', linewidth=lw)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['a'][:,1], label="species2", color='red',  linewidth=lw)
plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', label='Optima', linewidth=lw)

plt.ylim(1.0, 9.0)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36, labelpad=15)
axis.set_ylabel("Mean Phenotype $z̅$", fontsize=36, labelpad=15)
plt.tight_layout()
plt.title("No evolvable plast, high sig_eps", fontsize=36)
plt.show()

dumsav = 'images/' + subfold + '/' + fn + '.jpg'
figure.savefig(dumsav, dpi=550)

# --- Now plot the final tolerance curves for Condition 3 ---
plot_tolerance_curves(alpha, Gaa, Gbb, sig_e, sig_s, A, B, r, rho, tau,
                      avg_fin=1000, 
                      fig_title="No evolvable plast, high sig_eps)",
                      save_path='images/{}/{}_fitness_curves.jpg'.format(subfold, fn))


#%% Condition 4: Evolvable plasticity
# Fig 1D, 1E and 4C

plast = np.array([1, 1])
tot = 60000
CD_point = 52000
CDnp_point = 56000
a0 = np.array([5.01, 4.99])
sig_eps = np.sqrt(8)
fn = "EvolveablePlasticity_SameLifeHistory"

a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main(
    a0, b0, n0, plast, grow, A, B, Gaa, Gbb, kar, rho, tau, r, 
    sig_s, sig_u, sig_e, sig_eps, tot, seed
)
alpha = organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2)

n1 = np.mean(n[-avg_fin:,0])
n2 = np.mean(n[-avg_fin:,1])
sig_z1 = np.mean(np.sqrt(Gaa[0] + Gbb[0] * epss[-avg_fin:] ** 2 + sig_e ** 2))
sig_z2 = np.mean(np.sqrt(Gaa[1] + Gbb[1] * epss[-avg_fin:] ** 2 + sig_e ** 2))


# Main plot
fig, axis = plt.subplots(figsize=(10,8))
plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', label='Optima', linewidth=lw)
axis.plot(np.arange(0,tot/1000,0.001), alpha['a'][:,0], color='blue', label="$a̅$ species1", linewidth=lw)
axis.plot(np.arange(0,tot/1000,0.001), alpha['a'][:,1], color='red',  label="$a̅$ species2", linewidth=lw)
plt.ylim(1.0, 9.0)
plt.title("Evolvable plast, same taus", fontsize = 36)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36)
axis.set_ylabel("Phenotype in reference \n environment ($a̅$)", fontsize=36)

axis.annotate('', xy=(CD_point/1000, A[0]), 
              xytext=(CD_point/1000, A[0]+CD1), 
              arrowprops=dict(arrowstyle='<-', linestyle="-", linewidth=2, 
                              mutation_scale=30, color="blue"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), 
              xytext=(CDnp_point/1000, A[0]+CDnp1), 
              arrowprops=dict(arrowstyle='<-', linestyle=(0,(5,4)), linewidth=2, 
                              mutation_scale=30, color="blue"))
axis.annotate('', xy=(CD_point/1000, A[0]), 
              xytext=(CD_point/1000, A[0]-CD2), 
              arrowprops=dict(arrowstyle='<-', linestyle="-", linewidth=2, 
                              mutation_scale=30, color="red"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), 
              xytext=(CDnp_point/1000, A[0]-CDnp2), 
              arrowprops=dict(arrowstyle='<-', linestyle=(0,(5,4)), linewidth=2, 
                              mutation_scale=30, color="red"))

# Inset
inset_axis = inset_axes(axis, width="23%", height="23%", loc='lower left', borderpad=6.5)
inset_axis.set_facecolor('white')  
inset_axis.plot(np.arange(0,tot/1000,0.001), 
                alpha['a'][:,0] + alpha['b'][:,0]*alpha["environmental_deviation_species1"], 
                color='blue', linewidth=2, alpha=0.5)
inset_axis.plot(np.arange(0,tot/1000,0.001), 
                alpha['a'][:,1] + alpha['b'][:,1]*alpha["environmental_deviation_species2"], 
                color='red',  linewidth=2, alpha=0.5)

inset_axis.tick_params(axis='both', which='major', labelsize=16)
inset_axis.set_xlabel("Generations (x1000)", fontsize=18)
inset_axis.set_ylabel("Phenotype $z̅$", fontsize=18)
plt.show()

dumsav = 'images/' + subfold + '/' + fn + '.jpg'
fig.savefig(dumsav, dpi=550)

# Plasticity figure
fig, axis = plt.subplots(figsize=(10,8))
axis.tick_params(axis='both', which='major', labelsize=26)
plt.title("Evolvable plast, same taus", fontsize = 36)
axis.set_xlabel("Generations (x1000)", fontsize=36)
axis.set_ylabel("Mean plasticity ($b̅$)", fontsize=36)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['b'][:,0], linewidth=lw, color='blue')
axis.plot(np.arange(0, tot/1000, 0.001), alpha['b'][:,1], linewidth=lw, color='red')
plt.hlines(y=B[0]*rho, xmin=0, xmax=tot/1000, linestyle = (0, (1, 3)), color='black', 
           linewidth=lw, label="expected plasticity")
axis.legend(fontsize=26)
plt.tight_layout()
plt.show()

dumsav = 'images/' + subfold + '/' + fn +'_plasticity.jpg'
fig.savefig(dumsav, dpi=550)

# --- Now plot the final tolerance curves for Condition 4 ---
plot_tolerance_curves(alpha, Gaa, Gbb, sig_e, sig_s, A, B, r, rho, tau,
                      avg_fin=1000, 
                      fig_title="Evolvable plast, same taus",
                      save_path='images/{}/{}_fitness_curves.jpg'.format(subfold, fn))

#%% Condition 5: Evolvable plasticity with different predictabilities
# Fig 4D

plast = np.array([1, 1])
tot = 60000
CD_point = 52000
CDnp_point = 56000
a0 = np.array([5.01, 4.99])
sig_eps = np.sqrt(2)
tau = [0.9, 0.3]      # time between maturation and development
fn = "EvolveablePlasticity_diffLifeHistory"

a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2 = run_main(
    a0, b0, n0, plast, grow, A, B, Gaa, Gbb, kar, rho, tau, r, 
    sig_s, sig_u, sig_e, sig_eps, tot, seed
)
alpha = organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2)

n1 = np.mean(n[-avg_fin:,0])
n2 = np.mean(n[-avg_fin:,1])
sig_z1 = np.mean(np.sqrt(Gaa[0] + Gbb[0] * epss[-avg_fin:] ** 2 + sig_e ** 2))
sig_z2 = np.mean(np.sqrt(Gaa[1] + Gbb[1] * epss[-avg_fin:] ** 2 + sig_e ** 2))

# Main plot
fig, axis = plt.subplots(figsize=(10,8))
plt.title("Evolvable plast, diff taus", fontsize = 36)
plt.hlines(y=A[0], xmin=0, xmax=tot/1000, linestyle='--', color='black', label='Optima', linewidth=lw)
axis.plot(np.arange(0,tot/1000,0.001), alpha['a'][:,0], color='blue', label="$a̅$ species1", linewidth=lw)
axis.plot(np.arange(0,tot/1000,0.001), alpha['a'][:,1], color='red',  label="$a̅$ species2", linewidth=lw)
plt.ylim(1.0, 11.0)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36)
axis.set_ylabel("Phenotype in reference \n environment ($a̅$)", fontsize=36)

axis.annotate('', xy=(CD_point/1000, A[0]), 
              xytext=(CD_point/1000, A[0]+CD1), 
              arrowprops=dict(arrowstyle='<-', linestyle="-", linewidth=2, 
                              mutation_scale=30, color="blue"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), 
              xytext=(CDnp_point/1000, A[0]+CDnp1), 
              arrowprops=dict(arrowstyle='<-', linestyle=(0,(5,4)), linewidth=2, 
                              mutation_scale=30, color="blue"))
axis.annotate('', xy=(CD_point/1000, A[0]), 
              xytext=(CD_point/1000, A[0]-CD2), 
              arrowprops=dict(arrowstyle='<-', linestyle="-", linewidth=2, 
                              mutation_scale=30, color="red"))
axis.annotate('', xy=(CDnp_point/1000, A[0]), 
              xytext=(CDnp_point/1000, A[0]-CDnp2), 
              arrowprops=dict(arrowstyle='<-', linestyle=(0,(5,4)), linewidth=2, 
                              mutation_scale=30, color="red"))

# Inset
inset_axis = inset_axes(axis, width="23%", height="23%", loc='lower left', borderpad=6.5)
inset_axis.set_facecolor('white')  
inset_axis.plot(np.arange(0,tot/1000,0.001), 
                alpha['a'][:,0] + alpha['b'][:,0]*alpha["environmental_deviation_species1"], 
                color='blue', linewidth=2, alpha=0.5)
inset_axis.plot(np.arange(0,tot/1000,0.001), 
                alpha['a'][:,1] + alpha['b'][:,1]*alpha["environmental_deviation_species2"], 
                color='red',  linewidth=2, alpha=0.5)

inset_axis.tick_params(axis='both', which='major', labelsize=16)
inset_axis.set_xlabel("Generations (x1000)", fontsize=18)
inset_axis.set_ylabel("Phenotype $z̅$", fontsize=18)
plt.show()

dumsav = 'images/' + subfold + '/' + fn + '.jpg'
fig.savefig(dumsav, dpi=550)

# Plasticity figure
fig, axis = plt.subplots(figsize=(10,8))
plt.title("Evolvable plast, diff taus", fontsize = 36)
axis.tick_params(axis='both', which='major', labelsize=26)
axis.set_xlabel("Generations (x1000)", fontsize=36)
axis.set_ylabel("Mean plasticity ($b̅$)", fontsize=36)
axis.plot(np.arange(0, tot/1000, 0.001), alpha['b'][:,0], linewidth=lw, color='blue')
axis.plot(np.arange(0, tot/1000, 0.001), alpha['b'][:,1], linewidth=lw, color='red')
plt.hlines(y=B[0]*rho, xmin=0, xmax=tot/1000, linestyle = (0, (1, 3)), color='black', 
           linewidth=lw, label="expected plasticity")
axis.legend(fontsize=26)
plt.tight_layout()
plt.show()

dumsav = 'images/' + subfold + '/' + fn + '_plasticity.jpg'
fig.savefig(dumsav, dpi=550)

plot_tolerance_curves(alpha, Gaa, Gbb, sig_e, sig_s, A, B, r, rho, tau,
                      avg_fin=1000, 
                      fig_title="Evolvable plast, diff taus",
                      save_path='images/{}/{}_fitness_curves.jpg'.format(subfold, fn))
