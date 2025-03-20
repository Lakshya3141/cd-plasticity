# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:21:21 2023
These functions are used to simulate a single run of the model
@author: laksh
"""

import numpy as np
import numba as nb
import time
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
import matplotlib as mpl
import math
import matplotlib.pyplot as plt

# Increase the path chunksize limit
mpl.rcParams['agg.path.chunksize'] = 1000  # You can adjust the value as needed

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

# Forms the next two steps of autocorrelated environment values for varying tau values
# used in run_main_tau -> {further used in single_run, Exploration_VariedTau}
@nb.jit(nopython=True, nogil=True)
def env_tau(sel, rho, tau1, tau2, sig_eps):
    if tau1 > tau2:
        b = rho ** ((1 - tau1) / tau1) * sel + np.sqrt(1 - rho ** (2 * (1 - tau1) / tau1)) * np.random.normal(0, sig_eps)
        c = rho ** ((tau1 - tau2) / tau1) * b + np.sqrt(1 - rho ** (2 * (tau1 - tau2) / tau1)) * np.random.normal(0, sig_eps)
        d = rho ** ( tau2 / tau1) * c + np.sqrt(1 - rho ** (2 * tau2 / tau1)) * np.random.normal(0, sig_eps)
        return b, c, d 
    elif tau1 <= tau2:
        b = rho ** ((1 - tau2) / tau1) * sel + np.sqrt(1 - rho ** (2 * (1 - tau2) / tau1)) * np.random.normal(0, sig_eps)
        c = rho ** ((tau2 - tau1) / tau1) * b + np.sqrt(1 - rho ** (2 * (tau2 - tau1) / tau1)) * np.random.normal(0, sig_eps)
        d = rho * c + np.sqrt(1 - rho ** 2) * np.random.normal(0, sig_eps)
        return c, b, d

# Gives development time for a particular tau given previous selection time
# Used in exploration of things where taus is same for both species!
@nb.jit(nopython=True, nogil=True)
def dev_env(sel,rho,tau,sig_eps):
    return(rho ** ((1 - tau) / tau) * sel
    + np.sqrt(1 - rho ** (2 * (1 - tau) / tau)) * np.random.normal(0, sig_eps))

# Gives selection time for a particular tau given previous development time
# Used in exploration of things where taus is same for both species!
@nb.jit(nopython=True, nogil=True)
def sel_env(dev,rho,tau,sig_eps):
    return(rho * dev + np.sqrt(1 - rho ** 2) * np.random.normal(0, sig_eps))

# Calculates dist-from-optimal selection component of malthusian fitness
@nb.jit(nopython=True, nogil=True)
def mls_val(z1, theta1, sig_z, sig_s):
    return (-(sig_z**2 + (theta1 - z1)**2)/(2*sig_s**2))

# Calculates competition component of malthusian fitness
@nb.jit(nopython=True, nogil=True)
def mlc_val(z1, z2, n1, n2, r, kar, sig_u, sig_z):
    return (-r / kar) * np.sqrt(sig_u**2 / (sig_u**2 + sig_z**2)) * (n1 + n2 * np.exp(-(z1 - z2)**2 / (4 * (sig_u**2 + sig_z**2))))

# Calculates SG from dist-from-optimal selection component    
@nb.jit(nopython=True, nogil=True)
def sgs_val(z1, theta1, sig_s):
    return (theta1-z1)/(sig_s**2)

# Calculates SG from competition component 
@nb.jit(nopython=True, nogil=True)
def sgc_val(z1, z2, n2, r, kar, sig_u, sig_z):
    return (n2 / kar) * (np.exp(-(z1 - z2)**2 / (4 * (sig_u**2 + sig_z**2))) * r * (z1 - z2) * sig_u) / (2 * (sig_u**2 + sig_z**2)**(3/2))

# Calculates population level at next time point for species given
# malthusian fitness and grow[i] == 1
# species i's population remains constant if grow = [i] == 0
@nb.jit(nopython=True, nogil=True)
def pop_grow(n, grow, ml):
    return [np.exp(ml[0])*n[0] if grow[0] == 1 else n[0],
            np.exp(ml[1])*n[1] if grow[1] == 1 else n[1]]
   
 
# Function to simulate population
def run_main(a0, b0, n0, plast, grow, A, B, Gaa, Gbb, kar, rho, tau, r, sig_s, sig_u, sig_e, sig_eps, tot, seed):
    
    tau1 = tau[0]
    tau2 = tau[1]
    
    np.random.seed(int(time.time())) if seed < 0 else np.random.seed(seed)
    
    B_local = np.copy(B)
    Gbb_local = np.copy(Gbb)
    b0_local = np.copy(b0)
    
    # Modify local copies based on plast
    for i in range(0, 2):
        if plast[i] <= 0: Gbb_local[i] = 0  # Modify local Gbb
        if plast[i] < 0: b0_local[i] = 0  # Modify local b0
        if plast[i] == -2: B_local[i] = 0  # Modify local B

    a = [a0]
    b = [b0_local]  # Use local b0
    n = [n0]
    mls = []
    mlc = []
    sgs = []
    sgc = []
    z = []
    theta = []
    dinit = np.random.normal(0, sig_eps)
    eps1 = [dinit]
    eps2 = [dinit]
    epss = []
    epsd1 = []
    epsd2 = []

    start = time.time()
    for i in range(tot):
        
        d1, d2, sng = env_tau(eps1[-1], rho, tau1, tau2, sig_eps)
        
        eps1.append(d1)
        eps2.append(d2)
        eps1.append(sng)
        eps2.append(sng)
        
        epsd1.append(eps1[-2])
        epsd2.append(eps2[-2])
        
        epss.append(sng)
        
        d_dum = np.array([eps1[-2], eps2[-2]])
        
        theta.append(list(A + B_local * epss[-1]))  # Use local B
        z.append(a[-1] + b[-1] * d_dum)
        sig_z = np.sqrt(Gaa + Gbb_local * d_dum ** 2 + sig_e ** 2)  # Use local Gbb
        
        mls.append(np.array([mls_val(z[-1][0], theta[-1][0], sig_z[0], sig_s),
                            mls_val(z[-1][1], theta[-1][1], sig_z[1], sig_s)]))
        
        mlc.append(np.array([mlc_val(z[-1][0], z[-1][1], n[-1][0], n[-1][1], r, kar[0], sig_u, sig_z[0]),
                             mlc_val(z[-1][1], z[-1][0], n[-1][1], n[-1][0], r, kar[1], sig_u, sig_z[1])]))
        
        sgs.append(np.array([sgs_val(z[-1][0], theta[-1][0], sig_s),
                             sgs_val(z[-1][1], theta[-1][1], sig_s)]))
        
        sgc.append(np.array([sgc_val(z[-1][0], z[-1][1], n[-1][1], r, kar[0], sig_u, sig_z[0]),
                             sgc_val(z[-1][1], z[-1][0], n[-1][0], r, kar[1], sig_u, sig_z[1])]))
        
        a.append(a[-1] + (sgs[-1] + sgc[-1]) * Gaa)
        b.append(b[-1] + (sgs[-1] + sgc[-1]) * Gbb_local * d_dum)  # Use local Gbb
        n.append(pop_grow(n[-1], grow, r + mls[-1] + mlc[-1]))
    
    t_run = time.time() - start
    print(f'For loop: {t_run} seconds')
    
    a = np.array(a[1:])
    b = np.array(b[1:])
    n = np.array(n[1:])
    mls = np.array(mls)
    mlc = np.array(mlc)
    sgs = np.array(sgs)
    sgc = np.array(sgc)
    epss = np.array(epss)
    epsd1 = np.array(epsd1)
    epsd2 = np.array(epsd2)
    eps1 = eps1[1:]
    eps2 = eps2[1:]
    
    return a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2

# Function to find the fundamental niche of an evolved population
# Not used, additional
def niche_finder_fund(af_, bf_, nf_, epsran, A, B, Gaa, Gbb, r, kar, sig_eps, sig_e, sig_s, sig_u):
    mlf1 = []
    mlf2 = []
    for i, e in enumerate(epsran):
        z1 = af_[0] + bf_[0]*e
        z2 = af_[1] + bf_[1]*e
        thet1 = A[0] + B[0]*e
        thet2 = A[1] + B[1]*e
        sig_z = Gaa + Gbb*sig_eps**2 + sig_e**2
        dum1 = mls_val(z1, thet1, sig_z[0], sig_s)
        mlf1.append(r + dum1)
        
        dum1 = mls_val(z2, thet2, sig_z[1], sig_s)
        mlf2.append(r + dum1)
    return mlf1, mlf2

# Function to find niche an evolved population under competition!!
# Not used, additional
def niche_finder_comp(af_, bf_, nf_, epsran, A, B, Gaa, Gbb, r, kar, sig_eps, sig_e, sig_s, sig_u):
    mlf1 = []
    mlf2 = []
    for i, e in enumerate(epsran):
        z1 = af_[0] + bf_[0]*e
        z2 = af_[1] + bf_[1]*e
        thet1 = A[0] + B[0]*e
        thet2 = A[1] + B[1]*e
        n1 = nf_[0]
        n2 = nf_[1]
        sig_z = Gaa + Gbb*sig_eps**2 + sig_e**2
        dum1 = mls_val(z1, thet1, sig_z[0], sig_s)
        dum2 = mlc_val(z1, z2, n1, n2, r, kar[0], sig_u, sig_z[0])
        mlf1.append(r + dum1 + dum2)
        
        dum1 = mls_val(z2, thet2, sig_z[1], sig_s)
        dum2 = mlc_val(z2, z1, n2, n1, r, kar[1], sig_u, sig_z[1])
        mlf2.append(r + dum1 + dum2)
    return mlf1, mlf2

# Function to organize simulation output in a more accessible fashion
def organize_simulation_output(a, b, n, mls, mlc, sgs, sgc, epss, epsd1, epsd2, t_run, eps1, eps2):
    output = {
        "a": a,
        "b": b,
        "n": n,
        "mean_phenotype_species1": mls,
        "mean_phenotype_species2": mlc,
        "genetic_variance_species1": sgs,
        "genetic_variance_species2": sgc,
        "environmental_variance": epss,
        "environmental_deviation_species1": epsd1,
        "environmental_deviation_species2": epsd2,
        "generation_number": t_run,
        "environmental_variance_species1": eps1,
        "environmental_variance_species2": eps2
    }
    return output

#%% Functions used in plotting figures for manuscript for Fig 1

# the expected character displacement, based on eq. (11) combined 
# with the mean population size computed over the last 1000 generations in the simulation
def no_fluc_CD(sigma_u, sigma_z, sigma_s, r, N, K):
    term1 = sigma_u**2 + sigma_z**2
    term2 = (sigma_u * sigma_s**2) / (sigma_u**2 + sigma_z**2)**(3/2)
    inner_sqrt = term1 * math.log((r * N / K) * term2)
    result = math.sqrt(inner_sqrt)
    return result

# the expected character displacement, based on the approximation in eq. (15)
# that assumes the population size is at the equilibrium from eq. (13) in a non-fluctuating environment
def NoFluc_noPop(sigma_u, sigma_z, sigma_s, r):
    numerator = 2 * sigma_s**2 * r - sigma_z**2
    denominator = 4 * (sigma_u**2 + sigma_z**2)
    inner_log = numerator / denominator
    if inner_log <= 0:
        return None  # Return None for imaginary results
    else:
        result = math.sqrt((sigma_u**2 + sigma_z**2) * math.log(inner_log))
        return result
    
# the expected character displacement, based on the approximation in eq. (15)
# that assumes the population size is at the equilibrium from eq. (17) in a fluctuating environment
def Fluc_noPop(sigma_u, sigma_z, sigma_s, r, Bi, sigma_eps):
    numerator = 2 * sigma_s**2 * r - sigma_z**2 - (Bi**2)*(sigma_eps**2)
    denominator = 4 * (sigma_u**2 + sigma_z**2)
    inner_log = numerator / denominator
    if inner_log <= 0:
        return None  # Return None for imaginary results
    else:
        result = math.sqrt((sigma_u**2 + sigma_z**2) * math.log(inner_log))
        return result
    
# Function to plot multiplicative fitness from equilibrium state 
# averaged over the last avg_fin generations
def plot_tolerance_curves(alpha, Gaa, Gbb, sig_e, sigma_s, A, B, r, rho, tau, avg_fin=1000, fig_title="Tolerance Curves", save_path=None):  # <-- Add r here if you want to incorporate it in the exponent
    species_colors=['blue','red']
    species_labels=['Species 1','Species 2']
    # 1) Grab the last `avg_fin` data
    a_vals = alpha['a'][-avg_fin:]   # shape (avg_fin, 2)
    b_vals = alpha['b'][-avg_fin:]   # shape (avg_fin, 2)
    epss   = alpha['environmental_variance'][-avg_fin:]  # shape (avg_fin, )

    # 2) Compute the mean a, b, sigma_z for each species
    mean_a = []
    mean_b = []
    mean_sigz = []
    
    for i in range(2):
        # Averages over last avg_fin steps
        a_i = np.mean(a_vals[:, i])
        b_i = np.mean(b_vals[:, i])
        # compute sigma_z(t) = sqrt(Gaa[i] + Gbb[i]*epss(t)^2 + sig_e^2), then average
        sigma_z_over_time = np.sqrt(Gaa[i] + Gbb[i] * epss**2 + sig_e**2)
        sigz_i = np.mean(sigma_z_over_time)
        
        mean_a.append(a_i)
        mean_b.append(b_i)
        mean_sigz.append(sigz_i)

        # ---- Print debugging info for each species ----
        # print(f"\n[{species_labels[i]} Debug Info]")
        # print(f"  mean_a   = {a_i:.4f}")
        # print(f"  mean_b   = {b_i:.4f}")
        # print(f"  mean_sigz= {sigz_i:.4f}")
        # print(f"  A        = {A[i]:.4f}")
        # print(f"  B        = {B[i]:.4f}")
        # print(f"  r        = {r:.4f}")
        # print(f"  sigma_s  = {sigma_s:.4f}")
        # print("-------------------------------------------------")

    # 3) Evaluate Wbar_i(eps) on a range of eps from -20 to 20
    eps_range = np.linspace(-20, 20, 200)  # 200 points in [-20,20]
    
    exp_b = [B[0]*rho, B[1]*rho**(tau[1]/tau[0])]
    fig, ax = plt.subplots(figsize=(10,8))
    for i in range(2):
        a_i    = mean_a[i]
        b_i    = mean_b[i]
        sigz_i = mean_sigz[i]

        # Full formula: W_i(eps) = exp( r - [((a_i - A_i) + (b_i - B_i)*eps)^2 + sigz_i^2]/(2*sigma_s^2) )
        numerator = ((a_i - A[i]) + (b_i - B[i])*eps_range)**2 + sigz_i**2
        denom     = 2*(sigma_s**2)
        exponent  = r - numerator/denom
        W_i       = np.exp(exponent)

        ax.plot(eps_range, W_i, color=species_colors[i], label=species_labels[i], lw=3)
        # ax.plot(eps_range, W_i, color=species_colors[i], lw=3)
        
    for i in range(2):
        a_i    = mean_a[i]
        b_i    = exp_b[i]
        sigz_i = mean_sigz[i]

        # Full formula: W_i(eps) = exp( r - [((a_i - A_i) + (b_i - B_i)*eps)^2 + sigz_i^2]/(2*sigma_s^2) )
        numerator = ((a_i - A[i]) + (b_i - B[i])*eps_range)**2 + sigz_i**2
        denom     = 2*(sigma_s**2)
        exponent  = r - numerator/denom
        W_i       = np.exp(exponent)

        ax.plot(eps_range, W_i, color=species_colors[i], label="with exp. plasticity", lw=3, linestyle = "--")
        # ax.plot(eps_range, W_i, color=species_colors[i], lw=3, linestyle = "--")

    ax.set_title(fig_title, fontsize=36)
    ax.set_xlabel("Environment (ε)", fontsize=36, labelpad=15)
    ax.set_ylabel("Mean Multiplicative Fitness", fontsize=36, labelpad=15)
    ax.set_ylim([0.0, 1.2])  # as requested
    ax.set_xlim([-20, 20])
    ax.legend(fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=550)
    plt.show()