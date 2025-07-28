#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 13:41:50 2024

@author: 1m
"""

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import leastsq
import urllib.request
import pandas as pd


# Step 1: Download from GitHub
url = "https://raw.githubusercontent.com/Itsyimi2002/uni/main/coding_samples/amrdata.txt"
response = urllib.request.urlopen(url)

# Step 2: Load using pandas
df = pd.read_csv(response, delim_whitespace=True, header=None)  # 如果是以空格/Tab 分隔的就用这个

# Step 3: Convert to numpy array
amrdata = df.values

# Step 4: Split columns
time = amrdata[:, 0]
resistant_ratio = amrdata[:, 1]

# --- Given parameters ---
r = 0.5  # h^-1
N_max = 1e7  # CFU/L
d = 0.025  # h^-1
E_max = 2
H = 2
MIC_S = 8  # µg/L
MIC_R = 2000  # µg/L
S0 = 9e6  # CFU/L
R0 = 1e5  # CFU/L
A0 = 5.6  # µg/L

# --- Defines the model equations ---
def amr_ode(y, t, params):
    S, R, A = y
    b, a, dA = params
    N = S + R

    E_S = 1 - E_max * A**H / (MIC_S + A**H)
    E_R = 1 - E_max * A**H / (MIC_R + A**H)

    dSdt = r * (1 - N / N_max) * E_S * S - d * S - (b * S * R) / N
    dRdt = r * (1 - a) * (1 - N / N_max) * E_R * R - d * R + (b * S * R) / N
    dAdt = -dA * A

    return [dSdt, dRdt, dAdt]

# --- Simulation function ---
def amr_run(pars, t):
    ode_params = pars
    ode_starts = [S0, R0, A0]
    out = odeint(amr_ode, ode_starts, t, args=(ode_params,))
    return out[:, 1] / (out[:, 0] + out[:, 1]) * 100  # Returns the resistant ratio

# --- Log likelihood function ---
def amr_loglik(pars, t, data, sig):
    simulated_ratio = amr_run(pars, t)
    return np.sum(norm.logpdf(simulated_ratio, data, sig))

# --- MCMC settings ---
reps = 1000
npars = 3  # We are estimating three parameters: b, a, and dA
sigma = resistant_ratio / 10  # 10% standard deviation assumption

par_out = np.ones(shape=(reps, npars))
par_out[0,] = [10**(-6.5), 0.5, 1/(4*7*24)]  # Starting values

accept = np.zeros(shape=(reps, npars))
propsigma = [0.5, 1.5, 1.5]  # Proposal sigmas

# --- MCMC loop ---
for i in range(1, reps):
    par_out[i,] = par_out[i-1,]
    for j in range(npars):
        proposed = np.copy(par_out[i,])
        if j == 0:  # log-normal for b
            proposed[j] = np.exp(np.log(proposed[j]) + np.random.normal(0, propsigma[j]))
        else:
            proposed[j] += np.random.normal(0, propsigma[j])

        if proposed[1] < 0 or proposed[1] > 1:
            continue  # reject invalid 'a'

        alpha = np.exp(amr_loglik(proposed, time, resistant_ratio, sigma) -
                       amr_loglik(par_out[i,], time, resistant_ratio, sigma))
        if np.random.rand() < alpha:
            par_out[i, j] = proposed[j]
            accept[i, j] = 1

# --- Plot: simulated vs. observed ---
plt.plot(time, resistant_ratio, 'o', label='Observed Data')
plt.plot(time, amr_run(par_out[-1,:], time), '-', label='Simulated (Last Sample)')
plt.legend()
plt.xlabel('Time (hours)')
plt.ylabel('Resistant Ratio (%)')
plt.show()

# --- Trace plots of MCMC samples ---
param_names = ['b', 'a', 'dA']
fig, axs = plt.subplots(npars, figsize=(12, 7))
for i in range(npars):
    axs[i].plot(par_out[:, i], label=f'{param_names[i]} Trace')
    axs[i].set_title(f'{param_names[i]} Trace')
    axs[i].set_xlabel('Iterations')
    axs[i].set_ylabel(param_names[i])
    axs[i].legend()
plt.tight_layout()
plt.suptitle('Test Simulation with 1000 Replicates', y=1.05)
plt.show()

# --- Sliding window acceptance plots ---
def calculate_sliding_acceptance(acceptance_array, window=50):
    return np.convolve(acceptance_array, np.ones(window)/window, mode='valid')

fig, axs = plt.subplots(npars, figsize=(10, 10))
for i in range(npars):
    sliding_acceptance = calculate_sliding_acceptance(accept[:, i], window=50)
    axs[i].plot(sliding_acceptance, label=f'Sliding Acceptance for {param_names[i]}')
    axs[i].set_title(f'Acceptance plot for {param_names[i]}')
    axs[i].set_xlabel('Iteration')
    axs[i].set_ylabel('Acceptance Rate')
    axs[i].legend()

    overall_acceptance_rate = np.mean(accept[:, i])
    text_box = f'Overall acceptance rate: {overall_acceptance_rate:.2f}'
    axs[i].text(0.05, 0.95, text_box, transform=axs[i].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.tight_layout()
plt.show()
