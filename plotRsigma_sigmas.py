import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as stats
from sklearn.utils import shuffle
from Circuit import * 
from plotting_functions import * 

import multiprocessing
from multiprocessing.pool import ThreadPool

from itertools import repeat

from functools import partial
import time


if __name__ == "__main__":

    seed = 123 
    beta = 0.1 
    dt = 0.1
    stimulus_duration = 10
    wP = np.sqrt(((2-beta)/beta)) # np.sqrt(1.95/0.1)
    sigmas=np.arange(0.1,2.1,0.2)
    circuit = Circuit()
    circuit.dt = 0.1
    circuit.R_neuron= True
    circuit.single_PV = False
    #circuit.PV_mu = False
    #circuit.NMDA = True
    circuit.beta_P = beta
    circuit.wPY1 = np.array([wP]) # small intitial weights
    circuit.wPR = np.array([wP]) # small intitial weights
    circuit.wPS_P = np.array([wP])
    circuit.wPS_N = np.array([wP])
    circuit.error_weighting=1.0
    sim = Sim(stimulus_duration=stimulus_duration*4,number_of_samples=200000)
    with multiprocessing.Pool(processes=5) as pool:
        sigma_results=pool.starmap(sim.run_fakePV, zip(repeat(circuit),repeat(5),sigmas,repeat(seed)))  

    R_std = np.empty(len(sigmas))
    wRX_std = np.empty(len(sigmas))

    for i,sigma in enumerate(sigmas):
        R_std[i] = np.std(sigma_results[i]['rRa'][100000:])
        wRX_std[i] = np.std(sigma_results[i]['wRX1'][100000:])

    plt.figure(figsize=(6,3))
    #plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))
    a2 = plt.subplot(121)
    plt.errorbar(sigmas, wRX_std, linestyle='',marker='.',color='k')
    a2.spines['top'].set_visible(False)
    a2.spines['right'].set_visible(False)
    plt.xlim(-.1,2.0)
    plt.ylim(-.1,2.0)
    plt.xticks(np.arange(0.5,2.0,.5),[0.5,1.0,1.5],fontsize=16)
    plt.yticks(np.arange(0.5,2.0,.5),[0.5,1.0,1.5],fontsize=16)

    plt.xlabel(r'$\sigma_s$',fontsize=16)
    plt.ylabel(r'$\sigma$ of $w_{R,a}$',fontsize=16)
    a3 = plt.subplot(122)
    #plt.plot(sigmas**2, PV_avg, color='k')
    plt.errorbar(sigmas, R_std, linestyle='',marker='.',color='k')


    plt.xlim(-.1,2.0)
    plt.ylim(-.1,2.0)
    a3.spines['top'].set_visible(False)
    a3.spines['right'].set_visible(False)
    plt.xticks(np.arange(0.5,2.0,.5),[0.5,1.0,1.5],fontsize=16)
    plt.yticks(np.arange(0.5,2.0,.5),[0.5,1.0,1.5],fontsize=16)
    plt.xlabel(r'$\sigma_s$',fontsize=16)
    plt.ylabel(r'$\sigma$ of $r_{R}$',fontsize=16)
    plt.tight_layout()
    plt.savefig('./Rsigma_fakePVEW1dt01.png', bbox_inches='tight')
    plt.savefig('./Rsigma_fakePVEW1dt01.pdf', bbox_inches='tight')