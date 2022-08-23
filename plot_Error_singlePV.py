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



def plot_erroractivity(mismatch_results,training_mean,mean_range,sigma_range,name):
    #mean_range=mismatch_results.keys()
    #sigma_range=mismatch_results[mean_range[0]].keys()
    
    E_P_rates=np.empty((len(mean_range),len(sigma_range)))
    E_N_rates=np.empty((len(mean_range),len(sigma_range)))
    E_P_analytical=np.empty((len(mean_range),len(sigma_range)))
    E_N_analytical=np.empty((len(mean_range),len(sigma_range)))
    for i,mean in enumerate(mean_range):
        for j,sigma in enumerate(sigma_range):
            print(mismatch_results[sigma][i]['E_P_avg'])
            E_P_rates[i,j] = mismatch_results[sigma][i]['E_P_avg']
            E_N_rates[i,j] = mismatch_results[sigma][i]['E_N_avg']
            #E_P_std[i,j] = mismatch_results[mean][sigma]['E_P_std']
            #E_N_std[i,j] = mismatch_results[mean][sigma]['E_N_std']
            diff_E = np.max((mean-training_mean,0))
            diff_N = np.max((training_mean-mean,0))
            E_P_analytical[i,j] = (1/(1.0+(0.9*sigma)**2))*diff_E
            E_N_analytical[i,j] = (1/(1.0+(0.9*sigma)**2))*diff_N

    plt.figure(figsize=(8,5))
    #plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))
    plt.subplot(111)
    for k,sigma in enumerate(sigma_range):
        print(1-k*0.1)
        #plt.plot(mean_range, E_P_analytical[:,k], color='k')

        plt.plot(mean_range, E_P_rates[:,k], '.', color=cm.viridis(1-k*0.1))
        #plt.plot(mean_range, E_N_analytical[:,k], color='k')
        plt.plot(mean_range, E_N_rates[:,k], color=cm.viridis(1-k*0.1),label=r'$\sigma=%.1f$'%sigma)


    # works for wP=3.0
    #plt.xlim(0.1,1.0)
    #plt.ylim(0.1,1.0)
    plt.xticks(mean_range,mean_range-training_mean,fontsize=16)
    plt.yticks(np.arange(0,2.1,0.5),np.arange(0,2.1,0.5),fontsize=16)
    plt.xlabel(r'$s-\mu$',fontsize=20)
    plt.ylabel(r'$\Upsilon\  rate$',fontsize=20)
    #plt.legend(bbox_to_anchor=(1,1), fontsize=16, loc="upper left")
    plt.legend(bbox_to_anchor=(1,1), fontsize=16, loc="upper left")

    plt.tight_layout()
    plt.savefig('./Eratemeanssigmas%s.png'%name, bbox_inches='tight')
    plt.savefig('./Eratemeanssigmas%s.pdf'%name, bbox_inches='tight')
    
    plt.figure(figsize=(7,3))
    #plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))
    plt.subplot(121)
    for k,mean in enumerate(mean_range):
        if mean>5.0:
            plt.plot(sigma_range, E_P_rates[k,:], color=cm.magma((mean-5.0)*0.2),label=r'$|s-\mu|=%.f$'%np.abs(mean-training_mean))
            #plt.plot(sigma_range, E_N_rates[k,:], color=cm.magma(mean*0.1),label=r'$E^- \mu=%.f$'%mean)
            #plt.plot(sigma_range, E_P_analytical[k,:], '--',color=cm.viridis((mean-5.0)*0.1))
        #plt.plot(sigma_range, E_N_analytical[k,:], '--', color=cm.magma(mean*0.1))

    # works for wP=3.0
    #plt.xlim(0.1,1.0)
    #plt.ylim(0.1,1.0)
    plt.xticks([.1,.9,1.7],[.1,.9,1.7],fontsize=16)
    plt.yticks(np.arange(0,3,1),np.arange(0,3,1),fontsize=16)
    plt.legend(bbox_to_anchor=(1,1), fontsize=16, loc="upper left")

    plt.xlabel(r'$\sigma$',fontsize=20)
    plt.ylabel(r'$\Upsilon^+$ rate',fontsize=20)
    #plt.legend(bbox_to_anchor=(1,1), fontsize=16, loc="upper left")

    plt.subplot(122)
    for k,mean in enumerate(mean_range[mean_range<5.0]):
        plt.plot(sigma_range, E_N_rates[k,:], '--',color=cm.magma((5.0-mean)*0.2))
        #plt.plot(sigma_range, E_N_analytical[k,:], '--', color=cm.viridis((5.0-mean)*0.1))

    # works for wP=3.0
    #plt.xlim(0.1,1.0)
    #plt.ylim(0.1,1.0)
    plt.xticks([.1,.9,1.7],[.1,.9,1.7],fontsize=16)
    plt.yticks(np.arange(0,3,1),np.arange(0,3,1),fontsize=16)
    plt.xlabel(r'$\sigma$',fontsize=20)
    plt.ylabel(r'$\Upsilon^-$ rate',fontsize=20)
    plt.tight_layout()
    plt.savefig('./Eratesigmasmeans%s.png'%name, bbox_inches='tight')
    plt.savefig('./Eratesigmasmeans%s.pdf'%name, bbox_inches='tight')


if __name__ == "__main__":

    seed = 123
    training_mean = 5.0 
    sigmas = np.arange(0.1,1.1,0.3)
    beta = 0.1
    wP = np.sqrt(((2-beta)/beta)*(1/2)) # np.sqrt(1.95/0.1)
    circuit = Circuit()
    circuit.PV_mu = False
    circuit.wPY1 = np.array([wP]) # small intitial weights
    circuit.wPR = np.array([wP]) # small intitial weights
    circuit.wPS_P = np.array([wP])
    circuit.wPS_N = np.array([wP])
    sim = Sim(stimulus_duration=1,number_of_samples=200000)
    with multiprocessing.Pool(processes=5) as pool:
        training_results=pool.starmap(sim.run, zip(repeat(circuit),repeat(training_mean),sigmas,repeat(seed)))

    means=np.arange(1.0,9.0,1.0)
    mismatch_results = {}
    for i,sigma in enumerate(sigmas):
        print(np.mean(training_results[i]['wPX1'][-100000:]))
        print(training_results[i]['wPX1'][-1])
        circuit = Circuit()
        circuit.PV_mu = False
        circuit.Rrate = training_mean
        circuit.plastic_PX=False
        circuit.wPY1 = np.array([wP]) # small intitial weights
        circuit.wPR = np.array([wP]) # small intitial weights
        circuit.wPS_P = np.array([wP])
        circuit.wPS_N = np.array([wP])
        circuit.wRX1 = training_results[i]['wRX1'][-1]
        circuit.wPX1 = training_results[i]['wPX1'][-1]
        #circuit.wPX1_N = training_results[i]['wPX1_N'][-1]
        sim = Sim(stimulus_duration=3,number_of_samples=200000)
        with multiprocessing.Pool(processes=5) as pool:
            mismatch_results[sigma]=pool.starmap(sim.run, zip(repeat(circuit),means,repeat(0.0),repeat(seed)))     

    plot_erroractivity(mismatch_results,training_mean,means,sigmas,'')
