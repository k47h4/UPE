from Circuit import * 
from plotting_functions import * 

import multiprocessing
from multiprocessing.pool import ThreadPool

from itertools import repeat

from functools import partial
import time



if __name__ == "__main__":
    seed = 123
    Y1_mean = 5.0
    Y2_mean = 5.0
    Y1_sigma = 0.8
    Y2_sigma = 0.4
    mean = 5.0
    sigma=0.5
    #wP = np.sqrt(1.95/0.1)
    betas = np.array([0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.5,0.9])
    #wPs = np.empty_like(betas)
    #for i,beta in enumerate(betas):
    #    wPs[i] = np.sqrt((2-beta)/beta) 
        
        #results1 = run_pPEnetwork(mean= Y1_mean,sigma=Y1_sigma,new_rule=False,wP=wP,beta_P=beta)
        #results2 = run_pPEnetwork(mean= Y2_mean,sigma=Y2_sigma,new_rule=False,wP=wP,beta_P=beta)
        

    circuits = []
    for beta in betas:
        circuit = Circuit()
        circuit.beta_P = beta
        wP = np.sqrt((2-beta)/beta) 
        circuit.wPY1 = np.array([wP]) # small intitial weights
        circuit.wPS_P = np.array([wP])
        circuits.append(circuit)
    sim = Sim(stimulus_duration=1, number_of_samples=600000)
    with multiprocessing.Pool(processes=5) as pool:
        results=pool.starmap(sim.run_pPE, zip(circuits,repeat(mean),repeat(sigma),repeat(seed)))


    PVa_avg = np.empty_like(betas)
    PVa_std = np.empty_like(betas)
    wPVa_avg = np.empty_like(betas)
    wPVa_std = np.empty_like(betas)
    for i in range(len(betas)):
        PVa_avg[i] = np.mean(results[i]['rPa'][300000:])
        PVa_std[i] = np.std(results[i]['rPa'][300000:])
        wPVa_avg[i] = np.mean(results[i]['wPX1'][300000:])
        wPVa_std[i] = np.std(results[i]['wPX1'][300000:])
        


    plt.figure(figsize=(5,4))
    a1 = plt.subplot(111)
    #a1.text(-0.1, 1.15, 'A', transform=a1.transAxes,
    #          fontsize=16, va='top', ha='right')
    a1.errorbar(betas, PVa_avg, yerr=PVa_std, linestyle='',marker='.',color='k')
    a1.plot(np.arange(0,3),np.ones(3)*sigma**2,'--',color ='k',label=r'$\sigma^2$')
    a1.set_xscale('log')

    #plt.xlim(-.1,1.1)
    #plt.ylim(-.1,1.1)

    #plt.ylim(-.1,2.0)
    #a1.set_xticklabels(fontsize=16)
    a1.set_yticks(np.arange(0.0,0.6,.5))
    a1.set_yticklabels([0.0,.5],fontsize=16)
    a1.set_xlim(0,1.0)
    a1.set_xlabel(r'$\beta$',fontsize=16)
    a1.set_ylabel(r'$r_{PV}(a)$',fontsize=16)
    a1.spines['top'].set_visible(False)
    a1.spines['right'].set_visible(False)
    a1.set_ylim(0,0.52)
    a2 = a1.twinx()  # instantiate a second axes that shares the same x-axis
    a2.errorbar(betas, wPVa_avg, yerr=wPVa_std, linestyle='',marker='+',color=cm.viridis(0.8))
    a2.plot(np.arange(0,3),np.ones(3)*sigma,'--',color =cm.viridis(.8),label=r'$\sigma^2$')
    a2.set_ylabel(r'$w_{PV,a}$',fontsize=16)
    a2.set_ylim(0,0.52)
    a2.set_yticks(np.arange(0.0,0.6,.5))
    a2.set_yticklabels([0.0,.5],fontsize=16)
    a1.tick_params(axis='x', labelsize=16)
    a2.spines['top'].set_visible(False)



    plt.legend(bbox_to_anchor=(1,1), fontsize=16, loc="upper left")

    plt.tight_layout()

    plt.savefig('./beta.png', bbox_inches='tight')
    plt.savefig('./beta.pdf', bbox_inches='tight')