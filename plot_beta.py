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
        


    a6 = plt.subplot(111)
    a6.text(-0.1, 1.15, 'A', transform=a6.transAxes,
              fontsize=16, va='top', ha='right')
    plt.errorbar(betas, PVa_avg, yerr=PVa_std, linestyle='',marker='.',color='k')
    plt.errorbar(betas, wPVa_avg, yerr=wPVa_std, linestyle='',marker='+',color=cm.viridis(0.8))
    plt.plot(np.arange(0,2),np.ones(2)*sigma**2,'--',color ='k',label=r'$\sigma^2$')
    plt.plot(np.arange(0,2),np.ones(2)*sigma,'--',color =cm.viridis(.8),label=r'$\sigma^2$')
    a6.set_xscale('log')

    #plt.xlim(-.1,1.1)
    #plt.ylim(-.1,1.1)

    #plt.ylim(-.1,2.0)
    #plt.xticks(np.arange(0.0,1.2,.5),[0.0,.5,1.0],fontsize=16)
    #plt.yticks(np.arange(0.0,1.2,.5),[0.0,.5,1.0],fontsize=16)

    plt.xlabel(r'$\beta$',fontsize=16)
    plt.ylabel(r'$r_{PV}(a)$ / $w_{PV,a}$',fontsize=16)
    a6.spines['top'].set_visible(False)
    a6.spines['right'].set_visible(False)


    plt.tight_layout()

    plt.savefig('./beta.png', bbox_inches='tight')
    plt.savefig('./beta.pdf', bbox_inches='tight')