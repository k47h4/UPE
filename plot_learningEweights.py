import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from plotting_functions import * 
from Circuit import * 
import multiprocessing
from multiprocessing.pool import ThreadPool
from itertools import repeat
from functools import partial
import time


if __name__ == "__main__":


	mean = 1.0
	sigma = 0.1
	seed = 123
	start = 100000
	wEs = np.arange(1.0,6.0,1.0)
	wES_Ps_std = np.empty_like(wEs)
	wES_Ps = np.empty_like(wEs)
	circuits = []

	for i,wE in enumerate(wEs):
	    circuit = Circuit()
	    circuit.plastic_ES = True
	    circuit.wEY1_P = wE
	    circuit.wER_N = wE
	    circuit.wES_P = 0.01
	    circuit.wES_N = 0.01

	    circuits.append(circuit)

	    sim = Sim(stimulus_duration=3, number_of_samples=200000)
	    #results = sim.run_pPE(circuit,mean=mean,sigma=sigma,seed=seed)
	    #wPSs[i] = results['wPS_P'][-1]
	    #wPSs_std[i] = np.std(results['wPS_P'][100000:])
	    #print('wP')
	    #print(circuit.wP)
	    #print(wPSs)
	    #print(wPSs_std)

	with multiprocessing.Pool(processes=5) as pool:
	    results=pool.starmap(sim.run, zip(circuits,repeat(mean),repeat(sigma),repeat(seed)))


	wES_Ps_std = np.empty_like(wEs)
	wES_Ps = np.empty_like(wEs)
	wES_Ns_std = np.empty_like(wEs)
	wES_Ns = np.empty_like(wEs)

	for i,wE in enumerate(wEs):
	    wES_Ps[i] = np.mean(results[i]['wES_P'][100000:])
	    wES_Ps_std[i] = np.std(results[i]['wES_P'][100000:])
	    wES_Ns[i] = np.mean(results[i]['wES_N'][100000:])
	    wES_Ns_std[i] = np.std(results[i]['wES_N'][100000:])

	name = ''

	plt.figure(figsize=(6,3))
	#plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))
	plt.subplot(121)
	plt.errorbar(wEs,wES_Ps, yerr=wES_Ps_std, linestyle='',marker='_',color='k')

	plt.xlim(0.5,5.5)
	plt.ylim(0.5,5.5)#
	plt.xticks(np.arange(1,6),[1,2,3,4,5],fontsize=12)
	plt.yticks(np.arange(1,6),[1,2,3,4,5],fontsize=12)

	plt.xlabel(r'$w_{UPE^+,s}$',fontsize=16)
	plt.ylabel(r'$w_{UPE^+,SST}$',fontsize=16)
	plt.subplot(122)
	plt.errorbar(wEs,wES_Ns, yerr=wES_Ns_std, linestyle='',marker='_',color='k')

	plt.xlim(0.5,5.5)
	plt.ylim(0.5,5.5)
	plt.xticks(np.arange(1,6),[1,2,3,4,5],fontsize=12)
	plt.yticks(np.arange(1,6),[1,2,3,4,5],fontsize=12)

	plt.xlabel(r'$w_{UPE^-,R}$',fontsize=16)
	plt.ylabel(r'$w_{UPE^-,SST}$',fontsize=16)

	plt.tight_layout()
	plt.savefig('./LearningUPEweights.png', bbox_inches='tight')
	plt.savefig('./LearningUPEweights.pdf', bbox_inches='tight')
