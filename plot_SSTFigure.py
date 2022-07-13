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
	seed= 123
	beta = 0.1
	means = np.arange(1.0,6.0) 
	sigma = 1.0
	wP = np.sqrt((2-beta)/beta) # np.sqrt(1.95/0.1)
	#sigmas = np.arange(0.1,1.1,0.1)

	circuit = Circuit()
	circuit.beta_P = beta
	circuit.wPY1 = np.array([wP]) # small intitial weights
	circuit.wPS_P = np.array([wP])
	circuit.plastic_SX = True
	sim = Sim(stimulus_duration=1)
	with multiprocessing.Pool(processes=5) as pool:
	    results=pool.starmap(sim.run_pPE, zip(repeat(circuit),means,repeat(sigma),repeat(seed)))

	SST_avg = np.empty_like(means)
	SST_std = np.empty_like(means)
	wSX_avg = np.empty_like(means)
	wSX_std = np.empty_like(means)
	for i,mean in enumerate(means):
	    print(i)
	    print(np.mean(results[i]['wSX'][100000:]))
	    SST_avg[i] = np.mean(results[i]['rS_P'][100000:])
	    SST_std[i] = np.std(results[i]['rS_P'][100000:])
	    wSX_avg[i] = np.mean(results[i]['wSX'][100000:])
	    wSX_std[i] = np.std(results[i]['wSX'][100000:])


	plt.figure(figsize=(6,3))
	#plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))
	plt.subplot(121)
	plt.errorbar(means, wSX_avg, yerr=wSX_std, linestyle='',marker='.',color='k')

	# works for wP=3.0
	#plt.xlim(0.0,1.0)
	#plt.ylim(0.1,1.0)
	#plt.xticks(np.arange(0.0,1.1,.2),[0.0,.2,.4,.6,.8,1.0],fontsize=12)
	#plt.yticks(np.arange(0,1.2,.2),[0.0,.2,.4,.6,.8,1.0],fontsize=12)

	plt.xlabel(r'$\mu$',fontsize=16)
	plt.ylabel(r'$w_{SST,a}$',fontsize=16)
	plt.subplot(122)
	#plt.plot(sigmas**2, PV_avg, color='k')
	plt.errorbar(means, SST_avg, yerr=SST_std, linestyle='',marker='.',color='k')

	#plt.xlim(-.1,1.0)
	#plt.ylim(-.1,1.0)

	#plt.ylim(-.1,2.0)
	#plt.xticks(np.arange(0.0,1.1,.2),[0.0,.2,.4,.6,.8,1.0],fontsize=12)
	#plt.yticks(np.arange(0.0,1.1,.2),[0.0,.2,.4,.6,.8,1.0],fontsize=12)

	plt.xlabel(r'$\mu$',fontsize=16)
	plt.ylabel(r'$r_{SST}$',fontsize=16)
	plt.tight_layout()
	plt.savefig('./SSTerrorbarsscaling.png', bbox_inches='tight')
	plt.savefig('./SSTerrorbarsscaling.pdf', bbox_inches='tight')