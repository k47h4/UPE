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

	mean = 5.0
	sigma = 0.1
	seed = 123
	start= 100000
	wPs = np.arange(1.0,6.0,1.0)
	wPSs = np.empty_like(wPs)
	wPSs_std = np.empty_like(wPs)

	circuits = []
	for i,wP in enumerate(wPs):
		circuit = Circuit()
		circuit.plastic_PS = True
		circuit.wPY1 = wP
		circuit.wPS_P = 0.01
		circuits.append(circuit)

	sim = Sim(stimulus_duration=1, number_of_samples=200000)
	with multiprocessing.Pool(processes=5) as pool:
	    results=pool.starmap(sim.run_pPE, zip(circuits,repeat(mean),repeat(sigma),repeat(start),repeat(seed)))



	# plot everything in one plot:

	for i,wP in enumerate(wPs):
		wPSs[i] = np.mean(results[i]['wPS_P'][100000:])
		wPSs_std[i] = np.std(results[i]['wPS_P'][100000:])

	name = ''

	plt.figure(figsize=(3,3))
	#plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))
	plt.subplot(111)
	plt.errorbar(wPs,wPSs, yerr=wPSs_std, linestyle='',marker='_',color='k')

	plt.xlim(0.5,5.5)
	plt.ylim(0.5,5.5)
	plt.xticks(np.arange(1,6),[1,2,3,4,5],fontsize=12)
	plt.yticks(np.arange(1,6),[1,2,3,4,5],fontsize=12)

	plt.xlabel(r'$w_{P,s}$',fontsize=16)
	plt.ylabel(r'$w_{PV,SST}$',fontsize=16)
	plt.tight_layout()
	plt.savefig('./LearningPVweights.png', bbox_inches='tight')
	plt.savefig('./LearningPVweights.pdf', bbox_inches='tight')