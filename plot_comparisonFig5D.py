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
	mean = 5.0
	sigma_1 = .5
	sigma_2 = .9
	tau_I =0.2
	tau_E = 1.0
	seed=574858
	beta = 0.1 
	eta_R = 0.1
	wP = np.sqrt(((2-beta)/beta))#*(1/2)) # np.sqrt(1.95/0.1)

	circuitUPE = Circuit()
	#circuitUPE.dt = 0.1
	circuitUPE.tau_P = tau_I
	circuitUPE.tau_S = tau_I
	circuitUPE.tau_E = tau_E
	circuitUPE.tau_R = tau_E
	circuitUPE.single_PV = False # 
	circuitUPE.plastic_PS = False#True
	circuitUPE.NMDA=True # 
	circuitUPE.wPX1 = sigma_1 # 
	circuitUPE.wPX1_P = sigma_1 #
	circuitUPE.wPX1_N = sigma_1 #
	circuitUPE.plastic_PX=False #
	circuitUPE.R_neuron=True #
	#circuitUPE.beta_P = beta
	circuitUPE.wRX1 = np.array([0.1]) 
	circuitUPE.wPY1 = np.array([wP]) #
	circuitUPE.wPR = np.array([wP]) #
	circuitUPE.wPS_P = np.array([wP]) #
	circuitUPE.wPS_N = np.array([wP]) # 
	#circuitUPE.error_weighting=1.0
	sim = Sim(stimulus_duration=4,number_of_samples=200) # 
	sim.eta_R = eta_R
	UPE_results_1=sim.run(circuitUPE,mean,sigma_1,seed)

	minlim=0
	maxlim=800
	plt.figure(figsize=(10,10))
	plt.subplot(311)
	plt.plot(UPE_results_1['rY'],'.',color ='grey',linewidth=2,zorder=-10,label='WHISKER')
	plt.plot(UPE_results_1['rP_P'],color =cm.viridis(.9),linewidth=2,zorder=-10,label='PV+')
	plt.plot(UPE_results_1['rS_P'],color =cm.viridis(.1),linewidth=2,zorder=-10,label='SST+')
	plt.plot(UPE_results_1['rR'],color =cm.viridis(.5),linewidth=2,zorder=-10,label='R')
	plt.plot(UPE_results_1['rE_P'],color ='k',linewidth=2,zorder=-10,label='UPE+')
	plt.legend(bbox_to_anchor=(1,1))
	plt.xlim(minlim,maxlim)
	plt.ylim(0,8)
	plt.yticks(np.arange(0,9,2),np.arange(0,9,2),fontsize=16)

	plt.grid()
	plt.subplot(312)
	plt.plot(UPE_results_1['wRX1'],color ='r',linewidth=2,zorder=-10,label=r'$w_{R,a}$')
	#plt.ylim(0,.5)
	plt.xlim(minlim,maxlim)
	plt.xticks(np.arange(0,900,200),np.arange(0,900,200),fontsize=16)

	plt.grid()
	plt.legend(bbox_to_anchor=(1,1))
	plt.subplot(313)

	plt.plot(UPE_results_1['rY'],'.',color ='grey',linewidth=2,zorder=-10,label='WHISKER')

	plt.plot(UPE_results_1['rP_N'],color =cm.viridis(.9),linewidth=2,zorder=-10,label='PV-')
	plt.plot(UPE_results_1['rS_N'],color ='grey',linewidth=2,zorder=-10,label='SST-')
	plt.plot(UPE_results_1['rE_N'],color ='k',linewidth=2,zorder=-10,label='UPE-')
	plt.legend(bbox_to_anchor=(1,1))
	plt.xlim(minlim,maxlim)
	plt.yticks(np.arange(0,9,2),np.arange(0,9,2),fontsize=16)
	plt.xticks(np.arange(0,900,200),np.arange(0,900,200),fontsize=16)

	plt.grid()
	plt.savefig('./allreates%s.pdf'%'5D', bbox_inches='tight')
	plt.savefig('./allreates%s.png'%'5D', bbox_inches='tight')
