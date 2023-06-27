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
	sigma_1 = 0.1
	sigma_2 = 3.0

	seed=984985
	beta = 0.1 
	eta_R = 0.1*0.1
	dt = 0.1
	tau_I = 2.0
	tau_E = 10.0
	stim_duration = 10
	wP = np.sqrt(((2-beta)/beta))#*(1/2)) # np.sqrt(1.95/0.1)

	weighting = (1.0/2.0)*((1/(1+sigma_1**2)) + (1/(1+sigma_2**2)))

	circuitUPE = Circuit()
	circuitUPE.dt = dt
	circuitUPE.tau_P = tau_I
	circuitUPE.tau_S = tau_I
	circuitUPE.tau_E = tau_E
	circuitUPE.tau_R = tau_E
	circuitUPE.single_PV = False
	circuitUPE.plastic_PS = True
	circuitUPE.R_neuron=True
	circuitUPE.wRX1 = np.array([0.1]) 
	circuitUPE.wPS_P = np.array([wP]) 
	circuitUPE.wPS_N = np.array([wP]) 
	circuitUPE.error_weighting=1.0
	sim = Sim(stimulus_duration=stim_duration,number_of_samples=10000)
	sim.eta_R = eta_R
	sim.eta_P = 0.001*0.1
	sim.eta_S = 0.1*0.1
	sim.eta_ES = 0.01*0.1
	sim.eta_PS = 0.0001*0.1

	UPE_results_1=sim.run_fakePV(circuitUPE,mean,sigma_1,seed)


	circuitUPE2 = Circuit()
	circuitUPE2.dt = dt
	circuitUPE2.tau_P = tau_I
	circuitUPE2.tau_S = tau_I
	circuitUPE2.tau_E = tau_E
	circuitUPE2.tau_R = tau_E
	circuitUPE2.single_PV = False
	circuitUPE2.plastic_PS = True
	circuitUPE2.R_neuron=True
	circuitUPE2.wRX1 = np.array([0.1]) 
	circuitUPE2.wPS_P = np.array([wP]) 
	circuitUPE2.wPS_N = np.array([wP]) 
	circuitUPE2.error_weighting=1.0
	sim = Sim(stimulus_duration=stim_duration,number_of_samples=10000)
	sim.eta_R = eta_R
	sim.eta_P = 0.001*0.1
	sim.eta_S = 0.1*0.1
	sim.eta_ES = 0.01*0.1
	sim.eta_PS = 0.0001*0.1
	UPE_results_2=sim.run_fakePV(circuitUPE2,mean,sigma_2,seed)

	circuitN = Circuit()
	circuitN.dt = dt
	circuitN.tau_P = tau_I
	circuitN.tau_S = tau_I
	circuitN.tau_E = tau_E
	circuitN.tau_R = tau_E
	circuitN.single_PV = False
	circuitN.plastic_PS = True
	circuitN.R_neuron=True
	circuitN.uncertainty_weighted=False
	circuitN.weighting = weighting
	circuitN.wRX1 = np.array([0.1]) 
	circuitN.wPS_P = np.array([wP]) 
	circuitN.wPS_N = np.array([wP]) 
	circuitN.error_weighting=1.0
	sim = Sim(stimulus_duration=stim_duration,number_of_samples=10000)
	sim.eta_R = eta_R
	sim.eta_P = 0.001*0.1
	sim.eta_S = 0.1*0.1
	sim.eta_ES = 0.01*0.1
	sim.eta_PS = 0.0001*0.1
	N_results_1=sim.run_fakePV(circuitN,mean,sigma_1,seed)

	circuitN2 = Circuit()
	circuitN2.dt = dt
	circuitN2.tau_P = tau_I
	circuitN2.tau_S = tau_I
	circuitN2.tau_E = tau_E
	circuitN2.tau_R = tau_E
	circuitN2.single_PV = False
	circuitN2.plastic_PS = True
	circuitN2.R_neuron=True
	circuitN2.uncertainty_weighted=False
	circuitN2.weighting = weighting
	circuitN2.wRX1 = np.array([0.1]) 
	circuitN2.wPS_P = np.array([wP]) 
	circuitN2.wPS_N = np.array([wP]) 
	circuitN2.error_weighting=1.0
	sim = Sim(stimulus_duration=stim_duration,number_of_samples=10000)
	sim.eta_R = eta_R
	sim.eta_P = 0.001*0.1
	sim.eta_S = 0.1*0.1
	sim.eta_ES = 0.01*0.1
	sim.eta_PS = 0.0001*0.1
	N_results_2=sim.run_fakePV(circuitN2,mean,sigma_2,seed)


	name = 'difftaus'
	plt.figure(figsize=(12,4))
	a1=plt.subplot(131)
	plt.plot(UPE_results_1['rRa'],color =cm.magma(.6),linewidth=2,zorder=-10,label='UPE')
	plt.plot(N_results_1['rRa'], color ='k',linewidth=2,zorder=-10,label='unweighted')
	#plt.plot(UPE_results_1['rRa'],color =cm.viridis(.5),linewidth=2,zorder=-10,label='UPE')
	#plt.ylim(4.5,5.5)
	plt.xlim(0,2000)
	plt.xticks([0,1000,2000],[0,1,2],fontsize=16)
	plt.ylim(0,7)
	plt.yticks([0,5],[0,5],fontsize=16)
	a1.spines['top'].set_visible(False)
	a1.spines['right'].set_visible(False)
	plt.xlabel('time [s]',fontsize=16)
	plt.title('low uncertainty',fontsize=16)
	plt.ylabel('firing rate',fontsize=16)
	#plt.ylim(4.5,5.5)
	a2=plt.subplot(132)
	plt.plot(N_results_2['rRa'], color ='k',linewidth=2,zorder=-10,label='unweighted')
	#plt.xticks([10000,10100],[10000,10100],fontsize=16)
	plt.xticks([0,2000,4000],[0,2,4],fontsize=16)

	plt.yticks([0,5],[0,5],fontsize=16)
	plt.ylim(0,7)
	plt.plot(UPE_results_2['rRa'],color =cm.magma(.6),linewidth=2,zorder=-10,label='UPE')
	plt.xlim(0,5000)
	a2.spines['top'].set_visible(False)
	a2.spines['right'].set_visible(False)
	plt.title('high uncertainty',fontsize=16)
	#plt.legend(bbox_to_anchor=(1,1),fontsize=11)

	a3=plt.subplot(133)
	plt.plot(N_results_2['rRa'], color ='k',linewidth=2,zorder=-10,label='unweighted')
	#plt.xticks([10000,10100],[10000,10100],fontsize=16)
	plt.xticks([5000,10000,15000,20000],[5,10,15,20],fontsize=16)

	plt.yticks([0,5],[0,5],fontsize=16)
	plt.ylim(0,7)
	plt.plot(UPE_results_2['rRa'],color =cm.magma(.6),linewidth=2,zorder=-10,label='UPE')
	plt.xlim(5000,20000)
	a3.spines['top'].set_visible(False)
	a3.spines['right'].set_visible(False)
	plt.title('high uncertainty',fontsize=16)
	plt.legend(bbox_to_anchor=(1,1),fontsize=11)

	plt.tight_layout()
	plt.savefig('./fakefunctionality_timeevolutionofratesstimdur3_dt01%s%s%s.pdf'%(str(seed),str(eta_R),str(sigma_2)), bbox_inches='tight')


	exponent = 1 # if 2 it's variance 
	plt.figure(figsize=(6,3))
	a1 = plt.subplot(121)
	plt.bar([0,1],[np.std(UPE_results_1['rRa'][90000:])**exponent,np.std(N_results_1['rRa'][90000:])**exponent],color=[cm.magma(.6),'k'],width=0.3)
	plt.ylabel(r'$\sigma$ of firing rate',fontsize=16)
	plt.xticks([0,1],['UPE','unscaled'],fontsize=16)
	plt.xlabel('low uncertainty',fontsize=16)
	plt.yticks(np.arange(0,1.0,0.2),[0,0.2,.4,.6,.8],fontsize=16)

	#axins = a1.inset_axes([0.5, 0.5, 0.47, 0.47])
	#axins.bar([0,1],[np.std(UPE_results_1['rRa'][9000:])**exponent,np.std(N_results_1['rRa'][9000:])**exponent],color=[cm.magma(.6),'k'],width=0.3)
	#axins.set_yticks(np.arange(0,.0012,0.0002),[0,0.0002,.0004,.0006,.0008,.0001])
	#axins.set_xticks([0,1])
	#axins.set_xticklabels(['UPE','unscaled'])

	#plt.yticks(np.arange(0,.12,0.02),[0,0.02,.04,.06,.08,.1],fontsize=16)

	#plt.yticks(np.arange(0,.0015,0.0005),[0,0.5e-3,1e-3],fontsize=16)
	#plt.yticks([0,1],[0,1],fontsize=16)
	a1.spines['top'].set_visible(False)
	a1.spines['right'].set_visible(False)
	plt.ylim(0,0.6)
	#plt.xlim(-0.5,1.5)
	a2 = plt.subplot(122)
	plt.bar([0,1],[np.std(UPE_results_2['rRa'][90000:])**exponent,np.std(N_results_2['rRa'][90000:])**exponent],color=[cm.magma(.6),'k'],width=0.3)
	plt.ylabel(r'$\sigma$ of firing rate',fontsize=16)
	plt.xticks([0,1],['UPE','unscaled'],fontsize=16)
	plt.xlabel('high uncertainty',fontsize=16)
	plt.yticks(np.arange(0,1.0,0.2),[0,0.2,.4,.6,.8],fontsize=16)
	#plt.yticks([0,1],[0,1],fontsize=16)
	a2.spines['top'].set_visible(False)
	a2.spines['right'].set_visible(False)
	plt.ylim(0,0.6)
	#plt.xlim(-0.5,1.5)

	plt.tight_layout()
	plt.savefig('./fakefunctionalityfigure_dt01%s.png'%(name), bbox_inches='tight')





