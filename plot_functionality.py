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

	seed=123
	beta = 0.1 
	eta_R = 0.1
	wP = np.sqrt(((2-beta)/beta))#*(1/2)) # np.sqrt(1.95/0.1)

	weighting = (1.0/2.0)*((1/(1+sigma_1**2)) + (1/(1+sigma_2**2)))

	circuitUPE = Circuit()
	circuitUPE.dt = 0.1
	circuitUPE.single_PV = False
	circuitUPE.plastic_PS = True
	circuitUPE.R_neuron=True
	circuitUPE.wRX1 = np.array([0.1]) 
	circuitUPE.wPS_P = np.array([wP]) 
	circuitUPE.wPS_N = np.array([wP]) 
	circuitUPE.error_weighting=1.0
	sim = Sim(stimulus_duration=3,number_of_samples=200000)
	sim.eta_R = eta_R
	UPE_results_1=sim.run_fakePV(circuitUPE,mean,sigma_1,seed)


	circuitUPE2 = Circuit()
	circuitUPE2.dt = 0.1
	circuitUPE2.single_PV = False
	circuitUPE2.plastic_PS = True
	circuitUPE2.R_neuron=True
	circuitUPE2.wRX1 = np.array([0.1]) 
	circuitUPE2.wPS_P = np.array([wP]) 
	circuitUPE2.wPS_N = np.array([wP]) 
	circuitUPE2.error_weighting=1.0
	sim = Sim(stimulus_duration=3,number_of_samples=200000)
	sim.eta_R = eta_R
	UPE_results_2=sim.run_fakePV(circuitUPE2,mean,sigma_2,seed)

	circuitN = Circuit()
	circuitN.dt = 0.1
	circuitN.single_PV = False
	circuitN.plastic_PS = True
	circuitN.R_neuron=True
	circuitN.uncertainty_weighted=False
	circuitN.weighting = weighting
	circuitN.wRX1 = np.array([0.1]) 
	circuitN.wPS_P = np.array([wP]) 
	circuitN.wPS_N = np.array([wP]) 
	circuitN.error_weighting=1.0
	sim = Sim(stimulus_duration=3,number_of_samples=200000)
	sim.eta_R = eta_R
	N_results_1=sim.run_fakePV(circuitN,mean,sigma_1,seed)

	circuitN2 = Circuit()
	circuitN2.dt = 0.1
	circuitN2.single_PV = False
	circuitN2.plastic_PS = True
	circuitN2.R_neuron=True
	circuitN2.uncertainty_weighted=False
	circuitN2.weighting = weighting
	circuitN2.wRX1 = np.array([0.1]) 
	circuitN2.wPS_P = np.array([wP]) 
	circuitN2.wPS_N = np.array([wP]) 
	circuitN2.error_weighting=1.0
	sim = Sim(stimulus_duration=3,number_of_samples=200000)
	sim.eta_R = eta_R
	N_results_2=sim.run_fakePV(circuitN2,mean,sigma_2,seed)


	plt.figure(figsize=(12,4))
	a1=plt.subplot(131)
	plt.plot(UPE_results_1['rRa'],color =cm.magma(.6),linewidth=2,zorder=-10,label='UPE')
	plt.plot(N_results_1['rRa'], color ='k',linewidth=2,zorder=-10,label='unweighted')
	#plt.plot(UPE_results_1['rRa'],color =cm.viridis(.5),linewidth=2,zorder=-10,label='UPE')
	#plt.ylim(4.5,5.5)
	plt.xlim(0,500)
	plt.xticks([0,500],[0,500],fontsize=16)
	plt.ylim(0,8)
	plt.yticks([0,5],[0,5],fontsize=16)
	a1.spines['top'].set_visible(False)
	a1.spines['right'].set_visible(False)
	plt.xlabel('timesteps',fontsize=16)
	plt.title('low uncertainty')
	plt.ylabel('firing rate',fontsize=16)
	#plt.ylim(4.5,5.5)
	a2=plt.subplot(132)
	plt.plot(N_results_2['rRa'], color ='k',linewidth=2,zorder=-10,label='unweighted')
	#plt.xticks([10000,10100],[10000,10100],fontsize=16)
	plt.xticks([0,500],[0,500],fontsize=16)

	plt.yticks([0,5],[0,5],fontsize=16)
	plt.ylim(0,8)
	plt.plot(UPE_results_2['rRa'],color =cm.magma(.6),linewidth=2,zorder=-10,label='UPE')
	plt.xlim(0,500)
	a2.spines['top'].set_visible(False)
	a2.spines['right'].set_visible(False)
	plt.title('high uncertainty')
	#plt.legend(bbox_to_anchor=(1,1),fontsize=11)

	a3=plt.subplot(133)
	plt.plot(N_results_2['rRa'], color ='k',linewidth=2,zorder=-10,label='unweighted')
	#plt.xticks([10000,10100],[10000,10100],fontsize=16)
	plt.xticks([1000,1500],[1000,1500],fontsize=16)

	plt.yticks([0,5],[0,5],fontsize=16)
	plt.ylim(0,8)
	plt.plot(UPE_results_2['rRa'],color =cm.magma(.6),linewidth=2,zorder=-10,label='UPE')
	plt.xlim(1000,1500)
	a2.spines['top'].set_visible(False)
	a2.spines['right'].set_visible(False)
	plt.title('high uncertainty')
	plt.legend(bbox_to_anchor=(1,1),fontsize=11)

	plt.tight_layout()
	plt.savefig('./timeevolutionofratesstimdur3_dt0.1%s%s%s.pdf'%(str(seed),str(eta_R),str(sigma_2)), bbox_inches='tight')


	exponent = 1 # if 2 it's variance 
	plt.figure(figsize=(6,3))
	a1 = plt.subplot(121)
	plt.bar([0,1],[np.std(UPE_results_1['rRa'][100000:])**exponent,np.std(N_results_1['rRa'][100000:])**exponent],color=[cm.magma(.6),'k'],width=0.3)
	plt.ylabel('variance of firing rate',fontsize=16)
	plt.xticks([0,1],['UPE','unscaled'],fontsize=16)
	plt.xlabel('low uncertainty',fontsize=16)
	plt.yticks(np.arange(0,1.0,0.2),[0,0.2,.4,.6,.8],fontsize=16)

	axins = a1.inset_axes([0.5, 0.5, 0.47, 0.47])
	axins.bar([0,1],[np.std(UPE_results_1['rRa'][100000:])**exponent,np.std(N_results_1['rRa'][100000:])**exponent],color=[cm.magma(.6),'k'],width=0.3)
	axins.set_yticks(np.arange(0,.0012,0.0002),[0,0.0002,.0004,.0006,.0008,.0001])
	axins.set_xticks([0,1])
	axins.set_xticklabels(['UPE','unscaled'])

	#plt.yticks(np.arange(0,.12,0.02),[0,0.02,.04,.06,.08,.1],fontsize=16)

	#plt.yticks(np.arange(0,.0015,0.0005),[0,0.5e-3,1e-3],fontsize=16)
	#plt.yticks([0,1],[0,1],fontsize=16)
	a1.spines['top'].set_visible(False)
	a1.spines['right'].set_visible(False)
	plt.ylim(0,0.8)
	#plt.xlim(-0.5,1.5)
	a2 = plt.subplot(122)
	plt.bar([0,1],[np.std(UPE_results_2['rRa'][100000:])**exponent,np.std(N_results_2['rRa'][100000:])**exponent],color=[cm.magma(.6),'k'],width=0.3)
	plt.ylabel('variance of firing rate',fontsize=16)
	plt.xticks([0,1],['UPE','unscaled'],fontsize=16)
	plt.xlabel('high uncertainty',fontsize=16)
	plt.yticks(np.arange(0,1.0,0.2),[0,0.2,.4,.6,.8],fontsize=16)
	#plt.yticks([0,1],[0,1],fontsize=16)
	a2.spines['top'].set_visible(False)
	a2.spines['right'].set_visible(False)
	plt.ylim(0,0.8)
	#plt.xlim(-0.5,1.5)

	plt.tight_layout()
	#plt.legend(fontsize=11)
	plt.savefig('./Functionality_barplotdt0.1%s%s%s.png'%(str(seed),str(eta_R),str(sigma_2)), bbox_inches='tight')
	plt.savefig('./Functionality_barplotdt0.1%s%s%s.pdf'%(str(seed),str(eta_R),str(sigma_2)), bbox_inches='tight')






