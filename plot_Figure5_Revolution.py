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

	seed=574858
	beta = 0.1 
	eta_R = 0.1
	wP = np.sqrt(((2-beta)/beta)*(1/2)) # np.sqrt(1.95/0.1)

	circuitUPE = Circuit()
	circuitUPE.single_PV = True
	#circuitUPE.plastic_PS = True
	#circuitUPE.beta_P = 0
	#circuitUPE.NMDA=True
	circuitUPE.wPX1 = sigma_1
	circuitUPE.wPX1_P = sigma_1
	circuitUPE.wPX1_N = sigma_1
	circuitUPE.plastic_PX=False
	circuitUPE.R_neuron=True
	#circuitUPE.beta_P = beta
	circuitUPE.wRX1 = np.array([0.1]) 
	circuitUPE.wPY1 = np.array([wP]) 
	circuitUPE.wPR = np.array([wP])
	circuitUPE.wPS_P = np.array([wP]) 
	circuitUPE.wPS_N = np.array([wP]) 
	#circuitUPE.error_weighting=1.0
	sim = Sim(stimulus_duration=4,number_of_samples=200)
	sim.eta_R = eta_R
	UPE_results_1=sim.run(circuitUPE,mean,sigma_1,seed)

	maxlim=800
	minlim=0
	plt.figure(figsize=(10,10))
	plt.subplot(311)
	plt.plot(UPE_results_1['rP'],color =cm.viridis(.9),linewidth=2,zorder=-10,label='PV+')
	plt.plot(UPE_results_1['rS_P'],color =cm.viridis(.1),linewidth=2,zorder=-10,label='SST+')
	plt.plot(UPE_results_1['rR'],color =cm.viridis(.5),linewidth=2,zorder=-10,label='R')
	plt.plot(UPE_results_1['rE_P'],color ='k',linewidth=2,zorder=-10,label='UPE+')
	plt.plot(UPE_results_1['rY'],'.',color ='grey',linewidth=2,zorder=-10,label='WHISKER')
	plt.legend(bbox_to_anchor=(1,1))
	plt.xlim(minlim,maxlim)
	plt.xticks(np.arange(minlim,maxlim+1,100),np.arange(minlim,maxlim+1,100),fontsize=16)
	plt.ylim(0,8)
	plt.yticks(np.arange(9),np.arange(9),fontsize=16)
	plt.grid()
	plt.subplot(312)
	plt.plot(UPE_results_1['wRX1'],color ='r',linewidth=2,zorder=-10,label=r'$w_{R,a}$')
	#plt.ylim(0,.5)
	plt.xlim(minlim,maxlim)
	plt.grid()
	plt.legend(bbox_to_anchor=(1,1))
	plt.subplot(313)
	plt.plot(UPE_results_1['rP'],color =cm.viridis(.9),linewidth=2,zorder=-10,label='PV-')
	plt.plot(UPE_results_1['rS_N'],color ='grey',linewidth=2,zorder=-10,label='SST-')
	plt.plot(UPE_results_1['rE_N'],color ='k',linewidth=2,zorder=-10,label='UPE-')
	plt.legend(bbox_to_anchor=(1,1))
	plt.xlim(minlim,maxlim)
	plt.grid()
	plt.savefig('./allreates%s.pdf'%str('singlePV'), bbox_inches='tight')
	plt.savefig('./allreates%s.png'%str('singlePV'), bbox_inches='tight')

	mean = 5.0
	sigma_1 = .1
	sigma_2 = 5.0

	seed=574858
	beta = 0.1 
	eta_R = 0.1
	wP = np.sqrt(((2-beta)/beta))#*(1/2)) # np.sqrt(1.95/0.1)

	circuitUPE = Circuit()
	circuitUPE.single_PV = False
	circuitUPE.plastic_PS = True
	circuitUPE.R_neuron=True
	#circuitUPE.NMDA=True
	circuitUPE.PV_mu = False

	circuitUPE.beta_P = beta
	circuitUPE.wRX1 = np.array([0.1]) 
	circuitUPE.wPY1 = np.array([wP]) 
	circuitUPE.wPR = np.array([wP])
	circuitUPE.wPS_P = np.array([wP]) 
	circuitUPE.wPS_N = np.array([wP]) 
	#circuitUPE.error_weighting=1.0
	sim = Sim(stimulus_duration=4,number_of_samples=200000)
	sim.eta_R = eta_R
	UPE_results_1=sim.run_fakePV(circuitUPE,mean,sigma_1,seed)


	circuitUPE2 = Circuit()
	circuitUPE2.single_PV = False
	circuitUPE2.plastic_PS = True
	circuitUPE2.R_neuron=True
	#circuitUPE2.NMDA=True
	circuitUPE2.wRX1 = np.array([0.1]) 
	circuitUPE2.wPS_P = np.array([wP]) 
	circuitUPE2.wPS_N = np.array([wP]) 
	circuitUPE2.wPY1 = np.array([wP]) 
	circuitUPE2.wPR = np.array([wP])
	#circuitUPE2.error_weighting=1.0
	sim = Sim(stimulus_duration=3,number_of_samples=200000)
	sim.eta_R = eta_R
	UPE_results_2=sim.run_fakePV(circuitUPE2,mean,sigma_2,seed)

	PV_1 = np.mean(UPE_results_1['rP_P'])
	PV_2 = np.mean(UPE_results_2['rP_P'])
	PVN_1 = np.mean(UPE_results_1['rP_N'])
	PVN_2 = np.mean(UPE_results_2['rP_N'])
	print(PV_1)
	print(PV_2)
	print(PVN_1)
	print(PVN_2)
	PV1 = np.mean([PV_1,PVN_1])
	PV2 = np.mean([PV_2,PVN_2])



	weighting = (1.0/2.0)*((1/(1+sigma_1**2)) + (1/(1+sigma_2**2)))
	print(weighting)

	weighting = (1/2)*((1/(1+PV1)) + (1/(1+PV2)))

	print(weighting)

	seed=574858
	beta = 0.1 
	eta_R = 0.1
	wP = np.sqrt(((2-beta)/beta))#*(1/2)) # np.sqrt(1.95/0.1)

	circuitN = Circuit()
	circuitN.single_PV = False
	circuitN.weighting= weighting
	#circuitUPE.plastic_PS = True
	circuitN.beta_P = 0
	circuitN.NMDA=True
	circuitN.wPX1 = sigma_1
	circuitN.wPX1_P = sigma_1
	circuitN.wPX1_N = sigma_1
	circuitN.plastic_PX=False
	circuitN.R_neuron=True
	#circuitUPE.beta_P = beta
	circuitN.wRX1 = np.array([0.1]) 
	circuitN.wPY1 = np.array([wP]) 
	circuitN.wPR = np.array([wP])
	circuitN.wPS_P = np.array([wP]) 
	circuitN.wPS_N = np.array([wP]) 
	#circuitUPE.error_weighting=1.0
	sim = Sim(stimulus_duration=4,number_of_samples=200)
	sim.eta_R = eta_R
	N_results_1=sim.run(circuitN,mean,sigma_1,seed)


	plt.figure(figsize=(10,10))
	plt.subplot(311)
	plt.plot(N_results_1['rP_P'],color =cm.viridis(.9),linewidth=2,zorder=-10,label=r'$\sigma^2$')
	plt.plot(N_results_1['rS_P'],color =cm.viridis(.1),linewidth=2,zorder=-10,label='SST+')
	plt.plot(N_results_1['rR'],color =cm.viridis(.5),linewidth=2,zorder=-10,label='R')
	plt.plot(N_results_1['rE_P'],color ='k',linewidth=2,zorder=-10,label='UPE+')
	plt.plot(N_results_1['rY'],'.',color ='grey',linewidth=2,zorder=-10,label='WHISKER')
	plt.legend(bbox_to_anchor=(1,1))
	plt.xlim(minlim,maxlim)
	plt.xticks(np.arange(minlim,maxlim+1,100),np.arange(minlim,maxlim+1,100),fontsize=16)

	plt.yticks(np.arange(9),np.arange(9),fontsize=16)

	plt.ylim(0,8)
	plt.grid()
	plt.subplot(312)
	plt.plot(UPE_results_1['wRX1'],color ='r',linewidth=2,zorder=-10,label=r'$w_{R,a}$')
	#plt.ylim(0,.5)
	plt.xlim(minlim,maxlim)
	plt.grid()
	plt.legend(bbox_to_anchor=(1,1))
	plt.subplot(313)
	plt.plot(N_results_1['rP_N'],color =cm.viridis(.9),linewidth=2,zorder=-10,label='PV-')
	plt.plot(N_results_1['rS_N'],color ='grey',linewidth=2,zorder=-10,label='SST-')
	plt.plot(N_results_1['rE_N'],color ='k',linewidth=2,zorder=-10,label='UPE-')
	plt.legend(bbox_to_anchor=(1,1))
	plt.xlim(minlim,maxlim)
	plt.grid()
	plt.savefig('./allreates%s.pdf'%str('NMDA_N'), bbox_inches='tight')
	plt.savefig('./allreates%s.png'%str('NMDA_N'), bbox_inches='tight')