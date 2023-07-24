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
	stim_duration = 100
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

	UPE_results_1=sim.run_fakePV(circuitUPE,mean,sigma_1,seed=seed)


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
	UPE_results_2=sim.run_fakePV(circuitUPE2,mean,sigma_2,seed=seed)

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
	N_results_1=sim.run_fakePV(circuitN,mean,sigma_1,seed=seed)

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
	N_results_2=sim.run_fakePV(circuitN2,mean,sigma_2,seed=seed)


	tau_I = 2.0
	tau_E = 10.0
	seed = 123 
	beta = 0.1 
	dt = 0.1
	stimulus_duration=100 # 10
	wP = np.sqrt(((2-beta)/beta)) # np.sqrt(1.95/0.1)
	sigmas=np.arange(0.1,2.1,0.2)
	circuit = Circuit()
	circuit.dt = dt
	circuit.R_neuron= True
	circuit.single_PV = False
	circuit.tau_P = tau_I
	circuit.tau_S = tau_I
	circuit.tau_E = tau_E
	circuit.tau_R = tau_E
	circuit.plastic_PS = True
	#circuit.PV_mu = False
	#circuit.NMDA = True
	circuit.beta_P = beta
	circuit.wPY1 = np.array([wP]) # small intitial weights
	circuit.wPR = np.array([wP]) # small intitial weights
	circuit.wPS_P = np.array([wP])
	circuit.wPS_N = np.array([wP])
	circuit.error_weighting=1.0
	sim = Sim(stimulus_duration=stimulus_duration,number_of_samples=20000)

	# run for eta_R = 0.01
	sim.eta_R = 0.1
	with multiprocessing.Pool(processes=4) as pool:
	    sigma_results=pool.starmap(sim.run_fakePV, zip(repeat(circuit),repeat(5),sigmas,repeat(seed)))  

	R_std = np.empty(len(sigmas))
	wRX_std = np.empty(len(sigmas))

	for i,sigma in enumerate(sigmas):
	    R_std[i] = np.std(sigma_results[i]['rRa'][100000:])
	    wRX_std[i] = np.std(sigma_results[i]['wRX1'][100000:])

	# run for eta_R = 0.1
	# sim.eta_R = 0.1
	# with multiprocessing.Pool(processes=5) as pool:
	#     sigma_results_2=pool.starmap(sim.run_fakePV, zip(repeat(circuit),repeat(5),sigmas,repeat(seed)))  

	# R_std2 = np.empty(len(sigmas))
	# wRX_std2 = np.empty(len(sigmas))

	# for i,sigma in enumerate(sigmas):
	#     R_std2[i] = np.std(sigma_results_2[i]['rRa'][100000:])
	#     wRX_std2[i] = np.std(sigma_results_2[i]['wRX1'][100000:])

	weighting = 1.0

	circuitNs = Circuit()
	circuitNs.dt = dt
	circuitNs.tau_P = tau_I
	circuitNs.tau_S = tau_I
	circuitNs.tau_E = tau_E
	circuitNs.tau_R = tau_E
	circuitNs.single_PV = False
	circuitNs.plastic_PS = True
	circuitNs.R_neuron=True
	circuitNs.uncertainty_weighted=False
	circuitNs.weighting = weighting
	circuitNs.wPY1 = np.array([wP]) 
	circuitNs.wPR = np.array([wP]) 
	circuitNs.wPS_P = np.array([wP]) 
	circuitNs.wPS_N = np.array([wP]) 
	circuitNs.error_weighting=1.0
	sim = Sim(stimulus_duration=stimulus_duration,number_of_samples=20000)

	sim.eta_R = 0.1
	with multiprocessing.Pool(processes=4) as pool:
	    sigma_results_2=pool.starmap(sim.run_fakePV, zip(repeat(circuitNs),repeat(5),sigmas,repeat(seed)))  

	R_std2 = np.empty(len(sigmas))
	wRX_std2 = np.empty(len(sigmas))
	for i,sigma in enumerate(sigmas):
	    R_std2[i] = np.std(sigma_results_2[i]['rRa'][100000:])
	    wRX_std2[i] = np.std(sigma_results_2[i]['wRX1'][100000:])




	# sample_mean_1 = np.zeros((len(N_results_1['rY1'])))
	# sample_mean_2 = np.zeros((len(N_results_2['rY1'])))
	# N=0
	# for i, x in enumerate(N_results_1['rY1']):
	# 	N+=1
	# 	sample_mean_1[i] = (1/N) * np.sum(N_results_1['rY1'][:i+1])
	# 	sample_mean_2[i] = (1/N) * np.sum(N_results_2['rY1'][:i+1])

	name = 'difftaus'
	plt.figure(figsize=(12,8))
	a1=plt.subplot(231)
	a1.text(-0.1, 1.15, 'A', transform=a1.transAxes,fontsize=16, va='top', ha='right')

	plt.plot(N_results_1['rY1'],'.',color ='lightgrey',linewidth=2,zorder=-10,label='WHISKER')
	#plt.plot(UPE_results_1['rY1'],'.',color ='green',linewidth=2,zorder=-10,label='WHISKER')

	plt.plot(UPE_results_1['rRa'],color =cm.cividis(.7),linewidth=2,zorder=-10,label='UPE')
	plt.plot(N_results_1['rRa'], color ='k',linewidth=2,zorder=-10,label='unweighted')
	#plt.plot(sample_mean_1, color = 'blue')
	#plt.plot(UPE_results_1['rRa'],color =cm.viridis(.5),linewidth=2,zorder=-10,label='UPE')
	#plt.ylim(4.5,5.5)
	plt.xlim(0,20000)
	plt.xticks([0,20000],[0,20000],fontsize=16)
	plt.ylim(0,10)
	plt.yticks([0,5],[0,5],fontsize=16)
	a1.spines['top'].set_visible(False)
	a1.spines['right'].set_visible(False)
	plt.xlabel('timesteps',fontsize=16)
	plt.title('low uncertainty',fontsize=16)
	plt.ylabel('firing rate',fontsize=16)
	#plt.ylim(4.5,5.5)
	a1.set_rasterization_zorder(0)

	a2=plt.subplot(232)
	a2.text(-0.1, 1.15, 'B', transform=a2.transAxes,fontsize=16, va='top', ha='right')

	plt.plot(N_results_2['rY1'],'.',color ='lightgrey',linewidth=2,zorder=-10,label='WHISKER')
	#plt.plot(UPE_results_2['rY1'],'.',color ='green',linewidth=2,zorder=-10,label='WHISKER')

	plt.plot(N_results_2['rRa'], color ='k',linewidth=2,zorder=-10,label='unweighted')
	#plt.plot(sample_mean_2, color = 'blue')

	#plt.xticks([10000,10100],[10000,10100],fontsize=16)
	plt.xticks([0,50000],[0,50000],fontsize=16)

	plt.yticks([0,5],[0,5],fontsize=16)
	plt.ylim(0,10)
	plt.plot(UPE_results_2['rRa'],color =cm.cividis(.7),linewidth=2,zorder=-10,label='UPE')
	plt.xlim(0,50000)
	a2.spines['top'].set_visible(False)
	a2.spines['right'].set_visible(False)
	a2.set_rasterization_zorder(0)

	plt.title('high uncertainty',fontsize=16)
	#plt.legend(bbox_to_anchor=(1,1),fontsize=11)

	a3=plt.subplot(233)
	a3.text(-0.1, 1.15, 'C', transform=a3.transAxes,fontsize=16, va='top', ha='right')

	plt.plot(N_results_2['rY1'],'.',color ='lightgrey',linewidth=2,zorder=-10,label='WHISKER')

	plt.plot(N_results_2['rRa'], color ='k',linewidth=2,zorder=-10,label='unweighted')
	#plt.plot(sample_mean_2, color = 'blue')

	#plt.xticks([10000,10100],[10000,10100],fontsize=16)
	plt.xticks([50000,100000],[50000,100000],fontsize=16)

	plt.yticks([0,5],[0,5],fontsize=16)
	plt.ylim(0,10)
	plt.plot(UPE_results_2['rRa'],color =cm.cividis(.7),linewidth=2,zorder=-10,label='UPE')
	#plt.xlim(50000,200000)
	plt.xlim(50000,100000)
	a3.spines['top'].set_visible(False)
	a3.spines['right'].set_visible(False)
	plt.title('high uncertainty',fontsize=16)
	plt.legend(bbox_to_anchor=(1,1),fontsize=11)
	a3.set_rasterization_zorder(0)

	exponent = 1 # if 2 it's variance 
	a4 = plt.subplot(234)
	a4.text(-0.1, 1.15, 'D', transform=a4.transAxes,fontsize=16, va='top', ha='right')

	plt.bar([0,1],[np.std(UPE_results_1['rRa'][100000:])**exponent,np.std(N_results_1['rRa'][100000:])**exponent],color=[cm.cividis(.7),'k'],width=0.3)
	plt.ylabel(r'$\sigma$ of firing rate',fontsize=16)
	plt.xticks([0,1],['UPE','unscaled'],fontsize=16)
	plt.xlabel('low uncertainty',fontsize=16)
	plt.yticks(np.arange(0,1.0,0.2),[0,0.2,.4,.6,.8],fontsize=16)
	a4.set_rasterization_zorder(0)

	axins = a4.inset_axes([0.5, 0.5, 0.47, 0.47])
	axins.bar([0,1],[np.std(UPE_results_1['rRa'][100000:])**exponent,np.std(N_results_1['rRa'][100000:])**exponent],color=[cm.cividis(.7),'k'],width=0.3)
	#axins.set_yticks(np.arange(0,.0012,0.0002),[0,0.0002,.0004,.0006,.0008,.0001])
	axins.set_xticks([0,1])
	axins.set_xticklabels(['UPE','unscaled'])

	#plt.yticks(np.arange(0,.12,0.02),[0,0.02,.04,.06,.08,.1],fontsize=16)

	#plt.yticks(np.arange(0,.0015,0.0005),[0,0.5e-3,1e-3],fontsize=16)
	#plt.yticks([0,1],[0,1],fontsize=16)
	a4.spines['top'].set_visible(False)
	a4.spines['right'].set_visible(False)
	plt.ylim(0,0.6)
	#plt.xlim(-0.5,1.5)
	a5 = plt.subplot(235)
	a5.text(-0.1, 1.15, 'E', transform=a5.transAxes,fontsize=16, va='top', ha='right')

	plt.bar([0,1],[np.std(UPE_results_2['rRa'][100000:])**exponent,np.std(N_results_2['rRa'][100000:])**exponent],color=[cm.cividis(.7),'k'],width=0.3)
	plt.ylabel(r'$\sigma$ of firing rate',fontsize=16)
	plt.xticks([0,1],['UPE','unscaled'],fontsize=16)
	plt.xlabel('high uncertainty',fontsize=16)
	plt.yticks(np.arange(0,1.0,0.2),[0,0.2,.4,.6,.8],fontsize=16)
	#plt.yticks([0,1],[0,1],fontsize=16)
	a5.spines['top'].set_visible(False)
	a5.spines['right'].set_visible(False)
	plt.ylim(0,0.6)
	#plt.xlim(-0.5,1.5)
	a5.set_rasterization_zorder(0)



	a6 = plt.subplot(236)
	a6.text(-0.1, 1.15, 'F', transform=a6.transAxes,fontsize=16, va='top', ha='right')

	#plt.plot(sigmas**2, PV_avg, color='k')
	plt.errorbar(sigmas, R_std2, linestyle='',marker='.',color=cm.cividis(.0))#,label=r'$\eta_R = 0.1$')
	plt.errorbar(sigmas, R_std, linestyle='',marker='.',color=cm.cividis(.7))#,label=r'$\eta_R = 0.01$')


	plt.xlim(-.1,2.0)
	plt.ylim(-.1,2.0)
	a6.spines['top'].set_visible(False)
	a6.spines['right'].set_visible(False)
	plt.xticks(np.arange(0.5,2.0,.5),[0.5,1.0,1.5],fontsize=16)
	plt.yticks(np.arange(0.5,2.0,.5),[0.5,1.0,1.5],fontsize=16)
	plt.xlabel(r'$\sigma_s$',fontsize=16)
	plt.ylabel(r'$\sigma$ of $r_{R}$',fontsize=16)
	#plt.legend(bbox_to_anchor=(1.0,0.6,0.5,0.5))




	plt.tight_layout()
	plt.savefig('./functionalityfigure_dt01%s.png'%(name), bbox_inches='tight')
	plt.savefig('./functionalityfigure_dt01%s.pdf'%(name), bbox_inches='tight')





