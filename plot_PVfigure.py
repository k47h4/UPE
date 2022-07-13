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


	# run sim with two different sigmas:
	Y1_mean = 5.0
	Y2_mean = 5.0
	Y1_sigma = 0.8
	Y2_sigma = 0.4
	beta=0.1
	seed = 123
	circuit1 = Circuit()
	circuit1.beta_P = beta
	wP = np.sqrt((2-beta)/beta) # np.sqrt(1.95/0.1)
	circuit1.wPY1 = np.array([wP]) # small intitial weights
	circuit1.wPS_P = np.array([wP])
	sim = Sim(stimulus_duration=1)
	results1=sim.run_pPE(circuit1,mean=Y1_mean,sigma=Y1_sigma,seed=seed)


	circuit2 = Circuit()
	circuit2.beta_P = beta
	wP = np.sqrt((2-beta)/beta) # np.sqrt(1.95/0.1)
	circuit2.wPY1 = np.array([wP]) # small intitial weights
	circuit2.wPS_P = np.array([wP])
	sim = Sim(stimulus_duration=1)
	results2=sim.run_pPE(circuit2,mean=Y2_mean,sigma=Y2_sigma,seed=seed)


	# run sim with 10 different sigmas in parallel
	beta = 0.1
	mean = 5.0
	wP = np.sqrt((2-beta)/beta) # np.sqrt(1.95/0.1)
	sigmas = np.arange(0.1,1.1,0.1)
	PV_avg = np.empty_like(sigmas)
	PV_std = np.empty_like(sigmas)
	wPX_avg = np.empty_like(sigmas)
	wPX_std = np.empty_like(sigmas)
	circuit = Circuit()
	circuit.beta_P = beta
	circuit.wPY1 = np.array([wP]) # small intitial weights
	circuit.wPS_P = np.array([wP])
	sim = Sim(stimulus_duration=1)
	with multiprocessing.Pool(processes=5) as pool:
	    results=pool.starmap(sim.run_pPE, zip(repeat(circuit),repeat(mean),sigmas,repeat(seed)))


	for i,sigma in enumerate(sigmas):
	    PV_avg[i] = np.mean(results[i]['rPa'][100000:])
	    PV_std[i] = np.std(results[i]['rPa'][100000:])
	    wPX_avg[i] = np.mean(results[i]['wPX1'][100000:])
	    wPX_std[i] = np.std(results[i]['wPX1'][100000:])



	# plot everything in one plot:

	name = ''
	sigma1=0.8
	sigma2=0.4
	sim_time= 400000
	PV_avg1 = np.mean(results1['rPa'][100000:])
	PV_avg2 = np.mean(results2['rPa'][100000:])
	PV_std1 = np.std(results1['rPa'][100000:])
	PV_std2 = np.std(results2['rPa'][100000:])

	plt.figure(figsize=(7,10))
	a1 = plt.subplot(321)
	a1.text(-0.1, 1.15, 'A', transform=a1.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(results1['wPX1'], label='high', color =cm.viridis(.1),linewidth=2)
	plt.plot(results2['wPX1'], label='low',color =cm.viridis(.5),linewidth=2)
	plt.plot(np.arange(-30000,len(results1['wPX1'])),np.ones(len(results1['wPX1'])+30000)*sigma1,'--',color =cm.viridis(.1),label=r'$\sigma$ high')
	plt.plot(np.arange(-30000,len(results1['wPX1'])),np.ones(len(results1['wPX1'])+30000)*sigma2,'--',color =cm.viridis(.5),label=r'$\sigma$ low')

	plt.ylabel(r'$w_{PV,a}$',fontsize=16)
	plt.xlabel('time',fontsize=16)
	plt.yticks(np.arange(0,1.1,0.2),[0,0.2,.4,.6,.8,1.0],fontsize=16)
	plt.xticks([0,sim_time],[0,sim_time],fontsize=16)
	plt.ylim(0,1.0)
	plt.xlim(-30000,sim_time)
	lgd = plt.legend(loc='lower right',fontsize=11)
	a1.spines['top'].set_visible(False)
	a1.spines['right'].set_visible(False)

	a2 = plt.subplot(322)
	a2.text(-0.1, 1.15, 'B', transform=a2.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(np.arange(-1,3),np.ones(4)*sigma1**2,'--',color =cm.viridis(.1),label=r'$\sigma^2$ high')
	plt.plot(np.arange(-1,3),np.ones(4)*sigma2**2,'--',color =cm.viridis(.5),label=r'$\sigma^2$ low')
	plt.bar([0,1],[results1['PV_avg'],results2['PV_avg']],yerr=[results1['PV_std'],results2['PV_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)
	plt.ylabel(r'$r_{PV}$',fontsize=16)
	plt.xticks([0,1],['high','low'],fontsize=16)
	plt.xlabel('uncertainty',fontsize=16)
	plt.yticks(np.arange(0,1.3,0.2),[0,0.2,.4,.6,.8,1.0,1.2],fontsize=16)
	plt.yticks([0,1],[0,1],fontsize=16)
	a2.spines['top'].set_visible(False)
	a2.spines['right'].set_visible(False)
	plt.ylim(0,1.3)
	plt.xlim(-0.5,1.5)
	#plt.tight_layout()
	plt.legend(fontsize=11)

	a3 = plt.subplot(323)
	a3.text(-0.1, 1.15, 'C', transform=a3.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(results1['rPa'], label='high', color =cm.viridis(.1),linewidth=2)
	plt.plot(results2['rPa'], label='low',color =cm.viridis(.5),linewidth=2)
	plt.plot(np.arange(-30000,len(results1['rPa'])),np.ones(len(results1['rPa'])+30000)*sigma1**2,'--',color =cm.viridis(.1),label=r'$\sigma^2$ high')
	plt.plot(np.arange(-30000,len(results1['rPa'])),np.ones(len(results1['rPa'])+30000)*sigma2**2,'--',color =cm.viridis(.5),label=r'$\sigma^2$ low')

	plt.ylabel(r'$r_{PV}(a)$',fontsize=16)
	plt.xlabel('time',fontsize=16)
	plt.yticks(np.arange(0,1.1,0.2),[0,0.2,.4,.6,.8,1.0],fontsize=16)
	plt.xticks([0,sim_time],[0,sim_time],fontsize=16)
	plt.ylim(0,1.0)
	plt.xlim(-30000,sim_time)
	lgd = plt.legend(loc='lower right',fontsize=11)
	a3.spines['top'].set_visible(False)
	a3.spines['right'].set_visible(False)



	a4=plt.subplot(324)
	a4.text(-0.1, 1.15, 'D', transform=a4.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(np.arange(-1,3),np.ones(4)*sigma1**2,'--',color =cm.viridis(.1),label=r'$\sigma^2$ high')
	plt.plot(np.arange(-1,3),np.ones(4)*sigma2**2,'--',color =cm.viridis(.5),label=r'$\sigma^2$ low')
	plt.bar([0,1],[PV_avg1,PV_avg2],yerr=[PV_std1,PV_std2],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)
	plt.ylabel(r'$r_{PV}(a)$',fontsize=16)
	plt.xticks([0,1],['high','low'],fontsize=16)
	plt.xlabel('uncertainty',fontsize=16)
	plt.yticks(np.arange(0,1.3,0.2),[0,0.2,.4,.6,.8,1.0,1.2],fontsize=16)
	plt.yticks([0,1],[0,1],fontsize=16)
	a4.spines['top'].set_visible(False)
	a4.spines['right'].set_visible(False)
	plt.ylim(0,1.0)
	plt.xlim(-0.5,1.5)
	#plt.tight_layout()
	plt.legend(fontsize=11)

	#plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))

	a5 = plt.subplot(325)
	a5.text(-0.1, 1.15, 'E', transform=a5.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.errorbar(sigmas, wPX_avg, yerr=wPX_std, linestyle='',marker='.',color='k')

	# works for wP=3.0
	plt.xlim(0.0,1.1)
	plt.ylim(0.1,1.1)
	plt.xticks(np.arange(0.0,1.1,.5),[0.0,.5,1.0],fontsize=16)
	plt.yticks(np.arange(0,1.2,.5),[0.0,.5,1.0],fontsize=16)

	plt.xlabel(r'$\sigma$',fontsize=16)
	plt.ylabel(r'$w_{PV,a}$',fontsize=16)
	a5.spines['top'].set_visible(False)
	a5.spines['right'].set_visible(False)

	a6 = plt.subplot(326)
	a6.text(-0.1, 1.15, 'E', transform=a6.transAxes,
	          fontsize=16, va='top', ha='right')
	#plt.plot(sigmas**2, PV_avg, color='k')
	plt.errorbar(sigmas**2, PV_avg, yerr=PV_std, linestyle='',marker='.',color='k')

	plt.xlim(-.1,1.1)
	plt.ylim(-.1,1.1)

	#plt.ylim(-.1,2.0)
	plt.xticks(np.arange(0.0,1.2,.5),[0.0,.5,1.0],fontsize=16)
	plt.yticks(np.arange(0.0,1.2,.5),[0.0,.5,1.0],fontsize=16)

	plt.xlabel(r'$\sigma^2$',fontsize=16)
	plt.ylabel(r'$r_{PV}(a)$',fontsize=16)
	a6.spines['top'].set_visible(False)
	a6.spines['right'].set_visible(False)

	plt.tight_layout()

	plt.savefig('./PVratesandweights_all2%s.png'%name, bbox_inches='tight')
	plt.savefig('./PVratesandweights_all2%s.pdf'%name, bbox_inches='tight')

