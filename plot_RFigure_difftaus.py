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
	seed = 1345
	no_samples = 100000
	stimulus_duration = 10
	start = (no_samples*stimulus_duration)-100000
	beta = 0.1 
	wP = np.sqrt(((2-beta)/beta)*(1/2)) # np.sqrt(1.95/0.1)
	means=np.arange(0.0,5.1,1.0)
	tau_E = 10.0
	tau_I = 2.0
	circuit = Circuit()
	circuit.dt = 1.0
	circuit.R_neuron= True
	circuit.single_PV = True
	circuit.PV_mu = False
	#circuit.NMDA = True
	circuit.beta_P = beta
	circuit.wPY1 = np.array([wP]) # small intitial weights
	circuit.wPR = np.array([wP]) # small intitial weights
	circuit.wPS_P = np.array([wP])
	circuit.wPS_N = np.array([wP])
	sim = Sim(stimulus_duration=stimulus_duration,number_of_samples=no_samples)
	sim.eta_R = 0.1*0.1
	sim.eta_P = 0.001*0.1
	sim.eta_S = 0.1*0.1
	sim.eta_ES = 0.01*0.1
	sim.eta_PS = 0.0001*0.1
	with multiprocessing.Pool(processes=5) as pool:
	    mean_results=pool.starmap(sim.run, zip(repeat(circuit),means,repeat(0.4),repeat(seed)))  


	R_avg = np.empty(len(means))
	R_std = np.empty(len(means))
	wRX_avg = np.empty(len(means))
	wRX_std = np.empty(len(means))

	for i,mean in enumerate(means):
	    R_avg[i] = np.mean(mean_results[i]['rRa'][start:])
	    R_std[i] = np.std(mean_results[i]['rRa'][start:])
	    wRX_avg[i] = np.mean(mean_results[i]['wRX1'][start:])
	    wRX_std[i] = np.std(mean_results[i]['wRX1'][start:])


	name = 'difftaus'
	mean1=1.0
	mean2=5.0
	sim_time= no_samples*stimulus_duration
	R_avg1 = np.mean(mean_results[1]['rRa'][start:])
	R_avg2 = np.mean(mean_results[-1]['rRa'][start:])
	R_std1 = np.std(mean_results[1]['rRa'][start:])
	R_std2 = np.std(mean_results[-1]['rRa'][start:])

	plt.figure(figsize=(7.5,10))
	a1 = plt.subplot(321)
	a1.text(-0.1, 1.15, 'A', transform=a1.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(mean_results[1]['wRX1'], label=r'$\mu=1$', color =cm.viridis(.1),linewidth=2)
	plt.plot(mean_results[-1]['wRX1'], label=r'$\mu=5$',color =cm.viridis(.5),linewidth=2)
	plt.plot(np.arange(-30000,len(mean_results[1]['wRX1'])),np.ones(len(mean_results[1]['wRX1'])+30000)*mean1,'--',color =cm.viridis(.1),label=r'$\mu=1$')
	plt.plot(np.arange(-30000,len(mean_results[-1]['wRX1'])),np.ones(len(mean_results[-1]['wRX1'])+30000)*mean2,'--',color =cm.viridis(.5),label=r'$\mu=5$')

	plt.ylabel(r'$w_{R,a}$',fontsize=16)
	plt.xlabel('time',fontsize=16)
	plt.yticks(np.arange(0,5.1,1.0),[0,1,2,3,4,5],fontsize=16)
	plt.xticks([0,sim_time],[0,sim_time],fontsize=16)
	plt.ylim(0,5.5)
	plt.xlim(-30000,sim_time)
	#lgd = plt.legend(bbox_to_anchor=(1,1),fontsize=11)
	a1.spines['top'].set_visible(False)
	a1.spines['right'].set_visible(False)

	a2 = plt.subplot(322)
	a2.text(-0.1, 1.15, 'B', transform=a2.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(np.arange(-1,3),np.ones(4)*mean1,'--',color =cm.viridis(.1),label=r'$\mu=1$')
	plt.plot(np.arange(-1,3),np.ones(4)*mean2,'--',color =cm.viridis(.5),label=r'$\mu=5$')
	plt.bar([0,1],[mean_results[1]['R_avg'],mean_results[-1]['R_avg']],yerr=[mean_results[1]['R_std'],mean_results[-1]['R_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)
	plt.ylabel(r'$r_{R}$',fontsize=16)
	plt.xticks([0,1],[r'$\mu=1$',r'$\mu=5$'],fontsize=16)
	plt.xlabel('mean',fontsize=16)
	plt.yticks(np.arange(0,5.1,1.0),[0,1,2,3,4,5],fontsize=16)

	#plt.yticks(np.arange(0,1.3,0.2),[0,0.2,.4,.6,.8,1.0,1.2],fontsize=16)
	#plt.yticks([0,1],[0,1],fontsize=16)
	a2.spines['top'].set_visible(False)
	a2.spines['right'].set_visible(False)
	plt.ylim(0,5.5)
	plt.xlim(-0.5,1.5)
	#plt.tight_layout()
	plt.legend(bbox_to_anchor=(1,1),fontsize=11)

	a3 = plt.subplot(323)
	a3.text(-0.1, 1.15, 'C', transform=a3.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(mean_results[1]['rRa'], label=r'$\mu=1$', color =cm.viridis(.1),linewidth=2)
	plt.plot(mean_results[-1]['rRa'], label=r'$\mu=5$',color =cm.viridis(.5),linewidth=2)
	plt.plot(np.arange(-30000,len(mean_results[1]['rRa'])),np.ones(len(mean_results[1]['rRa'])+30000)*mean1,'--',color =cm.viridis(.1))#,label=r'$\mu=5$')
	plt.plot(np.arange(-30000,len(mean_results[-1]['rRa'])),np.ones(len(mean_results[-1]['rRa'])+30000)*mean2,'--',color =cm.viridis(.5))#,label=r'$\mu=1$')

	plt.ylabel(r'$r_{R}(a)$',fontsize=16)
	plt.xlabel('time',fontsize=16)
	plt.yticks(np.arange(0,5.1,1.0),[0,1,2,3,4,5],fontsize=16)

	#plt.yticks(np.arange(0,1.1,0.2),[0,0.2,.4,.6,.8,1.0],fontsize=16)
	plt.xticks([0,sim_time],[0,sim_time],fontsize=16)
	plt.ylim(0,5.5)
	plt.xlim(-30000,sim_time)
	#lgd = plt.legend(loc='lower right',fontsize=11)
	a3.spines['top'].set_visible(False)
	a3.spines['right'].set_visible(False)



	a4=plt.subplot(324)
	a4.text(-0.1, 1.15, 'D', transform=a4.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(np.arange(-1,3),np.ones(4)*mean1,'--',color =cm.viridis(.1),label=r'$\mu=1$')
	plt.plot(np.arange(-1,3),np.ones(4)*mean2,'--',color =cm.viridis(.5),label=r'$\mu=5$')
	plt.bar([0,1],[R_avg1,R_avg2],yerr=[R_std1,R_std2],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)
	plt.ylabel(r'$r_{R}(a)$',fontsize=16)
	plt.xticks([0,1],[r'$\mu=1$',r'$\mu=5$'],fontsize=16)
	plt.xlabel('mean',fontsize=16)
	plt.yticks(np.arange(0,5.1,1.0),[0,1,2,3,4,5],fontsize=16)
	#plt.yticks([0,1],[0,1],fontsize=16)
	a4.spines['top'].set_visible(False)
	a4.spines['right'].set_visible(False)
	plt.ylim(0,5.5)
	plt.xlim(-0.5,1.5)
	#plt.tight_layout()
	#plt.legend(fontsize=11)

	#plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))

	a5 = plt.subplot(325)
	a5.text(-0.1, 1.15, 'E', transform=a5.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.errorbar(means, wRX_avg, yerr=wRX_std, linestyle='',marker='.',color='k')

	# works for wP=3.0
	plt.xlim(-0.1,5.1)
	plt.ylim(-0.1,5.1)
	plt.xticks(np.arange(0.0,5.1,1.0),[0,1,2,3,4,5],fontsize=16)
	plt.yticks(np.arange(0,5.2,1.0),[0,1,2,3,4,5],fontsize=16)

	plt.xlabel(r'$\mu$',fontsize=16)
	plt.ylabel(r'$w_{R,a}$',fontsize=16)
	a5.spines['top'].set_visible(False)
	a5.spines['right'].set_visible(False)

	a6 = plt.subplot(326)
	a6.text(-0.1, 1.15, 'F', transform=a6.transAxes,
	          fontsize=16, va='top', ha='right')
	#plt.plot(sigmas**2, PV_avg, color='k')
	plt.errorbar(means, R_avg, yerr=R_std, linestyle='',marker='.',color='k')

	plt.xlim(-0.1,5.1)
	plt.ylim(-0.1,5.1)
	plt.xticks(np.arange(0.0,5.1,1.0),[0,1,2,3,4,5],fontsize=16)
	plt.yticks(np.arange(0,5.2,1.0),[0,1,2,3,4,5],fontsize=16)

	plt.xlabel(r'$\mu$',fontsize=16)
	plt.ylabel(r'$r_{R}(a)$',fontsize=16)
	a6.spines['top'].set_visible(False)
	a6.spines['right'].set_visible(False)

	plt.tight_layout()

	plt.savefig('./Rratesandweights_all2dt%s%s.png'%(str(circuit.dt),name), bbox_inches='tight')
	plt.savefig('./Rratesandweights_all2dt%s%s.pdf'%(str(circuit.dt),name), bbox_inches='tight')
    