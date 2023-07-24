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
	sigma = 0.5
	wP = np.sqrt((2-beta)/beta) # np.sqrt(1.95/0.1)
	#sigmas = np.arange(0.1,1.1,0.1)
	sim_time= 10000
	start = sim_time-1000

	circuit = Circuit()
	circuit.dt = 0.1
	circuit.beta_P = beta
	circuit.wPY1 = np.array([wP]) # small intitial weights
	circuit.wPS_P = np.array([wP])
	circuit.plastic_SX = True
	sim = Sim(stimulus_duration=10, number_of_samples=1000)
	with multiprocessing.Pool(processes=2) as pool:
	    results=pool.starmap(sim.run_pPE, zip(repeat(circuit),means,repeat(sigma),repeat(seed)))

	SST_avg = np.empty_like(means)
	SST_std = np.empty_like(means)
	wSX_avg = np.empty_like(means)
	wSX_std = np.empty_like(means)
	for i,mean in enumerate(means):
	    print(i)
	    print(np.mean(results[i]['wSX'][start:]))
	    SST_avg[i] = np.mean(results[i]['rS_P'][start:])
	    SST_std[i] = np.std(results[i]['rS_P'][start:])
	    wSX_avg[i] = np.mean(results[i]['wSX'][start:])
	    wSX_std[i] = np.std(results[i]['wSX'][start:])


	x_values = np.arange(0, 10, 0.001)
	y1_values = stats.norm(means[1], sigma)
	y2_values = stats.norm(means[-1], sigma)

	plt.figure(figsize=(9,6))
	#plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))
	a0 = plt.subplot(231)
	a0.text(-0.1, 1.15, 'A', transform=a0.transAxes,
	          fontsize=16, va='top', ha='right')

	a = plt.subplot(232)
	a.text(-0.1, 1.15, 'B', transform=a.transAxes,fontsize=16, va='top', ha='right')
	plt.plot(x_values, y1_values.pdf(x_values), color = cm.magma(0.8),linewidth =3,label=r'$\mu = $%d'%means[1])
	plt.plot(x_values, y2_values.pdf(x_values), color = cm.magma(0.6),linewidth =3,label=r'$\mu = $%d'%means[-1])
	#plt.plot(np.ones((100))*7,np.arange(0,1,0.01),color = 'k',linewidth=3, label = 'positive MM')
	#plt.plot(np.ones((100))*3,np.arange(0,1,0.01),color = 'gray',linewidth=3, label = 'negative MM')

	ldg = plt.legend()
	plt.xlim(0,7)
	plt.ylim(0,1.0)
	plt.yticks([0,1],[0,1],fontsize=16)
	plt.xticks([2,5],[2,5],fontsize=16)
	plt.xlabel('stimulus', fontsize = 16)
	a.spines['top'].set_visible(False)
	a.spines['right'].set_visible(False)
	#a.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=False)
	#a.legend(loc='upper left',markerscale=0.5,handlelength=1.0)#, bbox_to_anchor=(1, 0.5))

	a2 = plt.subplot(233)
	a2.text(-0.1, 1.15, 'C', transform=a2.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(np.arange(-1,3),np.ones(4)*means[1],'--',color =cm.magma(.8),label=r'$\mu=$%d'%means[1])
	plt.plot(np.arange(-1,3),np.ones(4)*means[-1],'--',color =cm.magma(.6),label=r'$\mu=$%d'%means[-1])
	plt.bar([0,1],[SST_avg[1],SST_avg[-1]],yerr=[SST_std[1],SST_std[-1]],color=[cm.magma(.8),cm.magma(.6)],width=0.3)
	plt.ylabel(r'$r_{SST}$',fontsize=16)
	plt.xticks([0,1],[r'$\mu=$%d'%means[1],r'$\mu=$%d'%means[-1]],fontsize=16)
	plt.xlabel('mean',fontsize=16)
	plt.yticks(np.arange(0,6),[0,1,2,3,4,5],fontsize=16)

	a2.spines['top'].set_visible(False)
	a2.spines['right'].set_visible(False)
	plt.ylim(0,5.5)
	plt.xlim(-0.5,1.5)
	#plt.tight_layout()
	#plt.legend(fontsize=11)


	a3 = plt.subplot(234)
	a3.text(-0.1, 1.15, 'D', transform=a3.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.plot(results[1]['rS_P'], label='high', color =cm.magma(.8),linewidth=2)
	plt.plot(results[-1]['rS_P'], label='low',color =cm.magma(.6),linewidth=2)
	plt.plot(np.arange(-30000,len(results[1]['rS_P'])),np.ones(len(results[1]['rS_P'])+30000)*means[1],'--',color =cm.magma(.8),label=r'$\mu = $%d'%means[1])
	plt.plot(np.arange(-30000,len(results[-1]['rS_P'])),np.ones(len(results[-1]['rS_P'])+30000)*means[-1],'--',color =cm.magma(.6),label=r'$\mu = $%d'%means[-1])

	plt.ylabel(r'$r_{SST}$',fontsize=16)
	plt.xlabel('time',fontsize=16)
	plt.yticks(np.arange(0,6),[0,1,2,3,4,5],fontsize=16)
	plt.xticks([0,sim_time],[0,sim_time],fontsize=16)
	plt.ylim(0,5.5)
	plt.xlim(-500,sim_time)
	#lgd = plt.legend(loc='lower right',fontsize=11)
	a3.spines['top'].set_visible(False)
	a3.spines['right'].set_visible(False)
	#plt.savefig('./SSTrates_dt01.png', bbox_inches='tight')
	#plt.savefig('./SSTrates_dt01.pdf', bbox_inches='tight')




	


	a5 = plt.subplot(235)
	a5.text(-0.1, 1.15, 'E', transform=a5.transAxes,
	          fontsize=16, va='top', ha='right')
	plt.errorbar(means, wSX_avg, yerr=wSX_std, linestyle='',marker='.',color='k')

	# works for wP=3.0
	#plt.xlim(0.0,1.0)
	#plt.ylim(0.1,1.0)
	#plt.xticks(np.arange(0.0,1.1,.2),[0.0,.2,.4,.6,.8,1.0],fontsize=12)
	#plt.yticks(np.arange(0,1.2,.2),[0.0,.2,.4,.6,.8,1.0],fontsize=12)
	plt.xticks(np.arange(1,6,1),[1,2,3,4,5],fontsize=16)
	plt.yticks(np.arange(1,6,1),[1,2,3,4,5],fontsize=16)
	plt.xlabel(r'$\mu$',fontsize=16)
	plt.ylabel(r'$w_{SST,a}$',fontsize=16)
	a5.spines['top'].set_visible(False)
	a5.spines['right'].set_visible(False)

	
	a6 = plt.subplot(236)
	a6.text(-0.1, 1.15, 'F', transform=a6.transAxes,
	          fontsize=16, va='top', ha='right')
	#plt.plot(sigmas**2, PV_avg, color='k')
	plt.errorbar(means, SST_avg, yerr=SST_std, linestyle='',marker='.',color='k')
	print(SST_std)
	print('look up')
	#plt.xlim(-.1,1.0)
	#plt.ylim(-.1,1.0)

	#plt.ylim(-.1,2.0)
	plt.xticks(np.arange(1,6,1),[1,2,3,4,5],fontsize=16)
	plt.yticks(np.arange(1,6,1),[1,2,3,4,5],fontsize=16)

	#plt.yticks(np.arange(0.0,1.1,.2),[0.0,.2,.4,.6,.8,1.0],fontsize=12)

	plt.xlabel(r'$\mu$',fontsize=16)
	plt.ylabel(r'$r_{SST}$',fontsize=16)
	a6.spines['top'].set_visible(False)
	a6.spines['right'].set_visible(False)
	plt.tight_layout()
	#plt.savefig('./SSTerrorbarsscaling.png', bbox_inches='tight')
	#plt.savefig('./SSTerrorbarsscaling.pdf', bbox_inches='tight')




	plt.savefig('./SSTplot_dt01.png', bbox_inches='tight')
	plt.savefig('./SSTplot_dt01.pdf', bbox_inches='tight')
