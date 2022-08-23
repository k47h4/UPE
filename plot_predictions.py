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

def plot_mismatch_activity(results,singlePV=False):
    training_mean= 5.0
    sigmas = list(mismatch_results.keys())
    print(sigmas)
    sigma_low = sigmas[0]
    sigma_high = sigmas[-1]
    print(sigma_low)
    print(sigma_high)
    x_values = np.arange(0, 10, 0.001)
    y1_values = stats.norm(training_mean, sigma_low)
    y2_values = stats.norm(training_mean, sigma_high)

    plt.figure(figsize=(12.5,5.5))

    a = plt.subplot(261)

    a.text(-0.1, 1.15, 'B', transform=a.transAxes,
          fontsize=16, va='top', ha='right')
    plt.plot(x_values, y1_values.pdf(x_values), color = cm.viridis(0.5),linewidth =3,label='low uncertainty')
    plt.plot(x_values, y2_values.pdf(x_values), color = cm.viridis(0.1),linewidth =3,label='high uncertainty')
    plt.plot(np.ones((100))*7,np.arange(0,1,0.01),color = 'k',linewidth=3, label = 'positive MM')
    #plt.plot(np.ones((100))*3,np.arange(0,1,0.01),color = 'gray',linewidth=3, label = 'negative MM')

    ldg = plt.legend()
    plt.xlim(0,10)
    plt.ylim(0,2.5)
    plt.yticks([0,1],[0,1],fontsize=16)
    plt.xticks([5],[5],fontsize=16)
    plt.xlabel('stimulus', fontsize = 16)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    #a.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=False)
    a.legend(loc='upper left',markerscale=0.5,handlelength=1.0)#, bbox_to_anchor=(1, 0.5))


    #plt.legend(loc='lower right')
    a3 = plt.subplot(262)
    a3.text(-0.1, 1.15, 'H', transform=a3.transAxes,
          fontsize=16, va='top', ha='right')
    plt.bar([0,1],[results[sigma_high][-1]['SST_P_avg'],results[sigma_low][-1]['SST_P_avg']],yerr=[results[sigma_high][-1]['SST_P_std'],results[sigma_low][-1]['SST_P_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    plt.ylim(0,8)
    plt.xlim(-0.5,1.5)
    plt.yticks([0,5],[0,5],fontsize=16)
    a3.spines['top'].set_visible(False)
    a3.spines['right'].set_visible(False)
    plt.ylabel(r'SST$^+$ rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.xlabel('uncertainty',fontsize=16)

    a4 = plt.subplot(263)
    a4.text(-0.1, 1.15, 'H', transform=a4.transAxes,
          fontsize=16, va='top', ha='right')
    plt.bar([0,1],[results[sigma_high][-1]['SST_N_avg'],results[sigma_low][-1]['SST_N_avg']],yerr=[results[sigma_high][-1]['SST_N_std'],results[sigma_low][-1]['SST_N_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    plt.ylim(0,8)
    plt.xlim(-0.5,1.5)
    plt.yticks([0,7],[0,7],fontsize=16)
    a4.spines['top'].set_visible(False)
    a4.spines['right'].set_visible(False)
    plt.ylabel(r'SST$^-$ rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.xlabel('uncertainty',fontsize=16)
    


    """a7 = plt.subplot(255)
    a7.text(-0.1, 1.15, 'E', transform=a7.transAxes,
          fontsize=16, va='top', ha='right')
    plt.errorbar(mean_range,SST_rates_mean, yerr=SST_std_mean, color = 'k', fmt='.')

    plt.xlabel(r'$\mu$',fontsize=15)
    plt.ylabel('SST rate',fontsize=15)
    plt.xticks([0,5],[0,5],fontsize=16)
    plt.yticks([0,5],[0,5],fontsize=16)
    plt.ylim(-0.2,6)
    a7.spines['top'].set_visible(False)
    a7.spines['right'].set_visible(False)
    """

    
    a2 = plt.subplot(264)
    a2.text(-0.1, 1.15, 'G', transform=a2.transAxes,
          fontsize=16, va='top', ha='right')
    if singlePV:
        plt.bar([0,1],[results[sigma_high][-1]['PV_avg'],results[sigma_low][-1]['PV_avg']],yerr=[results[sigma_high][-1]['PV_std'],results[sigma_low][-1]['PV_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    else:
        plt.bar([0,1],[results[sigma_high][-1]['PV_P_avg'],results[sigma_low][-1]['PV_P_avg']],yerr=[results[sigma_high][-1]['PV_P_std'],results[sigma_low][-1]['PV_P_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    plt.ylabel('PV rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.yticks([0,1,2],[0,1,2],fontsize=16)
    plt.ylim(0,2)
    a2.spines['top'].set_visible(False)
    a2.spines['right'].set_visible(False)
    plt.xlim(-0.5,1.5)
    plt.xlabel('uncertainty',fontsize=16)

    a1 = plt.subplot(265)
    a1.text(-0.1, 1.15, 'F', transform=a1.transAxes,
          fontsize=16, va='top', ha='right')
    plt.bar([0,1],[results[sigma_high][-1]['E_P_avg'],results[sigma_low][-1]['E_P_avg']],yerr=[results[sigma_high][-1]['E_P_std'],results[sigma_low][-1]['E_P_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    plt.ylabel(r'$\Upsilon^+$ rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.yticks([0,1,2],[0,1,2],fontsize=16)
    a1.spines['top'].set_visible(False)
    a1.spines['right'].set_visible(False)
    plt.ylim(0,2)
    plt.xlim(-0.5,1.5)
    plt.xlabel('uncertainty',fontsize=16)

    a5 = plt.subplot(266)
    a5.text(-0.1, 1.15, 'F', transform=a5.transAxes,
          fontsize=16, va='top', ha='right')
    plt.bar([0,1],[results[sigma_high][-1]['E_N_avg'],results[sigma_low][-1]['E_N_avg']],yerr=[results[sigma_high][-1]['E_N_std'],results[sigma_low][-1]['E_N_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    plt.ylabel(r'$\Upsilon^-$ rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.yticks([0,1,2],[0,1,2],fontsize=16)
    a5.spines['top'].set_visible(False)
    a5.spines['right'].set_visible(False)
    plt.ylim(0,2)
    plt.xlim(-0.5,1.5)
    plt.xlabel('uncertainty',fontsize=16)
    

    an = plt.subplot(267)
    an.text(-0.1, 1.15, 'B', transform=an.transAxes,
          fontsize=16, va='top', ha='right')
    plt.plot(x_values, y1_values.pdf(x_values), color = cm.viridis(0.5),linewidth =3,label='low uncertainty')
    plt.plot(x_values, y2_values.pdf(x_values), color = cm.viridis(0.1),linewidth =3,label='high uncertainty')
    #plt.plot(np.ones((100))*7,np.arange(0,1,0.01),color = 'k',linewidth=3, label = 'positive MM')
    plt.plot(np.ones((100))*3,np.arange(0,1,0.01),color = 'gray',linewidth=3, label = 'negative MM')

    ldg = plt.legend()
    plt.xlim(0,10)
    plt.ylim(0,2.5)
    plt.yticks([0,1],[0,1],fontsize=16)
    plt.xticks([5],[5],fontsize=16)
    plt.xlabel('stimulus', fontsize = 16)
    an.spines['top'].set_visible(False)
    an.spines['right'].set_visible(False)
    #a.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=False)
    a.legend(loc='upper left',markerscale=0.5,handlelength=1.0)#, bbox_to_anchor=(1, 0.5))


    #plt.legend(loc='lower right')
    a3n = plt.subplot(268)
    a3n.text(-0.1, 1.15, 'H', transform=a3n.transAxes,
          fontsize=16, va='top', ha='right')
    plt.bar([0,1],[results[sigma_high][0]['SST_P_avg'],results[sigma_low][0]['SST_P_avg']],yerr=[results[sigma_high][0]['SST_P_std'],results[sigma_low][0]['SST_P_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    plt.ylim(0,8)
    plt.xlim(-0.5,1.5)
    plt.yticks([0,5],[0,5],fontsize=16)
    a3n.spines['top'].set_visible(False)
    a3n.spines['right'].set_visible(False)
    plt.ylabel(r'SST$^+$ rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.xlabel('uncertainty',fontsize=16)

    a4n = plt.subplot(269)
    a4n.text(-0.1, 1.15, 'H', transform=a4n.transAxes,
          fontsize=16, va='top', ha='right')
    plt.bar([0,1],[results[sigma_high][0]['SST_N_avg'],results[sigma_low][0]['SST_N_avg']],yerr=[results[sigma_high][0]['SST_N_std'],results[sigma_low][0]['SST_N_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    plt.ylim(0,8)
    plt.xlim(-0.5,1.5)
    plt.yticks([0,3],[0,3],fontsize=16)
    a4n.spines['top'].set_visible(False)
    a4n.spines['right'].set_visible(False)
    plt.ylabel(r'SST$^-$ rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.xlabel('uncertainty',fontsize=16)
    


    """a7 = plt.subplot(255)
    a7.text(-0.1, 1.15, 'E', transform=a7.transAxes,
          fontsize=16, va='top', ha='right')
    plt.errorbar(mean_range,SST_rates_mean, yerr=SST_std_mean, color = 'k', fmt='.')

    plt.xlabel(r'$\mu$',fontsize=15)
    plt.ylabel('SST rate',fontsize=15)
    plt.xticks([0,5],[0,5],fontsize=16)
    plt.yticks([0,5],[0,5],fontsize=16)
    plt.ylim(-0.2,6)
    a7.spines['top'].set_visible(False)
    a7.spines['right'].set_visible(False)
    """

    
    a2n = plt.subplot(2,6,10)
    a2n.text(-0.1, 1.15, 'G', transform=a2n.transAxes,
          fontsize=16, va='top', ha='right')
    if singlePV:
        plt.bar([0,1],[results[sigma_high][0]['PV_avg'],results[sigma_low][0]['PV_avg']],yerr=[results[sigma_high][0]['PV_std'],results[sigma_low][0]['PV_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    else:
        plt.bar([0,1],[results[sigma_high][0]['PV_N_avg'],results[sigma_low][0]['PV_N_avg']],yerr=[results[sigma_high][0]['PV_N_std'],results[sigma_low][0]['PV_N_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  

    plt.ylabel('PV rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.yticks([0,1,2],[0,1,2],fontsize=16)
    plt.ylim(0,2)
    a2n.spines['top'].set_visible(False)
    a2n.spines['right'].set_visible(False)
    plt.xlim(-0.5,1.5)
    plt.xlabel('uncertainty',fontsize=16)

    a1n = plt.subplot(2,6,11)
    a1n.text(-0.1, 1.15, 'F', transform=a1n.transAxes,
          fontsize=16, va='top', ha='right')
    plt.bar([0,1],[results[sigma_high][0]['E_P_avg'],results[sigma_low][0]['E_P_avg']],yerr=[results[sigma_high][0]['E_P_std'],results[sigma_low][0]['E_P_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    plt.ylabel(r'$\Upsilon^+$ rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.yticks([0,1,2],[0,1,2],fontsize=16)
    a1n.spines['top'].set_visible(False)
    a1n.spines['right'].set_visible(False)
    plt.ylim(0,2)
    plt.xlim(-0.5,1.5)
    plt.xlabel('uncertainty',fontsize=16)

    a5n = plt.subplot(2,6,12)
    a5n.text(-0.1, 1.15, 'F', transform=a5n.transAxes,
          fontsize=16, va='top', ha='right')
    plt.bar([0,1],[results[sigma_high][0]['E_N_avg'],results[sigma_low][0]['E_N_avg']],yerr=[results[sigma_high][0]['E_N_std'],results[sigma_low][0]['E_N_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)  
    plt.ylabel(r'$\Upsilon^-$ rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.yticks([0,1,2],[0,1,2],fontsize=16)
    a5n.spines['top'].set_visible(False)
    a5n.spines['right'].set_visible(False)
    plt.ylim(0,2)
    plt.xlim(-0.5,1.5)
    plt.xlabel('uncertainty',fontsize=16)


    """


    a3 = plt.subplot(2,5,10)
    a3.text(-0.1, 1.15, 'I', transform=a3.transAxes,
          fontsize=16, va='top', ha='right')
    plt.errorbar(sigma_range**2,PV_rates_sigma, yerr=PV_std_sigma, color = 'k',fmt='.')
    plt.xlabel(r'$\sigma^2$',fontsize=15)
    plt.ylabel('PV rate',fontsize=15)
    plt.xticks([0.0,1.0],[0,1],fontsize=16)
    plt.yticks([0,2],[0,2],fontsize=16)
    a3.spines['top'].set_visible(False)
    a3.spines['right'].set_visible(False)
    """
    plt.tight_layout()

    plt.savefig('./NMDAcircuitmismatch.pdf', bbox_inches='tight')



if __name__ == "__main__":

	seed = 123
	training_mean = 5.0 
	sigmas = np.arange(0.1,1.1,0.7)
	beta = 0.1
	wP = np.sqrt(((2-beta)/beta)) # np.sqrt(1.95/0.1)
	circuit = Circuit()
	circuit.single_PV = False
	circuit.wPY1 = np.array([wP]) # small intitial weights
	circuit.wPR = np.array([wP]) # small intitial weights
	circuit.wPS_P = np.array([wP])
	circuit.wPS_N = np.array([wP])
	circuit.NMDA = True

	sim = Sim(stimulus_duration=1,number_of_samples=200000)
	with multiprocessing.Pool(processes=5) as pool:
	    training_results=pool.starmap(sim.run, zip(repeat(circuit),repeat(training_mean),sigmas,repeat(seed)))


	means=np.arange(3.0,8.0,2.0)
	mismatch_results = {}
	for i,sigma in enumerate(sigmas):
	    print(np.mean(training_results[i]['wPX1_P'][-100000:]))
	    print(training_results[i]['wPX1_P'][-1])
	    circuit = Circuit()
	    circuit.single_PV = False
	    circuit.Rrate = training_mean
	    circuit.plastic_PX=False
	    circuit.NMDA = True
	    circuit.wPY1 = np.array([wP]) # small intitial weights
	    circuit.wPR = np.array([wP]) # small intitial weights
	    circuit.wPS_P = np.array([wP])
	    circuit.wPS_N = np.array([wP])
	    circuit.wRX1 = training_results[i]['wRX1'][-1]
	    circuit.wPX1_P = training_results[i]['wPX1_P'][-1]
	    circuit.wPX1_N = training_results[i]['wPX1_N'][-1]
	    sim = Sim(stimulus_duration=3,number_of_samples=200000)
	    with multiprocessing.Pool(processes=5) as pool:
	        mismatch_results[sigma]=pool.starmap(sim.run, zip(repeat(circuit),means,repeat(0.0),repeat(seed)))     


	plot_mismatch_activity(mismatch_results)
	        