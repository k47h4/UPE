

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_PVratesandweights(results1,results2,sigma1,sigma2,name):
    plt.figure(figsize=(7,4))
    a5 = plt.subplot(121)
    plt.plot(results1['wPX1'], label='high', color =cm.viridis(.1),linewidth=2, zorder = -1)
    plt.plot(results2['wPX1'], label='low',color =cm.viridis(.5),linewidth=2, zorder = -1)
    plt.plot(np.arange(len(results1['wPX1'])),np.ones(len(results1['wPX1']))*sigma1,'--',color =cm.viridis(.1),label=r'$\sigma$ high')
    plt.plot(np.arange(len(results1['wPX1'])),np.ones(len(results1['wPX1']))*sigma2,'--',color =cm.viridis(.5),label=r'$\sigma$ low')

    plt.ylabel('PV weights',fontsize=16)
    plt.xlabel('time',fontsize=16)
    plt.yticks(np.arange(0,1.1,0.2),[0,0.2,.4,.6,.8,1.0],fontsize=16)
    plt.xticks([0,300000],[0,300000],fontsize=16)
    plt.ylim(0,1.0)
    #plt.xlim(0,11000)
    lgd = plt.legend(loc='lower right',fontsize=11)
    a5.spines['top'].set_visible(False)
    a5.spines['right'].set_visible(False)
    a5.set_rasterization_zorder(0)


    a2 = plt.subplot(122)
    plt.plot(np.arange(-1,3),np.ones(4)*sigma1**2,'--',color =cm.viridis(.1),label=r'$\sigma^2$ high')
    plt.plot(np.arange(-1,3),np.ones(4)*sigma2**2,'--',color =cm.viridis(.5),label=r'$\sigma^2$ low')
    plt.bar([0,1],[results1['PV_avg'],results2['PV_avg']],yerr=[results1['PV_std'],results2['PV_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)
    plt.ylabel('PV rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.xlabel('uncertainty',fontsize=16)
    plt.yticks(np.arange(0,1.3,0.2),[0,0.2,.4,.6,.8,1.0,1.2],fontsize=16)
    plt.yticks([0,1],[0,1],fontsize=16)
    a2.spines['top'].set_visible(False)
    a2.spines['right'].set_visible(False)
    plt.ylim(0,1.3)
    plt.xlim(-0.5,1.5)
    plt.tight_layout()
    plt.legend(fontsize=11)
    plt.savefig('./PVratesandweights_%s.png'%name, bbox_inches='tight')
    plt.savefig('./PVratesandweights_%s.pdf'%name, bbox_inches='tight')

def plot_2PVratesandweights(results1,results2,sigma1,sigma2,name):
    plt.figure(figsize=(7,4))
    a5 = plt.subplot(221)
    plt.plot(results1['wPX1_P'], label='high', color =cm.viridis(.1),linewidth=2, zorder = -1)
    plt.plot(results2['wPX1_P'], label='low',color =cm.viridis(.5),linewidth=2, zorder = -1)
    plt.plot(np.arange(len(results1['wPX1'])),np.ones(len(results1['wPX1']))*sigma1,'--',color =cm.viridis(.1),label=r'$\sigma$ high')
    plt.plot(np.arange(len(results1['wPX1'])),np.ones(len(results1['wPX1']))*sigma2,'--',color =cm.viridis(.5),label=r'$\sigma$ low')

    plt.ylabel('PV+ weights',fontsize=16)
    plt.xlabel('time',fontsize=16)
    plt.yticks(np.arange(0,1.1,0.2),[0,0.2,.4,.6,.8,1.0],fontsize=16)
    plt.xticks([0,300000],[0,300000],fontsize=16)
    plt.ylim(0,1.0)
    #plt.xlim(0,11000)
    lgd = plt.legend(loc='lower right',fontsize=11)
    a5.spines['top'].set_visible(False)
    a5.spines['right'].set_visible(False)
    a5.set_rasterization_zorder(0)


    a2 = plt.subplot(222)
    plt.plot(np.arange(-1,3),np.ones(4)*sigma1**2,'--',color =cm.viridis(.1),label=r'$\sigma^2$ high')
    plt.plot(np.arange(-1,3),np.ones(4)*sigma2**2,'--',color =cm.viridis(.5),label=r'$\sigma^2$ low')
    plt.bar([0,1],[results1['PV_P_avg'],results2['PV_P_avg']],yerr=[results1['PV_P_std'],results2['PV_P_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)
    plt.ylabel('PV+ rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.xlabel('uncertainty',fontsize=16)
    plt.yticks(np.arange(0,1.3,0.2),[0,0.2,.4,.6,.8,1.0,1.2],fontsize=16)
    plt.yticks([0,1],[0,1],fontsize=16)
    a2.spines['top'].set_visible(False)
    a2.spines['right'].set_visible(False)
    plt.ylim(0,1.3)
    plt.xlim(-0.5,1.5)

    a6 = plt.subplot(223)
    plt.plot(results1['wPX1_N'], label='high', color =cm.viridis(.1),linewidth=2, zorder = -1)
    plt.plot(results2['wPX1_N'], label='low',color =cm.viridis(.5),linewidth=2, zorder = -1)
    plt.plot(np.arange(len(results1['wPX1_N'])),np.ones(len(results1['wPX1_N']))*sigma1,'--',color =cm.viridis(.1),label=r'$\sigma$ high')
    plt.plot(np.arange(len(results1['wPX1_N'])),np.ones(len(results1['wPX1_N']))*sigma2,'--',color =cm.viridis(.5),label=r'$\sigma$ low')

    plt.ylabel('PV- weights',fontsize=16)
    plt.xlabel('time',fontsize=16)
    plt.yticks(np.arange(0,1.1,0.2),[0,0.2,.4,.6,.8,1.0],fontsize=16)
    plt.xticks([0,300000],[0,300000],fontsize=16)
    plt.ylim(0,1.0)
    #plt.xlim(0,11000)
    lgd = plt.legend(loc='lower right',fontsize=11)
    a6.spines['top'].set_visible(False)
    a6.spines['right'].set_visible(False)
    a6.set_rasterization_zorder(0)


    a3 = plt.subplot(224)
    plt.plot(np.arange(-1,3),np.ones(4)*sigma1**2,'--',color =cm.viridis(.1),label=r'$\sigma^2$ high')
    plt.plot(np.arange(-1,3),np.ones(4)*sigma2**2,'--',color =cm.viridis(.5),label=r'$\sigma^2$ low')
    plt.bar([0,1],[results1['PV_N_avg'],results2['PV_N_avg']],yerr=[results1['PV_N_std'],results2['PV_N_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)
    plt.ylabel('PV- rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.xlabel('uncertainty',fontsize=16)
    plt.yticks(np.arange(0,1.3,0.2),[0,0.2,.4,.6,.8,1.0,1.2],fontsize=16)
    plt.yticks([0,1],[0,1],fontsize=16)
    a3.spines['top'].set_visible(False)
    a3.spines['right'].set_visible(False)
    plt.ylim(0,1.3)
    plt.xlim(-0.5,1.5)



    plt.tight_layout()
    plt.legend(fontsize=11)
    plt.savefig('./PV2ratesandweights_%s.png'%name, bbox_inches='tight')
    plt.savefig('./PV2ratesandweights_%s.pdf'%name, bbox_inches='tight')

def plot_Error(mismatch_results,training_mean,mean_range,sigma_range,name):
    #mean_range=mismatch_results.keys()
    #sigma_range=mismatch_results[mean_range[0]].keys()
    
    E_P_rates=np.empty((len(mean_range),len(sigma_range)))
    E_N_rates=np.empty((len(mean_range),len(sigma_range)))
    E_P_analytical=np.empty((len(mean_range),len(sigma_range)))
    E_N_analytical=np.empty((len(mean_range),len(sigma_range)))
    for i,mean in enumerate(mean_range):
        for j,sigma in enumerate(sigma_range):
            print(mismatch_results[mean][sigma]['E_P_avg'])
            E_P_rates[i,j] = mismatch_results[mean][sigma]['E_P_avg']
            E_N_rates[i,j] = mismatch_results[mean][sigma]['E_N_avg']
            #E_P_std[i,j] = mismatch_results[mean][sigma]['E_P_std']
            #E_N_std[i,j] = mismatch_results[mean][sigma]['E_N_std']
            diff_E = np.max((mean-training_mean,0))
            diff_N = np.max((training_mean-mean,0))
            E_P_analytical[i,j] = (1/(1.0+(0.9*sigma)**2))*diff_E
            E_N_analytical[i,j] = (1/(1.0+(0.9*sigma)**2))*diff_N

    plt.figure(figsize=(8,5))
    #plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))
    plt.subplot(111)
    for k,sigma in enumerate(sigma_range):
        print(1-k*0.1)
        #plt.plot(mean_range, E_P_analytical[:,k], color='k')

        plt.plot(mean_range, E_P_rates[:,k], '--', color=cm.viridis(1-k*0.1))
        #plt.plot(mean_range, E_N_analytical[:,k], color='k')
        plt.plot(mean_range, E_N_rates[:,k], color=cm.viridis(1-k*0.1),label=r'$\sigma=%.1f$'%sigma)


    # works for wP=3.0
    #plt.xlim(0.1,1.0)
    #plt.ylim(0.1,1.0)
    plt.xticks(mean_range,mean_range-training_mean,fontsize=16)
    plt.yticks(np.arange(0,2.1,0.5),np.arange(0,2.1,0.5),fontsize=16)
    plt.xlabel(r'$s-\mu$',fontsize=20)
    plt.ylabel(r'$\Upsilon\  rate$',fontsize=20)
    #plt.legend(bbox_to_anchor=(1,1), fontsize=16, loc="upper left")
    plt.legend(bbox_to_anchor=(1,1), fontsize=16, loc="upper left")

    plt.tight_layout()
    plt.savefig('./Eratemeanssigmas%s.png'%name, bbox_inches='tight')
    plt.savefig('./Eratemeanssigmas%s.pdf'%name, bbox_inches='tight')
    
    plt.figure(figsize=(7,3))
    #plt.plot(sigmas, PV_avg*(1/np.sqrt(2)))
    plt.subplot(121)
    for k,mean in enumerate(mean_range):
        if mean>5.0:
            plt.plot(sigma_range, E_P_rates[k,:], color=cm.magma((mean-5.0)*0.2),label=r'$|s-\mu|=%.f$'%np.abs(mean-training_mean))
            #plt.plot(sigma_range, E_N_rates[k,:], color=cm.magma(mean*0.1),label=r'$E^- \mu=%.f$'%mean)
            #plt.plot(sigma_range, E_P_analytical[k,:], '--',color=cm.viridis((mean-5.0)*0.1))
        #plt.plot(sigma_range, E_N_analytical[k,:], '--', color=cm.magma(mean*0.1))

    # works for wP=3.0
    #plt.xlim(0.1,1.0)
    #plt.ylim(0.1,1.0)
    plt.xticks([.1,.9,1.7],[.1,.9,1.7],fontsize=16)
    plt.yticks(np.arange(0,3,1),np.arange(0,3,1),fontsize=16)
    plt.legend(bbox_to_anchor=(1,1), fontsize=16, loc="upper left")

    plt.xlabel(r'$\sigma$',fontsize=20)
    plt.ylabel(r'$\Upsilon^+$ rate',fontsize=20)
    #plt.legend(bbox_to_anchor=(1,1), fontsize=16, loc="upper left")

    plt.subplot(122)
    for k,mean in enumerate(mean_range[mean_range<5.0]):
        plt.plot(sigma_range, E_N_rates[k,:], '--',color=cm.magma((5.0-mean)*0.2))
        #plt.plot(sigma_range, E_N_analytical[k,:], '--', color=cm.viridis((5.0-mean)*0.1))

    # works for wP=3.0
    #plt.xlim(0.1,1.0)
    #plt.ylim(0.1,1.0)
    plt.xticks([.1,.9,1.7],[.1,.9,1.7],fontsize=16)
    plt.yticks(np.arange(0,3,1),np.arange(0,3,1),fontsize=16)
    plt.xlabel(r'$\sigma$',fontsize=20)
    plt.ylabel(r'$\Upsilon^-$ rate',fontsize=20)
    plt.tight_layout()
    plt.savefig('./Eratesigmasmeans%s.png'%name, bbox_inches='tight')
    plt.savefig('./Eratesigmasmeans%s.pdf'%name, bbox_inches='tight')

def plot_all_rates(results1,xmin=0,xmax=10000,name='defaultname'):
    plt.figure(figsize=(15,10))

    a0 = plt.subplot(231)
    plt.plot(results1['rY1'][:-1], label='stimulus[t]',color ='k',linewidth=2)
    plt.plot(results1['rS_N'][1:], label='SST_N[t+1]', color =cm.viridis(.5),linewidth=2)
    plt.ylim(3,7)
    plt.plot(results1['rR'][:-1], label='mean prediction', color ='g',linewidth=2)
    plt.plot(results1['rS_P'][1:], label='SST+', color =cm.viridis(.9),linewidth=2)

    #plt.ylabel('stimulus',fontsize=16)
    plt.xlabel('time',fontsize=16)
    #plt.yticks([0,1],[0,1],fontsize=16)
    #plt.xticks([0,10000],[0,10000],fontsize=16)
    #plt.ylim(0,1.25)
    plt.xlim(xmin,xmax)
    plt.legend(loc='lower right')
    a0.spines['top'].set_visible(False)
    a0.spines['right'].set_visible(False)

    a0 = plt.subplot(232)
    plt.plot(results1['rP'], color ='k',linewidth=2)
    try:
        plt.plot(results1['rC1'], color=cm.magma(.5), label='C1')
        plt.plot(results1['rC2'], color=cm.viridis(.3), label='C2')
        plt.plot(np.nonzero(results1['rC1']==1)[0], abs(results1['rY1'][results1['rC1']==1]), 'o', color=cm.magma(.5), label='C1 sample')
        plt.plot(np.nonzero(results1['rC2']==1)[0], abs(results1['rY1'][results1['rC2']==1]), 'o', color=cm.viridis(.3), label='C2 sample')
    except:
        print('no context')
    plt.ylabel('PV rates',fontsize=16)
    plt.xlabel('time',fontsize=16)
    #plt.yticks([0,1],[0,1],fontsize=16)
    #plt.xticks([0,10000],[0,10000],fontsize=16)
    plt.ylim(0,1.25)
    #plt.xlim(0,11000)
    plt.xlim(xmin,xmax)

    plt.legend(loc='lower right')
    a0.spines['top'].set_visible(False)
    a0.spines['right'].set_visible(False)


    a0 = plt.subplot(233)
    plt.plot(results1['rE_P'], color =cm.viridis(.1),linewidth=2,label='rE_P')
    #plt.plot(results1['rE_N'], label='stimulus', color =cm.viridis(.5),linewidth=2,label='rE_N')

    plt.ylabel('E_P rates',fontsize=16)
    plt.xlabel('time',fontsize=16)
    #plt.yticks([0,1],[0,1],fontsize=16)
    #plt.xticks([0,10000],[0,10000],fontsize=16)
    #plt.ylim(0,1.25)
    #plt.xlim(0,11000)
    plt.xlim(xmin,xmax)

    plt.legend(loc='lower right')
    a0.spines['top'].set_visible(False)
    a0.spines['right'].set_visible(False)

    a0 = plt.subplot(234)
    plt.plot(results1['rS_N'][:-1], label='SST+', color =cm.viridis(.5),linewidth=2)
    plt.plot(results1['rR'][1:], label='mean prediction', color ='k',linewidth=2)

    plt.ylabel('rates',fontsize=16)
    plt.xlabel('time',fontsize=16)
    #plt.yticks([0,1],[0,1],fontsize=16)
    #plt.xticks([0,10000],[0,10000],fontsize=16)
    plt.ylim(3,7)
    #plt.xlim(0,11000)
    plt.xlim(xmin,xmax)

    plt.legend(loc='lower right')
    a0.spines['top'].set_visible(False)
    a0.spines['right'].set_visible(False)
    
    
    a12 = plt.subplot(235)
    plt.plot(results1['PV_in'], color =cm.viridis(.1),linewidth=2)
    plt.ylabel('PV input',fontsize=16)
    plt.xlabel('time',fontsize=16)
    #plt.yticks([0,1],[0,1],fontsize=16)
    #plt.xticks([0,10000],[0,10000],fontsize=16)
    #plt.ylim(0,1.25)
    #plt.xlim(0,11000)
    plt.xlim(xmin,xmax)

    plt.legend(loc='lower right')
    a0.spines['top'].set_visible(False)
    a0.spines['right'].set_visible(False)
    
    a13 = plt.subplot(236)
    plt.plot(results1['rE_N'], color =cm.viridis(.1),linewidth=2)
    plt.ylabel('E_N rates',fontsize=16)
    plt.xlabel('time',fontsize=16)
    plt.xlim(xmin,xmax)

    #plt.yticks([0,1],[0,1],fontsize=16)
    #plt.xticks([0,10000],[0,10000],fontsize=16)
    #plt.ylim(0,1.25)
    #plt.xlim(0,11000)
    plt.legend(loc='lower right')
    a0.spines['top'].set_visible(False)
    a0.spines['right'].set_visible(False)
    plt.tight_layout()


def plot_Rratesandweights(results1,results2,mean1,mean2,name):
    plt.figure(figsize=(7,4))
    a5 = plt.subplot(121)
    plt.plot(results1['wRX1'], label=r'$\mu=%d$'%mean1, color =cm.viridis(.1),linewidth=2,zorder=-10)
    plt.plot(results2['wRX1'], label=r'$\mu=%d$'%mean2,color =cm.viridis(.5),linewidth=2,zorder=-10)
    plt.plot(np.arange(len(results1['wRX1'])),np.ones(len(results1['wRX1']))*mean1,'--',color =cm.viridis(.1))#,label=r'$\mu=%d$'%mean1)
    plt.plot(np.arange(len(results1['wRX1'])),np.ones(len(results1['wRX1']))*mean2,'--',color =cm.viridis(.5))#,label=r'$\mu=%d$'%mean1)

    plt.ylabel('R weights',fontsize=16)
    plt.xlabel('time',fontsize=16)
    #plt.yticks(np.arange(0,1.1,0.2),[0,0.2,.4,.6,.8,1.0],fontsize=16)
    plt.xticks([0,300000],[0,300000],fontsize=16)
    #plt.ylim(0,1.25)
    #plt.xlim(0,11000)
    lgd = plt.legend(loc='lower right',fontsize=11)
    a5.spines['top'].set_visible(False)
    a5.spines['right'].set_visible(False)
    a5.set_rasterization_zorder(0)

    a2 = plt.subplot(122)
    plt.plot(np.arange(-1,3),np.ones(4)*mean1,'--',color =cm.viridis(.1))#,label=r'$\mu=%d$'%mean1)
    plt.plot(np.arange(-1,3),np.ones(4)*mean2,'--',color =cm.viridis(.5))#,label=r'$\mu=%d$'%mean2)
    plt.bar([0,1],[results1['R_avg'],results2['R_avg']],yerr=[results1['R_std'],results2['R_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)
    plt.ylabel('R rate',fontsize=16)
    plt.xticks([0,1],[r'$\mu=%d$'%mean1,r'$\mu=%d$'%mean2],fontsize=16)
    plt.xlabel('Stimulus',fontsize=16)
    #plt.yticks(np.arange(0,1.3,0.2),[0,0.2,.4,.6,.8,1.0,1.2],fontsize=16)
    #plt.yticks([0,2],[0,2],fontsize=16)
    a2.spines['top'].set_visible(False)
    a2.spines['right'].set_visible(False)
    #plt.ylim(0,2.3)
    plt.xlim(-0.5,1.5)
    plt.tight_layout()
    #plt.legend(fontsize=11)
    plt.savefig('./Rratesandweights_%s.png'%name, bbox_inches='tight')
    plt.savefig('./Rratesandweights_%s.pdf'%name, bbox_inches='tight')


def plot_Rrate(resultsUPE,resultsN,name):
    a1 = plt.subplot(111)
    a1.text(-0.1, 1.15, 'A', transform=a1.transAxes,
          fontsize=16, va='top', ha='right')
    plt.plot(resultsN['rR'], label='unscaled',color =cm.viridis(.5),linewidth=2)
    plt.plot(resultsUPE['rR'], label='UPE', color =cm.viridis(.1),linewidth=2)
    plt.ylabel(r'R rate',fontsize=16)
    plt.xlabel('time',fontsize=16)
    #plt.legend(loc='lower right')
    plt.xlim(0,1000)
    plt.xticks([0,1000],[0,1000],fontsize=16)
    plt.ylim(0,6)
    a1.spines['top'].set_visible(False)
    a1.spines['right'].set_visible(False)

    plt.savefig('./Rrates_%s.png'%name, bbox_inches='tight')
    plt.savefig('./Rrates_%s.pdf'%name, bbox_inches='tight')


def plot_wPS(results, name):
    
    plt.figure(figsize=(3,3))
    a3 = plt.subplot(111)
    a3.text(-0.1, 1.15, 'B', transform=a3.transAxes,
          fontsize=16, va='top', ha='right')
    plt.plot(results['wPS_P'], label='+', color =cm.magma(.5),linewidth=2)
    plt.plot(results['wPS_N'], label='-',color =cm.viridis(.3),linewidth=2)
    plt.ylabel('S to PV weights',fontsize=16)
    plt.xlabel('time',fontsize=16)
    plt.yticks([0,6],[0,6],fontsize=16)
    plt.xticks([0,300000],[0,300000],fontsize=16)
    #plt.ylim(0,1.25)
    plt.xlim(0,310000)
    plt.legend(loc='lower right')
    a3.spines['top'].set_visible(False)
    a3.spines['right'].set_visible(False)
    
    
    plt.tight_layout()
    plt.savefig('./wPS_results_%s.png'%name, bbox_inches='tight')

