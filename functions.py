import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as stats

def phi(x_,case=1):
    x = np.array([x_])
    if case == 1:
        x[x<=0] = 0
        x[x>20] = 20
    return x[0]

def phi_square(_x):
    x = np.array([_x])
    x[x<=0] = 0
    x[x>0] = x[x>0]**2 
    x[x>20] = 20
    return x[0]

def rectify(x):
    return x * (x>0)

def gamma_phi_square(_x,gamma=5.0):
    x = np.array([_x])
    x[x<=0] = 0
    x[x>0] = gamma * (x[x>0]**2) 
    x[x>100] = 100
    return x[0]

def plot_PVratesandweights(results1,results2,sigma1,sigma2,name):
    plt.figure(figsize=(7,4))
    a5 = plt.subplot(121)
    plt.plot(results1['wPX1'], label='high', color =cm.viridis(.1),linewidth=2)
    plt.plot(results2['wPX1'], label='low',color =cm.viridis(.5),linewidth=2)
    plt.plot(np.arange(len(results1['wPX1'])),np.ones(len(results1['wPX1']))*sigma1,'--',color =cm.viridis(.1),label=r'$\sigma$ high')
    plt.plot(np.arange(len(results1['wPX1'])),np.ones(len(results1['wPX1']))*sigma2,'--',color =cm.viridis(.5),label=r'$\sigma$ low')

    plt.ylabel('PV weights',fontsize=16)
    plt.xlabel('time',fontsize=16)
    plt.yticks(np.arange(0,1.1,0.2),[0,0.2,.4,.6,.8,1.0],fontsize=16)
    plt.xticks([0,300000],[0,300000],fontsize=16)
    #plt.ylim(0,1.25)
    #plt.xlim(0,11000)
    lgd = plt.legend(loc='lower right',fontsize=11)
    a5.spines['top'].set_visible(False)
    a5.spines['right'].set_visible(False)

    a2 = plt.subplot(122)
    plt.plot(np.arange(-1,3),np.ones(4)*sigma1**2,'--',color =cm.viridis(.1),label=r'$\sigma^2$ high')
    plt.plot(np.arange(-1,3),np.ones(4)*sigma2**2,'--',color =cm.viridis(.5),label=r'$\sigma^2$ low')
    plt.bar([0,1],[results1['PV_avg'],results2['PV_avg']],yerr=[results1['PV_std'],results2['PV_std']],color=[cm.viridis(.1),cm.viridis(.5)],width=0.3)
    plt.ylabel('PV rate',fontsize=16)
    plt.xticks([0,1],['high','low'],fontsize=16)
    plt.xlabel('uncertainty',fontsize=16)
    plt.yticks(np.arange(0,1.3,0.2),[0,0.2,.4,.6,.8,1.0,1.2],fontsize=16)
    #plt.yticks([0,2],[0,2],fontsize=16)
    a2.spines['top'].set_visible(False)
    a2.spines['right'].set_visible(False)
    #plt.ylim(0,2.3)
    plt.xlim(-0.5,1.5)
    plt.tight_layout()
    plt.legend(fontsize=11)
    plt.savefig('./PVratesandweights_%s.png'%name, bbox_inches='tight',rasterized=True)
    
def plot_Rratesandweights(results1,results2,mean1,mean2,name):
    plt.figure(figsize=(7,4))
    a5 = plt.subplot(121)
    plt.plot(results1['wRX1'], label=r'$\mu=%d$'%mean1, color =cm.viridis(.1),linewidth=2)
    plt.plot(results2['wRX1'], label=r'$\mu=%d$'%mean2,color =cm.viridis(.5),linewidth=2)
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
    plt.savefig('./Rratesandweights_%s.png'%name, bbox_inches='tight',rasterized=True)