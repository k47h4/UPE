import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm
import scipy.stats as stats
from sklearn.utils import shuffle
from activation_functions import * 
from plotting_functions import * 
from functions import *


class Circuit:
	def __init__(self):
		"""
		wP,wE : float
			connection weight from R/s/SST to PV and from  to E
		w_init_PVX,PVS,R,ES : float
			initial weight from X to PV
			initial weight from X to R
			initial weight from S to PV
			initial weight from S to E
		plastic,_PX,PS,ES : boolean
			truth value indicates whether plasticity is switched on or off
		PV_mu: boolean
			 truth value determines whether PVs get input from SST_P and R
		R_neuron : boolean
			truth value indicates whether there is a representation neuron
		rectified: boolean
			truth value indicates whether activation functions are rectified
		"""
		self.uncertainty_weighted = True
		self.weighting = 1.0
		self.error_weighting = 0.1 
		self.PV_weighting = 1.0 
		self.wP = np.sqrt((2-0.1)/0.1)#np.sqrt((1.95/0.1)*(1/2)) for single PV
		self.wE = 1.0
		self.w_init_PVX = 0.01 
		self.w_init_PVS = 0.01
		self.w_init_R = 0.1 
		self.w_init_ES = 0.01 
		self.w_init_PVC1 = 0.01
		self.w_init_PVC2 = 0.01
		self.plastic = True
		self.plastic_PX = True
		self.plastic_PS = False
		self.plastic_ES = False
		self.plastic_EP = False
		self.plastic_SX = False
		self.R_neuron = False
		self.single_PV = True
		self.PV_mu = True
		self.NMDA = False
		self.exponent = 2
		self.rectified = 1
		self.Rrate = None
		self.SSTrate = None
		# time constants
		self.tau_P = 1.0
		self.tau_S = 1.0
		self.tau_E = 1.0
		self.tau_R = 1.0
		self.tau_V = 1.0
		self.dt = 1
		# nudging parameter
		self.beta_R = 0.1 
		self.beta_P = 0.1
		self.beta_S = 0.1
		self.sigma = None
		# weights
		# to PV
		self.wPX1 = np.array([self.w_init_PVX]) # small intitial weights
		self.wPY1 = np.array([self.wP]) # small intitial weights
		self.wPR = np.array([self.wP]) # small intitial weights

		self.wPX1_P = np.array([self.w_init_PVX]) # small intitial weights
		self.wPX1_N = np.array([self.w_init_PVX]) # small intitial weights

		if self.plastic_PS:
			self.wPS_P = np.array([self.w_init_PVS])
			self.wPS_N = np.array([self.w_init_PVS])
		else:
			self.wPS_P = np.array([self.wP])
			self.wPS_N = np.array([self.wP])

		# to Representation neuron
		self.wRX1 = np.array([self.w_init_R]) # small intitial weight
		self.wRY1 = np.array([1.0])
		self.wRE_P = np.array([1.0])
		self.wRE_N = np.array([1.0])

		# to SST
		self.wSR_P = np.array([1.0])
		self.wSY1_N = np.array([1.0])
		# in pPE
		self.wSY1 = np.array([1.0])
		self.wSX = np.array([0.01])

		# to E
		self.wEP_P = np.array([1.0])
		self.wEP_N = np.array([1.0])
		self.wER_N = np.array([self.wE])
		self.wEY1_P = np.array([self.wE])

		if self.plastic_ES:
			self.wES_P = np.array([self.w_init_ES])
			self.wES_N = np.array([self.w_init_ES])
		else:
			self.wES_P = np.array([self.wE])
			self.wES_N = np.array([self.wE])
			
		self.wPC1 = np.array([self.w_init_PVC1]) # small intitial weights
		self.wPC2 = np.array([self.w_init_PVC2]) # small intitial weights

		self.VIP = False
		



class Sim:
	def __init__(self, stimulus_duration = 1, number_of_samples=400000, stimulus_gap=False):
		self.stimulus_duration = stimulus_duration
		self.number_of_samples = number_of_samples
		if stimulus_gap:
			self.T = self.number_of_samples * (self.stimulus_duration +3)
		else:
			self.T = self.number_of_samples * self.stimulus_duration
		print(self.T)
		print('sim_init')
		self.sigma = 0.5
		self.mean= 5.0
		#learning rate
		self.eta_R = 0.5
		self.eta_P = 0.001
		self.eta_S = 0.1
		self.eta_ES = 0.01
		self.eta_PS = 0.0001

		# monitors
		T = self.T
		self.rP_monitor = np.empty((T))
		self.rPa_monitor = np.empty((T))
		self.rP_P_monitor = np.empty((T))
		self.rP_N_monitor = np.empty((T))
		self.rE_P_monitor = np.empty((T))
		self.rS_P_monitor = np.empty((T))
		self.rE_N_monitor = np.empty((T))
		self.rS_N_monitor = np.empty((T))
		self.rR_monitor = np.empty((T))
		self.rRa_monitor = np.empty((T))

		self.wPX1_monitor = np.empty((T))
		self.wSX_monitor = np.empty((T))
		self.wPX1_P_monitor = np.empty((T))
		self.wPX1_N_monitor = np.empty((T))
		self.wRX1_monitor = np.empty((T))
		self.wPS_P_monitor = np.empty((T))
		self.wES_P_monitor = np.empty((T))
		self.wPS_N_monitor = np.empty((T))
		self.wES_N_monitor = np.empty((T))

		self.PV_in = np.empty(T)
		self.EP_in = np.empty(T)
		self.EN_in = np.empty(T)
		self.PV_in_a = np.empty(T)

		self.sigma_mon = np.empty(T)



	def get_stimuli(self,mean,sigma, stimulus_gap=False):
		# whisker
		if stimulus_gap:
			self.T = self.number_of_samples * (self.stimulus_duration + 3)
		print(self.T)
		rY1 = np.zeros(self.T)
		Y1 = sigma*np.random.randn(self.number_of_samples)+mean
		for i in range(self.stimulus_duration):
			rY1[i::self.stimulus_duration+stimulus_gap*3] = Y1
		if stimulus_gap: 
			rY1[:-3] = rY1[3:]
			rY1[-3:] = np.zeros((3))
		print(rY1)
		return rY1


	def monitor_weights(self,circuit,t):
		self.wPX1_monitor[t] = circuit.wPX1
		if circuit.single_PV == False:
			self.wPX1_P_monitor[t] = circuit.wPX1_P
			self.wPX1_N_monitor[t] = circuit.wPX1_N
		self.wSX_monitor[t] = circuit.wSX

		self.wRX1_monitor[t] = circuit.wRX1
		self.wPS_P_monitor[t] = circuit.wPS_P
		self.wPS_N_monitor[t] = circuit.wPS_N
		self.wES_P_monitor[t] = circuit.wES_P
		self.wES_N_monitor[t] = circuit.wES_N


	def run(self, circuit, mean=5.0, sigma=0.5, seed=None):
		print('mean'+str(mean))
		print('sigma'+str(sigma))
		print('seed'+str(seed))
		
		if seed is not None:
			np.random.seed(seed)

		#sim parameters
		self.sigma = sigma
		self.mean = mean

		self.T = self.number_of_samples * self.stimulus_duration 
		T = self.T

		#self.T = self.number_of_samples * self.stimulus_duration # number of time steps
		# inputs
		# sound
		rX1 = np.ones((T))
		
		# get whisker stimuli
		rY1 = self.get_stimuli(mean,sigma)

		if circuit.Rrate is None:
			rRrate = np.ones(T)*self.mean
		else:
			rRrate = np.ones(T)*circuit.Rrate


		#initial rates
		R = 0.0
		rR=0.0
		rP = 0.0
		rP_P = 0.0
		rP_N = 0.0
		rS_P = 0.0
		rS_N = 0.0
		rE_P = 0.0
		rE_N = 0.0


		for t in range(T):


			# store monitors
			self.rP_monitor[t] = rP
			if not circuit.single_PV:
				self.rP_P_monitor[t] = rP_P
				self.rP_N_monitor[t] = rP_N

			self.sigma_mon[t] = self.sigma

			self.rS_P_monitor[t] = rS_P
			self.rS_N_monitor[t] = rS_N
			self.rE_P_monitor[t] = rE_P
			self.rE_N_monitor[t] = rE_N
			self.rR_monitor[t] = R
			self.rRa_monitor[t] = phi(circuit.wRX1 * rX1[t])



			self.monitor_weights(circuit,t)


			if circuit.R_neuron:
				rR = R
			else:
				rR=rRrate[t]

			drS_P = (-rS_P + phi(circuit.wSR_P * rR,case=circuit.rectified))/circuit.tau_S
			drS_N = (-rS_N + phi(circuit.wSY1_N * rY1[t],case=circuit.rectified))/circuit.tau_S

			if circuit.single_PV:
				drP = (-rP + phi_square(((1-circuit.beta_P)*(circuit.wPX1 * rX1[t]) + circuit.beta_P*(circuit.wPY1 * rY1[t] - circuit.PV_mu*(circuit.wPS_P * rS_P) + circuit.PV_mu*(circuit.wPR * rR) - circuit.wPS_N * rS_N))))/circuit.tau_P
			else:
				drP_P = (-rP_P + phi_square((1-circuit.beta_P)*(circuit.wPX1_P * rX1[t]) + circuit.beta_P*(circuit.wPY1 * rY1[t] - circuit.wPS_P * rS_P)))/circuit.tau_P
				#drP_P = (-rP_P + phi_square((1-circuit.beta_P)*(circuit.wPX1_P * rX1[t]) + circuit.beta_P*(circuit.wPY1 * rY1[t] - circuit.wPS_P * rR)))/circuit.tau_P

				drP_N = (-rP_N + phi_square((1-circuit.beta_P)*(circuit.wPX1_N * rX1[t]) + circuit.beta_P*(circuit.wPR * rR - circuit.wPS_N * rS_N)))/circuit.tau_P

			self.PV_in_a[t] = (1-circuit.beta_P)*(circuit.wPX1 * rX1[t])
			self.PV_in[t] = (circuit.wPY1 * rY1[t] - circuit.PV_mu*(circuit.wPS_P * rS_P) + circuit.PV_mu*(circuit.wPR * rR) - circuit.wPS_N * rS_N)
			if circuit.single_PV:
				drE_P = (-rE_P + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_P * rP))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * (circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P),case=circuit.rectified))/circuit.tau_E                    
				drE_N = (-rE_N + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_N * rP))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * (circuit.wER_N * rR - circuit.wES_N * rS_N),case=circuit.rectified))/circuit.tau_E
			else:
				if circuit.NMDA:
					drE_P = (-rE_P + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_P * rP_P))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * rectify(circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P)**circuit.exponent,case=circuit.rectified))/circuit.tau_E                    
					drE_N = (-rE_N + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_N * rP_N))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * rectify(circuit.wER_N * rR - circuit.wES_N * rS_N)**circuit.exponent,case=circuit.rectified))/circuit.tau_E
				else:
					drE_P = (-rE_P + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_P * rP_P))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * (circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P),case=circuit.rectified))/circuit.tau_E                    
					drE_N = (-rE_N + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_N * rP_N))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * (circuit.wER_N * rR - circuit.wES_N * rS_N),case=circuit.rectified))/circuit.tau_E
			if circuit.VIP:
				K = (1.0/(1.0+rP_P))
				drV = (-K*rV + phi(I,case=circuit.rectified))/circuit.tau_V 


			self.EN_in[t] = circuit.wER_N * rR - circuit.wES_N * rS_N
			self.EP_in[t] = circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P

			drR = (-R + phi((circuit.wRX1 * rX1[t]) + circuit.error_weighting*(circuit.wRE_P * rE_P) - circuit.error_weighting*(circuit.wRE_N * rE_N),case=circuit.rectified))/circuit.tau_R # no convex combination, input from error neuron

				
				
				



			# weight changes
			#v = (1/circuit.beta_P)*(np.sqrt(rP)-((1-circuit.beta_P)/2)*circuit.wPX1*rX1[t])
			#dwPX1 = circuit.plastic_PX * self.eta_P * ((np.sqrt(17)*v - (circuit.wPX1*rX1[t])) * rX1[t])
			dwPX1 = circuit.plastic_PX * self.eta_P * ((rP - phi_square(circuit.wPX1*rX1[t])) * rX1[t])
			if not circuit.single_PV:
				dwPX1_P = circuit.plastic_PX * self.eta_P * ((rP_P - phi_square(circuit.wPX1_P*rX1[t])) * rX1[t])
				dwPX1_N = circuit.plastic_PX * self.eta_P * ((rP_N - phi_square(circuit.wPX1_N*rX1[t])) * rX1[t])

			dwRX1 = circuit.plastic * self.eta_R * ((R - (circuit.wRX1*rX1[t])) * rX1[t])
			dwPS_P = circuit.plastic_PS * self.eta_PS * (circuit.wPY1 * rY1[t] - circuit.wPS_P * rS_P)*rS_P
			dwPS_N = circuit.plastic_PS * self.eta_PS * (circuit.wPR * rR - circuit.wPS_N * rS_N)*rS_N

			dwES_P = circuit.plastic_ES * self.eta_ES * ((circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P))*rS_P
			dwES_N = circuit.plastic_ES * self.eta_ES * ((circuit.wER_N * rR - circuit.wES_N * rS_N))*rS_N
			
			
			# rate changes
			R += circuit.dt*drR
			rS_P += circuit.dt*drS_P
			rS_N += circuit.dt*drS_N
			if circuit.single_PV:
				rP += circuit.dt*drP
			else:
				rP_P += circuit.dt*drP_P
				rP_N += circuit.dt*drP_N

			rE_P += circuit.dt*drE_P
			rE_N += circuit.dt*drE_N
			
			
			#wRE_P += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_P
			#wRE_N += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_N


			circuit.wPX1 += circuit.dt*dwPX1
			if not circuit.single_PV:
				circuit.wPX1_P += circuit.dt*dwPX1_P
				circuit.wPX1_N += circuit.dt*dwPX1_N

			circuit.wRX1 += circuit.dt*dwRX1
			if circuit.plastic_PS:
				circuit.wPS_P += circuit.dt*dwPS_P
				circuit.wPS_N += circuit.dt*dwPS_N
			if circuit.plastic_ES:
				circuit.wES_P += circuit.dt*dwES_P
				circuit.wES_N += circuit.dt*dwES_N


		results = {
		'PV_avg' : np.mean(self.rP_monitor[100:]),
		'PV_P_avg' : np.mean(self.rP_P_monitor[100000:]),
		'PV_N_avg' : np.mean(self.rP_N_monitor[100000:]),
		'SST_P_avg': np.mean(self.rS_P_monitor[100:]),
		'E_P_avg' : np.mean(self.rE_P_monitor[100:]),
		'SST_N_avg': np.mean(self.rS_N_monitor[100:]),
		'E_N_avg' : np.mean(self.rE_N_monitor[100:]),
		'R_avg' : np.mean(self.rR_monitor[100000:]),
		'PV_std' : np.std(self.rP_monitor[100:]),
		'PV_P_std' : np.std(self.rP_P_monitor[100000:]),	
		'PV_N_std' : np.std(self.rP_N_monitor[100000:]),	
		'SST_P_std': np.std(self.rS_P_monitor[100:]),
		'E_P_std' : np.std(self.rE_P_monitor[100:]),
		'SST_N_std': np.std(self.rS_N_monitor[100:]),
		'E_N_std' : np.std(self.rE_N_monitor[100:]),
		'R_std' : np.std(self.rR_monitor[100000:]),
		'wPX1' : self.wPX1_monitor,
		'wPX1_P' : self.wPX1_P_monitor,
		'wPX1_N' : self.wPX1_N_monitor,

		'wRX1' : self.wRX1_monitor, 
		'wPS_P' : self.wPS_P_monitor, 
		'wPS_N' : self.wPS_N_monitor, 
		'wES_P' : self.wES_P_monitor, 
		'wES_N' : self.wES_N_monitor, 
		#'wRE' : wRE_monitor, 
		'rE_P' : self.rE_P_monitor, 
		'rS_P' : self.rS_P_monitor, 
		'rE_N' : self.rE_N_monitor, 
		'rS_N' : self.rS_N_monitor, 
		'rP' : self.rP_monitor, 
		'rP_P' : self.rP_P_monitor, 
		'rP_N' : self.rP_N_monitor, 
		'rX' : rX1,
		'rY' : rY1,

		'rR' : self.rR_monitor, 
		'rRa' : self.rRa_monitor, 
		'rY1' : rY1,
		'PV_in':self.PV_in,
		'PV_in_a':self.PV_in_a,

		'EP_in':self.EP_in,
		'EN_in':self.EN_in,
		'sigma':self.sigma_mon

		}

		return results


	def run_pPE(self, circuit, mean=5.0, sigma=0.5, start=100000,seed=None):
		print('mean'+str(mean))
		print('sigma'+str(sigma))
		print('seed'+str(seed))
		
		if seed is not None:
			np.random.seed(seed)

		#sim parameters
		self.sigma = sigma
		self.mean = mean

		self.T = self.number_of_samples * self.stimulus_duration 
		T = self.T
		#self.T = self.number_of_samples * self.stimulus_duration # number of time steps
		# inputs
		# sound
		rX1 = np.ones((T))
		
		# get whisker stimuli
		rY1 = self.get_stimuli(mean,sigma)

		#initial rates
		rP = 0.0
		rS_P = 0.0
		rE_P = 0.0


		wa_mu_mon = np.empty((T))
		wa_sigma_mon = np.empty((T))
		ws_sigma_mon = np.empty((T))
		diff_mon = np.empty((T))
		beta_dyn_mon = np.empty((T))
		sigma_e_mon = np.empty((T))

		ratio = np.empty((T))
		phase1 = np.empty((T))
		n = 1
		wa_mu = 0.001
		wa_sigma = 0.05
		ws_sigma = 0.001
		sigma_e = .1
		beta_dyn = 0.1

		for t in range(T):
			if circuit.plastic_SX:
				drS_P = (-rS_P + phi((1-circuit.beta_S)*(circuit.wSX * rX1[t]) + circuit.beta_S*(circuit.wSY1 * rY1[t]),case=circuit.rectified))/circuit.tau_S
			else:
				drS_P = (-rS_P + phi(circuit.wSR_P * mean,case=circuit.rectified))/circuit.tau_S
			drP = (-rP + phi_square(((1-circuit.beta_P)*(circuit.wPX1 * rX1[t]) + circuit.beta_P*(circuit.wPY1 * rY1[t] - circuit.wPS_P * rS_P))))/circuit.tau_P
			drE_P = (-rE_P + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_P * rP))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * (circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P),case=circuit.rectified))/circuit.tau_E                    

				
			# store monitors
			self.rP_monitor[t] = rP
			self.rPa_monitor[t] = phi_square(circuit.wPX1 * rX1[t])
			self.sigma_mon[t] = self.sigma

			self.rS_P_monitor[t] = rS_P
			self.rE_P_monitor[t] = rE_P

			wa_mu_mon[t] = wa_mu
			wa_sigma_mon[t] = wa_sigma
			ws_sigma_mon[t] = ws_sigma
			self.monitor_weights(circuit,t)


			# weight changes
			dwPX1 = circuit.plastic_PX * self.eta_P * ((rP - phi_square(circuit.wPX1*rX1[t])) * rX1[t])
			dwPS_P = circuit.plastic_PS * self.eta_PS * (circuit.wPY1 * rY1[t] - circuit.wPS_P * rS_P)*rS_P
			dwES_P = circuit.plastic_ES * self.eta_ES * ((circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P))*rS_P
			dwSX = circuit.plastic_SX * self.eta_S * ((rS_P - (circuit.wSX*rX1[t])) * rX1[t])

			# estimate variability of w_a a
			wa_mu += (1/n) * (circuit.wPX1 - wa_mu)
			wa_sigma += (1/n) * ((circuit.wPX1 - wa_mu)**2 - wa_sigma)

			ws_sigma += (1/n) * ((rY1[t] - rS_P)**2 - ws_sigma)
			sigma_e_mon[t] = sigma_e

			if ((1/n) * (circuit.wPX1 - wa_mu)) > 1e-6:
				beta_dyn = (.1/(.1+wa_mu))
				phase1[t]=1 
			else:
				phase1[t]=0
				#print(((1/n) * (circuit.wPX1 - wa_mu)))
				beta_dyn = (sigma_e/(sigma_e+wa_mu))
				if t>10:
					sigma_e = (1-beta_dyn)*sigma_e  

			ratio[t] = (1/wa_mu) 
			#beta_dyn= (1/wa_mu**2)/((1/wa_mu**2) + (1/wa_sigma**2))
			#ratio[t] = (1/wa_mu**2)/((1/wa_mu**2) + (1/(wa_sigma**2+1e-10)))
			#beta_dyn+= (-beta_dyn + ((1/wa_mu**2)/((1/wa_mu**2) + (1/(wa_sigma**2+1e-3)))))/10.0

			beta_dyn_mon[t] = beta_dyn
			diff_mon[t] = (rY1[t] - rS_P)
			n += 1




			# rate changes
			rS_P += circuit.dt*drS_P
			rP += circuit.dt*drP
			rE_P += circuit.dt*drE_P

			circuit.wPX1 += circuit.dt*dwPX1
			if circuit.plastic_SX:
				circuit.wSX += circuit.dt*dwSX
			if circuit.plastic_PS:
				circuit.wPS_P += circuit.dt*dwPS_P
			if circuit.plastic_ES:
				circuit.wES_P += circuit.dt*dwES_P



		results = {

		'PV_avg' : np.mean(self.rP_monitor[start:]),
		'SST_P_avg': np.mean(self.rS_P_monitor[start:]),
		'E_P_avg' : np.mean(self.rE_P_monitor[start:]),
		'PV_std' : np.std(self.rP_monitor[start:]),
		'PV_P_std' : np.std(self.rP_P_monitor[start:]),	
		'SST_P_std': np.std(self.rS_P_monitor[start:]),
		'E_P_std' : np.std(self.rE_P_monitor[start:]),
		'wPX1' : self.wPX1_monitor,
		'wPX1_P' : self.wPX1_P_monitor,
		'wSX' : self.wSX_monitor,
		'wPS_P' : self.wPS_P_monitor, 
		'wES_P' : self.wES_P_monitor, 
		'rE_P' : self.rE_P_monitor, 
		'rS_P' : self.rS_P_monitor, 
		'rP' : self.rP_monitor, 
		'rPa' : self.rPa_monitor, 
		'rY1' : rY1,
		'sigma':self.sigma_mon,
		'wa_mu':wa_mu_mon,
		'wa_sigma':wa_sigma_mon,
		'ws_sigma':ws_sigma_mon,
		'beta_dyn':beta_dyn_mon,
		'ratio' : ratio,
		'phase1' : phase1,
		'sigma_e': sigma_e_mon
		}

		return results


	def run_nPE(self, circuit, mean=5.0, sigma=0.5, start=100000,seed=None):
		print('mean'+str(mean))
		print('sigma'+str(sigma))
		print('seed'+str(seed))
		
		if seed is not None:
			np.random.seed(seed)

		#sim parameters
		self.sigma = sigma
		self.mean = mean

		self.T = self.number_of_samples * self.stimulus_duration 
		T = self.T
		#self.T = self.number_of_samples * self.stimulus_duration # number of time steps
		# inputs
		# sound
		rX1 = np.ones((T))
		
		# get whisker stimuli
		rY1 = self.get_stimuli(mean,sigma)

		#initial rates
		rP = 0.0
		rS_N = 0.0
		rE_N = 0.0

		if circuit.Rrate is None:
			rR = np.ones(T)*mean
		else:
			rR = np.ones(T)*circuit.Rrate


		for t in range(T):

			drS_N = (-rS_N + phi(circuit.wSY1 * rY1[t],case=circuit.rectified))/circuit.tau_S
			drP = (-rP + phi_square(((1-circuit.beta_P)*(circuit.wPX1 * rX1[t]) + circuit.beta_P*(circuit.wPR * rR[t] - circuit.wPS_P * rS_N))))/circuit.tau_P
			drE_N = (-rE_N + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_N * rP))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * (circuit.wER_N * rR[t] - circuit.wES_N * rS_N),case=circuit.rectified))/circuit.tau_E                    

				
			# store monitors
			self.rP_monitor[t] = rP
			self.rPa_monitor[t] = phi_square(circuit.wPX1 * rX1[t])
			self.sigma_mon[t] = self.sigma

			self.rS_N_monitor[t] = rS_N
			self.rE_N_monitor[t] = rE_N
			self.monitor_weights(circuit,t)


			# weight changes
			dwPX1 = circuit.plastic_PX * self.eta_P * ((rP - phi_square(circuit.wPX1*rX1[t])) * rX1[t])
			dwPS_N = circuit.plastic_PS * self.eta_PS * (circuit.wPR * rR[t] - circuit.wPS_N * rS_N)*rS_N
			#dwRX1 = circuit.plastic_RX * self.eta_R * ((rR[t] - (circuit.wRX1*rX1[t])) * rX1[t])

			
			# rate changes
			rS_N += circuit.dt*drS_N
			rP += circuit.dt*drP
			rE_N += circuit.dt*drE_N

			circuit.wPX1 += circuit.dt*dwPX1
			#if circuit.plastic_RX:
			#	circuit.wRX1 += circuit.dt*dwRX1
			if circuit.plastic_PS:
				circuit.wPS_N += circuit.dt*dwPS_N
			if circuit.plastic_ES:
				circuit.wES_N += circuit.dt*dwES_N



		results = {

		'PV_avg' : np.mean(self.rP_monitor[start:]),
		'SST_N_avg': np.mean(self.rS_N_monitor[start:]),
		'E_N_avg' : np.mean(self.rE_N_monitor[start:]),
		'PV_std' : np.std(self.rP_monitor[start:]),
		'PV_N_std' : np.std(self.rP_N_monitor[start:]),	
		'SST_N_std': np.std(self.rS_N_monitor[start:]),
		'E_N_std' : np.std(self.rE_N_monitor[start:]),
		'wPX1' : self.wPX1_monitor,
		'wPX1_N' : self.wPX1_N_monitor,
		'wSX' : self.wSX_monitor,
		'wPS_N' : self.wPS_N_monitor, 
		'wES_N' : self.wES_N_monitor, 
		'rE_N' : self.rE_N_monitor, 
		'rS_N' : self.rS_N_monitor, 
		'rP' : self.rP_monitor, 
		'rPa' : self.rPa_monitor, 
		'rY1' : rY1,
		'sigma':self.sigma_mon

		}

		return results

	

	def run_fakePV(self, circuit, mean=5.0, sigma=0.5, stimulus_gap = False, seed=None):
		print('mean'+str(mean))
		print('sigma'+str(sigma))
		print('seed'+str(seed))
		
		if seed is not None:
			np.random.seed(seed)

		#sim parameters
		self.sigma = sigma
		self.mean = mean

		if stimulus_gap:
			self.T = self.number_of_samples * (self.stimulus_duration +3)
		else:
			self.T = self.number_of_samples * self.stimulus_duration

		print('T')
		print(self.T)

		T = self.T
		#self.T = self.number_of_samples * self.stimulus_duration # number of time steps
		# inputs
		# sound
		rX1 = np.ones((T))
		
		# get whisker stimuli
		rY1 = self.get_stimuli(mean,sigma, stimulus_gap)

		if circuit.Rrate is None:
			rRrate = np.ones(T)*self.mean
		else:
			rRrate = np.ones(T)*circuit.Rrate


		#initial rates
		R = 0.0
		rR=0.0
		rP = self.sigma**2
		rP_P = 0.0
		rP_N = 0.0
		rS_P = 0.0
		rS_N = 0.0
		rE_P = 0.0
		rE_N = 0.0


		for t in range(T):

			if circuit.R_neuron:
				rR = R
			else:
				rR=rRrate[t]

			drS_P = (-rS_P + phi(circuit.wSR_P * rR,case=circuit.rectified))/circuit.tau_S
			drS_N = (-rS_N + phi(circuit.wSY1_N * rY1[t],case=circuit.rectified))/circuit.tau_S

			drE_P = (-rE_P + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_P * rP))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * (circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P),case=circuit.rectified))/circuit.tau_E                    
			drE_N = (-rE_N + phi((circuit.uncertainty_weighted*(1.0/(circuit.PV_weighting + (circuit.wEP_N * rP))) + ((1-circuit.uncertainty_weighted)*circuit.weighting)) * (circuit.wER_N * rR - circuit.wES_N * rS_N),case=circuit.rectified))/circuit.tau_E


			self.EN_in[t] = circuit.wER_N * rR - circuit.wES_N * rS_N
			self.EP_in[t] = circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P

			drR = (-R + phi((circuit.wRX1 * rX1[t]) + circuit.error_weighting*(circuit.wRE_P * rE_P) - circuit.error_weighting*(circuit.wRE_N * rE_N),case=circuit.rectified))/circuit.tau_R # no convex combination, input from error neuron


				
				
				
			# store monitors
			self.rP_monitor[t] = rP

			self.sigma_mon[t] = self.sigma

			self.rS_P_monitor[t] = rS_P
			self.rS_N_monitor[t] = rS_N
			self.rE_P_monitor[t] = rE_P
			self.rE_N_monitor[t] = rE_N
			self.rR_monitor[t] = R
			self.rRa_monitor[t] = phi(circuit.wRX1 * rX1[t])


			self.monitor_weights(circuit,t)



			# weight changes

			dwRX1 = circuit.plastic * self.eta_R * ((R - (circuit.wRX1*rX1[t])) * rX1[t])
			dwPS_P = circuit.plastic_PS * self.eta_PS * (circuit.wPY1 * rY1[t] - circuit.wPS_P * rS_P)*rS_P
			dwPS_N = circuit.plastic_PS * self.eta_PS * (circuit.wPR * rR - circuit.wPS_N * rS_N)*rS_N

			dwES_P = circuit.plastic_ES * self.eta_ES * ((circuit.wEY1_P * rY1[t] - circuit.wES_P * rS_P))*rS_P
			dwES_N = circuit.plastic_ES * self.eta_ES * ((circuit.wER_N * rR - circuit.wES_N * rS_N))*rS_N
			
			
			# rate changes
			R += circuit.dt*drR
			rS_P += circuit.dt*drS_P
			rS_N += circuit.dt*drS_N
			rE_P += circuit.dt*drE_P
			rE_N += circuit.dt*drE_N
			
			

			circuit.wRX1 += circuit.dt*dwRX1
			if circuit.plastic_PS:
				circuit.wPS_P += circuit.dt*dwPS_P
				circuit.wPS_N += circuit.dt*dwPS_N
			if circuit.plastic_ES:
				circuit.wES_P += circuit.dt*dwES_P
				circuit.wES_N += circuit.dt*dwES_N



		results = {
		'PV_avg' : np.mean(self.rP_monitor[100:]),
		'PV_P_avg' : np.mean(self.rP_P_monitor[100000:]),
		'PV_N_avg' : np.mean(self.rP_N_monitor[100000:]),
		'SST_P_avg': np.mean(self.rS_P_monitor[100:]),
		'E_P_avg' : np.mean(self.rE_P_monitor[100:]),
		'SST_N_avg': np.mean(self.rS_N_monitor[100:]),
		'E_N_avg' : np.mean(self.rE_N_monitor[100:]),
		'R_avg' : np.mean(self.rR_monitor[100000:]),
		'PV_std' : np.std(self.rP_monitor[100:]),
		'PV_P_std' : np.std(self.rP_P_monitor[100000:]),	
		'PV_N_std' : np.std(self.rP_N_monitor[100000:]),	
		'SST_P_std': np.std(self.rS_P_monitor[100:]),
		'E_P_std' : np.std(self.rE_P_monitor[100:]),
		'SST_N_std': np.std(self.rS_N_monitor[100:]),
		'E_N_std' : np.std(self.rE_N_monitor[100:]),
		'R_std' : np.std(self.rR_monitor[100000:]),
		'wPX1' : self.wPX1_monitor,
		'wPX1_P' : self.wPX1_P_monitor,
		'wPX1_N' : self.wPX1_N_monitor,

		'wRX1' : self.wRX1_monitor, 
		'wPS_P' : self.wPS_P_monitor, 
		'wPS_N' : self.wPS_N_monitor, 
		'wES_P' : self.wES_P_monitor, 
		'wES_N' : self.wES_N_monitor, 
		#'wRE' : wRE_monitor, 
		'rE_P' : self.rE_P_monitor, 
		'rS_P' : self.rS_P_monitor, 
		'rE_N' : self.rE_N_monitor, 
		'rS_N' : self.rS_N_monitor, 
		'rP' : self.rP_monitor, 
		'rP_P' : self.rP_N_monitor, 
		'rP_N' : self.rP_N_monitor, 

		'rR' : self.rR_monitor, 
		'rRa' : self.rRa_monitor, 
		'rY1' : rY1,
		'PV_in':self.PV_in,
		'PV_in_a':self.PV_in_a,

		'EP_in':self.EP_in,
		'EN_in':self.EN_in,
		'sigma':self.sigma_mon

		}

		return results


	


	def run_oldfakePV(self, mean=5.0, sigma=0.5, stimulus_duration=1, seed=None):
		print('mean'+str(mean))
		print('sigma'+str(sigma))
		print('seed'+str(seed))
		print('wPX1:'+str(self.wPX1))
		
		
		if seed is not None:
			np.random.seed(seed)

		#sim parameters

		number_of_samples = 400000

		T = number_of_samples * stimulus_duration # number of time steps
		# inputs
		# sound
		rX1 = np.ones((T))
		rY1 = np.zeros(T)


		# whisker
		Y1_mean = mean
		Y1_sigma = sigma
		self.sigma = sigma

		Y1 = Y1_sigma*np.random.randn(number_of_samples)+Y1_mean

		for i in range(stimulus_duration):
			rY1[i::stimulus_duration] = Y1


		#rS_P= np.ones(T)*Y1_mean
		# test
		if self.Rrate is None:
			rRrate = np.ones(T)*Y1_mean
		else:
			rRrate = np.ones(T)*self.Rrate

		self.PV_rate = sigma**2

		#learning rate
		eta_R = 0.1
		eta_P = 0.001
		eta_ES = 0.01
		eta_PS = 0.0001

		#initial rates
		R = 0.0
		rR=0.0
		rP = self.PV_rate
		rP_P = 0.0
		rP_N = 0.0
		rS_P = 0.0
		rS_N = 0.0
		rE_P = 0.0
		rE_N = 0.0


		# monitors
		rP_monitor = np.empty((T))
		rP_P_monitor = np.empty((T))
		rP_N_monitor = np.empty((T))
		rE_P_monitor = np.empty((T))
		rS_P_monitor = np.empty((T))
		rE_N_monitor = np.empty((T))
		rS_N_monitor = np.empty((T))
		rR_monitor = np.empty((T))
		rRa_monitor = np.empty((T))

		wPX1_monitor = np.empty((T))
		wPX1_P_monitor = np.empty((T))
		wPX1_N_monitor = np.empty((T))
		wRX1_monitor = np.empty((T))
		wPS_P_monitor = np.empty((T))
		wES_P_monitor = np.empty((T))
		wPS_N_monitor = np.empty((T))
		wES_N_monitor = np.empty((T))

		PV_in = np.empty(T)
		EP_in = np.empty(T)
		EN_in = np.empty(T)
		PV_in_a = np.empty(T)

		sigma_mon = np.empty(T)


		for t in range(T):

			if self.R_neuron:
				rR = R
			else:
				rR=rRrate[t]

			drS_P = (-rS_P + phi(self.wSR_P * R,case=self.rectified))/self.tau_S
			drS_N = (-rS_N + phi(self.wSY1_N * rY1[t],case=self.rectified))/self.tau_S


			drE_P = (-rE_P + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_P * rP))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wEY1_P * rY1[t] - self.wES_P * rS_P),case=self.rectified))/self.tau_E                    
			drE_N = (-rE_N + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_N * rP))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wER_N * R - self.wES_N * rS_N),case=self.rectified))/self.tau_E


			EN_in[t] = self.wER_N * rR - self.wES_N * rS_N
			EP_in[t] = self.wEY1_P * rY1[t] - self.wES_P * rS_P
			#drR = (-R + phi((1-self.beta_R)*(self.wRX1 * rX1[t]) + self.beta_R*((self.wRE_P * rE_P) - (self.wRE_N * rE_N)),case=self.rectified))/self.tau_R # convex combination, input from error neuron
			drR = (-R + phi((self.wRX1 * rX1[t]) + ((self.wRE_P * rE_P) - (self.wRE_N * rE_N)),case=self.rectified))/self.tau_R # no convex combination, input from error neuron

			#drR = (-R + phi((self.wRX1 * rX1[t]) + self.error_weighting*(self.wRE_P * rE_P) - self.error_weighting*(self.wRE_N * rE_N),case=self.rectified))/self.tau_R # no convex combination, input from error neuron

				
				
				
			# store monitors
			rP_monitor[t] = rP
			if not self.single_PV:
				rP_P_monitor[t] = rP_P
				rP_N_monitor[t] = rP_N

			sigma_mon[t] = self.sigma

			rS_P_monitor[t] = rS_P
			rS_N_monitor[t] = rS_N
			rE_P_monitor[t] = rE_P
			rE_N_monitor[t] = rE_N
			rR_monitor[t] = R
			rRa_monitor[t] = phi(self.wRX1 * rX1[t])

			wPX1_monitor[t] = self.wPX1
			if self.single_PV == False:
				wPX1_P_monitor[t] = self.wPX1_P
				wPX1_N_monitor[t] = self.wPX1_N

			wRX1_monitor[t] = self.wRX1
			wPS_P_monitor[t] = self.wPS_P
			wPS_N_monitor[t] = self.wPS_N
			wES_P_monitor[t] = self.wES_P
			wES_N_monitor[t] = self.wES_N



			
			
			# weight changes
			#v = (1/self.beta_P)*(np.sqrt(rP)-((1-self.beta_P)/2)*self.wPX1*rX1[t])
			#dwPX1 = self.plastic_PX * eta_P * ((np.sqrt(17)*v - (self.wPX1*rX1[t])) * rX1[t])
			dwPX1 = self.plastic_PX * eta_P * ((rP - phi_square(self.wPX1*rX1[t])) * rX1[t])
			if not self.single_PV:
				dwPX1_P = self.plastic_PX * eta_P * ((rP_P - phi_square(self.wPX1_P*rX1[t])) * rX1[t])
				dwPX1_N = self.plastic_PX * eta_P * ((rP_N - phi_square(self.wPX1_N*rX1[t])) * rX1[t])

			#dwRX1 = self.plastic * eta_R * ((R - (self.wRX1*rX1[t])) * rX1[t])
			dwRX1 = self.plastic * eta_R * ((rE_P - rE_N) * rX1[t])
			dwPS_P = self.plastic_PS * eta_PS * (self.wPY1 * rY1[t] - self.wPS_P * rS_P)*rS_P
			dwPS_N = self.plastic_PS * eta_PS * (self.wPR * rR - self.wPS_N * rS_N)*rS_N

			dwES_P = self.plastic_ES * eta_ES * ((self.wEY1_P * rY1[t] - self.wES_P * rS_P))*rS_P
			dwES_N = self.plastic_ES * eta_ES * ((self.wER_N * rR - self.wES_N * rS_N))*rS_N
			
			
			# rate changes
			R += self.dt*drR
			rS_P += self.dt*drS_P
			rS_N += self.dt*drS_N

			rE_P += self.dt*drE_P
			rE_N += self.dt*drE_N
			
			
			#wRE_P += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_P
			#wRE_N += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_N

			self.wPX1 += self.dt*dwPX1
			if not self.single_PV:
				self.wPX1_P += self.dt*dwPX1_P
				self.wPX1_N += self.dt*dwPX1_N

			self.wRX1 += self.dt*dwRX1
			if self.plastic_PS:
				self.wPS_P += self.dt*dwPS_P
				self.wPS_N += self.dt*dwPS_N
			if self.plastic_ES:
				self.wES_P += self.dt*dwES_P
				self.wES_N += self.dt*dwES_N



		results = {
		'PV_avg' : np.mean(rP_monitor[100:]),
		'PV_P_avg' : np.mean(rP_P_monitor[100000:]),
		'PV_N_avg' : np.mean(rP_N_monitor[100000:]),
		'SST_P_avg': np.mean(rS_P_monitor[100:]),
		'E_P_avg' : np.mean(rE_P_monitor[100:]),
		'SST_N_avg': np.mean(rS_N_monitor[100:]),
		'E_N_avg' : np.mean(rE_N_monitor[100:]),
		'R_avg' : np.mean(rR_monitor[100:]),
		'PV_std' : np.std(rP_monitor[100:]),
		'PV_P_std' : np.std(rP_P_monitor[100000:]),	
		'PV_N_std' : np.std(rP_N_monitor[100000:]),	
		'SST_P_std': np.std(rS_P_monitor[100:]),
		'E_P_std' : np.std(rE_P_monitor[100:]),
		'SST_N_std': np.std(rS_N_monitor[100:]),
		'E_N_std' : np.std(rE_N_monitor[100:]),
		'R_std' : np.std(rR_monitor[100:]),
		'wPX1' : wPX1_monitor,
		'wPX1_P' : wPX1_P_monitor,
		'wPX1_N' : wPX1_N_monitor,

		'wRX1' : wRX1_monitor, 
		'wPS_P' : wPS_P_monitor, 
		'wPS_N' : wPS_N_monitor, 
		'wES_P' : wES_P_monitor, 
		'wES_N' : wES_N_monitor, 
		#'wRE' : wRE_monitor, 
		'rE_P' : rE_P_monitor, 
		'rS_P' : rS_P_monitor, 
		'rE_N' : rE_N_monitor, 
		'rS_N' : rS_N_monitor, 
		'rP' : rP_monitor, 
		'rP_P' : rP_N_monitor, 
		'rP_N' : rP_N_monitor, 

		'rR' : rR_monitor, 
		'rRa' : rRa_monitor, 
		'rY1' : rY1,
		'PV_in':PV_in,
		'PV_in_a':PV_in_a,

		'EP_in':EP_in,
		'EN_in':EN_in,
		'sigma':sigma_mon

		}

		return results


	def run_Kalman(self, mean=5.0, sigma=0.5, stimulus_duration=1, seed=None):
		print('mean'+str(mean))
		print('sigma'+str(sigma))
		print('seed'+str(seed))
		print('wPX1:'+str(self.wPX1))
		
		
		if seed is not None:
			np.random.seed(seed)

		#sim parameters

		number_of_samples = 100000

		T = number_of_samples * stimulus_duration # number of time steps
		# inputs
		# sound
		rX1 = np.ones((T))
		rY1 = np.zeros(T)


		# whisker
		Y1_mean = mean
		Y1_sigma = sigma
		self.sigma = sigma

		Y1 = Y1_sigma*np.random.randn(number_of_samples)+Y1_mean

		for i in range(stimulus_duration):
			rY1[i::stimulus_duration] = Y1


		#rS_P= np.ones(T)*Y1_mean
		# test
		if self.Rrate is None:
			rRrate = np.ones(T)*Y1_mean
		else:
			rRrate = np.ones(T)*self.Rrate

		#learning rate
		eta_R = 0.1
		eta_P = 0.001
		eta_ES = 0.01
		eta_PS = 0.0001

		#initial rates
		R = 0.0
		rR=0.0
		rP = 0.0
		rP_P = 0.0
		rP_N = 0.0
		rS_P = 0.0
		rS_N = 0.0
		rE_P = 0.0
		rE_N = 0.0
		rV = 1.0 


		# monitors
		rP_monitor = np.empty((T))
		rP_P_monitor = np.empty((T))
		rP_N_monitor = np.empty((T))
		rE_P_monitor = np.empty((T))
		rS_P_monitor = np.empty((T))
		rE_N_monitor = np.empty((T))
		rS_N_monitor = np.empty((T))
		rR_monitor = np.empty((T))
		rV_monitor = np.empty((T))
		K_monitor = np.empty((T))

		wPX1_monitor = np.empty((T))
		wPX1_P_monitor = np.empty((T))
		wPX1_N_monitor = np.empty((T))
		wRX1_monitor = np.empty((T))
		wPS_P_monitor = np.empty((T))
		wES_P_monitor = np.empty((T))
		wPS_N_monitor = np.empty((T))
		wES_N_monitor = np.empty((T))

		PV_in = np.empty(T)
		EP_in = np.empty(T)
		EN_in = np.empty(T)
		PV_in_a = np.empty(T)

		sigma_mon = np.empty(T)


		for t in range(T):

			if self.R_neuron:
				rR = R
			else:
				rR=rRrate[t]

			drS_P = (-rS_P + phi(self.wSR_P * R,case=self.rectified))/self.tau_S
			drS_N = (-rS_N + phi(self.wSY1_N * rY1[t],case=self.rectified))/self.tau_S

			if self.single_PV:
				drP = (-rP + phi_square(((1-self.beta_P)*(self.wPX1 * rX1[t]) + self.beta_P*(self.wPY1 * rY1[t] - self.PV_mu*(self.wPS_P * rS_P) + self.PV_mu*(self.wPR * rR) - self.wPS_N * rS_N))))/self.tau_P
			else:
				drP_P = (-rP_P + phi_square((1-self.beta_P)*(self.wPX1_P * rX1[t]) + self.beta_P*(self.wPY1 * rY1[t] - self.wPS_P * rS_P)/(rV+0.001)))/self.tau_P
				#drP_P = (-rP_P + phi_square((1-self.beta_P)*(self.wPX1_P * rX1[t]) + self.beta_P*(self.wPY1 * rY1[t] - self.wPS_P * rR)))/self.tau_P

				drP_N = (-rP_N + phi_square((1-self.beta_P)*(self.wPX1_N * rX1[t]) + self.beta_P*(self.wPR * rR - self.wPS_N * rS_N)/(rV+0.001)))/self.tau_P

			PV_in_a[t] = (1-self.beta_P)*(self.wPX1 * rX1[t])
			PV_in[t] = (self.wPY1 * rY1[t] - self.PV_mu*(self.wPS_P * rS_P) + self.PV_mu*(self.wPR * rR) - self.wPS_N * rS_N)


			if self.single_PV:
				K_P = 1.0/(1.0 + (self.wEP_P * rP))
				K_N = 1.0/(1.0 + (self.wEP_N * rP))
				drE_P = (-rE_P + phi((self.uncertainty_weighted*(K_P) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wEY1_P * rY1[t] - self.wES_P * rS_P),case=self.rectified))/self.tau_E                    
				drE_N = (-rE_N + phi((self.uncertainty_weighted*(K_N) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wER_N * R - self.wES_N * rS_N),case=self.rectified))/self.tau_E
			else:
				K_P = 1.0/(1.0 + (self.wEP_P * rP_P))
				K_N = 1.0/(1.0 + (self.wEP_N * rP_N))
				drE_P = (-rE_P + phi((self.uncertainty_weighted*(K_P) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wEY1_P * rY1[t] - self.wES_P * rS_P),case=self.rectified))/self.tau_E                    
				drE_N = (-rE_N + phi((self.uncertainty_weighted*(K_N) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wER_N * R - self.wES_N * rS_N),case=self.rectified))/self.tau_E


			if t>=100000:
				drV = (-K_P*rV)/self.tau_V 
				if t == 100000:
					rV=2
			else: 
				drV = 0
			#drV = (-K_P*rV + phi(I,case=self.rectified))/self.tau_V 
			if ((t%100000)==0) or ((t%100001)==0) or ((t%100002)==0) :
				print(t)
				print('rP'+str(rP_P))
				print('K'+str(K_P))
				print('rV' + str(rV))
				print('drV' + str(drV))
				print('rPV' + str(rP_P))

			EN_in[t] = self.wER_N * rR - self.wES_N * rS_N
			EP_in[t] = self.wEY1_P * rY1[t] - self.wES_P * rS_P

			drR = (-R + phi((self.wRX1 * rX1[t]) + self.error_weighting*(self.wRE_P * rE_P) - self.error_weighting*(self.wRE_N * rE_N),case=self.rectified))/self.tau_R # no convex combination, input from error neuron

				
				
				
			# store monitors
			rP_monitor[t] = rP
			if not self.single_PV:
				rP_P_monitor[t] = rP_P
				rP_N_monitor[t] = rP_N

			sigma_mon[t] = self.sigma

			rS_P_monitor[t] = rS_P
			rS_N_monitor[t] = rS_N
			rE_P_monitor[t] = rE_P
			rE_N_monitor[t] = rE_N
			rR_monitor[t] = R
			rV_monitor[t] = rV
			K_monitor[t] = K_P

			wPX1_monitor[t] = self.wPX1
			if self.single_PV == False:
				wPX1_P_monitor[t] = self.wPX1_P
				wPX1_N_monitor[t] = self.wPX1_N

			wRX1_monitor[t] = self.wRX1
			wPS_P_monitor[t] = self.wPS_P
			wPS_N_monitor[t] = self.wPS_N
			wES_P_monitor[t] = self.wES_P
			wES_N_monitor[t] = self.wES_N



			
			
			# weight changes
			#v = (1/self.beta_P)*(np.sqrt(rP)-((1-self.beta_P)/2)*self.wPX1*rX1[t])
			#dwPX1 = self.plastic_PX * eta_P * ((np.sqrt(17)*v - (self.wPX1*rX1[t])) * rX1[t])
			dwPX1 = self.plastic_PX * eta_P * ((rP - phi_square(self.wPX1*rX1[t])) * rX1[t])
			if not self.single_PV:
				dwPX1_P = self.plastic_PX * eta_P * ((rP_P - phi_square(self.wPX1_P*rX1[t])) * rX1[t])
				dwPX1_N = self.plastic_PX * eta_P * ((rP_N - phi_square(self.wPX1_N*rX1[t])) * rX1[t])

			dwRX1 = self.plastic * eta_R * ((R - (self.wRX1*rX1[t])) * rX1[t])
			dwPS_P = self.plastic_PS * eta_PS * (self.wPY1 * rY1[t] - self.wPS_P * rS_P)*rS_P
			dwPS_N = self.plastic_PS * eta_PS * (self.wPR * rR - self.wPS_N * rS_N)*rS_N

			dwES_P = self.plastic_ES * eta_ES * ((self.wEY1_P * rY1[t] - self.wES_P * rS_P))*rS_P
			dwES_N = self.plastic_ES * eta_ES * ((self.wER_N * rR - self.wES_N * rS_N))*rS_N
			
			
			# rate changes
			R += self.dt*drR
			rS_P += self.dt*drS_P
			rS_N += self.dt*drS_N
			rV += self.dt*drV
			if ((t%100000)==0) or ((t%100001)==0) or ((t%100002)==0):
				print('rV unten' + str(rV))
			if self.single_PV:
				rP += self.dt*drP
			else:
				rP_P += self.dt*drP_P
				rP_N += self.dt*drP_N

			rE_P += self.dt*drE_P
			rE_N += self.dt*drE_N
			
			
			#wRE_P += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_P
			#wRE_N += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_N

			self.wPX1 += self.dt*dwPX1
			if not self.single_PV:
				self.wPX1_P += self.dt*dwPX1_P
				self.wPX1_N += self.dt*dwPX1_N

			self.wRX1 += self.dt*dwRX1
			if self.plastic_PS:
				self.wPS_P += self.dt*dwPS_P
				self.wPS_N += self.dt*dwPS_N
			if self.plastic_ES:
				self.wES_P += self.dt*dwES_P
				self.wES_N += self.dt*dwES_N



		results = {
		'PV_avg' : np.mean(rP_monitor[100:]),
		'PV_P_avg' : np.mean(rP_P_monitor[100000:]),
		'PV_N_avg' : np.mean(rP_N_monitor[100000:]),
		'SST_P_avg': np.mean(rS_P_monitor[100:]),
		'E_P_avg' : np.mean(rE_P_monitor[100:]),
		'SST_N_avg': np.mean(rS_N_monitor[100:]),
		'E_N_avg' : np.mean(rE_N_monitor[100:]),
		'R_avg' : np.mean(rR_monitor[100:]),
		'PV_std' : np.std(rP_monitor[100:]),
		'PV_P_std' : np.std(rP_P_monitor[100000:]),	
		'PV_N_std' : np.std(rP_N_monitor[100000:]),	
		'SST_P_std': np.std(rS_P_monitor[100:]),
		'E_P_std' : np.std(rE_P_monitor[100:]),
		'SST_N_std': np.std(rS_N_monitor[100:]),
		'E_N_std' : np.std(rE_N_monitor[100:]),
		'R_std' : np.std(rR_monitor[100:]),
		'wPX1' : wPX1_monitor,
		'wPX1_P' : wPX1_P_monitor,
		'wPX1_N' : wPX1_N_monitor,

		'wRX1' : wRX1_monitor, 
		'wPS_P' : wPS_P_monitor, 
		'wPS_N' : wPS_N_monitor, 
		'wES_P' : wES_P_monitor, 
		'wES_N' : wES_N_monitor, 
		#'wRE' : wRE_monitor, 
		'rE_P' : rE_P_monitor, 
		'rS_P' : rS_P_monitor, 
		'rE_N' : rE_N_monitor, 
		'rS_N' : rS_N_monitor, 
		'rP' : rP_monitor, 
		'rP_P' : rP_N_monitor, 
		'rP_N' : rP_N_monitor, 
		'rV' : rV_monitor, 
		'K' : K_monitor, 
		'rR' : rR_monitor, 
		'rY1' : rY1,
		'PV_in':PV_in,
		'PV_in_a':PV_in_a,

		'EP_in':EP_in,
		'EN_in':EN_in,
		'sigma':sigma_mon

		}

		return results

	def run_PVtobothE(self, mean=5.0, sigma=0.5, stimulus_duration=1, seed=None):
		print('mean'+str(mean))
		print('sigma'+str(sigma))
		print('seed'+str(seed))
		print('wPX1:'+str(self.wPX1))
		
		
		if seed is not None:
			np.random.seed(seed)

		#sim parameters
		number_of_samples = 400000

		T = number_of_samples * stimulus_duration # number of time steps
		# inputs
		# sound
		rX1 = np.ones((T))
		rY1 = np.zeros(T)


		# whisker
		Y1_mean = mean
		Y1_sigma = sigma
		self.sigma = sigma

		Y1 = Y1_sigma*np.random.randn(number_of_samples)+Y1_mean

		for i in range(stimulus_duration):
			rY1[i::stimulus_duration] = Y1


		#rS_P= np.ones(T)*Y1_mean
		# test
		if self.Rrate is None:
			rRrate = np.ones(T)*Y1_mean
		else:
			rRrate = np.ones(T)*self.Rrate

		#learning rate
		eta_R = 0.1
		eta_P = 0.001
		eta_ES = 0.01
		eta_PS = 0.0001

		#initial rates
		R = 0.0
		rR=0.0
		rP = 0.0
		rP_P = 0.0
		rP_N = 0.0
		rS_P = 0.0
		rS_N = 0.0
		rE_P = 0.0
		rE_N = 0.0


		# monitors
		rP_monitor = np.empty((T))
		rP_P_monitor = np.empty((T))
		rP_N_monitor = np.empty((T))
		rE_P_monitor = np.empty((T))
		rS_P_monitor = np.empty((T))
		rE_N_monitor = np.empty((T))
		rS_N_monitor = np.empty((T))
		rR_monitor = np.empty((T))

		wPX1_monitor = np.empty((T))
		wPX1_P_monitor = np.empty((T))
		wPX1_N_monitor = np.empty((T))
		wRX1_monitor = np.empty((T))
		wPS_P_monitor = np.empty((T))
		wES_P_monitor = np.empty((T))
		wPS_N_monitor = np.empty((T))
		wES_N_monitor = np.empty((T))

		PV_in = np.empty(T)
		EP_in = np.empty(T)
		EN_in = np.empty(T)
		PV_in_a = np.empty(T)

		sigma_mon = np.empty(T)


		for t in range(T):

			if self.R_neuron:
				rR = R
			else:
				rR=rRrate[t]

			drS_P = (-rS_P + phi(self.wSR_P * R,case=self.rectified))/self.tau_S
			drS_N = (-rS_N + phi(self.wSY1_N * rY1[t],case=self.rectified))/self.tau_S

			if self.single_PV:
				raise ValueError
			else:
				drP_P = (-rP_P + phi_square((1-self.beta_P)*(self.wPX1_P * rX1[t]) + self.beta_P*(self.wPY1 * rY1[t] - self.wPS_P * rS_P)))/self.tau_P
				#drP_P = (-rP_P + phi_square((1-self.beta_P)*(self.wPX1_P * rX1[t]) + self.beta_P*(self.wPY1 * rY1[t] - self.wPS_P * rR)))/self.tau_P

				drP_N = (-rP_N + phi_square((1-self.beta_P)*(self.wPX1_N * rX1[t]) + self.beta_P*(self.wPR * rR - self.wPS_N * rS_N)))/self.tau_P

			PV_in_a[t] = (1-self.beta_P)*(self.wPX1 * rX1[t])
			PV_in[t] = (self.wPY1 * rY1[t] - self.PV_mu*(self.wPS_P * rS_P) + self.PV_mu*(self.wPR * rR) - self.wPS_N * rS_N)
			if self.single_PV:
				raise ValueError
			else:
				drE_P = (-rE_P + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_P * rP_P) + (self.wEP_N * rP_N))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wEY1_P * rY1[t] - self.wES_P * rS_P),case=self.rectified))/self.tau_E                    
				drE_N = (-rE_N + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_P * rP_P) + (self.wEP_N * rP_N))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wER_N * R - self.wES_N * rS_N),case=self.rectified))/self.tau_E


			EN_in[t] = self.wER_N * rR - self.wES_N * rS_N
			EP_in[t] = self.wEY1_P * rY1[t] - self.wES_P * rS_P

			drR = (-R + phi((self.wRX1 * rX1[t]) + self.error_weighting*(self.wRE_P * rE_P) - self.error_weighting*(self.wRE_N * rE_N),case=self.rectified))/self.tau_R # no convex combination, input from error neuron

				
				
				
			# store monitors
			rP_monitor[t] = rP
			if not self.single_PV:
				rP_P_monitor[t] = rP_P
				rP_N_monitor[t] = rP_N

			sigma_mon[t] = self.sigma

			rS_P_monitor[t] = rS_P
			rS_N_monitor[t] = rS_N
			rE_P_monitor[t] = rE_P
			rE_N_monitor[t] = rE_N
			rR_monitor[t] = R
			wPX1_monitor[t] = self.wPX1
			if self.single_PV == False:
				wPX1_P_monitor[t] = self.wPX1_P
				wPX1_N_monitor[t] = self.wPX1_N

			wRX1_monitor[t] = self.wRX1
			wPS_P_monitor[t] = self.wPS_P
			wPS_N_monitor[t] = self.wPS_N
			wES_P_monitor[t] = self.wES_P
			wES_N_monitor[t] = self.wES_N



			
			
			# weight changes
			#v = (1/self.beta_P)*(np.sqrt(rP)-((1-self.beta_P)/2)*self.wPX1*rX1[t])
			#dwPX1 = self.plastic_PX * eta_P * ((np.sqrt(17)*v - (self.wPX1*rX1[t])) * rX1[t])
			dwPX1 = self.plastic_PX * eta_P * ((rP - phi_square(self.wPX1*rX1[t])) * rX1[t])
			if not self.single_PV:
				dwPX1_P = self.plastic_PX * eta_P * ((rP_P - phi_square(self.wPX1_P*rX1[t])) * rX1[t])
				dwPX1_N = self.plastic_PX * eta_P * ((rP_N - phi_square(self.wPX1_N*rX1[t])) * rX1[t])

			dwRX1 = self.plastic * eta_R * ((R - (self.wRX1*rX1[t])) * rX1[t])
			dwPS_P = self.plastic_PS * eta_PS * (self.wPY1 * rY1[t] - self.wPS_P * rS_P)*rS_P
			dwPS_N = self.plastic_PS * eta_PS * (self.wPR * rR - self.wPS_N * rS_N)*rS_N

			dwES_P = self.plastic_ES * eta_ES * ((self.wEY1_P * rY1[t] - self.wES_P * rS_P))*rS_P
			dwES_N = self.plastic_ES * eta_ES * ((self.wER_N * rR - self.wES_N * rS_N))*rS_N
			
			
			# rate changes
			R += self.dt*drR
			rS_P += self.dt*drS_P
			rS_N += self.dt*drS_N
			if self.single_PV:
				rP += self.dt*drP
			else:
				rP_P += self.dt*drP_P
				rP_N += self.dt*drP_N

			rE_P += self.dt*drE_P
			rE_N += self.dt*drE_N
			
			
			#wRE_P += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_P
			#wRE_N += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_N

			self.wPX1 += self.dt*dwPX1
			if not self.single_PV:
				self.wPX1_P += self.dt*dwPX1_P
				self.wPX1_N += self.dt*dwPX1_N

			self.wRX1 += self.dt*dwRX1
			if self.plastic_PS:
				self.wPS_P += self.dt*dwPS_P
				self.wPS_N += self.dt*dwPS_N
			if self.plastic_ES:
				self.wES_P += self.dt*dwES_P
				self.wES_N += self.dt*dwES_N



		results = {
		'PV_avg' : np.mean(rP_monitor[100:]),
		'PV_P_avg' : np.mean(rP_P_monitor[100000:]),
		'PV_N_avg' : np.mean(rP_N_monitor[100000:]),
		'SST_P_avg': np.mean(rS_P_monitor[100:]),
		'E_P_avg' : np.mean(rE_P_monitor[100:]),
		'SST_N_avg': np.mean(rS_N_monitor[100:]),
		'E_N_avg' : np.mean(rE_N_monitor[100:]),
		'R_avg' : np.mean(rR_monitor[100:]),
		'PV_std' : np.std(rP_monitor[100:]),
		'PV_P_std' : np.std(rP_P_monitor[100000:]),	
		'PV_N_std' : np.std(rP_N_monitor[100000:]),	
		'SST_P_std': np.std(rS_P_monitor[100:]),
		'E_P_std' : np.std(rE_P_monitor[100:]),
		'SST_N_std': np.std(rS_N_monitor[100:]),
		'E_N_std' : np.std(rE_N_monitor[100:]),
		'R_std' : np.std(rR_monitor[100:]),
		'wPX1' : wPX1_monitor,
		'wPX1_P' : wPX1_P_monitor,
		'wPX1_N' : wPX1_N_monitor,

		'wRX1' : wRX1_monitor, 
		'wPS_P' : wPS_P_monitor, 
		'wPS_N' : wPS_N_monitor, 
		'wES_P' : wES_P_monitor, 
		'wES_N' : wES_N_monitor, 
		#'wRE' : wRE_monitor, 
		'rE_P' : rE_P_monitor, 
		'rS_P' : rS_P_monitor, 
		'rE_N' : rE_N_monitor, 
		'rS_N' : rS_N_monitor, 
		'rP' : rP_monitor, 
		'rP_P' : rP_N_monitor, 
		'rP_N' : rP_N_monitor, 

		'rR' : rR_monitor, 
		'rY1' : rY1,
		'PV_in':PV_in,
		'PV_in_a':PV_in_a,

		'EP_in':EP_in,
		'EN_in':EN_in,
		'sigma':sigma_mon

		}

		return results


	def run_crossPV(self, mean=5.0, sigma=0.5, stimulus_duration=1, seed=None):
		print('mean'+str(mean))
		print('sigma'+str(sigma))
		print('seed'+str(seed))
		print('wPX1:'+str(self.wPX1))
		
		
		if seed is not None:
			np.random.seed(seed)

		#sim parameters
		number_of_samples = 400000

		T = number_of_samples * stimulus_duration # number of time steps
		# inputs
		# sound
		rX1 = np.ones((T))
		rY1 = np.zeros(T)


		# whisker
		Y1_mean = mean
		Y1_sigma = sigma
		self.sigma = sigma

		Y1 = Y1_sigma*np.random.randn(number_of_samples)+Y1_mean

		for i in range(stimulus_duration):
			rY1[i::stimulus_duration] = Y1


		#rS_P= np.ones(T)*Y1_mean
		# test
		if self.Rrate is None:
			rRrate = np.ones(T)*Y1_mean
		else:
			rRrate = np.ones(T)*self.Rrate

		#learning rate
		eta_R = 0.1
		eta_P = 0.001
		eta_ES = 0.001
		eta_PS = 0.0001

		#initial rates
		R = 0.0
		rR=0.0
		rP = 0.0
		rP_P = 0.0
		rP_N = 0.0
		rS_P = 0.0
		rS_N = 0.0
		rE_P = 0.0
		rE_N = 0.0


		# monitors
		rP_monitor = np.empty((T))
		rP_P_monitor = np.empty((T))
		rP_N_monitor = np.empty((T))
		rE_P_monitor = np.empty((T))
		rS_P_monitor = np.empty((T))
		rE_N_monitor = np.empty((T))
		rS_N_monitor = np.empty((T))
		rR_monitor = np.empty((T))

		wPX1_monitor = np.empty((T))
		wPX1_P_monitor = np.empty((T))
		wPX1_N_monitor = np.empty((T))
		wRX1_monitor = np.empty((T))
		wPS_P_monitor = np.empty((T))
		wES_P_monitor = np.empty((T))
		wPS_N_monitor = np.empty((T))
		wES_N_monitor = np.empty((T))

		PV_in = np.empty(T)
		EP_in = np.empty(T)
		EN_in = np.empty(T)
		PV_in_a = np.empty(T)

		sigma_mon = np.empty(T)


		for t in range(T):

			if self.R_neuron:
				rR = R
			else:
				rR=rRrate[t]

			drS_P = (-rS_P + phi(self.wSR_P * R,case=self.rectified))/self.tau_S
			drS_N = (-rS_N + phi(self.wSY1_N * rY1[t],case=self.rectified))/self.tau_S

			if self.single_PV:
				drP = (-rP + phi_square(((1-self.beta_P)*(self.wPX1 * rX1[t]) + self.beta_P*(self.wPY1 * rY1[t] - self.PV_mu*(self.wPS_P * rS_P)))))/self.tau_P
			else:
				#drP_P = (-rP_P + phi_square((1-self.beta_P)*(self.wPX1_P * rX1[t]) + self.beta_P*(self.wPY1 * rY1[t] - self.wPS_P * rS_P)))/self.tau_P
				drP_P = (-rP_P + phi_square((1-self.beta_P)*(self.wPX1_P * rX1[t]) + self.beta_P*(self.wPR * rR - self.wPS_N * rS_N)))/self.tau_P

				drP_N = (-rP_N + phi_square((1-self.beta_P)*(self.wPX1_N * rX1[t]) + self.beta_P*(self.wPY1 * rY1[t] - self.wPS_P * rS_P)))/self.tau_P

			PV_in_a[t] = (1-self.beta_P)*(self.wPX1 * rX1[t])
			PV_in[t] = (self.wPY1 * rY1[t] - self.PV_mu*(self.wPS_P * rS_P) + self.PV_mu*(self.wPR * rR) - self.wPS_N * rS_N)
			if self.single_PV:
				drE_P = (-rE_P + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_P * rP))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wEY1_P * rY1[t] - self.wES_P * rS_P),case=self.rectified))/self.tau_E                    
				drE_N = (-rE_N + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_N * rP))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wER_N * R - self.wES_N * rS_N),case=self.rectified))/self.tau_E
			else:
				drE_P = (-rE_P + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_P * rP_P))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wEY1_P * rY1[t] - self.wES_P * rS_P),case=self.rectified))/self.tau_E                    
				drE_N = (-rE_N + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_N * rP_N))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wER_N * R - self.wES_N * rS_N),case=self.rectified))/self.tau_E


			EN_in[t] = self.wER_N * rR - self.wES_N * rS_N
			EP_in[t] = self.wEY1_P * rY1[t] - self.wES_P * rS_P

			drR = (-R + phi((self.wRX1 * rX1[t]) + self.error_weighting*(self.wRE_P * rE_P) - self.error_weighting*(self.wRE_N * rE_N),case=self.rectified))/self.tau_R # no convex combination, input from error neuron

				
				
				
			# store monitors
			rP_monitor[t] = rP
			if not self.single_PV:
				rP_P_monitor[t] = rP_P
				rP_N_monitor[t] = rP_N

			sigma_mon[t] = self.sigma

			rS_P_monitor[t] = rS_P
			rS_N_monitor[t] = rS_N
			rE_P_monitor[t] = rE_P
			rE_N_monitor[t] = rE_N
			rR_monitor[t] = R
			wPX1_monitor[t] = self.wPX1
			if self.single_PV == False:
				wPX1_P_monitor[t] = self.wPX1_P
				wPX1_N_monitor[t] = self.wPX1_N

			wRX1_monitor[t] = self.wRX1
			wPS_P_monitor[t] = self.wPS_P
			wPS_N_monitor[t] = self.wPS_N
			wES_P_monitor[t] = self.wES_P
			wES_N_monitor[t] = self.wES_N



			
			
			# weight changes
			#v = (1/self.beta_P)*(np.sqrt(rP)-((1-self.beta_P)/2)*self.wPX1*rX1[t])
			#dwPX1 = self.plastic_PX * eta_P * ((np.sqrt(17)*v - (self.wPX1*rX1[t])) * rX1[t])
			dwPX1 = self.plastic_PX * eta_P * ((rP - phi_square(self.wPX1*rX1[t])) * rX1[t])
			if not self.single_PV:
				dwPX1_P = self.plastic_PX * eta_P * ((rP_P - phi_square(self.wPX1_P*rX1[t])) * rX1[t])
				dwPX1_N = self.plastic_PX * eta_P * ((rP_N - phi_square(self.wPX1_N*rX1[t])) * rX1[t])

			dwRX1 = self.plastic * eta_R * ((R - (self.wRX1*rX1[t])) * rX1[t])
			dwPS_P = self.plastic_PS * eta_PS * (self.wPY1 * rY1[t] - self.wPS_P * rS_P)*rS_P
			dwPS_N = self.plastic_PS * eta_PS * (self.wPR * rR - self.wPS_N * rS_N)*rS_N
			dwES_P = self.plastic_ES * eta_ES * ((self.wEY1_P * rY1[t] - self.wES_P * rS_P))*rS_P
			dwES_N = self.plastic_ES * eta_ES * ((self.wER_N * rR - self.wES_N * rS_N))*rS_N
			
			
			# rate changes
			R += self.dt*drR
			rS_P += self.dt*drS_P
			rS_N += self.dt*drS_N
			if self.single_PV:
				rP += self.dt*drP
			else:
				rP_P += self.dt*drP_P
				rP_N += self.dt*drP_N

			rE_P += self.dt*drE_P
			rE_N += self.dt*drE_N
			
			
			#wRE_P += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_P
			#wRE_N += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_N

			self.wPX1 += self.dt*dwPX1
			if not self.single_PV:
				self.wPX1_P += self.dt*dwPX1_P
				self.wPX1_N += self.dt*dwPX1_N

			self.wRX1 += self.dt*dwRX1
			if self.plastic_PS:
				self.wPS_P += self.dt*dwPS_P
				self.wPS_N += self.dt*dwPS_N
			if self.plastic_ES:
				self.wES_P += self.dt*dwES_P
				self.wES_N += self.dt*dwES_N



		results = {
		'PV_avg' : np.mean(rP_monitor[100:]),
		'PV_P_avg' : np.mean(rP_P_monitor[100000:]),
		'PV_N_avg' : np.mean(rP_N_monitor[100000:]),
		'SST_P_avg': np.mean(rS_P_monitor[100:]),
		'E_P_avg' : np.mean(rE_P_monitor[100:]),
		'SST_N_avg': np.mean(rS_N_monitor[100:]),
		'E_N_avg' : np.mean(rE_N_monitor[100:]),
		'R_avg' : np.mean(rR_monitor[100:]),
		'PV_std' : np.std(rP_monitor[100:]),
		'PV_P_std' : np.std(rP_P_monitor[100000:]),	
		'PV_N_std' : np.std(rP_N_monitor[100000:]),	
		'SST_P_std': np.std(rS_P_monitor[100:]),
		'E_P_std' : np.std(rE_P_monitor[100:]),
		'SST_N_std': np.std(rS_N_monitor[100:]),
		'E_N_std' : np.std(rE_N_monitor[100:]),
		'R_std' : np.std(rR_monitor[100:]),
		'wPX1' : wPX1_monitor,
		'wPX1_P' : wPX1_P_monitor,
		'wPX1_N' : wPX1_N_monitor,

		'wRX1' : wRX1_monitor, 
		'wPS_P' : wPS_P_monitor, 
		'wPS_N' : wPS_N_monitor, 
		'wES_P' : wES_P_monitor, 
		'wES_N' : wES_N_monitor, 
		#'wRE' : wRE_monitor, 
		'rE_P' : rE_P_monitor, 
		'rS_P' : rS_P_monitor, 
		'rE_N' : rE_N_monitor, 
		'rS_N' : rS_N_monitor, 
		'rP' : rP_monitor, 
		'rP_P' : rP_N_monitor, 
		'rP_N' : rP_N_monitor, 

		'rR' : rR_monitor, 
		'rY1' : rY1,
		'PV_in':PV_in,
		'PV_in_a':PV_in_a,

		'EP_in':EP_in,
		'EN_in':EN_in,
		'sigma':sigma_mon

		}

		return results


	def run_contexts(self,Y1_mean,Y1_sigma,Y2_sigma,stimulus_duration=4, PV_weighting= 0.2, error_weighting=0.1, Y=None, seed=None, rho_0 = 0.205):

		if seed is not None:
			np.random.seed(seed)

		#sim parameters
		number_of_contextsamples = 50000
		dt = 1
		T = number_of_contextsamples*stimulus_duration # number of time steps
		# inputs
		# sound
		rX1 = np.ones((T))
		
		# whisker
		#Y1 = Y1_sigma*np.random.randn(int(T/2))+Y1_mean
		#Y2 = Y2_sigma*np.random.randn(int(T/2))+Y1_mean
		#Y = np.concatenate((Y1,Y2))

		# context
		C1 = np.ones((number_of_contextsamples))
		C1[int(len(C1)/2):] = 0 
		C2 = np.ones((number_of_contextsamples))
		C2[:int(len(C1)/2)] = 0
		
		
		C1s, C2s = shuffle(C1,C2)
		
		rC1 = np.zeros(T)
		rC2 = np.zeros(T)
		rY1 = np.zeros(T)

		for i in range(stimulus_duration):
			rC1[i::stimulus_duration] = C1s
			rC2[i::stimulus_duration] = C2s
		
		rY1[(rC1==1)]= Y1_sigma*np.random.randn(int(T/2))+Y1_mean
		rY1[(rC2==1)]= Y2_sigma*np.random.randn(int(T/2))+Y1_mean

		if not self.uncertainty_weighted:
			rC1 = np.ones(T)
			rC2 = np.zeros(T)
		
		
		idx=np.nonzero(np.diff(rC1==1)) # look where context changes, the same for C1 and C2
		#idx=idx+np.ones(len(idx)) # take these indices and add one to get index for rY1 at timepoint t where there is a new context
		#idx_int = np.array(idx, dtype='int') # make an int array out of this
		idx_1=idx-np.ones(len(idx)) # take these indices and add one to get index for rY1 at timepoint t where there is a new context
		idx_1 = np.array(idx_1, dtype='int') # make an int array out of this
		rY1[idx_1]=Y1_mean # set stimulus input to 0 where context changes.
		rY1[idx]=Y1_mean # set stimulus input to 0 where context changes.
		
		
		plt.plot(rC1, color=cm.magma(.5),linewidth=0.5, label='C1: high uncertainty')
		plt.plot(rC2, color=cm.viridis(.3),linewidth=0.5, label='C2: low uncertainty')
		plt.plot(np.nonzero(rC1==1)[0], abs(rY1[rC1==1]), 'o', color=cm.magma(.5), label='C1 sample')
		plt.plot(np.nonzero(rC2==1)[0], abs(rY1[rC2==1]), 'o', color=cm.viridis(.3), label='C2 sample')
		plt.xlim(20,100)
		plt.show()

		
		#learning rate
		eta_R = 0.1
		eta_P = 0.001
		eta_PC = 0.001
		eta_ES = 0.001
		eta_ES_N = 0.001
		eta_RE = 0.001
		
		
		#initial rates
		rP = 0.0#phi_square(rB[0])
		rS_P = 0.0
		rE_P = 0.0
		rS_N = 0.0
		rE_N = 0.0

		rR = 0.0

		# monitors
		rP_monitor = np.empty((T))
		rE_P_monitor = np.empty((T))
		rS_P_monitor = np.empty((T))
		rE_N_monitor = np.empty((T))
		rS_N_monitor = np.empty((T))

		rR_monitor = np.empty((T))

		wPC1_monitor = np.empty((T))
		wPC2_monitor = np.empty((T))
		wRX1_monitor = np.empty((T))
		wPS_P_monitor = np.empty((T))
		wPS_N_monitor = np.empty((T))
		wES_P_monitor = np.empty((T))
		wES_N_monitor = np.empty((T))
		wEP_P_monitor = np.empty((T))
		wEP_N_monitor = np.empty((T))
		wRE_P_monitor = np.empty((T))
		wRE_N_monitor = np.empty((T))
		scaling_factor = np.empty((T))

		PV_in_target = np.empty((T))
		PV_in_wX = np.empty((T))
		PV_in = np.empty((T))
		PV_deltaw = np.empty((T))


		for t in range(T):                 

			if not self.R_neuron:
				rR=rRrate[t]

			drS_P = (-rS_P + phi(self.wSR_P * rR,case=self.rectified))/self.tau_S
			drS_N = (-rS_N + phi(self.wSY1_N * rY1[t],case=self.rectified))/self.tau_S

			drP = (-rP + phi_square(((1-self.beta_P)*(self.wPC1 * rC1[t] + self.wPC2 * rC2[t]) + self.beta_P*(self.wPY1 * rY1[t] - self.PV_mu*(self.wPS_P * rS_P) + self.PV_mu*(self.wPR * rR) - self.wPS_N * rS_N))))/self.tau_P
			PV_in[t] = (self.wPY1 * rY1[t] - self.PV_mu*(self.wPS_P * rS_P) + self.PV_mu*(self.wPR * rR) - self.wPS_N * rS_N)
			#print('Y1')
			#print(self.wPY1 * rY1[t])
			#print('SSTP off')
			#print(self.PV_mu*(self.wPS_P * rS_P))
			#print('R off')
			#print(self.PV_mu*(self.wPR * rR))
			#print('SSTN')
			#print(self.wPS_N * rS_N)
			#print(self.PV_mu*(self.wPS_P * rS_P))
			#print(self.PV_mu*(self.wPR * rR))
			drE_P = (-rE_P + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_P * rP))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wEY1_P * rY1[t] - self.wES_P * rS_P),case=self.rectified))/self.tau_E                    
			drE_N = (-rE_N + phi((self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_N * rP))) + ((1-self.uncertainty_weighted)*self.weighting)) * (self.wER_N * rR - self.wES_N * rS_N),case=self.rectified))/self.tau_E
			scaling_factor[t] = (self.uncertainty_weighted*(1.0/(self.PV_weighting + (self.wEP_P * rP))) + ((1-self.uncertainty_weighted)*self.weighting))
			if self.R_neuron:
				drR = (-rR + phi((self.wRX1 * rX1[t]) + self.error_weighting*(self.wRE_P * rE_P) - self.error_weighting*(self.wRE_N * rE_N),case=self.rectified))/self.tau_R # no convex combination, input from error neuron


			# store monitors
			rP_monitor[t] = rP
			rS_P_monitor[t] = rS_P
			rE_P_monitor[t] = rE_P
			rS_N_monitor[t] = rS_N
			rE_N_monitor[t] = rE_N
			if self.R_neuron:
				rR_monitor[t] = rR
			wPC1_monitor[t] = self.wPC1
			wPC2_monitor[t] = self.wPC2

			wRX1_monitor[t] = self.wRX1
			wPS_P_monitor[t] = self.wPS_P
			wPS_N_monitor[t] = self.wPS_N
			wES_P_monitor[t] = self.wES_P
			wES_N_monitor[t] = self.wES_N
			wRE_P_monitor[t] = self.wRE_P
			wRE_N_monitor[t] = self.wRE_N
			wEP_P_monitor[t] = self.wEP_P
			wEP_N_monitor[t] = self.wEP_N

			

			
			
			# weight changes
			#if t > 100:
			#wPX1_P += plastic_PV * eta_P * ((rP_P - ((wPX1_P*rX1[t])**2)) * rX1[t])
			#wPX1_N += plastic_PV * eta_P * ((rP_N - ((wPX1_N*rX1[t])**2)) * rX1[t])

			dwPC1 = self.plastic_PX * eta_PC * ((rP - phi_square(self.wPC1*rC1[t] + self.wPC2*rC2[t])) * rC1[t])
			dwPC2 = self.plastic_PX * eta_PC * ((rP - phi_square(self.wPC1*rC1[t] + self.wPC2*rC2[t])) * rC2[t])
			#print(plastic * eta_P * ((rP_P - ((wPX1*rX1[t])**2)) * rX1[t]))
			dwRX1 = self.plastic * eta_R * ((rR - (self.wRX1*rX1[t])) * rX1[t])
			if self.PV_mu:
				dwPS_P = self.plastic_PS * eta_P * ((self.wPY1 * rY1[t] - self.wPS_P * rS_P)-0)*rS_P
				dwPS_N = self.plastic_PS * eta_P * ((self.wPR * rR - self.wPS_N * rS_N)-0)*rS_N
			else:
				dwPS_P = 0.0
				dwPS_N = self.plastic_PS * eta_P * ((self.wPY1 * rY1[t] - self.wPS_N * rS_N)-0)*rS_N
				
			dwEP_N = self.plastic_EP * eta_P * ((rE_P - rho_0)*rP)
			dwEP_P = self.plastic_EP * eta_P * ((rE_N - rho_0)*rP)
			
			#wES_N += plastic * eta_ES_N * ((wER_N * rR - wES_N * rS_N)-0)*rS_N
			#wES_P += plastic * eta_ES * ((wEY1_P * rY1[t] - wES_P * rS_P)-0)*rS_P
			#wRE_P += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_P
			#wRE_N += plastic * eta_RE * (wRE_P*rE_P - wRE_N*rE_N) * rE_N

			# rate changes
			if self.R_neuron:
				rR += dt*drR
			rS_P += dt*drS_P
			rS_N += dt*drS_N
			rP += dt*drP
			rE_P += dt*drE_P
			rE_N += dt*drE_N
			
			self.wPC1 += dt*dwPC1
			self.wPC2 += dt*dwPC2
			self.wRX1 += dt*dwRX1
			if self.plastic_PS:
				self.wPS_P += dt*dwPS_P
				self.wPS_N += dt*dwPS_N
					
			if self.plastic_EP:
				self.wEP_P += dt*dwEP_P
				self.wEP_N += dt*dwEP_N


		results = {
		'PV_avg' : np.mean(rP_monitor[100:]),
		'SST_P_avg': np.mean(rS_P_monitor[100:]),
		'E_P_avg' : np.mean(rE_P_monitor[100:]),
		'R_avg' : np.mean(rR_monitor[100:]),
		'SST_N_avg' : np.mean(rS_N_monitor[100:]),
		'E_N_avg' : np.mean(rE_N_monitor[100:]),
		'PV_std' : np.std(rP_monitor[100:]),
		'SST_P_std': np.std(rS_P_monitor[100:]),
		'E_P_std' : np.std(rE_P_monitor[100:]),
		'R_std' : np.std(rR_monitor[100:]),
		'SST_N_std' : np.std(rS_N_monitor[100:]),
		'E_N_std' : np.std(rE_N_monitor[100:]),        
		'wPC1' : wPC1_monitor, 
		'wPC2' : wPC2_monitor, 
		'scaling_factor' : scaling_factor, 

		'wRX1' : wRX1_monitor, 
		'wPS_P' : wPS_P_monitor, 
		'wPS_N' : wPS_N_monitor, 
		'wEP_P' : wEP_P_monitor, 
		'wEP_N' : wEP_N_monitor, 

		'wES_P' : wES_P_monitor, 
		'wES_N' : wES_N_monitor, 
		'wRE_P' : wRE_P_monitor, 
		'wRE_N' : wRE_N_monitor, 
		'rE_P' : rE_P_monitor, 
		'rS_P' : rS_P_monitor, 
		'rP' : rP_monitor, 
		'rE_N' : rE_N_monitor, 
		'rS_N' : rS_N_monitor, 
		'rR' : rR_monitor, 
		'rY1' : rY1,
		'rC1' : rC1,
		'rC2' : rC2,        
		'PV_in' : PV_in
		}


		return results


if __name__ == "__main__":

	"""
	circuit1 = Circuit()
	circuit2 = Circuit()
	circuit1.single_PV = False
	circuit2.single_PV = False
	results_lowsigma = circuit1.run(sigma=0.4)
	results_highsigma = circuit2.run(sigma=0.8)

	plot_2PVratesandweights(results_lowsigma,results_highsigma,0.4,0.8,'testpyfile')
	"""
	seed = 378784
	training_results={}
	training_mean=5.0
	sigma_range=np.arange(0.3,1.0,0.4)
	beta_P = 0.01

	for sigma in sigma_range:
		circuit = Circuit()
		circuit.single_PV = False
		circuit.beta_P = beta_P
		wP = np.sqrt((1-beta_P)/beta_P)
		circuit.wPY1 = np.array([wP]) # small intitial weights
		circuit.wPR = np.array([wP]) # small intitial weights
		circuit.wPS_P = np.array([wP])
		circuit.wPS_N = np.array([wP])
		training_results[sigma]=circuit.run(mean=training_mean,sigma=sigma,seed=seed)

	mismatch_results={}
	mean_range=np.arange(3.0,8.0,1.0)
	for mean in mean_range:
		print('mean'+str(mean))
		mismatch_results[mean]={}
		for i,sigma in enumerate(training_results.keys()):
			circuit = Circuit()
			circuit.single_PV=False
			circuit.beta_P = beta_P
			wP = np.sqrt((1-beta_P)/beta_P)
			circuit.wPY1 = np.array([wP]) # small intitial weights
			circuit.wPR = np.array([wP]) # small intitial weights
			circuit.wPS_P = np.array([wP])
			circuit.wPS_N = np.array([wP])
			circuit.Rrate= training_mean
			circuit.wRX1 = training_results[sigma]['wRX1'][-1]
			circuit.wPX1_P=training_results[sigma]['wPX1_P'][-1]
			circuit.wPX1_N=training_results[sigma]['wPX1_N'][-1]
			circuit.plastic_PX=False
			#print(training_results[sigma]['wRX1'][-1])
			#print(training_results[sigma]['wPX1'][-1])

			mismatch_results[mean][sigma_range[i]]=circuit.run(mean=mean,sigma=0.0,seed=seed)
			print('sigma'+str(sigma))


	with open('training_results.pickle', 'wb') as handle:
	    pickle.dump(training_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('mismatch_results.pickle', 'wb') as handle:
	    pickle.dump(mismatch_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

	#with open('mismatch_results.pickle', 'rb') as handle:
	#    mismatch_results = pickle.load(handle)


	plot_Error(mismatch_results,training_mean,mean_range,sigma_range,'mismatch2PVbeta001')
	plot_2PVratesandweights(training_results[sigma_range[0]],training_results[sigma_range[1]],sigma_range[0],sigma_range[1],'testpyfile')


