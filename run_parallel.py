from Circuit import * 

import multiprocessing
from itertools import repeat

from functools import partial
import time




def initialise_Circuit():
	beta_P = 0.05
	circuit = Circuit()
	circuit.single_PV = False
	circuit.beta_P = beta_P
	wP = np.sqrt((1-beta_P)/beta_P)
	circuit.wPY1 = np.array([wP]) # small intitial weights
	circuit.wPR = np.array([wP]) # small intitial weights
	circuit.wPS_P = np.array([wP])
	circuit.wPS_N = np.array([wP])

	return circuit 




if __name__ == "__main__":

	pool = multiprocessing.Pool(processes=4)
	
	sigma_range=np.arange(0.3,1.0,0.4).tolist()
	training_mean = 5.0
	seed = 378784
	circuit = initialise_Circuit()
	
	#L = pool.starmap(circuit.run, [(5, 1), (4, 2), (3, 3)])
	%%time

	training_results = pool.starmap(circuit.run, zip(repeat(training_mean),sigma_range,repeat(seed)))
	
	print(training_results[0])
	print(training_results[1])
    
	"""
	training_results={}
	sigma_range=np.arange(0.3,1.0,0.4)
	beta_P = 0.01

	for sigma in sigma_range:

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
	"""

