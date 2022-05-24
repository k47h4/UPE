if __name__ == "__main__":
	test_dict = {'key1':1, 'key2':2}


	with open('/scratch/snx3000/kwilmes/test_dict.pickle', 'wb') as handle:
	    pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)