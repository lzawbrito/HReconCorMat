import numpy as np
from os import path 

alex_dir = "../../data/alex_data/"

def get_dir(thermal, l, delta, beta):
	fn = "Thermal_Data" if thermal else "Zero_Temp_Data"

	if thermal:
		fn = path.join(fn, 
				f"Cormat_Data_L={l}_Delta={delta}_Spectral_Depth=20_beta={beta}")
	else: 
		fn = path.join(fn, f"Cormat_Data_L={l}_Delta={delta}")

	fn = path.join(alex_dir, fn)

	return fn


def get_eigen(thermal, l, delta, beta, trunc):
	fn = get_dir(thermal, l, delta, beta)
	fn = path.join(fn, f"Truncations={trunc}")
	evecs = np.genfromtxt(path.join(fn, 'Eigvecs.txt'), delimiter='\t')
	evals = np.genfromtxt(path.join(fn, 'Eigvals.txt'), delimiter='\t')
	
	if np.shape(evals) == (): 
		evals = np.array([evals])
	return evals, evecs


def get_hamiltonian(l, delta, beta): 
	fn = get_dir(True, l, delta, beta)
	fn = path.join(fn, "Extra_Data")

	h = np.genfromtxt(path.join(fn, 'Hvec.txt'), delimiter='\t')
	
	return h 

def get_boltzmann(l, delta, beta): 
	fn = get_dir(True, l, delta, beta)
	fn = path.join(fn, "Extra_Data")

	h = np.genfromtxt(path.join(fn, 'Boltzmann_weights.txt'), delimiter='\t')
	
	return h 


def get_spectrum(l, delta, beta):
	fn = get_dir(True, l, delta, beta)
	fn = path.join(fn, "Extra_Data")

	h = np.genfromtxt(path.join(fn, 'Raw_Spectrum.txt'), delimiter='\t')
	
	return h 


def get_cm(l, delta):
	fn = get_dir(False, l, delta, 0.0)
	fn = path.join(fn, "Correlation_Matrix.txt")

	h = np.genfromtxt(fn, delimiter='\t')
	
	return h 