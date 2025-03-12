# WANDA - Walking gait Adaptation N' Decouple Architecture
# ------------------- import modules ---------------------

# standard modules
import time, sys, os, glob
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
from scipy.fft import fft, fftfreq

# plot
import matplotlib.pyplot as plt 




path = 'utils/data/bone'
ndim = 3

files = glob.glob(path+'/*/**/***/****/****.npy')

conds = [[] for i in range(ndim)]

for file in files:
	cond = file.split('/')[len(path.split('/')):-2]

	for i in range(ndim):
		conds[i].append(int(cond[i]))
		conds[i] = sorted(list(set(conds[i])))

conds = [({v: k for k, v in dict(enumerate(conds[i])).items()}  ) for i in range(ndim)]

print(conds)

def emptylist(dshape):
	data = []
	for i in  range(dshape[0]):
		datai = []
		for j in  range(dshape[1]):
			dataj = [[] for k in  range(dshape[2])]
			datai.append(dataj)
		data.append(datai)
	return data

#conds = ['independent_withouttagotae','symetry_withouttagotae','symetry_withtagotae','allshare_withtagotae']

dshape = [len(cond) for cond in conds]
miniconds = []
rewards = emptylist(dshape)
bases = emptylist(dshape)
torques = emptylist(dshape)
sync_indices = emptylist(dshape)

rewardsd = deepcopy(rewards)


'''
 -----------------------------------------------------------------------------------------
                                    read data
 -----------------------------------------------------------------------------------------
'''

for file in files:
	
	cond = [int(ci) for ci in file.split('/')[3:-2]]
	
	data = np.load(file)
	if 'reward' in file:
		print(file)
		rewards[conds[0][cond[0]]][conds[1][cond[1]]][conds[2][cond[2]]].append(data)
		print(data.shape)
		cond = int(file.split('/')[5])
		colors = {0:'tab:blue',2000:'tab:cyan',500:'tab:orange',1000:'tab:red'}
		if cond in [1000]:
			plt.plot(np.mean((data[:,40:,0]),axis=1),label=file,alpha=0.5)

	'''
	elif 'basis' in file:
		bases[conds[0][cond[0]]][conds[1][cond[1]]][conds[2][cond[2]]].append(data)
	elif 'torque' in file:
		torques[conds[0][cond[0]]][conds[1][cond[1]]][conds[2][cond[2]]].append(data)
		sync_indices[conds[0][cond[0]]][conds[1][cond[1]]][conds[2][cond[2]]].append(0.0)
	'''

plt.legend()
plt.show()
sys.exit()
def compute_order_parameter(phases):
    """
    Compute the order parameter R from phases.
    
    Args:
        phases (ndarray): Array of phases (in radians) for each leg.
    
    Returns:
        R (float): Order parameter indicating the level of synchronization.
    """
    N = len(phases)
    order_param = np.abs(np.sum(np.exp(1j * phases)) / N)
    return order_param


def compute_relative_phases(torque_signals, sampling_rate):
	"""
	Compute the relative phases and synchronization index from torque signals.
	
	Args:
		torque_signals (ndarray): A 2D array where each row is the torque signal of a leg (shape: num_legs x num_samples).
		sampling_rate (float): Sampling rate of the torque signals in Hz.
	
	Returns:
		relative_phases (ndarray): Pairwise relative phases between legs (shape: num_legs x num_legs).
		sync_index (float): Average synchronization index across all leg pairs.
	"""
	num_legs, num_samples = torque_signals.shape
	
	# Fourier Transform to extract dominant frequency components
	fft_values = fft(torque_signals, axis=1)
	freqs = fftfreq(num_samples, d=1/sampling_rate)
	
	# Find the dominant frequency component (ignoring the DC component)
	dominant_indices = np.argmax(np.abs(fft_values[:, 1:]), axis=1) + 1
	dominant_phases = np.angle(fft_values[np.arange(num_legs), dominant_indices])
	
	# Compute pairwise relative phases
	relative_phases = np.zeros((num_legs, num_legs))
	for i in range(num_legs):
		for j in range(num_legs):
			relative_phases[i, j] = (dominant_phases[i] - dominant_phases[j]) % (2 * np.pi)
			if relative_phases[i, j] > np.pi:
				relative_phases[i, j] -= 2 * np.pi  # Wrap to [-π, π]
	
	# Compute Synchronization Index (SI)
	sync_sum = 0
	num_pairs = 0
	for i in range(num_legs):
		for j in range(i + 1, num_legs):
			sync_sum += np.cos(relative_phases[i, j])
			num_pairs += 1
	sync_index = sync_sum / num_pairs if num_pairs > 0 else 0
	
	return relative_phases, sync_index

#rewards = np.array(rewards,dtype=object)
#bases = np.array(bases,dtype=object)
for i in range(dshape[0]):
	for j in range(dshape[1]):
		for k in range(dshape[2]):
			#print(i,j,k)
			#print(i,j,k,np.mean(np.array(rewards[i][j][k]),axis=0).shape,np.array(rewards[i][j][k]).shape)
			#print(i,j,k,np.mean(np.array(rewards[i][j][k]),axis=0).shape,np.mean(np.array(bases[i][j][k]),axis=0).shape)
			#print(np.array(rewards[i][j][k]))
			if rewards[i][j][k] != []:
				#print(np.array(rewards[i][j][k]).shape)
				#goodid = np.where(np.sum(np.abs(np.array(rewards[i][j][k]))[:,-1,:,0],axis=(1))>0)
				#print(i,j,k,np.sum(np.abs(np.array(rewards[i][j][k]))[:,-1,:,0],axis=(1))>1e-6)
				#print(np.array(rewards[i][j][k]).shape,rewardsd[i][j][k])
				#print(np.array(rewards[i][j][k]).shape)
				rewardsd[i][j][k] = np.std(np.mean(np.array(rewards[i][j][k]),axis=-1),axis=0)
				#print(np.array(bases[i][j][k]).shape,np.array(rewards[i][j][k]).shape)
				rewards[i][j][k] = np.mean(np.array(rewards[i][j][k]),axis=0)
				#print(np.mean(np.array(rewards[i][j][k])[goodid],axis=0).shape,rewardsd[i][j][k])
				
				#bases[i][j][k] = np.mean(np.array(bases[i][j][k])[goodid],axis=0)
				

				


				#torque = np.array(torques[i][j][k])[goodid]
				#sync_indices_ = []
				#print(torque.shape)
				'''
				for n in range(torque.shape[0]):
					torquei = torque[n]
					phase, si = compute_relative_phases(np.transpose(torquei[1,:,1::3]),20)
					op = compute_order_parameter(phase)
					sync_indices_.append(op)
				sync_indices[i][j][k] = np.mean(np.array(sync_indices_))'''

				#torques[i][j][k] = np.mean(np.array(torques[i][j][k])[goodid],axis=0)

'''
for i,j,k in zip([0,0,0,0,0,1,1,2],[0,1,2,2,2,2,2,0],[0,0,0,2,4,2,4,0]):
	rewards[i][j][k] = np.zeros((500,30))
	rewardsd[i][j][k] = np.zeros((500))
	#bases[i][j][k] = np.zeros((500,30,4))
	#torques[i][j][k] = np.zeros((500,30,18))
	#sync_indices[i][j][k] = 0.0'''

'''
for i in range(dshape[0]):
	for j in range(dshape[1]):
		for k in range(dshape[2]):
			print(i,j,k,rewards[i][j][k])
			print(rewards[i][j][k].shape)'''

#print(np.array(rewards).shape)
rewards = np.array(rewards,dtype=np.float)
rewardsd = np.array(rewardsd,dtype=np.float)

#bases = np.array(bases,dtype=np.float)
#torques = np.array(torques,dtype=np.float)
#sync_indices = np.array(sync_indices,dtype=np.float)
#print(rewards.shape)
print(rewards[:,:,:,-50:].shape)
finalrewards = np.mean(np.mean(rewards[:,:,:,-10:,40:],axis=-2),axis=-2)[:,:,:,0]
print('final',finalrewards*20*100)
'''
for indexs in [[0,0,0]]:
	i,j,k = indexs

	if 1:
		#freqval = np.resize(np.argmax(bases[i,j,k],axis=-1),(500*30))

		rmean = np.mean(rewards[i,j,k][:,40:,0],-1)*20*100
		rstd = np.std(rewards[i,j,k][:,40:,0],-1)*20*100
		print(rmean.shape,rstd.shape)
		#print(indexs,sync_indices[i,j,k])
		print(rewards[i,j,k].shape)
		plt.plot(rmean,alpha=1,label=str(i)+','+str(j)+','+str(k))
		plt.fill_between(np.arange(0,rmean.shape[0]),rmean-rstd,rmean+rstd,alpha=0.1)


			

#plt.ylim([-3,50])
plt.legend()
plt.show()'''