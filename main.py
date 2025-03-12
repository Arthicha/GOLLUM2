# WANDA - Walking gait Adaptation N' Decouple Architecture
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser
from configparser import ConfigParser
from argparse import ArgumentParser


# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
import torch.nn.functional as F


# simulation
#from interface.vrepinterfaze import VrepInterfaze
#from interface.mujocointerfaze import MujocoInterfaze
#torch.cuda.empty_cache()

# visualization
import cv2
import matplotlib.pyplot as plt
import roboticstoolbox as rtb

# control
from network.SME import SequentialMotionExecutor
from network.FC import GaussianMixtureModel, FullyConnected, VQVAE_FC
from optimizer.agol import AddedGradientOnlineLearning
from optimizer.gd import GradientDescent
from utils.utils import TorchReplay as Replay
from utils.utils import make_transition, Dict
from utils.kinematics import Kinematics
from utils.utils import RunningMeanStd
from utils.kinematics import BoneKin as RobKin
from interface.mujocointerfaze import MujocoInterfaze

from agents.sac import SAC

runargv = sys.argv[1:]


# ------------------- config variables ---------------------
PATH = 'utils/data/'+runargv[0]+'/'+runargv[1]+'/'+runargv[2]+'/'+runargv[3]+'/'+runargv[4]

NJPERM = 12
NREPLAY = 8
NTIMESTEP = 700
T0 = 40
NREWARD = 3
NFC = 8
NOBS = 13#4#13
NFCOBS = 13+4*12
NEPISODE = 1000
HORIZON = 20
GAE = False
GAMMA = 0.95 
LAMBDA = 0.5

FRAMEWIDTH = 400
FRAMEHEIGHT = 400

RESET = False
ALPHA = 0.1
lr_tagotae = 1
CONNECTION = torch.FloatTensor(np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])).cuda()

# ------------------- auxiliary functions ---------------------

cumsumgain = torch.arange(NTIMESTEP,0,-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
coefficients = (LAMBDA) ** torch.arange(NTIMESTEP, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
if not os.path.exists(PATH):
	os.makedirs(PATH)

def numpy(x):
	return x.cpu().detach().numpy()

def tensor(x):
	return torch.FloatTensor(x).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def critic_func(x):
	if len(x.shape) == 4:
		return torch.concatenate([x[:,:,:,[0]],-torch.abs(x[:,:,:,1:])],dim=-1)
	else:
		return torch.concatenate([x[:,[0]],-torch.abs(x[:,1:])],dim=-1)

def critic_func2(x):
	return -torch.abs(x)

def compute_return(speed,reduce=torch.mean):
	global HORIZON
	returns_ = speed.clone()
	returns_ = returns_.unfold(1,HORIZON,1) 
	returns_ = reduce(returns_,dim=-1) 
	return returns_

def get_tdtarget(values,next_values, rewards):
	global GAMMA
	return rewards + GAMMA * next_values

def get_gae(values,next_values, rewards):
	global coefficients

	td_target = get_tdtarget(values,next_values,rewards)
	delta = td_target - values

	advantages_norm = torch.flip(torch.cumsum(torch.flip(delta, dims=[0]) * coefficients, dim=0), dims=[0])
	#advantages_denorm = torch.flip(torch.cumsum(coefficients, dim=0), dims=[0])
	
	return advantages_norm#/advantages_denorm


# ------------------- setup ---------------------

# initiliaze SME network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sme = SequentialMotionExecutor('network.ini',CONNECTION)
critic = FullyConnected(nin=NOBS+NFC, nout = NREWARD, nlayer=3,nneuron=64,last_actfunc=critic_func,stack=True,
	last_bias = False,zeroparams=True)
#error_critic = FullyConnected(nin=NOBS+NFC, nout = NREWARD, nlayer=3,nneuron=64,last_actfunc=critic_func,stack=True,
#   last_bias = False,zeroparams=True)


# init reflex net
fc = VQVAE_FC(input_dim=NFCOBS, hidden_dim=64, latent_dim=10, num_embeddings=NFC, commitment_cost=0.25).to(device)

# initialize AGOL learning algorithm
agol = AddedGradientOnlineLearning({'output':sme.mn.W},'optimizer.ini')
gd = GradientDescent({'output':sme.mn.W},'optimizer.ini') # initialzie GD learning algorithm for baseline estimation
#error_gd = GradientDescent({'output':sme.mn.W},'optimizer.ini') # initialzie GD learning algorithm for baseline estimation

agol.attach_returnfunction(compute_return) # set return function
agol.attach_valuenet(critic) # set value network (remove this if you want to use average baseline)

gd.attach_returnfunction(compute_return)  # set return function
gd.attach_valuenet(critic)

#error_gd.attach_returnfunction(compute_return)  # set return function
#error_gd.attach_valuenet(error_critic)

# initialize simulation interface
vrep = MujocoInterfaze(hz = 30, dt=0.001,print_step=None, render_mode="rgb_array",width=FRAMEWIDTH, height=FRAMEHEIGHT,camera_name='side')


# initialize experience replay
reward_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,NREWARD))
grad_replay = {'output':Replay(NREPLAY,shape= (NTIMESTEP,CONNECTION.shape[0]+NFC,NJPERM))}
weight_replay = {'output':Replay(NREPLAY,shape=(1,CONNECTION.shape[0]+NFC,NJPERM))}
observation_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,NOBS))
fcobservation_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,NFCOBS))
bases_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,CONNECTION.shape[0]))
old_embedding_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,NFC))
critical_replay = Replay(140,shape=(1,1,NOBS),fillfirst=False)

# initialize fc weight replay
optimizer = torch.optim.Adam(fc.parameters(), lr=1e-4)
recorded_reward = np.zeros((NEPISODE,NTIMESTEP+T0,NREWARD))

chcond = 2000
ncksample = 1

fc.load_state_dict(torch.load('utils/data/bone/1/10/'+str(chcond)+'/'+str(ncksample)+'/fc'),strict=False)
critic.load_state_dict(torch.load('utils/data/bone/1/10/'+str(chcond)+'/'+str(ncksample)+'/critic'),strict=False)
sme.set_weight(tensor(np.load('utils/data/bone/1/10/'+str(chcond)+'/'+str(ncksample)+'/weight.npy')))
agol.set_sigma(tensor(np.load('utils/data/bone/1/10/'+str(chcond)+'/'+str(ncksample)+'/sigma.npy')))




fcgain = float(runargv[3])/1000.0
sumemb = None
# ------------------- start locomotion learning ---------------------
for i in range(NEPISODE):
	print('episode',i)

	# episode-wise setup
	vrep.reset()

	prepose = vrep.get_robot_pose()

	wnoises = agol.wnoise()
	for key in wnoises.keys():
	   wnoises[key][:] *= 0.0

	sme.explore(wnoises['output'])
	#for i in range(10):
	#   print(sme.mn.W[4+i])
	#sys.exit()
	#with torch.no_grad():
	#   sme.mn.W[4:] *= 0
	#   sme.mn.Wn[4:] *= 0

	weight_replay['output'].add(sme.mn.Wn,convert=False)

	action = np.zeros((1,NJPERM)) #+ reflex
	mag_fcoutput = 0
	preembedding = None
	
	for t in range(NTIMESTEP+T0):

		gain = 1 if t >= T0 else t/T0
		
		if 1:#(i%10 == 0):# and (i%5 == 0):
			cv2.imshow("image", vrep.render())
			cv2.waitKey(1)

		# prepare network
		obs = torch.concatenate([torch.FloatTensor(vrep.get_observations()).to(device),sme.get_basis(torch=True)[0]],dim=0)
		qdq = np.expand_dims(vrep.get_obs()[3:],axis=0)
		q = qdq[:,::2]
		dq = qdq[:,1::2]
		error = tensor(action - q)
		derror = tensor(0 - dq)
		
		lphase = tensor(vrep.estimate_onlylegphase(np.reshape(q,(-1,3)))).unsqueeze(0)
		fcobs = torch.concatenate([obs.unsqueeze(0)[:,:-4],tensor(q),tensor(dq),error,derror,lphase],dim=1)

		encoding, _, _ = fc.forward(fcobs)
		embedding = fc.encoding2embedding(encoding)
		output = sme.forward(fcgain*embedding*0.1)

		# update environment
		myaction = output 
		action = gain*np.clip(np.array((myaction).detach().cpu()),-1.5,1.5)
		vrep.set_robot_joint(action)
		vrep.update(contact=None)

		if (t >= T0):
			# smeagol backprob
			sme.zero_grad()
			torch.sum(output).backward() 
			grad_replay['output'].add(sme.mn.W.grad.abs(),convert=False)

		# compute reward
		pose = vrep.get_robot_pose()
		dx = pose[0]-prepose[0]
		dy = pose[1]-prepose[1]

		qdq = np.expand_dims(vrep.get_obs()[3:],axis=0)
		q = qdq[:,::2]
		reward = dx*np.cos(prepose[-1]) + dy*np.sin(prepose[-1]) 
		rollpen = -np.abs(pose[3])
		pitchpen = -np.abs(pose[4])

		recorded_reward[i,t,0] = reward
		recorded_reward[i,t,1] = pose[3]
		recorded_reward[i,t,2] = pose[4]
		prepose = deepcopy(pose)

		# append experience replay
		if (t >= T0):
			fcobservation_replay.add(fcobs.unsqueeze(0).detach(),convert=False)
			bases_replay.add(sme.get_basis(torch=True).detach(),convert=False)
			reward_replay.add([[reward,rollpen,pitchpen]],convert=True)
			observation_replay.add(obs.unsqueeze(0).detach(),convert=False)
			old_embedding_replay.add(embedding.detach(),convert=False)
	#sys.exit()
	# print episode info
	print('\tepisodic reward',torch.sum(reward_replay.data()[-1,:,:,0]).item())
	print('\tpenalty',torch.min(reward_replay.data()[-1,:,:,1]).item(),torch.min(reward_replay.data()[-1,:,:,2]).item())
	sys.exit()
	

	'''
	# -------------------------------------------------------------------------------------------------
	
	enough, nsample = fc.is_enough(critical_replay.data()[:,0,0])
	if(enough):
		fcgain = 1.0
		fc.update(critical_replay.data()[-nsample:,0,0])
			
	print('\tcritical:',nsample)'''
	sid = -(i + 1) if i < (NREPLAY-1) else 0

	for it in range(1):
		optimizer.zero_grad()
		trainingdata = torch.reshape(fcobservation_replay.data()[-sid:],(-1,NFCOBS))
		rewarddata = torch.reshape(reward_replay.data()[-sid:],(-1,NREWARD))
		encodings, x_recon, vq_loss = fc.forward(trainingdata)
		recon_loss = F.mse_loss(x_recon, torch.concatenate([trainingdata,rewarddata],dim=-1))  # Reconstruction loss
		loss = recon_loss + vq_loss

		loss.backward()
		optimizer.step()

	encodings, _ , _ = fc.forward(torch.reshape(fcobservation_replay.data()[-NREPLAY:],(-1,NFCOBS)))
	embedding = torch.reshape(fc.encoding2embedding(encodings),(NREPLAY,NTIMESTEP,1,NFC)).detach()
	'''
	if i > 2:
		plt.clf()
		plt.plot(torch.reshape(embedding[-2:,:,0,:],(-1,NFC)).cpu().numpy())
		plt.pause(1)'''

	# -------------------------------------------------------------------------------------------------
	

	#print(observation_replay.data().shape,embedding.shape)
	gd.update(torch.concat([observation_replay.data(),embedding],dim=-1),reward_replay.data(),nepi=i)
	
	values, _ = critic(torch.concat([observation_replay.data(),embedding],dim=-1))
	values = values.detach()

	# roger & advantage
	rewards = reward_replay.data()
	reweight = agol.roger(rewards,values)
	Q_ = compute_return(torch.sum(reweight*rewards,-1,keepdim=True))
	V_ = compute_return(torch.sum(reweight*values,-1,keepdim=True))
	advantages = Q_ - V_

	# using new state from the embedding (old states can be obtained by grad)
	newstate = torch.concat([bases_replay.data(),embedding],dim=-1)
	state = torch.concat([bases_replay.data(),old_embedding_replay.data()],dim=-1)

	# update
	dws, dsig = agol.update(advantages, weight_replay,{'output':sme.mn.W}, grad_replay, state[:,:,0], 
		newstate[:,:,0],lrscale=0.1*float(runargv[2]),nepi=i,verbose=True,weightadjustment=False,horizon=0)
	with torch.no_grad():
		sme.mn.W[:4] += dws['output'][:4]
	agol.add_sigma(dsig,start=0,end=4,gain=1)

	#Q_ = compute_return(torch.sum(rewards[:,:,:,1:],-1,keepdim=True))
	#V_ = compute_return(torch.sum(values[:,:,:,1:],-1,keepdim=True))
	#advantages = Q_ - V_
	
	# update
	dws, dsig = agol.update(advantages, weight_replay,{'output':sme.mn.W}, grad_replay,state[:,:,0],
		newstate[:,:,0],lrscale=0.1*float(runargv[2]),nepi=i,verbose=False,weightadjustment=True,horizon=0)
	with torch.no_grad():
		sme.mn.W[4:] += 0.1*dws['output'][4:]
	agol.add_sigma(dsig,start=4,end=-1,gain=1)

	
	
if 1:#(i%100 == 0) and (i != 0):
	np.save(PATH+'/reward', recorded_reward)
	np.save(PATH+'/weight',sme.mn.W.detach().cpu().numpy())
	np.save(PATH+'/sigma',agol.get_sigma()['output'].detach().cpu().numpy())
	torch.save(critic.state_dict(), PATH+'/critic')
	torch.save(fc.state_dict(), PATH+'/fc')
	#np.save(PATH+'/cluster',fc.cluster.detach().cpu().numpy())

'''
for k in range(NMODULE):
	np.save(PATH+'/weight'+str(k),sme[k].mn.W.detach().cpu().numpy())'''
