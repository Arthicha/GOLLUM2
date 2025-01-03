# WANDA - Walking gait Adaptation N' Decouple Architecture
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array

# simulation
from interface.vrepinterfaze import VrepInterfaze

# control
from network.SME import SequentialMotionExecutor
from optimizer.agol import AddedGradientOnlineLearning
from optimizer.gd import GradientDescent
from utils.utils import TorchReplay as Replay

runargv = sys.argv[1:]

# ------------------- config variables ---------------------
PATH = 'utils/data/'+runargv[0]+'/'+runargv[1]+'/'+runargv[2]+'/'+runargv[3]

if int(runargv[0]) == 0:
	MORDER = np.array([0,0,0,0,0,0])
elif int(runargv[0]) == 1:
	MORDER = np.array([0,1,2,3,4,5])
elif int(runargv[0]) == 2:
	MORDER = np.array([0,1,2,0,1,2])

NMODULE = len(MORDER)
NJPERM = 3
NREPLAY = 8
NTIMESTEP = 30
NEPISODE = 500
RESET = False
ALPHA = 0.1
lr_tagotae = 1
CONNECTION = torch.FloatTensor(np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,0,0,0]])).cuda()

# ------------------- auxiliary functions ---------------------

cumsumgain = torch.arange(NTIMESTEP,0,-1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
if not os.path.exists(PATH):
	os.makedirs(PATH)


def numpy(x):
	return x.cpu().detach().numpy()

def tensor(x):
	return torch.FloatTensor(x).to(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def compute_return(speed):
	global cumsumgain
	speed_flip = torch.flip(speed,dims=[1])
	speed_fliptilend = torch.cumsum(speed_flip,dim=1)
	speed_tilend = torch.flip(speed_fliptilend,dims=[1])
	return speed_tilend/cumsumgain

# ------------------- setup ---------------------

# initiliaze SME network
sme = [SequentialMotionExecutor('network.ini',CONNECTION) for i in range(NMODULE)]

# initialize AGOL learning algorithm
agol = [AddedGradientOnlineLearning(sme[i].mn.W,'optimizer.ini') for i in range(NMODULE)]
gd = [GradientDescent(sme[i].vn.W,'optimizer.ini') for i in range(NMODULE)] # initialzie GD learning algorithm for baseline estimation

for i in range(NMODULE):
	agol[i].attach_returnfunction(compute_return) # set return function
	agol[i].attach_valuenet(sme[i].vn) # set value network (remove this if you want to use average baseline)

	gd[i].attach_returnfunction(compute_return)  # set return function
	gd[i].attach_valuenet(sme[i].vn)

# initialize simulation interface
vrep = VrepInterfaze(port=int(runargv[-1]))

# initialize experience replay
reward_replay = Replay(NREPLAY,shape=(NTIMESTEP,1,1))
grad_replay = [Replay(NREPLAY,shape= (NTIMESTEP,CONNECTION.shape[0],NJPERM)) for i in range(NMODULE)]
weight_replay = [Replay(NREPLAY,shape=(1,CONNECTION.shape[0],NJPERM)) for i in range(NMODULE)]
observation_replay = [Replay(NREPLAY,shape=(NTIMESTEP,1,CONNECTION.shape[0])) for i in range(NMODULE)]

ama_selforg = np.array([1,1])
ma_selforg = np.array([0,0])

recorded_reward = np.zeros((NEPISODE,NTIMESTEP))
recorded_basis = np.zeros((NEPISODE,NTIMESTEP,CONNECTION.shape[0]))
recorded_torque = np.zeros((NEPISODE,NTIMESTEP,len(MORDER)*NJPERM))


# ------------------- start locomotion learning ---------------------
for i in range(NEPISODE):
	print('episode',i)

	# episode-wise setup
	prepose = vrep.get_robot_pose()

	wnoises = [agol[k].wnoise() for k in range(NMODULE)]
	for k in range(NMODULE):
		sme[k].explore(wnoises[MORDER[k]])
		#sme[k].explore(agol[k].wnoise())
		weight_replay[k].add(sme[MORDER[k]].mn.Wn)

	gaitstd = 0
	sumdelta = 0
	presumdelta = 1
	torques = np.zeros((18))
	for t in range(NTIMESTEP):

		# update network
		mbasis = []
		output_k = []
		forcings = []

		torques = 0.5+vrep.get_jointtorque()+0.5*torques
		jointangles = vrep.get_jointangle()
		zfoots = np.zeros((NMODULE,))
		jacz = np.zeros((NMODULE,3))

		for k in range(6):
			theta2 = jointangles[3*k+1]
			theta3 = jointangles[3*k+2]
			zfoots[k] = 0.07*np.cos(theta2-0.261799) - 0.12*np.cos(-theta3+theta2-0.1309)
			#jacz[k,1] = -0.07*np.sin(theta2-0.261799) + 0.12*np.sin(-theta3+theta2-0.1309)
			#jacz[k,2] = -0.12*np.sin(-theta3+theta2-0.1309)

			alpha1 = (15+23)*np.pi/180
			alpha2 = (111.5)*np.pi/180 + theta2
			jacz[k,1] = -0.07*np.sin(theta2-alpha1) - 0.12*np.cos(-theta3+theta2-alpha1-alpha2)
			jacz[k,2] = 0.12*np.cos(-theta3+theta2-alpha1-alpha2)
		#zfoots = zfoots - np.mean(zfoots)
		zfoots /= (1e-6+np.max(np.abs(zfoots)))
		#print(jacz[0])
		recorded_torque[i,t] = torques
		#torques = torques[1::3]
		output = []
		
		for k in range(NMODULE):	
			force = 2*(torques[[3*k+1,3*k+2]]-ma_selforg)/(1e-6+ama_selforg)

			force = -force#np.clip(-force,None,0)
			out, delta = sme[k].forward(sensory=force, scaling=0.1*float(runargv[2])/presumdelta,jacz=jacz[k])
			out = torch.clamp(out,-1,1)
			sumdelta += torch.sum(torch.abs(delta))
			output_k.append(out)
			output += numpy(output_k[k]).flatten().tolist()
			basis = sme[k].get_basis(torch=True)
			mbasis.append(torch.argmax(basis[0]).item())
			observation_replay[k].add(basis)
			recorded_basis[i,t] = numpy(basis)[0]

			# backpropagate output gradient
			torch.sum(output_k[-1]).backward() 
			grad_replay[k].add(sme[k].mn.W.grad)
			forcings.append(numpy(sme[k].zpg.forcing))

		

		gaitstd += np.std(np.array(mbasis))


		# update environment
		#print(output)
		vrep.set_robot_joint(np.array(output))
		vrep.update()

		# compute reward
		pose = vrep.get_robot_pose()
		dx = pose[0]-prepose[0]
		dy = pose[1]-prepose[1]
		reward = dx*np.cos(prepose[-1]) + dy*np.sin(prepose[-1]) #- 0.001*sumdelta
		recorded_reward[i,t] = reward
		prepose = deepcopy(pose)

		# append experience replay
		reward_replay.add(tensor([reward]).unsqueeze(0))
		
	
	istart = 0 if i<8 else i-8
	
	print(recorded_torque.shape)
	ama_selforg = np.array([np.mean(np.abs(recorded_torque[istart:i+1,:,1::3])),np.mean(np.abs(recorded_torque[istart:i+1,:,2::3]))])
	ma_selforg = np.array([np.mean((recorded_torque[istart:i+1,:,1::3])),np.mean((recorded_torque[istart:i+1,:,2::3]))])
	
	presumdelta = 0.9*presumdelta + 0.1*sumdelta/(NTIMESTEP*NMODULE)
	print('\tepisodic reward',torch.sum(reward_replay.data()[-1]).item())
	print('\tgait std',gaitstd,sumdelta)

	# update the network
	for k in range(NMODULE):
		gd[k].update(observation_replay[k].data(),reward_replay.data())
		agol[k].update(observation_replay[k].data(),weight_replay[k].data(),reward_replay.data(),grad_replay[k].data(),lrscale=0.1*float(runargv[1]))
		

	# balancing the network
	for k in range(NMODULE):
		if k != MORDER[k]:
			sme[k].set_weight(sme[MORDER[k]].mn.W.detach())

			agol[k].set_sigma(agol[MORDER[k]].get_sigma().detach())


np.save(PATH+'/reward', recorded_reward)
np.save(PATH+'/torque', recorded_torque)
np.save(PATH+'/basis', recorded_basis)
