# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

# modular network
from optim import Optim

# experience replay
from utils.utils import TorchReplay as Replay

import matplotlib.pyplot as plt


# ------------------- configuration variables ---------------------
EPSILON = 1e-6
PLOT = False # plot actual return vs predicted value
# ------------------- class GD ---------------------

class GradientDescent(Optim):

	
	# -------------------- constructor -----------------------
	# (private)

	def setup(self,config):

		self.lr_gain = 1
		self.__lr = float(config["CRITICOPTIM"]["LR"])
		self.iteration = int(config["CRITICOPTIM"]["ITERATION"])
		

		# reset everything before use
		self.reset()

	def attach_valuenet(self,vnet):
		self.vnet = vnet
		self.critic_optimizer = optim.SGD(self.vnet.parameters(), lr=self.__lr)

	def attach_returnfunction(self,func):
		self.compute_return = func

	
	
	# ------------------------- update and learning ----------------------------
	# (public)

	
	def update(self,states, rewards,nepi=0, max_grad_norm = 0.5, update_withtd=False):

		for i in range(self.iteration if nepi > 2 else 100):



			self.critic_optimizer.zero_grad()

			enables = torch.FloatTensor(np.zeros((rewards.shape[0],1,1,1))).to(self.device)
			if nepi >= rewards.shape[0]:
				enables += 1
			else:
				enables[-1-nepi:] += 1

			values, _ = self.vnet(states)
				

			predicted_returns = self.compute_return(values)
			returns = self.compute_return(rewards)

			#predicted_returns = values
			#returns = (rewards + 0.95*next_values)
			loss = torch.sum(enables*torch.pow(returns-predicted_returns,2))/torch.sum(enables)
			
			
			loss.backward()

			nn.utils.clip_grad_norm_(self.vnet.parameters(), max_grad_norm)
			self.critic_optimizer.step()


		print("\tvalue loss:",loss.item())



		'''
		if PLOT:
			predicted_value = self.compute_return(self.vnet(states))
			plt.clf()
			plt.plot(np.transpose(self.numpy(values[:,:,0,0])),c='tab:blue')
			plt.plot(np.transpose(self.numpy(predicted_value[:,:,0,0])),c='tab:orange')
			plt.plot(np.transpose(self.numpy(torch.mean(values[:,:,0,0],dim=0))),c='tab:red')
			plt.savefig('value.jpg')
		'''

	
		




