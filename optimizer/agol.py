# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
from torch.distributions import Normal
import torch.nn as nn

# modular network
from optim import Optim

import matplotlib.pyplot as plt

# ------------------- configuration variables ---------------------
EPS = 1e-6
ENDINGCLIP = 5 # trow away n-last timestep

# ------------------- class AddedGradientOnlineLearning ---------------------
class AddedGradientOnlineLearning(Optim):
	# -------------------- constructor -----------------------
	# (private)

	def setup(self,config):
		self.vnet = None

		# initialize replay buffer
		self.__sigma = float(config["ACTOROPTIM"]["SIGMA"])
		self.__sigmas = {}

		for key in self.W.keys():
			if len(self.W[key].shape) == 1:
				self.__sigmas[key] = self.zeros(1,self.W[key].shape[0]) + self.__sigma
			else:
				self.__sigmas[key] = self.zeros(self.W[key].shape[0],self.W[key].shape[1]) + self.__sigma
		

		self.__min_grad = float(config["ACTOROPTIM"]["MINGRAD"])
		self.__lr = float(config["ACTOROPTIM"]["LR"])
		self.__adaptivegain = 1.0
		self.enables = None
		# reset everything before use
		self.reset()

	def attach_valuenet(self,vnet):
		self.vnet = vnet

	def attach_returnfunction(self,func):
		self.compute_return = func

	

	# ------------------------- update and learning ----------------------------
	# (public)

	def weighted_average(self,x,w,enable,dim=0,eps=0):
		return torch.sum(enable*x*w,dim,keepdim=True)/(eps+torch.sum(enable*w,dim,keepdim=True))

	
	def roger(self, rewards, predicted_rewards):

		safethreshold = torch.FloatTensor(np.array([0.2,0.2])).to(self.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # 2,2
		
		ch_mu = predicted_rewards[:,:,:,1:]


		#ch_sd = torch.std(predicted_rewards[:,:,:,1:],dim=(0,1,2),keepdim=True)
		#ch_sd = torch.std((rewards[:,:,:,1:]).pow(2),dim=(0,1,2),keepdim=True).sqrt()/3
		ch_sd = torch.mean((rewards[:,:,:,1:]-ch_mu).pow(2),dim=(0),keepdim=True).sqrt()
		ch_min = torch.clamp(ch_mu - 3*ch_sd,None,0.0)
		#print(ch_min.mean().item())
		reweight = rewards.clone()*0
		#reweight[:,:,:,1:] = torch.pow(torch.clamp(ch_min.unsqueeze(0)/safethreshold,-1,1),2) # closeness
		reweight[:,:,:,1:] = torch.pow(ch_min.unsqueeze(0)/safethreshold,2) # closeness

		rv = reweight[:,:,:,1:].sum(-1,keepdim=True)
		reweight[:,:,:,1:] = reweight[:,:,:,1:]*torch.clamp(rv,0,1)/(1e-6+rv)
		reweight[:,:,:,[0]] = 1-torch.clamp(rv,0,1)

		return reweight.detach()

	
	def update(self, advantages, exp_weight_replay, weights,grad_replay, state, newstate,lrscale=1,
		nepi=0,verbose=False,horizon=0,weightadjustment=False):
		

		if 1:#with torch.no_grad():	



			with torch.no_grad():
				param_update = {}	
				sigma_update = {}	

				# normalize advantage
				std = torch.sqrt(torch.mean(advantages.pow(2)))
				std_advantage = (advantages)/(std+EPS)
				std_advantage = torch.clamp(std_advantage,-3,3)
				
				# balance advantage
				corrected_advantage = std_advantage.clone()
				sumpos = corrected_advantage[(corrected_advantage) >= 0].abs().sum()
				sumneg = corrected_advantage[(corrected_advantage) < 0].abs().sum()
				negratio = 0.1*torch.clamp(sumpos/sumneg,0,1)
				corrected_advantage[corrected_advantage < 0] *= negratio

				# compute enable
				if self.enables is None:
					self.enables = torch.FloatTensor(np.zeros((advantages.shape[0],1,1,1))).to(self.device)

				if nepi >= advantages.shape[0]:
					self.enables += 1
				else:
					self.enables[-1-nepi:] += 1

			for key in exp_weight_replay.keys():



				


				# compute parameter update
				exploired_weights = exp_weight_replay[key].data()

				if weightadjustment:
					newweight = torch.randn_like(exp_weight_replay[key].data(), requires_grad=True)  # new weight to optimize
					with torch.no_grad():
						newweight = exploired_weights.clone()
					newweight = newweight.requires_grad_()

					# Define the optimizer (e.g., Adam)
					optimizer = torch.optim.Adam([newweight], lr=1e-3)
					
					dstate = newstate - state
					stateena = dstate.abs()
					for t in range(100):
						# Compute outputold and output using explored_weights and newweight
						# shape: [8, 70, 36]
						outputold = (exploired_weights * (stateena*state).unsqueeze(-1)).sum(-2).detach()
						output = (newweight * (stateena*newstate).unsqueeze(-1)).sum(-2)  # shape: [8, 70, 36]

						# Compute the error (loss function)
						loss = (outputold - output).pow(2).mean()
						
						# Backpropagation and optimization
						optimizer.zero_grad()  # Reset gradients
						loss.backward(retain_graph = True)  # Compute gradients
						optimizer.step()  # Update newweights
					print('\tweight_loss:',loss.item())
					adjweights_ = newweight.detach()
				else:
					adjweights_ = exploired_weights

				with torch.no_grad():
					
					weights_ = weights[key] if len(weights[key].shape) >= 2 else weights[key].unsqueeze(0) 
					exploration = (adjweights_-weights_)
					print('\tadj:',(adjweights_-exploired_weights).abs().max().item())
					#print(adjweights_.shape)
					#print(exploired_weights.shape)
					#print('\tds:',(newstate-state).abs().mean().item())

					rels = newstate.abs().unsqueeze(-1)

					update = (self.enables*rels*exploration)[:,horizon//2:corrected_advantage.shape[1]+horizon//2]* corrected_advantage
					#update = (self.enables*torch.abs(grad_replay[key].data())*exploration)* corrected_advantage
					
					dw = self.__adaptivegain*torch.mean(lrscale*self.__lr*update[:,:-ENDINGCLIP] ,dim=(0,1))#/torch.pow(self.__sigmas,2)
					#print(dw.shape,reweight.shape)
					#dw = torch.clamp(lrscale*self.__lr*0.1*1e-4*update,-self.__min_grad,self.__min_grad)
					

					
					dwnorm = torch.norm(dw[:4].flatten())
					if dwnorm >=  self.__min_grad:
						print('\tclip',dwnorm.item())
						dw[:4] = dw[:4]*(self.__min_grad/dwnorm).abs()
						#self.__adaptivegain *= 0.99

					dwnorm = torch.norm(dw[4:].flatten())
					if dwnorm >=  self.__min_grad:
						dw[4:] = dw[4:]*(self.__min_grad/dwnorm).abs()

					#print(print(((self.enables)*((torch.pow(exploration,2)-torch.pow(self.__sigmas[key],2))/torch.pow(self.__sigmas[key],1))).shape))
					# compute exploration update
					dsigma = std_advantage*(rels*(self.enables)*((torch.pow(exploration,2)-torch.pow(self.__sigmas[key],2))/torch.pow(self.__sigmas[key],1)))[:,horizon//2:corrected_advantage.shape[1]+horizon//2]
					#dsigma = std_advantage*(self.enables)*((torch.pow(exploration,2)-torch.pow(self.__sigmas[key],2))/torch.pow(self.__sigmas[key],1))
					
					#dsigma = self.__lr*1e-5*torch.sum(dsigma[:,:-ENDINGCLIP],dim=[0,1])
					dsigma = self.__lr*1e-5*torch.sum(dsigma[:,:-ENDINGCLIP],dim=[0,1])
					dsigma = torch.clamp(dsigma,-0.001,0.001)
					#print(self.__sigmas.mean().item())

					# apply the update
					with torch.no_grad():
						dw_ = dw if len(weights[key].shape) >= 2 else dw[0] 
						param_update[key] = dw_.detach()
						#self.W += (dw).detach()
						#self.__sigmas[key] = torch.clamp(self.__sigmas[key] + dsigma,0.01,self.__sigma)
						sigma_update[key] = (torch.clamp(self.__sigmas[key] + dsigma,0.01,self.__sigma) - self.__sigmas[key])

					if verbose:
						print('\ts',torch.mean(self.__sigmas[key][:4,2]).item())
						print('\tw',weights[key][:4,2].cpu().detach().numpy())
					else:
						print('\tw2',np.max(np.abs(weights[key][4:,2].cpu().detach().numpy())))
		return param_update, sigma_update

	# -------------------- apply noise -----------------------
	def wnoise(self):
		noises = {}
		for key in self.W.keys():
			self.dist = Normal(loc=0,scale=self.__sigmas[key])
			noise = self.dist.rsample()
			noises[key] = noise
		return noises

	# -------------------- set -----------------------
	def set_sigma(self,newsigma):
		with torch.no_grad():
			for key in self.__sigmas.keys():
				self.__sigmas[key] *= 0
				self.__sigmas[key] += newsigma

	# -------------------- get -----------------------
	def get_sigma(self):
		sigmas = {}
		with torch.no_grad():
			for key in self.__sigmas.keys():
				sigmas[key] = self.__sigmas[key].detach()
		return sigmas


	def add_sigma(self,dsig,start=0,end=-1,gain=1):
		with torch.no_grad():
			for key in self.__sigmas.keys():
				self.__sigmas[key][start:end] += gain*dsig[key][start:end].detach()


	
		




