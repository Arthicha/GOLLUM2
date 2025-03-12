
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy
import configparser

# math-related modules
import numpy as np # cpu array
import torch # cpu & gpu array
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# modular network
from modules.utils import HyperParams
from modules.torchNet import torchNet 
from modules.centralpattern import SequentialCentralPatternGenerator
from modules.basis import BasisNetwork 
from modules.motor import MotorNetwork 

#plot
import matplotlib.pyplot as plt

# ------------------- configuration variables ---------------------

EPSILON = 1e-6 # a very small value

# ------------------- class SDN ---------------------


def kmeans_clustering(batch, cluster_init, max_iters=100, tol=1e-4):
	"""
	Perform k-means clustering.
	
	Args:
		batch (torch.Tensor): Input data tensor of shape (batch, dim).
		n_clusters (int): Number of clusters.
		max_iters (int): Maximum number of iterations to run.
		tol (float): Tolerance for convergence based on cluster center changes.
	
	Returns:
		cluster_centers (torch.Tensor): Cluster centers of shape (n_clusters, dim).
		probabilities (torch.Tensor): Soft probabilities for each sample in the batch.
	"""
	# Initialize cluster centers randomly from the batch
	n_clusters = cluster_init.shape[0]
	cluster_centers = cluster_init#batch[torch.randint(0, batch.size(0), (n_clusters,))]

	for iteration in range(max_iters):
		# Calculate pairwise distances between each point and each cluster center
		distances = torch.cdist(batch, cluster_centers)
		
		# Find the closest cluster center for each data point (assign clusters)
		cluster_assignments = torch.argmin(distances, dim=-1)  # shape: (batch,)
		
		# Compute probabilities for each data point based on distance to clusters
		# Here we use softmax to make it a probability distribution
		#probabilities = F.softmax(-distances, dim=-1)  # shape: (batch, n_clusters)
		
		# Compute new cluster centers as the mean of the assigned points
		new_cluster_centers = torch.stack([
			batch[cluster_assignments == i].mean(dim=0) if (cluster_assignments == i).sum() > 0 else cluster_centers[i]
			for i in range(n_clusters)
		])
		
		# Check for convergence (if the cluster centers don't change significantly)
		if torch.all(torch.abs(new_cluster_centers - cluster_centers) < tol):
			break
		
		cluster_centers = new_cluster_centers
	
	return cluster_centers, cluster_assignments


def compute_probabilities(batch, cluster_centers):
	"""
	Compute the probability distribution for each data point to each cluster.
	
	Args:
		batch (torch.Tensor): Input data tensor of shape (batch, dim).
		cluster_centers (torch.Tensor): Cluster centers of shape (n_clusters, dim).
	
	Returns:
		probabilities (torch.Tensor): Soft probabilities for each sample in the batch,
									  with shape (batch, n_clusters).
	"""
	# Calculate pairwise distances between each point and each cluster center
	distances = torch.cdist(batch, cluster_centers)  # shape: (batch, n_clusters)
	
	# Compute probabilities using softmax on the negative distances (closer clusters should have higher probability)
	probabilities = F.softmax(-distances, dim=-1)  # shape: (batch, n_clusters)
	for i in range(5):
		probabilities = torch.clamp(probabilities - 0.1*probabilities.mean(-1,keepdim=True),0,None)
	probabilities = F.softmax(-distances, dim=-1)
	return probabilities



class GaussianMixtureModel(nn.Module):
	def __init__(self, n_clusters, n_features):
		super(GaussianMixtureModel, self).__init__()
		self.n_clusters = n_clusters
		self.n_features = n_features
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.first =  True

		# Initialize parameters
		self.mu = nn.Parameter(torch.randn(n_clusters, n_features, device=self.device))  # Means
		self.sigma = nn.Parameter(torch.ones(n_clusters, n_features, device=self.device))  # Variance


	def gaussian_pdf(self, x, mu, sigma):
		"""Compute Gaussian Probability Density Function."""
		const = 1.0 / (torch.sqrt(2 * torch.pi * sigma.sum(-1)))
		exp_term = torch.exp((-0.5 * ((x - mu) ** 2) / sigma).sum(-1))
		return torch.clamp(const * exp_term,0,1)


	def update(self,input_):
		# input_ = (Nbatch,Ndim)
		if self.first:
			random_indices = torch.randperm(input_.shape[0])[:self.n_clusters]
			mu_init = input_[random_indices].to(self.device)  # Select and move to device
			with torch.no_grad():
				self.mu.data = mu_init
			self.first = False

		with torch.no_grad():
			k_cluster, assignment = kmeans_clustering(input_,self.mu,max_iters=10)
			
			for k in range(0,self.n_clusters):
				
				#sys.exit()
				indices = (assignment == k)
				if indices.shape[0] != 0:
					sigma = (input_[indices] - k_cluster[assignment[k]]).pow(2).sum(-1).mean().sqrt()
					gain = 1.0
				else:
					sigma = 0.01
					gain = 0.0

				self.mu.data[k] = k_cluster[k]
				self.sigma.data[k] = torch.clamp(sigma,0.01,100.0)

	def is_enough(self,data):
		# data = (#batch,#dim)
		criteria = (data.abs().sum(-1) > 0).type(int).sum()
		if criteria > 2*self.n_clusters:
			return True
		else:
			return False


	def forward(self, input_):
		"""Compute responsibilities (γ), the probability of each point belonging to each cluster."""
		
		x = input_.unsqueeze(1)  # Shape: (N, 1, D)
		mu = self.mu.unsqueeze(0)  # Shape: (1, K, D)
		sigma = self.sigma.unsqueeze(0)  # Shape: (1, K, D)

		probs = self.gaussian_pdf(x, mu, sigma)  # (N, K, D)
		return probs

	def log_likelihood(self, x, gamma):
		"""Compute the log-likelihood of the Gaussian Mixture Model."""
		# Calculate the probability of each sample under each Gaussian component
		log_probs = torch.log(gamma.sum(dim=1) + 1e-8)  # Avoid log(0) with epsilon
		return log_probs.sum()
