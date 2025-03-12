
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

class FullyConnected(nn.Module):
	'''
	Sequential Motion Executor : Actor Network
	Parameters:
		connection/transition matrix from 'connection' 
		hyperparameter from a .init file at 'configfile'
	'''

	# ---------------------- constructor ------------------------ 
	def __init__(self, nin = 1, nout = 1,nlayer = 1, nneuron = 64,
		zeroparams = False,last_actfunc=lambda x: x,stack=False,last_bias = True,embedding_from=None):

		super(FullyConnected, self).__init__()

		self.embedding_id = embedding_from
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.last_actfunc = last_actfunc
		self.stack = stack
		self.last_name = 'layers.'+str(nlayer-1)
		self.layers = nn.ModuleList()
		self.nfc = 8
		
		#self.layers.append(ScaleLayer(nin, device=self.device))
		if nlayer > 1:
			for i in range(nlayer - 1):
				self.layers.append(nn.Linear(nin if i == 0 else nneuron, nneuron,bias=last_bias).to(self.device))
			self.layers.append(nn.Linear(nneuron+4+self.nfc if self.stack else nneuron, nout,bias=last_bias).to(self.device))
		else:
			self.layers.append(nn.Linear(nin, nout,bias=last_bias).to(self.device))

		if zeroparams:
			self.apply(self.zero_init)
		else:
			self.apply(self.xavier_init)

	# ---------------------- debugging   ------------------------ 

	def forward(self, input_,mask=None):

		nfc = self.nfc
		x0 = input_[:,:,:,-4-nfc:] if (len(input_.shape) == 4) else input_[:,-4-nfc:]
		x = input_ 
		embedding = None
		for i in range(len(self.layers)):

			if self.embedding_id == i:
				x = self.layers[i](x)
				#x = nn.functional.tanh(x)
				#embedding = torch.zeros_like(embedding_)
				#embedding.scatter_(-1, torch.argmax(embedding,dim=-1, keepdim=True), 1)
				x = torch.nn.functional.softmax(x,dim=-1)
				for k in range(10):
					x = torch.clamp(x - x.mean(-1,keepdim=True),1e-4,None)
					x = x / x.max(-1,keepdim=True).values
				embedding = x.clone()

			elif i == len(self.layers)-1:
				x = (self.layers[i](x)) if not self.stack else (self.layers[i](torch.concat([x,x0],dim=-1)))
				x = self.last_actfunc(x)
			else:
				x = (self.layers[i](x))
				x = nn.functional.tanh(x)

		return x, embedding

	def explore(self,noises,lastlayer_gain=1,firstlayer_bias = 0, firstlayer_gain=1):
		with torch.no_grad():
			for name, param in self.named_parameters():

				if self.last_name in name:
					gain = lastlayer_gain 
				else:
					gain = 1

				if param.shape == noises[name].shape:
					param.data += gain*noises[name]
				else:
					param.data += gain*noises[name][0]


	def zero_init(self,m):
		if isinstance(m, nn.Linear):  # Check if the module is a linear layer
			nn.init.xavier_uniform_(m.weight,gain=1.4)
			#nn.init.zeros_(m.weight)  # Initialize weights to zero
			#if m.bias is not None:
			#	nn.init.zeros_(m.bias)  # Initialize biases to zero
		
		with torch.no_grad():
			for name, param in self.named_parameters():
				#if self.last_name in name:
				param.data *= 0.1
	

	def xavier_init(self, layer):
		if isinstance(layer, nn.Linear):
			nn.init.xavier_uniform_(layer.weight,gain=1.4)
			#with torch.no_grad():
			#	layer.weight *= 0.1
			if layer.bias is not None:
				nn.init.zeros_(layer.bias)



class EncoderFC(nn.Module):
	""" Fully Connected Encoder """
	def __init__(self, input_dim=784, hidden_dim=512, latent_dim=64):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, latent_dim)

	def forward(self, x):
		x = F.tanh(self.fc1(x))
		z_e = self.fc2(x)  # Output latent representation
		return z_e

class VectorQuantizerFC(nn.Module):
	""" Fully Connected Vector Quantization Layer """
	def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
		super().__init__()
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.commitment_cost = commitment_cost

		# additional gains (for continual learning)
		self.enables = nn.Parameter(torch.ones(num_embeddings, 1)*0)
		self.trainables = nn.Parameter(torch.ones(num_embeddings, 1)*0)

		# Codebook (Embedding table)
		self.embedding = nn.Embedding(num_embeddings, embedding_dim)
		self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)  # Initialize

	def forward(self, z_e):
		# Compute L2 distance between encoded features and embeddings
		distances = (torch.sum(z_e ** 2, dim=1, keepdim=True) 
					 + torch.sum(self.embedding.weight ** 2, dim=1)
					 - 2 * torch.matmul(z_e, self.embedding.weight.t()))  # (Batch, num_embeddings)

		#encoding_indices = torch.argmin(distances, dim=1)  # Closest embedding index
		#z_q = self.embedding(encoding_indices)  # Retrieve quantized vector
		temperature = 0.01
		encoding_indices = F.softmax(-distances / temperature, dim=-1)
		z_q = torch.matmul(encoding_indices, self.embedding.weight)

		# Compute loss (commitment & embedding loss)
		loss = F.mse_loss(z_q.detach(), z_e) + self.commitment_cost * F.mse_loss(z_e.detach(), z_q)

		# Straight-through estimator (gradient trick)
		z_q = z_e + (z_q - z_e).detach()

		#latent_smoothness_loss = F.mse_loss(z_e[1:], z_e[:-1])  # Enforce similarity across timesteps

		return z_q, encoding_indices, loss# + 2*latent_smoothness_loss

	def clear_enabletrainable(self,encoding_indices,threshold=1e-2):
		pass


class DecoderFC(nn.Module):
	""" Fully Connected Decoder """
	def __init__(self, latent_dim=64, hidden_dim=512, output_dim=784):
		super().__init__()
		self.fc1 = nn.Linear(latent_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, z_q):
		x = F.tanh(self.fc1(z_q))
		x_recon = (self.fc2(x))  # Ensure output is in [0,1] range
		return x_recon

class VQVAE_FC(nn.Module):
	""" Full VQ-VAE model (Fully Connected) """
	def __init__(self, input_dim=784, hidden_dim=512, latent_dim=64, num_embeddings=512, commitment_cost=0.25):
		super().__init__()
		self.num_embeddings = num_embeddings
		self.encoder = EncoderFC(input_dim, hidden_dim, latent_dim)
		self.vq_layer = VectorQuantizerFC(num_embeddings, latent_dim, commitment_cost)
		self.decoder = DecoderFC(latent_dim, hidden_dim, input_dim+3)

	def forward(self, x):
		z_e = self.encoder(x)
		z_q, encodings, vq_loss = self.vq_layer(z_e)
		x_recon = self.decoder(z_q)
		return encodings, x_recon, vq_loss

	def encoding2embedding(self,encoding):
		embedding = encoding#F.one_hot(encoding,self.num_embeddings)
		return embedding


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

	distances = torch.cdist(batch, cluster_centers)
	cluster_assignments = torch.argmin(distances,dim=-1)

	'''
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
		
		cluster_centers = new_cluster_centers'''
	
	return cluster_centers, cluster_assignments

def compute_intra_cluster_distance(batch, centroids, assignments):
	"""
	Compute the intra-cluster distance using tensor operations (fully differentiable).

	Args:
		batch (torch.Tensor): Data points of shape (batch_size, n_features).
		centroids (torch.Tensor): Cluster centroids of shape (n_clusters, n_features).
		assignments (torch.Tensor): Indices of assigned clusters for each data point (batch_size).

	Returns:
		torch.Tensor: Intra-cluster distance, shape (n_clusters,)
	"""
	# Compute distances between each data point and each centroid
	distances = torch.cdist(batch, centroids)  # (batch_size, n_clusters)

	# Select only the distances for the assigned clusters
	intra_distances = torch.gather(distances, dim=1, index=assignments.unsqueeze(1)).squeeze(1)  # (batch_size,)

	# Sum intra-cluster distances for each cluster
	intra_cluster_dist = torch.zeros(centroids.size(0), device=batch.device).scatter_add_(
		0, assignments, intra_distances
	)  # (n_clusters,)

	return intra_cluster_dist

def compute_inter_cluster_distance(centroids):
	"""
	Compute the inter-cluster distance (fully differentiable).

	Args:
		centroids (torch.Tensor): Cluster centroids of shape (n_clusters, n_features).

	Returns:
		torch.Tensor: Pairwise inter-cluster distance matrix, shape (n_clusters, n_clusters).
	"""
	inter_distances = torch.cdist(centroids, centroids)  # (n_clusters, n_clusters)
	#inter_distances.fill_diagonal_(torch.inf)  # Avoid self-distance
	return inter_distances

def objective_function(batch, centroids, assignments, alpha=1.0, beta=1.0):
	"""
	Objective function to maximize inter-cluster distance and minimize intra-cluster distance.

	Args:
		batch (torch.Tensor): Data points of shape (batch_size, n_features).
		centroids (torch.Tensor): Cluster centroids of shape (n_clusters, n_features).
		assignments (torch.Tensor): Indices of assigned clusters for each data point (batch_size).
		alpha (float): Weight for minimizing intra-cluster distance.
		beta (float): Weight for maximizing inter-cluster distance.

	Returns:
		torch.Tensor: The objective value.
	"""
	# Compute intra-cluster distances (minimize)
	intra_dist = compute_intra_cluster_distance(batch, centroids, assignments)
	
	# Compute inter-cluster distances (maximize)
	inter_dist = compute_inter_cluster_distance(centroids)
	
	# Minimize intra-cluster distance (summed intra-cluster distance)
	intra_loss = intra_dist.sum()
	
	# Maximize inter-cluster distance (sum of minimum inter-cluster distances)
	inter_loss = inter_dist.min(dim=1)[0].sum()  # sum of minimum pairwise inter-distances
	
	# The objective is to minimize intra-loss and maximize inter-loss
	objective = alpha * intra_loss - beta * inter_loss
	
	return objective


class GaussianMixtureModel(nn.Module):
	def __init__(self, n_clusters, n_features):
		super(GaussianMixtureModel, self).__init__()
		self.n_clusters = n_clusters
		self.n_features = n_features
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.first =  True

		# Initialize parameters
		self.gain = nn.Parameter(torch.ones(n_clusters, 1, device=self.device))
		self.mu = nn.Parameter(torch.randn(n_clusters, n_features, device=self.device),requires_grad=True)  # Means
		self.sigma = nn.Parameter(torch.ones(n_clusters, n_features, device=self.device))  # Variance
		self.optimizer = torch.optim.Adam([self.mu], lr=0.001)

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

		

		for t in range(10):

			k_cluster, assignment = kmeans_clustering(input_,self.mu,max_iters=1)
			with torch.no_grad():
				self.mu.data = k_cluster


			self.optimizer.zero_grad()


			# Compute intra-cluster distances (minimize)
			intra_dist = compute_intra_cluster_distance(input_, self.mu, assignment)
			# Compute inter-cluster distances (maximize)
			inter_dist = compute_inter_cluster_distance(self.mu)
			# Minimize intra-cluster distance (summed intra-cluster distance)
			intra_loss = intra_dist.sum()
			# Maximize inter-cluster distance (sum of minimum inter-cluster distances)
			inter_loss = inter_dist.min(dim=1)[0].sum()  # sum of minimum pairwise inter-distances
			# The objective is to minimize intra-loss and maximize inter-loss
			objective = 1 * intra_loss - 1 * inter_loss
			objective.backward()
			
			self.optimizer.step()
		print(objective.item())


			
			
		for k in range(0,self.n_clusters):
			
			#sys.exit()
			indices = (assignment == k)
			if indices.shape[0] != 0:
				sigma = (input_[indices] - k_cluster[assignment[k]]).pow(2).mean(0).sqrt()
			else:
				sigma = 0.01

			#self.mu.data[k] = k_cluster[k]
			#print(sigma.shape)
			self.sigma.data[k] = torch.clamp(sigma,0.01,100.0)
			minid = torch.argmin((input_[indices] - self.mu.data[assignment[k]]).pow(2).sum(-1))
			self.gain.data[k] = self.gaussian_pdf(input_[indices][minid],self.mu.data[assignment[k]],self.sigma.data[assignment[k]])
			#print(self.gain.data[k])



	def is_enough(self,data):
		# data = (#batch,#dim)
		criteria = (data.abs().sum(-1) > 0).type(torch.int).sum()
		if criteria > (3*self.n_clusters):
			return True, criteria
		else:
			return False, criteria


	def forward(self, input_):
		"""Compute responsibilities (γ), the probability of each point belonging to each cluster."""
		
		x = input_.unsqueeze(1)  # Shape: (N, 1, D)
		mu = self.mu.unsqueeze(0)  # Shape: (1, K, D)
		sigma = self.sigma.unsqueeze(0)  # Shape: (1, K, D)
		probs = self.gaussian_pdf(x, mu, sigma)#(1e-8+self.gain.data.T)  # (N, K, D)
		return probs

	def log_likelihood(self, x, gamma):
		"""Compute the log-likelihood of the Gaussian Mixture Model."""
		# Calculate the probability of each sample under each Gaussian component
		log_probs = torch.log(gamma.sum(dim=1) + 1e-8)  # Avoid log(0) with epsilon
		return log_probs.sum()
