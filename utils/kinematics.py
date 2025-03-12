
# ------------------- import modules ---------------------

# standard modules
import time, sys, os
from copy import deepcopy

# math-related modules
import numpy as np # cpu array
import torch # pytorch
import matplotlib.pyplot as plt
import roboticstoolbox as rtb

# ------------------- configuration variables ---------------------


# ------------------- class TorchReplay ---------------------
class MorfKin:

	l1 = 0.07
	l2 = 0.12
	offset1 = 0.261799
	offset2 = 0.1309


	def __init__(self):

		a1 = 0.0
		d1 = 0  # Assuming no offset along the z-axis
		alpha1 = 0#np.pi/2  # 90 degrees, as assumed earlier

		# Link 2: l1link23 -> l1foot
		a2 = 0.05  # Assume an arbitrary length for demonstration (you can replace this with real value)
		d2 = 0.0
		alpha2 = np.pi/2  # -90 degrees, as assumed earlier

		# Link 3: l1foot -> final body
		a3 = 0.075  # Again, replace this with actual value
		d3 = 0
		alpha3 = -np.pi  # 90 degrees

		# Link 3: l1foot -> final body
		a4 = 0.12  # Again, replace this with actual value
		d4 = 0
		alpha4 = 0  # 90 degrees



		# Define the robot using the DH parameters
		L1 = rtb.RevoluteMDH(a=a1, d=d1, alpha=alpha1, offset=0)  # first link
		L2 = rtb.RevoluteMDH(a=a2, d=d2, alpha=alpha2, offset=np.pi/2)  # second link
		L3 = rtb.RevoluteMDH(a=a3, d=d3, alpha=alpha3, offset=-np.pi)  # third link
		L4 = rtb.RevoluteMDH(a=a4, d=d4, alpha=alpha4, offset=0)  # third link

		# Create the robotic arm with these links
		self.robot = rtb.SerialLink([L1, L2, L3,L4], name="ThreeDOFRobot")
		q = np.array([0,-0.4,-0.4,0])
		#self.robot.plot(q)
		#time.sleep(5)
		#sys.exit()


	def get_zfoot(self,theta1,theta2):
		zfoots = self.l1*np.cos(theta1-self.offset1) - self.l2*np.cos(-theta2+theta1-self.offset2)
		# zfoots[k] = 0.07*np.cos(theta2-0.261799) - 0.12*np.cos(-theta3+theta2-0.1309)
		zfootss = deepcopy(zfoots)
		for k in range(theta1.shape[0]):
			q = np.array([0, theta1[k], -theta2[k],0])  # Example joint angles (in radians)
			T = self.robot.fkine(q)  # Compute the forward kinematics
			
			zfoots[k] = T.t[2]

		return zfoots
	def get_jacz(self,theta1,theta2):
		
		# old 
		#jacz[k,1] = -0.07*np.sin(theta2-0.261799) + 0.12*np.sin(-theta3+theta2-0.1309)
		#jacz[k,2] = -0.12*np.sin(-theta3+theta2-0.1309)

		# new
		alpha1 = self.offset1 + 0.4 
		alpha2 = (111.5)*np.pi/180 + theta2
		jacz_theta1 = -self.l1*np.sin(theta2-alpha1) - self.l2*np.cos(-theta3+theta2-alpha1-alpha2)
		jacz_theta2 = self.l2*np.cos(-theta3+theta2-alpha1-alpha2)
		#alpha1 = (15+23)*np.pi/180
		#alpha2 = (111.5)*np.pi/180 + theta2
		#jacz[k,1] = -0.07*np.sin(theta2-alpha1) - 0.12*np.cos(-theta3+theta2-alpha1-alpha2)
		#jacz[k,2] = 0.12*np.cos(-theta3+theta2-alpha1-alpha2)



		return jacz_theta1, jacz_theta2


	def get_jacs(self,theta1,theta2):

		J = self.jacobian(0,theta1,-theta2,0,0)
		Jinv = deepcopy(J)*0+1

		for i in range(J.shape[0]):
			Jinv = np.linalg.inv(J)


		return J, Jinv



	def jacobian(self, q1, q2, q3, roll, pitch):
		
		J = []

		for k in range(q2.shape[0]):
			q = np.array([q1, q2[k], q3[k],0])  # Example joint angles (in radians)
			J_ = self.robot.jacobe(q,half='trans')[:,:-1]  # Compute the forward kinematics
			#J_ = J_[[2,0,1]]
			J.append(J_)

		J = np.array(J)

		return J



class BoneKin:

	l1 = 0.35
	l2 = 0.35
	offset1 = 0.261799
	offset2 = 0.1309
	hiplength = 0.11973
	bodywidth = 0.144
	bodylength = 0.657


	def __init__(self):
		pass


	def get_zfoot(self,theta1,theta2):
		q1 = 0
		q2 = theta1
		q3 = theta2
		roll = 0
		pitch = 0
		l1 = self.l1 
		l2 = self.l2
		a = self.hiplength*np.array([-1, 1, -1, 1])
		l = self.bodylength*np.array([1, 1, -1, -1])
		w = self.bodywidth*np.array([-1, 1, -1, 1])

		# Compute z
		sigma_5 = np.sin(q1) * np.sin(roll) - np.cos(pitch) * np.cos(q1) * np.cos(roll)
		sigma_3 = np.cos(q2) * sigma_5 + np.cos(roll) * np.sin(pitch) * np.sin(q2)
		zfoots = a * (np.cos(q1) * np.sin(roll) + np.cos(pitch) * np.cos(roll) * np.sin(q1)) + l2 * (np.cos(q3) * sigma_3 - np.sin(q3) * (np.sin(q2) * sigma_5 - np.cos(q2) * np.cos(roll) * np.sin(pitch))) + (w * np.sin(roll)) / 2 + l1 * sigma_3 - (l * np.cos(roll) * np.sin(pitch)) / 2
		
		return zfoots

	def get_jacz(self,theta1,theta2):

		J = self.jacobian(0,theta1,theta2,0,0)
		for i in range(J.shape[0]):
			J = np.linalg.inv(J)

		jacz_theta1 = J[:,2,1]
		jacz_theta2 = J[:,2,2]

		return jacz_theta1, jacz_theta2

	def get_jacs(self,theta1,theta2):

		J = self.jacobian(0,theta1,theta2,0,0)
		Jinv = deepcopy(J)*0+1

		for i in range(J.shape[0]):
			Jinv = np.linalg.inv(J)


		return J, Jinv

	def jacobian(self, q1, q2, q3, roll, pitch):
		l1 = self.l1
		l2 = self.l2
		a = self.hiplength*np.array([-1, 1, -1, 1])
		s = np.sin
		c = np.cos

		# Precompute common terms
		sigma1 = c(q1) * c(roll) - c(pitch) * s(q1) * s(roll)
		sigma2 = c(q1) * s(roll) + c(pitch) * c(roll) * s(q1)
		sigma6 = s(q2) * (s(q1) * s(roll) - c(pitch) * c(q1) * c(roll)) - c(q2) * c(roll) * s(pitch)
		sigma7 = s(q2) * (c(q1) * s(roll) + c(pitch) * c(roll) * s(q1)) + c(q2) * s(pitch) * s(roll)
		sigma8 = c(pitch) * c(q2) - c(q1) * s(pitch) * s(q2)
		sigma9 = s(q1) * s(roll) - c(pitch) * c(q1) * c(roll)
		sigma10 = c(roll) * s(q1) + c(pitch) * c(q1) * s(roll)

		sigma3 = l2 * (c(q3) * sigma6 + s(q3) * (c(q2) * sigma9 + c(roll) * s(pitch) * s(q2)))
		sigma4 = l2 * (c(q3) * sigma7 + s(q3) * (c(q2) * sigma10 - s(pitch) * s(q2) * s(roll)))
		sigma5 = l2 * (c(q3) * sigma8 - s(q3) * (c(pitch) * s(q2) + c(q1) * c(q2) * s(pitch)))

		# Compute the Jacobian tensor
		J11 = l2 * (c(q2) * c(q3) * s(pitch) * s(q1) - s(pitch) * s(q1) * s(q2) * s(q3)) + a * c(q1) * s(pitch) + l1 * c(q2) * s(pitch) * s(q1)
		J12 = -l1 * sigma8 - sigma5
		J13 = -sigma5

		J21 = l2 * (c(q2) * c(q3) * sigma1 - s(q2) * s(q3) * sigma1) - a * sigma10 + l1 * c(q2) * sigma1
		J22 = -sigma4 - l1 * sigma7
		J23 = -sigma4

		J31 = l2 * (c(q2) * c(q3) * sigma2 - s(q2) * s(q3) * sigma2) - a * sigma9 + l1 * c(q2) * sigma2
		J32 = -sigma3 - l1 * sigma6
		J33 = -sigma3

		# Assemble the Jacobian matrix
		jacobian = np.array([
			[J11, J12, J13],
			[J21, J22, J23],
			[J31, J32, J33]
		])

		J = np.transpose(jacobian,(2,0,1))
		return J



	


class Kinematics:

	def __init__(self,dhparam):
		self.dh_func = dhparam

	def dh_transform(self,theta, d, a, alpha):
		"""Compute the DH transformation matrix."""
		ct, st = np.cos(theta), np.sin(theta)
		ca, sa = np.cos(alpha), np.sin(alpha)
		return np.array([
			[ct, -st * ca,  st * sa, a * ct],
			[st,  ct * ca, -ct * sa, a * st],
			[0,       sa,      ca,     d],
			[0,        0,       0,     1]
		])

	def forward_kinematics(self,q):
		"""Compute forward kinematics given joint angles q = [q1, q2, q3]."""
		# DH Parameters: [theta, d, a, alpha]
		dh_params = self.dh_func(q)
		
		T = np.eye(4)  # Identity matrix (base frame)
		T_matrices = []
		joint_positions = [np.zeros(3)]  # Base at origin
		z_axes = [np.array([0, 0, 1])]  # z-axis for revolute joints
		
		for params in dh_params:
			T_i = self.dh_transform(*params)
			T = T @ T_i  # Accumulate transformations
			T_matrices.append(T.copy())
			joint_positions.append(T[:3, 3])
			z_axes.append(T[:3, 2])
		
		end_effector_pos = joint_positions[-1]
		return end_effector_pos#, T_matrices, joint_positions, z_axes

	def compute_jacobian(self,q):
		"""Compute the Jacobian for the given joint angles q."""
		_, _, joint_positions, z_axes = self.forward_kinematics(q)
		
		J_v = np.zeros((3, 3))  # Linear velocity Jacobian
		J_w = np.zeros((3, 3))  # Angular velocity Jacobian
		
		p_foot = joint_positions[-1]
		
		for i in range(3):
			J_v[:, i] = np.cross(z_axes[i], p_foot - joint_positions[i])
			J_w[:, i] = z_axes[i]
		
		J = np.vstack((J_v, J_w))
		return J






		


		


	
