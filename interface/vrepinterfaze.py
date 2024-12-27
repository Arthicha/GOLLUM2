# interface/__init__.py

'''
Class: VrepInterfaze
created by: arthicha srisuchinnawong
e-mail: arsri21@student.sdu.dk
date: 1 August 2022

This class provide easy direct interface between python3 and CoppeliaSim
aiming mainly for reinforcement learning of a hexapod robot (MORF) 

IMPORTANT: please follow this tutorial first: https://www.youtube.com/watch?v=SQont-mTnfM
NOTE THAT: this class use numpy array
'''

# ------------------- import modules ---------------------

# standard modules
import time, sys, os

# vrep interface module 
import sim, simConst

# math-related modules
import numpy as np # cpu array

# ------------------- configuration variables ---------------------
HZ = 30
MODE = sim.simx_opmode_oneshot_wait # regular operation mode
MODEF = sim.simx_opmode_streaming # fast operation mode
NLEG = 6 

# ------------------- class VrepInterface ---------------------

class VrepInterfaze:

	__joint_name = ['TC','CF','FT'] # predefined joint names
	__offset = np.array([0,-0.4,-0.4]) # joint offset

	
	
	# ---------------------- constructor ------------------------ 
	def __init__(self,ip='127.0.0.1',port=19997):

		# initialize interface
		sim.simxFinish(-1) # close all opened connections
		self.__ip = ip
		self.__port = port
		self.__clientID = sim.simxStart(self.__ip,self.__port,True,True,5000,5) # Connect to CoppeliaSim

		# robot handle
		_, self.__robot_ref_handle = sim.simxGetObjectHandle(self.__clientID,'morf_ref',MODE)
		_, self.__floor_handle = sim.simxGetObjectHandle(self.__clientID,'floor',MODE)

		self.__joint_handle = np.zeros((NLEG,len(self.__joint_name))).astype(int) # joint handle (leg l, joint j)
		self.__target_positions = np.zeros((NLEG,len(self.__joint_name))).astype(float) # joint target position (leg l, joint j)
		for l in range(0,NLEG):
			for j in range(0,len(self.__joint_name)): 
				_, self.__joint_handle[l,j] = sim.simxGetObjectHandle(self.__clientID,self.__joint_name[j]+str(l), MODE)
			
		# start the simulation
		self.reset()
		sim.simxSynchronous(self.__clientID,True)
		sim.simxStartSimulation(self.__clientID,MODE)
		print("INFO: VrepInterfaze is initialized successfully.")

	# ---------------------- actuation  ------------------------ 
	def set_robot_joint(self,target_pos):
		target_pos = target_pos.reshape((NLEG,len(self.__joint_name)))
		target_pos += self.__offset # offset
		self.__target_positions = target_pos

	def set_zero(self):
		self.set_robot_joint(np.zeros((NLEG*len(self.__joint_name))))

	# ---------------------- get simulation data  ------------------------
	def get_robot_pose(self):
		status, posarray = sim.simxGetObjectPosition(self.__clientID,self.__robot_ref_handle,self.__floor_handle,MODEF)
		status, orienarray = sim.simxGetObjectOrientation(self.__clientID,self.__robot_ref_handle,self.__floor_handle,MODEF)
		pose = np.array([posarray[0],posarray[1],posarray[2],orienarray[0],orienarray[1],orienarray[2]])
		return self.__redo(status,pose,self.get_robot_pose)

	def get_jointangle(self):
		positions = np.zeros((NLEG*len(self.__joint_name)))
		for l in range(0,NLEG):
			for j in range(0,len(self.__joint_name)):
				status, positions[len(self.__joint_name)*l+j] = sim.simxGetJointPosition(self.__clientID,self.__joint_handle[l,j],MODEF)
		return self.__redo(status,positions,self.get_jointangle)


	def get_jointtorque(self):
		torques = np.zeros((NLEG*len(self.__joint_name)))
		for l in range(0,NLEG):
			for j in range(0,len(self.__joint_name)):
				status, torques[len(self.__joint_name)*l+j] = sim.simxGetJointForce(self.__clientID,self.__joint_handle[l,j],MODEF)
		return self.__redo(status,torques,self.get_jointtorque)

	# ---------------------- simulation control  ------------------------

	def __redo(self,status,value,func):
		if status != 0:
			self.update()
			value = func()
		return value

	def update(self):
		for l in range(0,NLEG):
			for j in range(0,len(self.__joint_name)): 
				sim.simxSetJointTargetPosition(self.__clientID,int(self.__joint_handle[l,j]),self.__target_positions[l,j],MODEF)
		sim.simxSynchronousTrigger(self.__clientID)

	def reset(self,init_pose=[0.0,0.0,0.15,0.0,0.0,0.0],tsleep=0.5,zero=True):
		
		self.end()

		self.__clientID = sim.simxStart(self.__ip,self.__port,True,True,5000,5) # Connect to CoppeliaSim
		sim.simxSynchronous(self.__clientID,True)
		sim.simxStartSimulation(self.__clientID,MODE)

		if zero:
			self.set_zero()
		
		for i in range(10):
			self.update()
		#print("INFO: VrepInterfaze reset successfully.")
		
				
	def end(self):
		sim.simxStopSimulation(self.__clientID,MODE)
		sim.simxFinish(self.__clientID)


