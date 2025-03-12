import os, sys
import configparser



worker = 19997

sample = 30
while 1:
	sample += 1

	for lr in [10]:

		for robot in ['bone']:

			for tagotae_lr in [1000]:
		
				for gait_id in [1]:

					path = 'utils/data/'+str(robot)+"/"+ str(gait_id) +"/" +str(lr) +"/" + str(tagotae_lr)+"/"+str(sample)
					
					#if os.path.exists(path):
					#	print('exist, skillpp')
					#	continue

					print('run',path)
					#sys.exit())
					os.system("python3 main.py "+ str(robot)+" "+ str(gait_id) +" " +str(lr) +" " + str(tagotae_lr)+" "+str(sample)+" "+str(worker))
					sys.exit()

	if sample > 10:
		break
		sys.exit()

sys.exit()


