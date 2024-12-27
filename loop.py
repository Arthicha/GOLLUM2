import os, sys
import configparser



worker = 19997

sample = 0
while 1:
	sample += 1

	for lr in [10,20,5,30]:
		for tagotae_lr in [0,5,10,20]:

			gaittypes = [1,2] if tagotae_lr == 0 else [0,1,2]
			for gait_id in gaittypes:
				os.system("python3 main.py "+ str(gait_id) +" " +str(lr) +" " + str(tagotae_lr)+" "+str(sample)+" "+str(worker))
	
	if sample > 10:
		break
		sys.exit()

sys.exit()


