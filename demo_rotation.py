from partial_view import Partial_views
import torch
import os 
import numpy as np 
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


### This code is demonstrating on rotation
### around x-axis, y-axis, and z-axis sperately

if __name__ == '__main__':
	#betas = torch.rand(1,300)
	#pose_axisang = torch.rand(1,72)
	camera = {
		"H" :  576,
		"W" :  640,
		"R" :  np.array([90,0,90]),
		"T" :  torch.tensor([[-0.0000, 0.2, 2.7000]]),
		"K" :  torch.Tensor([502.9671325683594,503.04168701171875,322.229736328125,329.3377685546875]),
	}

	# Loading AMASS data
	dataset = 'ACCAD'
	seq_lvl_1 = 'Female1General_c3d'
	seq_lvl_2 = 'A7 - crouch_poses'
	data_path = os.path.join('data', dataset, seq_lvl_1,seq_lvl_2)

	d = np.load(data_path + '.npz')
	relevant_poses = d['poses']

	d_poses = torch.tensor(d['poses'][:,:72],dtype=torch.float32) 
	
	### There are 608 samples in this data set
	n_sample = d_poses.shape[0]
	
	####### In the demo file, I save and show 3 images for illustration.
	folder = "rotation"
	try:
		os.makedirs(folder)
	except:
		pass
	
	n_translation_smaple = 10

	for i in range(0,min(n_sample,1),50):
		pose = d_poses[i].reshape(1,-1)
		camera["R"] = np.array([90,0,90])
		pv = Partial_views(pose_axisang=pose,betas=None,camera=camera)
		pv.run(folder,filename="image"+str(i+1),show2D=False,save2D=True,show3D=False,save3D=True)
		
		### change the camera view:
		### translate along x-asix
		
		r = np.array([10,0,0])

		try:
			os.makedirs(folder+"/x")
		except:
			pass 
		for j in range(n_translation_smaple):
			camera["R"] += r
			pv.updateRun(folder+"/x",filename="image"+str(i+1) + "_x"+str(j+1),camera=camera,show2D=False,save2D=True,show3D=False,save3D=True)

		### translate along y-asix
		r = np.array([0,10,0])
		camera["R"] = np.array([90,0,90])

		try:
			os.makedirs(folder+"/y")
		except:
			pass 

		for j in range(n_translation_smaple):
			camera["R"] += r
			pv.updateRun(folder+"/y",filename="image"+str(i+1)+ "_y"+str(j+1),camera=camera,show2D=False,save2D=True,show3D=False,save3D=True)

		### translate along z-asix
		r = np.array([0,0,10])
		camera["R"] = np.array([90,0,90])

		try:
			os.makedirs(folder+"/z")
		except:
			pass 

		for j in range(n_translation_smaple):
			camera["R"] += r
			pv.updateRun(folder+"/z",filename="image"+str(i+1)+ "_z"+str(j+1),camera=camera,show2D=False,save2D=True,show3D=False,save3D=True)