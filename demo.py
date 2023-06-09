from partial_view import Partial_views
import torch
import os 
import numpy as np 
from smplpytorch.pytorch.smpl_layer import SMPL_Layer

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
	
	n_sample = d_poses.shape[0]
	
	####### In the demo file, I save and show 3 images for illustration.
	os.mkdir("result")
	for i in range(0,min(n_sample,30),10):
		pose = d_poses[i].reshape(1,-1)
		pv = Partial_views(pose_axisang=pose,betas=None,camera=camera)
		pv.run("result","image"+str(i+1))
		pv.show_3D_Partial_View()
