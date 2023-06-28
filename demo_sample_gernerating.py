from partial_view import Partial_views
import torch
import os 
import numpy as np 
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import time
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
	file_name = ['A7 - crouch_poses',"A1 - Stand_poses","A2 - Sway t2_poses","A2 - Sway_poses"]
	
	data_path = [os.path.join('data', dataset, seq_lvl_1,x ) for x in file_name]
	d = [np.load(y + '.npz') for y in data_path]
	d_poses = [ torch.tensor(z['poses'][:,:72],dtype=torch.float32) for z in d ]
	

	
	####### In the demo file, I save and show 3 images for illustration.
	folder = "result_sample"
	try:
		os.mkdir(folder)
	except:
		pass 
	
	
	point_cloud = []
	pv = Partial_views(pose_axisang=None,betas=None,camera=camera)
	for i in range(len(d_poses)):
		print("start in file " + file_name[i])
		for p in d_poses[i]:
			pose = p.reshape(1,-1)
			pv.updateMesh(pose_axisang=pose)
			pv.run(folder,"image"+str(i+1),show2D=False,save2D=False,show3D=False,save3D=False)
			point_cloud.append(np.asarray(pv.pcd.points))

	np.save("sample", np.asarray(point_cloud))