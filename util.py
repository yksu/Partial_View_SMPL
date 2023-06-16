import numpy as np 
import smplx
from scipy.spatial.transform import Rotation as Rotation
class SMPL_path:
	def __init__(self,gender=None):
		self.gender = gender
		path = "models/model.pkl"

		if gender == None:
			self.smpl = smplx.SMPL(path)
		elif gender == "M":
			self.smpl = smplx.SMPL("models/male.pkl")
		elif gender == "F":
			self.smpl = smplx.SMPL("models/female.pkl")
		elif gender == "N":
			self.smpl = smplx.SMPL("models/neutral.pkl")
		else:
			self.smpl = smplx.SMPL(path)
	def get_smpl(self):
		#print(self)
		return self.smpl

import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer import RasterizationSettings
from pytorch3d.renderer import MeshRasterizer

def set_Camera_Rasterizer(H,W,R,T,K):
	''' 
    	Set the the parameter of the camera
    	Args:
		H (int) : image height
		W (int) : image width
    		R (np.array): (3,) array storing the rotation angle
    		T (torch.Tensor): (3,) array storing the translation vector
    		K (torch.Tensor): (4,) array storing the camaera calibration matrix
	
	returns:
		cameras_front
	'''
	if H == None:	
		image_size_height = 576
	else:
		image_size_height = H
	
	if W == None:
		image_size_width = 640
	else:
		image_size_width = W

	image_size = torch.tensor(
		[image_size_height, image_size_width]
		).unsqueeze(0).cpu()
	
	try:
		if R.any() != None:
			r = Rotation.from_euler('xyz', R, degrees=True)
			R = r.as_matrix()
			R = torch.tensor(R).reshape(1,3,3)
			
		else:
			R = torch.tensor([[
				[-1.,  0.,  0.],
				[ 0.,  1.,  0.],
				[ 0.,  0., -1.]
				]]).cpu()
	except:
		if R == None:
			R = torch.tensor([[
				[-1.,  0.,  0.],
				[ 0.,  1.,  0.],
				[ 0.,  0., -1.]
				]]).cpu()

		else:
			r = Rotation.from_euler('xyz', R, degrees=True)
			R = r.as_matrix()
			R = torch.tensor(R).reshape(1,3,3)
	if T == None:
		T = torch.tensor([[-0.0000, 0.2, 2.7000]]).cpu()
	else:
		T = T
		# BEHAVE: Kinect azure depth camera 0 parameters
	if K == None:
		K = [502.9671325683594,503.04168701171875,322.229736328125,329.3377685546875]
		fx = K[0]
		fy = K[1]
		cx = K[2]
		cy = K[3]
		K = torch.tensor([[
						[fx, 0.0, cx, 0.0],
						[0.0, fy, cy, 0.0],
						[0.0, 0.0, 0.0, 1.0],
						[0.0, 0.0, 1.0, 0.0]]]).cpu()
	else:		
		fx = K[0]
		fy = K[1]
		cx = K[2]
		cy = K[3]
		K = torch.tensor([[
						[fx, 0.0, cx, 0.0],
						[0.0, fy, cy, 0.0],
						[0.0, 0.0, 0.0, 1.0],
						[0.0, 0.0, 1.0, 0.0]]]).cpu()

	cameras_front = PerspectiveCameras(
		R=R, T=T, 
		in_ndc=False, 
		K=K, 
		image_size=image_size).cpu()
	
	raster_settings_kinect = RasterizationSettings(
			image_size=(image_size_height, image_size_width),
			blur_radius=0,
			faces_per_pixel=1,  # number of faces to save per pixel
			bin_size=0
			)
	rasterizer = MeshRasterizer(
			cameras=cameras_front, 
			raster_settings=raster_settings_kinect
			)
	return rasterizer 
