from pytorch3d.structures import Meshes
import torch
import numpy as np
import open3d as o3d
import util
from util import set_Camera_Rasterizer
import pytorch3d.renderer
from pytorch3d.renderer import (
    look_at_view_transform, BlendParams,
    MeshRenderer, SoftPhongShader
)
import pytorch3d
import matplotlib.pyplot as plt
from pytorch3d.ops import interpolate_face_attributes
#import pymeshlab
from scipy.spatial.transform import Rotation as Rotation
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import warnings
warnings.filterwarnings('ignore')

class Partial_views:
	### Part 1 initialization
	def __init__(self, pose_axisang=None, betas=None,camera=None):
		''' 
		Initializing
		Args:
			pose_axisang (torch): (batch_size,72) 
			betas (torch): (batch_size,300) 
			camera (dict)
		'''	
		self.setRef(gender="M")
		self.setSMPL(pose_axisang=pose_axisang,betas=betas)

		### Set face location for mesh and ref_mesh
		self.face_location_attr_scan        =    self.mesh.verts_packed()[self.mesh.faces_packed()].reshape((13776, 3,-1))
		self.face_location_attr_scan_normal =    self.mesh.verts_normals_packed()[self.mesh.faces_packed()].reshape((13776, 3,-1))
		self.face_location_attr_ref        = self.mesh_ref.verts_packed()[self.mesh.faces_packed()].reshape((13776, 3,-1))
		self.face_location_attr_ref_normal = self.mesh_ref.verts_normals_packed()[self.mesh.faces_packed()].reshape((13776, 3,-1))
		
		self.camera=camera
	def setRef(self,gender="M"):
		''' 
		Set the reference SMPL object and the reference Mesh object.
		Args:
			gender (str) : gender of the object
		'''	
		if gender == "M":
			self.smpl = util.SMPL_path(gender="M").get_smpl()
		elif gender == "F":
			self.smpl = util.SMPL_path(gender="F").get_smpl()
		else:
			self.smpl = util.SMPL_path(gender="N").get_smpl()

		self.ref_faces    = self.smpl.faces.astype('float32')
		self.ref_vertices = self.smpl.shapedirs
		self.ref_faces	  = torch.from_numpy(self.ref_faces).float().cpu()[None,:]
		self.ref_vertices =  self.ref_vertices.float().cpu()[None,:]
		self.mesh_ref = Meshes(self.ref_vertices, self.ref_faces)
		
	def setSMPL(self,pose_axisang=None,betas=None):
		''' 
		Set the SMPL object from models by default use the male object
		Args:
			pose_axisang (torch): (batch_size,72) 
			betas (torch): (batch_size,300) 
		'''	
		
		self.test_smpl = SMPL_Layer(model_root="./models")
		test_vert = self.test_smpl.smpl_data["v_posed"]
		
		test_vert = np.array(test_vert)
		
		if pose_axisang==None:
			self.vertices = torch.from_numpy(test_vert).float().cpu()[None,:]
		else :
			self.vertices,_ = self.test_smpl(pose_axisang,betas)		

		self.mesh     = Meshes(self.vertices, self.ref_faces)

	def updateMesh(self,pose_axisang,betas=None):
		''' 
		Update the mesh by pose_axisang and betas
		Args:
			pose_axisang (torch): (batch_size,72) 
			betas (torch): (batch_size,300) 
		'''	
		self.vertices,_ = self.test_smpl(pose_axisang,betas)
		self.mesh       = Meshes(self.vertices, self.ref_faces)
		
	def setCamera(self):
		''' 
		setCamera : set the camera parameter (H , W , T , R , K) from the attributes self.camera
		'''
		
		if self.camera == None:
			return
		keys = self.camera.keys()
		if "H" in keys:
			self.H = self.camera['H']
		else:
			self.H = None

		if "W" in keys:
			self.W = self.camera['W']
		else:
			self.W = None	

		
		if "T" in keys:
			self.T = self.camera['T']
		else:
			self.T = None
		
		if "R" in keys:
			self.R = self.camera['R']
		else:
			self.R = None	
		
		if "K" in keys:
			self.K = self.camera['K']
		else:
			self.K = None	

	### Part 2: depth map creating and 2D image showing
	def updateCamera(self,camera):
		''' 
		updateCamera : set the camera parameter (H , W , T , R , K) from the the arguement camera

    		Args:
      			camera (dict) : with keys  ["H" , "W" , "T" , "R" , "K"]
		'''
		keys = self.camera.keys()
		if "H" in keys:
			self.H = self.camera['H']
		else:
			self.H = None

		if "W" in keys:
			self.W = self.camera['W']
		else:
			self.W = None	

		if "R" in keys:
			self.R = self.camera['R']
		else:
			self.R = None	
		
		if "T" in keys:
			self.T = self.camera['T']
		else:
			self.T = None

		if "K" in keys:
			self.K = self.camera['K']
		else:
			self.K = None		
	
	def show_image(self,folder,filename,show=True,save=True):
		''' 
    		Show the 2D depth map and save the image
      		Args:
      			folder (str) : folder name
	 		filename(str): filename
    			show=True(bool): decide whether to show the 2D depth map
       			show=True(bool): decide whether to save the 2D depth map
    
    		'''	
		#self.setCamera()
		#self.setDepth_map(H=self.H,W=self.W,R=self.R,T=self.T,K=self.K)

		depth = self.depth
		plt.imshow(self.depth[0, :,:, 0].cpu().numpy())
		if show:
			plt.show()
		if save:
			if folder == None:
				plt.savefig('output/'+filename)
			else:
				plt.savefig(folder+"/"+filename)	
		plt.close()

		# get partial pointcloud from rendering results
		# get correct order in data
	
	
	def setDepth_map(self,H=None,W=None,R=None,T=None,K=None):
		'''  
    		Create the depth map
    		Args:
			H (int) : image height
			W (int) : image width
    			R (np.array): (3,) array storing the rotation angle
    			T (torch.Tensor): (3,) array storing the translation vector
    			K (torch.Tensor): (4,) array storing the camaera calibration matrix
		
    		'''	
		
		self.rasterizer = set_Camera_Rasterizer(H,W,R,T,K)
		
		fragments = self.rasterizer(self.mesh)

		self.depth = fragments.zbuf
		self.pix_to_face = fragments.pix_to_face
		self.barycentric = fragments.bary_coords

		
	### Part 3: recover 3D partial view from the 2D depth map
	def recover3D(self):
		''' 
    		Recover the partial view from the 2D depth map
	    
    		'''	
		#pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated value of the face attribute for each pixel.
		self.pixel_location_scan = interpolate_face_attributes( self.pix_to_face, self.barycentric, self.face_location_attr_scan)
		self.pixel_normal_scan   = interpolate_face_attributes( self.pix_to_face, self.barycentric, self.face_location_attr_scan_normal)

		self.pixel_location_ref = interpolate_face_attributes(self.pix_to_face, self.barycentric, self.face_location_attr_ref)
		self.pixel_normal_ref = interpolate_face_attributes  (self.pix_to_face, self.barycentric, self.face_location_attr_ref_normal)

		
		points, idx, counts = torch.unique(self.pixel_location_scan.reshape(-1, 3), 
			sorted=True, return_inverse=True, return_counts=True, dim=0)
		
		n, ind_sorted = torch.sort(idx, stable=True)
		
		cum_sum = counts.cumsum(0)
		cum_sum = torch.cat((torch.tensor([0]).cpu(), cum_sum[:-1]))
		first_indicies = ind_sorted[cum_sum]

		nonZeroRows = torch.abs(points).sum(dim=1) > 0
		depth_points = points[nonZeroRows].cpu().numpy()
		
		nonZeroRows = torch.abs(self.pixel_normal_ref.reshape(-1, 3)[first_indicies]).sum(dim=1) > 0
		zero_idx = (nonZeroRows == False).nonzero(as_tuple=True)[0]

		self.pcd = o3d.geometry.PointCloud()
		self.pcd.points = o3d.utility.Vector3dVector(depth_points)
		
	def show_3D_Partial_View(self):
		''' 
    		Show the 3D Partial View
	    
    		'''
		o3d.visualization.draw_geometries([self.pcd])
	
	def save_3D_Partial_View(self,folder,filename):
		''' 
    		Show the 3D Partial View
      		Args:
			
			folder (str) : folder name
	 		filename(str): filename	    
    		'''
		name = folder + "/" + filename + ".ply"
		o3d.io.write_point_cloud( name, self.pcd)

	### For simplicity, this method would set the camera parameter and produce the 2D image and 3D object.
	def run(self,folder,filename,show2D=False,save2D=True,show3D=False,save3D=True):
		''' 
    		Set the camera parameter, compute the 2D depth map, produce the 2D image and 3D object.
      		Args:
			
			folder (str) : folder name
	 		filename(str): filename	   
    			show2D=False(bool): decide whether to show the 2D depth map 
       			save2D=True(bool): decide whether to save the 2D depth map 
	  		show3D=True(bool): decide whether to show the 3D object
     			save3D=True(bool): decide whether to save the 3D object
    		'''
		self.setCamera()
		self.setDepth_map(H=self.H,W=self.W,R=self.R,T=self.T,K=self.K)
		self.show_image(show=False, save=True,folder=folder,filename=filename)
		self.recover3D()
		if show3D: 
			self.show_3D_Partial_View()
		if save3D:
			self.save_3D_Partial_View(folder,filename)

	### This can update the camera parameter and produce the new 2D image and 3D object.
	def updateRun(self,folder,filename,camera,show2D=False,save2D=True,show3D=False,save3D=True):
		''' 
    		Update the camera parameter, compute the 2D depth map, produce the 2D image and 3D object.
      		Args:
			
			folder (str) : folder name
	 		filename(str): filename
    			camera(dict)
    			show2D=False(bool): decide whether to show the 2D depth map 
       			save2D=True(bool): decide whether to save the 2D depth map 
	  		show3D=True(bool): decide whether to show the 3D object
     			save3D=True(bool): decide whether to save the 3D object
    		'''
		
		self.updateCamera(camera)
		self.setDepth_map(H=self.H,W=self.W,R=self.R,T=self.T,K=self.K)
		self.show_image(show=False, save=True,folder=folder,filename=filename)
		self.recover3D()
		if show3D: 
			self.show_3D_Partial_View()
		if save3D:
			self.save_3D_Partial_View(folder,filename)
