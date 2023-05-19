from pytorch3d.structures import Meshes
import torch
import numpy as np
import open3d as o3d
import util
import pytorch3d.renderer
from pytorch3d.renderer import (
    PerspectiveCameras, look_at_view_transform,
    RasterizationSettings, BlendParams,
    MeshRenderer, MeshRasterizer, SoftPhongShader
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
	
	def __init__(self, pose_axisang, betas):
		self.setSMPL()
		self.setTestSMPL(pose_axisang,betas)
		self.setRefMesh()
		self.setMesh()
		self.setFaceLocationMesh()
		self.setFaceLocationRefMesh()
		
	
	def setSMPL(self):
		''' 
    	Set the SMPL object from models by default use the male object
    	'''	
		self.smpl = util.SMPL_path(gender="M").get_smpl()

	def setTestSMPL(self,pose_axisang,betas):
		''' 
    	Set the SMPL object
    
    	Args:
    	    pose_axisang (torch): (batch_size,72) 
    	    betas (torch): (batch_size,300) 
    	'''	
		test_smpl = SMPL_Layer(model_root="./models")
		test_vert = test_smpl.smpl_data["v_posed"]
		
		test_vert = np.array(test_vert)
		no_input_para=True

		if no_input_para:
			self.vertices = torch.from_numpy(test_vert).float().cpu()[None,:]
		else :
			self.vertices,_ = test_smpl(pose_axisang,betas)		


	def setMesh(self,mesh_vert=None):
		''' 

    	Set the Mesh object
    
    	'''	
		# 6890 vertices, 13776 faces

		self.mesh     = Meshes(self.vertices, self.ref_faces)


	def setRefMesh(self):
		''' 
    	Set the Mesh object for reference
	    
    	'''	
		self.ref_faces    = self.smpl.faces.astype('float32')
		self.ref_vertices = self.smpl.shapedirs
		self.ref_faces	  = torch.from_numpy(self.ref_faces).float().cpu()[None,:]
		self.ref_vertices =  self.ref_vertices.float().cpu()[None,:]
		self.mesh_ref = Meshes(self.ref_vertices, self.ref_faces)

	def setFaceLocationMesh(self):
		''' 
    	Set the face location of the Mesh object
    
    	
    	'''	
		self.face_location_attr_scan        =    self.mesh.verts_packed()[self.mesh.faces_packed()].reshape((13776, 3,-1))
		self.face_location_attr_scan_normal =    self.mesh.verts_normals_packed()[self.mesh.faces_packed()].reshape((13776, 3,-1))
		
	def setFaceLocationRefMesh(self):
		''' 
    	Set the face location of the reference Mesh object
     
    	'''	
		self.face_location_attr_ref        = self.mesh_ref.verts_packed()[self.mesh.faces_packed()].reshape((13776, 3,-1))
		self.face_location_attr_ref_normal = self.mesh_ref.verts_normals_packed()[self.mesh.faces_packed()].reshape((13776, 3,-1))
	
	def show_image(self):
		''' 
    	Show the 2D depth map and save the image
    
    	
    	'''	
		self.depth_map()

		depth = self.depth
		plt.imshow(self.depth[0, :,:, 0].cpu().numpy())
		#plt.show()
		plt.savefig('2D_object.png')
		plt.close()

		# get partial pointcloud from rendering results
		# get correct order in data

	def set_intrinsic(self, R=None, T=None, intrinsics=None):
		''' 
    	Set the intrinsic parameter for the camera
    
    	Args:
    	    R (np.ndarray): (3,) array storing the rotation angle
    	    T (np.ndarray): (3,) array storing the translation vector
    	    intrinsics (np.ndarray): (4,) array storing the camaera calibration matrix
    	'''
		if R == None:
			self.R = torch.tensor([[
					[-1.,  0.,  0.],
					[ 0.,  1.,  0.],
					[ 0.,  0., -1.]
					]]).cpu()
		else:
			r = Rotation.from_euler('zyx', R, degrees=True)
			self.R = r.as_matrix()
		if T == None:
			self.T = torch.tensor([[-0.0000, 0.2, 2.7000]]).cpu()
		else:
			self.T = T
		# BEHAVE: Kinect azure depth camera 0 parameters
		if intrinsics == None:
			intrinsics = [502.9671325683594,503.04168701171875,322.229736328125,329.3377685546875]
			fx = intrinsics[0]
			fy = intrinsics[1]
			cx = intrinsics[2]
			cy = intrinsics[3]
			self.intrinsics = torch.tensor([[
							[fx, 0.0, cx, 0.0],
							[0.0, fy, cy, 0.0],
							[0.0, 0.0, 0.0, 1.0],
							[0.0, 0.0, 1.0, 0.0]]]).cpu()
		else:		
			fx = intrinsics[0]
			fy = intrinsics[1]
			cx = intrinsics[2]
			cy = intrinsics[3]
			self.intrinsics = torch.tensor([[
							[fx, 0.0, cx, 0.0],
							[0.0, fy, cy, 0.0],
							[0.0, 0.0, 0.0, 1.0],
							[0.0, 0.0, 1.0, 0.0]]]).cpu()


	def set_Camera(self):
		''' 
    	Set the the parameter of the camera
    
		'''	
		self.image_size_height = 576
		self.image_size_width = 640
		self.image_size = torch.tensor([self.image_size_height, self.image_size_width]).unsqueeze(0).cpu()
		self.cameras_front = PerspectiveCameras(R=self.R, T=self.T, in_ndc=False, K=self.intrinsics, image_size=self.image_size).cpu()

	def setKinect(self):
		self.raster_settings_kinect = RasterizationSettings(
			image_size=(self.image_size_height, self.image_size_width),
			blur_radius=0,
			faces_per_pixel=1,  # number of faces to save per pixel
			bin_size=0
			)

	def setRasterizer(self):
		self.rasterizer = MeshRasterizer(
			cameras=self.cameras_front, 
			raster_settings=self.raster_settings_kinect
			)
	def depth_map(self):
		mesh = self.mesh
		
		T = torch.tensor([[-0.0000, 1.2, 2.7000]]).cpu()
		self.set_intrinsic(R=None,T=T)
		self.set_Camera()
		self.setKinect()
		self.setRasterizer()

		fragments = self.rasterizer(mesh)

		self.depth = fragments.zbuf
		self.pix_to_face = fragments.pix_to_face
		self.barycentric = fragments.bary_coords

	def setPixelScan(self):
		self.pixel_location_scan = interpolate_face_attributes( self.pix_to_face, self.barycentric, self.face_location_attr_scan)
		self.pixel_normal_scan   = interpolate_face_attributes( self.pix_to_face, self.barycentric, self.face_location_attr_scan_normal)

	def setPixelRef(self):
		self.pixel_location_ref = interpolate_face_attributes(self.pix_to_face, self.barycentric, self.face_location_attr_ref)
		self.pixel_normal_ref = interpolate_face_attributes  (self.pix_to_face, self.barycentric, self.face_location_attr_ref_normal)

	def recover3D(self):
		''' 
    	Recover the partial view from the 2D depth map
	    
    	'''	
		#pixel_vals: tensor of shape (N, H, W, K, D) giving the interpolated value of the face attribute for each pixel.
		self.setPixelScan()
		self.setPixelRef()
		
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

		pcd = o3d.geometry.PointCloud()
		
		pcd.points = o3d.utility.Vector3dVector(depth_points)
		self.pcd = pcd
		self.depth_points = depth_points
		o3d.visualization.draw_geometries([pcd])