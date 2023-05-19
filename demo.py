from partial_view import Partial_views
import torch

if __name__ == '__main__':
	betas = torch.rand(1,300)
	pose_axisang = torch.rand(1,72)
	pv = Partial_views(pose_axisang=pose_axisang,betas=betas)
	pv.show_image()
	pv.recover3D()
	print("done")