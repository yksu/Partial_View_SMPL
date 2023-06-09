from partial_view import Partial_views
import torch

if __name__ == '__main__':
	betas = torch.rand(1,300)
	pose_axisang = torch.rand(1,72)
	camera = {
		"H" = 576,
		"W" = 640,
		"R" = torch.Tensor([0,90,0]),
		"T" =  torch.tensor([[-0.0000, 0.2, 2.7000]]),
		"K" = torch.Tensor([502.9671325683594,503.04168701171875,322.229736328125,329.3377685546875]),
	}
	pv = Partial_views(pose_axisang=pose_axisang,betas=betas)
	pv.show_image()
	pv.recover3D()
	print("done")