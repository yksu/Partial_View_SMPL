import numpy as np 
import smplx
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
