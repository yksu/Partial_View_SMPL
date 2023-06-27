import numpy as np

def vertex_mapping(V1, V2):
	"""
	This method compute the mapping F1 : V1 x V2 -> Index(V1)

	Args
		V1 (np.ndarray): size Nx3
		V2 (np.ndarray): size Mx3
	returns:
		vec (np.ndarray , dtype = int): size Mx1, the closest index of vertex of V1 to the vertex in V2.
	"""
	r = []

	### For each p in V2, find the p' in V1 such that p' = argmin_( p' in V1 ) L2( p - p')
	for v in V2:
		
		d = np.sum((V1 - v )**2,axis=1)
		
		r.append(np.argmin(d))
	return np.asarray(r)

def face_mapping(V1,V2,F):
	"""
	This method compute the mapping F2 : V1 x V2 x F -> Index(F)

	Args
		V1 (np.ndarray): size Nx3
		V2 (np.ndarray): size Mx3
		F (np.ndarray): size Fx3
	returns:
		vec (np.ndarray , dtype = int): size Mx1, the closest index of face to the vertex in V2.

	"""
	### V_F of shape F x 3 x 3 
	V_F = V1[F]
	r = []

	### For each p in V2, find the triangle t in V_F such that t = argmin_( t in V1 ) L2( p - t)
	for v in V2:
		residual = V_F-v
		residual = np.sum(residual**2,axis=1)
		residual = np.sum(residual**2,axis=1)
		r.append(np.argmin(residual))

	### V_F_c of shape M x 3 x 3
	### Solving equation for each point: [ v1 v2 v3 ] x = v, 
	### which is av1 + bv2 + cv3 = v, x = [a b c]^T the barycentric coordinate.
	V_F_c = V_F[r]
	V_F_c = np.asarray([ b.T for b in V_F_c])
	b = np.linalg.solve(V_F_c,V2)

	return np.asarray(r) , b
