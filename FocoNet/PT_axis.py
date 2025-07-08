''' 
This script computes Pressure (P), Tension (T) and Null (B) axes
using fault geometries: Strike, Dip and Rake angles from Moment tensor
inversions or First motion P-wave arrivals. The figure is plotted in the 
lower hemisphere using an equal-area projection. 

Esteban J. Chaves - Volcanological and Seismological Observatory of Costa Rica, Universidad Nacional (OVSICORI-UNA)-
Copyright(c) Esteban J. Chaves 2016-2017

This is code is distributed under the License GNU LGPL. 
Please cite using DOI: 10.5281/zenodo.825322

Contact: esteban.chaves.sibaja@una.cr
Website: https://github.com/echavess / http://www.ovsicori.una.ac.cr/index.php/ovsicori/personal
'''

import numpy as np
import torch

class Tensor_axes(object):

	def __init__(self, foc_tensor):

		'''Important: Strike, dip and rake must be Numpy N arrays event with only 1 element''' 

		self.source = foc_tensor

	def  PTB(self):

		'''Lower hemisphere equal-area projection''' 
		projection = -1
		N = self.source.shape[0]
		strike = self.source[:,0]
		dip = self.source[:,1]
		rake = self.source[:,2]


		'''Computing Nomal (n) and Slip (u) vectors from Strike, Dip and Rake elements''' 
		n1 = -1*torch.sin(dip*torch.pi/180.0)*torch.sin(strike*torch.pi/180.0)
		n2 = torch.sin(dip*torch.pi/180.0)*torch.cos(strike*torch.pi/180.0)
		n3 = -1*torch.cos(dip*torch.pi/180.0)
		n = torch.concatenate((n1, n2, n3)).reshape(N, -1)

		u1 = torch.cos(rake*torch.pi/180.0)*torch.cos(strike*torch.pi/180.0) + torch.cos(dip*torch.pi/180.0)*torch.sin(rake*torch.pi/180.0)*torch.sin(strike*torch.pi/180.0)
		u2 = torch.cos(rake*torch.pi/180.0)*torch.sin(strike*torch.pi/180.0) - torch.cos(dip*torch.pi/180.0)*torch.sin(rake*torch.pi/180.0)*torch.cos(strike*torch.pi/180.0)
		u3 = -1*torch.sin(rake*torch.pi/180.0)*torch.sin(dip*torch.pi/180.0)
		u = torch.concatenate((u1, u2, u3)).reshape(N, -1)

		P_osa = torch.subtract(n, u) / torch.linalg.norm(torch.subtract(n, u))
		T_osa = torch.add(n, u) / torch.linalg.norm(torch.add(n, u))


		# if P_osa[2] > 0:
		# 	P_osa[0] = -1*P_osa[0]
		# 	P_osa[1] = -1*P_osa[1]
		# 	P_osa[2] = -1*P_osa[2]

		# if T_osa[0] > 0:
		# 	T_osa[0] = -1*T_osa[0]
		# 	T_osa[1] = -1*T_osa[1]
		# 	T_osa[2] = -1*T_osa[2]

		angle = (torch.arctan(torch.abs(P_osa[0]/P_osa[1]))*180.0)/torch.pi

		# if P_osa[0] > 0 and P_osa[1] > 0: 
		# 	P_azimuth = angle

		# elif P_osa[0] > 0 and P_osa[1] < 0:
		# 	P_azimuth = 180 - angle

		# elif P_osa[0] < 0 and P_osa[1] < 0:
		# 	P_azimuth = angle + 180

		# elif P_osa[0] < 0 and P_osa[1] > 0:
		# 	P_azimuth = 360 - angle

		P_theta = (torch.arccos(torch.abs(P_osa[2]))*180.0)/torch.pi


		angle_2 = (torch.arctan(torch.abs(T_osa[0]/T_osa[1]))*180.0)/torch.pi

		# if T_osa[0] > 0 and T_osa[1] > 0:
		# 	T_azimuth = angle_2

		# elif T_osa[0] > 0 and T_osa[1] < 0:
		# 	T_azimuth = 180 - angle_2

		# elif T_osa[0] < 0 and T_osa[1] < 0:
		# 	T_azimuth = angle_2 + 180

		# elif T_osa[0] < 0 and T_osa[1] > 0:
		# 	T_azimuth = 360 - angle_2

		T_theta = (torch.arccos(torch.abs(T_osa[2]))*180.0)/torch.pi

		P_x = torch.sqrt(2.0)*projection*torch.sin(P_theta*torch.pi/360.0)*torch.sin(P_azimuth*torch.pi/180)
		P_y = torch.sqrt(2.0)*projection*torch.sin(P_theta*torch.pi/360.0)*torch.cos(P_azimuth*torch.pi/180)

		P_axes = [P_x, P_y]
		P.append(P_axes)

		T_x = torch.sqrt(2.0)*projection*torch.sin(T_theta*torch.pi/360)*torch.sin(T_azimuth*torch.pi/180)
		T_y = torch.sqrt(2.0)*projection*torch.sin(T_theta*torch.pi/360)*torch.cos(T_azimuth*torch.pi/180)

		T_axes = [T_x, T_y]
		T.append(T_axes)

		print("Solving the Jacobian...")

		return P, T
