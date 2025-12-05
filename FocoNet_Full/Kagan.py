#!/usr/bin/env python

# import standard libraries
import numpy as np
from copy import deepcopy
import torch

def PTB2Kagan(label, predictions):

	N = predictions.shape[0]
    
	p_label = label[:, 0, :]
	t_label = label[:, 1, :]
	b_label = torch.cross(p_label, t_label, dim=1)
	p_pred = predictions[:, 0, :]
	t_pred = predictions[:, 1, :]
	b_pred = torch.cross(p_pred, t_pred, dim=1)

	pp = (p_label * p_pred).sum(dim=1)
	tt = (t_label * t_pred).sum(dim=1)
	bb = (b_label * b_pred).sum(dim=1)
	#print('**', pp.max(), tt.max(), bb.max())

	the = torch.zeros(N, 4)

	the[:, 0] = torch.arccos(torch.clamp(( pp + tt + bb - 1)/2., -0.99999, 0.99999))
	the[:, 1] = torch.arccos(torch.clamp(( pp - tt - bb - 1)/2., -0.99999, 0.99999))
	the[:, 2] = torch.arccos(torch.clamp((-pp - tt + bb - 1)/2., -0.99999, 0.99999))
	the[:, 3] = torch.arccos(torch.clamp((-pp + tt - bb - 1)/2., -0.99999, 0.99999))

	ang = torch.min(the, axis=1).values

	return ang

	


