#Layer-Norm Layer

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
	'''
	Inserted after self-attention layer and Position-wise FFN
	Takes as inputs residual connection as well as output of preceding
	layer
	layer norm module (details:https://arxiv.org/abs/1607.06450)
	'''
	def __init__(self,features,eps=1e-6):
		super(LayerNorm,self).__init__()
		self.a_2 = nn.Parameter(torch.ones(features))
		self.b_2 = nn.Parameter(torch.zeros(features))
		self.eps = eps
	
	def forward(self,x):
		mean = x.mean(-1,keepdim=True)
		std = x.std(-1,keepdim=True)
		
		output = (self.a_2 * (x-mean)/(std+self.eps)) + self.b_2
		
		return output
