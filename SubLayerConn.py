#Sublayer connection

import torch
import torch.nn as nn

from LayerNorm import LayerNorm

class SubLayerConnection(nn.Module):
	'''
	"Add & Norm" layer from paper 
	residual connection followed by a layer norm
	outputs dims d_model=512
	
	Eq - LayerNorm(x + SubLayer(x))
		SubLayer - function implemented by sublayer 
					e.g. Self-attention or FeedForwardNetwork
	'''
	def __init__(self,size,dropout):
		super(SubLayerConnection,self).__init__()
		self.norm = LayerNorm(size)
		self.dropout = nn.Dropout(dropout)
		
	def forward(self,x,sublayer):
		'''
		apply residual connection to any sublayer with the same size
		'''
		#import pdb; pdb.set_trace()
		n = self.norm(x)
		sub = sublayer(n)
		d = self.dropout(sub)
		return x + d
