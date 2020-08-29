#Position-wise FeedForward NN

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
	'''
	In addition to Multiheaded-attention layer
	forms a complete sublayer for encoders and decoders alike
	Works similar to a Multi-Layer Perceptron Network
	'''
	def __init__(self,d_model,d_ff,dropout=0.1):
		'''
		Args:
			d_model - input dim ~ 512
			d_ff - output dim of 1st linear layer ~ 2048
		'''
		super(PositionwiseFeedForward,self).__init__()
		self.w_1 = nn.Linear(d_model,d_ff)
		self.w_2 = nn.Linear(d_ff,d_model)
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, x):
		x = F.relu(self.w_1(x))
		x = self.dropout(x)
		w2 = self.w_2(x)
        
		return w2
