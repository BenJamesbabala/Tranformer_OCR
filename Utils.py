#Utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from math import sqrt
import numpy as np
import copy


def clones(module,N):
	'''
	Produce N identical copies of a given module
	'''
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
	def __init__(self,d_model,vocab):
		super(Embeddings,self).__init__()
		self.lut = nn.Embedding(vocab,d_model)
		self.d_model = d_model
	
	def forward(self,x):
		return self.lut(x) * sqrt(self.d_model)

def create_masks(tgt, tgt_pad=0):
	
	#src_mask = (src != src_pad).unsqueeze(-2)
	if tgt is not None:
		tgt_mask = (tgt != tgt_pad).unsqueeze(-2)
		size = tgt.size(1)
		np_mask = np.triu(np.ones(size),k=1).astype('uint8')
		nopeak_mask = Variable(subsequent_mask(size))
	
		tgt_mask = tgt_mask & nopeak_mask
	
	else:
		tgt_mask = None
	
	return tgt_mask

def create_tgt_mask(tgt,pad):	
	'''
	https://towardsdatascience.com/
	how-to-code-the-transformer-in-pytorch-24db27c8f9ec
	'''
	tgt_mask = (tgt != pad).unsqueeze(-2)
	size = tgt.size(1)
	nopeak_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
	nopeak_mask = Variable(subsequent_mask(size))
	
	mask = tgt_mask & nopeak_mask
	

def subsequent_mask(size):
	'''INCOMPLETE: Mask subsequent positions'''
	attn_shape = (1,size,size)
	subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
	
	return torch.from_numpy(subsequent_mask) == 0
