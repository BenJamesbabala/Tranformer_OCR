#Self-Attention
#

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from Utils import clones

def selfattention(query,key,value,mask=None,dropout=None):
	'''
	Scaled Dot-product of attention
	'''
	d_k = query.size(-1)
	scores = torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)
	#print("[INFO]Shape of scores is: {}".format(scores.shape))
	
	#Optional, from paper
	if mask is not None:
		#mask = mask.flatten(1)
		#print("[INFO]Shape of mask is: {}".format(mask.shape))
		#print("[INFO]Shape of scores is: {}".format(scores.shape))
		scores = scores.masked_fill_(mask==0,-1e9)
	
	attn = F.softmax(scores,dim=-1)
	attn = F.dropout(attn,p=dropout)
	
	out = torch.matmul(attn,value)
	
	return out, attn

class MultiHeadedAttn(nn.Module):
	'''
	Concatenation of self-attention blocks 
	'''
	def __init__(self, h=8, d_model=512, dropout=0.1):
		'''
		Args: 
			h: num of self-attention heads
			d_model: dim of input (possible)
		'''
		super(MultiHeadedAttn,self).__init__()
		assert d_model%h == 0, "h has value not compatible"
		#assume d_v == d_k
		self.d_k = d_model//h
		self.h = h
		self.dropout = dropout
		self.linears = clones(nn.Linear(d_model,d_model),4)
		self.attn = None

	def forward(self,query, key,value,mask=None):
		'''
		Input flow bottom-to-top of multiheaded attention block
		'''
		
		if mask is not None:
			#Apply mask to all h-heads
			mask = mask.unsqueeze(1)
			
		batch_size = query.size(0)
		#Linear projects in batch from d_model to (h x d_k)
		query, key, value = [l(x).view(batch_size,-1,self.h,self.d_k).transpose(1,2)
							for l, x in zip(self.linears,(query,key,value))]
		
		#Apply attention on all projected vectors in batch
		x, self.attn = selfattention(query,key,value,mask=mask,dropout=self.dropout)
		#Concat results and apply final linear
		x = x.transpose(1,2).contiguous().view(batch_size,-1,self.h*self.d_k)
		x = self.linears[-1](x)
		
		return x
