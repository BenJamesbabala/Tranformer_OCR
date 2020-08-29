#EncoderLayer

import torch
import torch.nn as nn

from Utils import clones
from LayerNorm import LayerNorm
from SublayerConn import SubLayerConnection

class EncoderLayer(nn.Module):
	'''
	Stack 2 sublayers: sublayer0 -> sublayer1 -> Out
		sublayer0 - [self-attn -> LayerNorm]
		sublayer1 - [FFN -> LayerNorm] 
	'''
	def __init__(self,size,self_attn,feed_forward,dropout):
		super(EncoderLayer,self).__init__()
		self.self_attn = self_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SubLayerConnection(size,dropout),2)
		self.size = size
	
	def forward(self,x,mask=None):
		
		#Apply self-attn, then LayerNorm layer on 1st sublayer
		x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
		
		#Combine prev sublayer out and FFN to 2nd sublayer
		out = self.sublayer[1](x,self.feed_forward)
		
		return out

class Encoder(nn.Module):
	'''
	Core encoder is a stack of N-Layers
		N- num of identical layers of encoder
		layer- type of layer to build
	'''
	def __init__(self,layer,N):
		super(Encoder,self).__init__()
		self.layers = clones(layer,N)
		self.norm = LayerNorm(layer.size)
		
	def forward(self,x,mask=None):
		'''
		standard forward pass
		'''
		for layer in self.layers:
			x = layer(x,mask)
		
		output = self.norm(x)
		return output
