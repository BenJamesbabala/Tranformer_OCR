#DecoderLayer and Stack

import torch
import torch.nn as nn
import torch.nn.functional as F

from SublayerConn import SubLayerConnection
from LayerNorm import LayerNorm
from Utils import clones
#from PositionwiseFeedForward import PositionwiseFeedForward

class DecoderLayer(nn.Module):
	'''
	Decoder is made of self-attn,src-attn and feed-forward
	Same concept as Encoder, except for addition of encoding from encoder-stack.
	such encoding is  labeled memory and used in sublayer1
	
	Stacks 3 sublayers: sublayer0 -> sublayer1 -> sublayer2 -> Out
		sublayer0 => [self-attn -> LayerNorm]
		sublayer1 => [encoder-attn -> LayerNorm]
		sublayer2 => [FFN -> LayerNorm]
	'''
	def __init__(self,size,self_attn,encoder_attn,feed_forward,dropout):
		super(DecoderLayer,self).__init__()
		self.size = size
		self.self_attn = self_attn
		self.encoder_attn = encoder_attn
		self.feed_forward = feed_forward
		self.sublayer = clones(SubLayerConnection(size,dropout),3)
		
	def forward(self,x,memory,tgt_mask,src_mask=None):
		#print("[INFO]Shape of x is: {}".format(x.shape))
		#print("[INFO]Shape of mem is: {}".format(memory.shape))
		m = memory
		x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,tgt_mask))
		x = self.sublayer[1](x, lambda x: self.encoder_attn(x,m,m,src_mask))
		x = self.sublayer[2](x, self.feed_forward)
		
		return x


class Decoder(nn.Module):
	'''
	N-layer decoder with masking
	'''
	def __init__(self,layer,N):
		super(Decoder,self).__init__()
		self.layers = clones(layer,N)
		self.norm = LayerNorm(layer.size)

	def forward(self,x,memory,src_mask,tgt_mask):
		for layer in self.layers:
			x = layer(x,memory,src_mask,tgt_mask)
		
		return self.norm(x)
