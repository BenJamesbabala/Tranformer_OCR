#Transformer module

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
from math import sqrt

from SelfAttention import MultiHeadedAttn
from EncoderLayer import Encoder, EncoderLayer
from DecoderLayer import Decoder, DecoderLayer
from PositionwiseFeedForward import PositionwiseFeedForward
from PostionalEncoding import PositionalEncoding
from Utils import clones, Embeddings

class Transformer(nn.Module):
	'''
	
	'''
	def __init__(self, tgt_size,N=6,d_model=512, d_ff=2048, h=8, dropout=0.1):
		'''
		
		'''
		super(Transformer,self).__init__()
		self.tgt_size = tgt_size
		c = copy.deepcopy
		self.attn = MultiHeadedAttn(h,d_model,dropout)
		self.ffn = PositionwiseFeedForward(d_model,d_ff,dropout)
		
		self.position_in = PositionalEncoding(d_model,dropout)
		self.position_out = PositionalEncoding(d_model,dropout)
		self.encoder = Encoder(EncoderLayer(d_model,c(self.attn),c(self.ffn),dropout),N)
		self.decoder = Decoder(DecoderLayer(d_model,c(self.attn),c(self.attn),c(self.ffn),dropout),N)
		self.tgt_embed = Embeddings(d_model,tgt_size)
		#self.std_embed = nn.Embedding(tgt_size,d_model)
	
	def forward(self,feat,tgt,tgt_mask=None):
		
		feat = feat.flatten(2).permute(0,2,1)
		encoder = self.encode(feat)
		decoder = self.decode(encoder,tgt,tgt_mask)
		
		return decoder
	
	def encode(self,feat):
		
		encoded = self.encoder(self.position_in(feat))

		return encoded
	
	def decode(self,memory,tgt,tgt_mask,src_mask=None):
		
		embeded_tgt = self.tgt_embed(tgt)#.permute(0,2,1)
		
		decoded = self.decoder( x=self.position_out(embeded_tgt),
								memory=memory,
								tgt_mask=tgt_mask,
								src_mask=None)
		
		return decoded
