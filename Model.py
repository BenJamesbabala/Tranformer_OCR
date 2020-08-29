#Image captioning Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from Transformer import Transformer

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN,self).__init__()
        model = models.vgg16(pretrained=True)
        head = list(model.children())[:-1]
        head = head[0]
        self.vgg16 = head[:25]
        self.output_size = self.vgg16[-1].out_channels
        self.conv_out = nn.Conv2d(self.output_size,self.output_size,1)

    def forward(self,image):
        with torch.no_grad():
            features = self.vgg16(image)

        features = self.conv_out(features)

        return features

class Decoder(nn.Module):
    def __init__(self,decoder_size):
        super(Decoder,self).__init__()
        self.transformer = Transformer(decoder_size)
        d_model = 512
        vocab_size=18
        self.linear = nn.Linear(d_model,vocab_size)
        
    def forward(self,feature,tgt,tgt_mask):
    	hiddens = self.transformer(feature,tgt,tgt_mask)
    	probs = F.log_softmax(self.linear(hiddens),dim=-1)
    	return probs
