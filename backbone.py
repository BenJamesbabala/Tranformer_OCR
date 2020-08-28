#Backbone module

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import models

model_names = ('vgg16','vgg13','vgg19','resnet18','resnet34','mobilenetv2')

def get_model(name,pretrained):
	'''
	returns model of desired CNN-backbone
	***determine what layers needed for each
	'''
	print("[INFO] Loading CNN-Backbone")
	if name == 'vgg16' or name == None:
		model = models.vgg16(pretrained)
		model = nn.Sequential(*(list(model.children())[:1]))
		return model
	elif name == 'vgg13':
		model = models.vgg13(pretrained)
		model = nn.Sequential(*(list(model.children())[:1]))
		return model
	elif name == 'vgg19':
		model = models.vgg19(pretrained)
		model = nn.Sequential(*(list(model.children())[:1]))
		return model
	elif name == 'resnet18':
		model = models.resnet18(pretrained)
		model = nn.Sequential(*(list(model.children())[:1]))
		return model
	elif name == 'resnet34':
		model = models.resnet34(pretrained)
		model = nn.Sequential(*(list(model.children())[:1]))
		return model
	elif name == 'mobilenetv2':
		model = models.mobilenet_v2(pretrained)
		model = nn.Sequential(*(list(model.children())[:1]))
		return model

class Backbone(nn.Module):
	'''
	CNN backbone
		Create custom
		*VGG13
		VGG16
		*VGG19
		ResNet-18
		ResNet-34
		MobileNet V2
	'''
	def __init__(self,backbone,training_mode=True,pretrained=True):
		super(Backbone,self).__init__()
		name = backbone if backbone.lower() in model_names else None
		self.backbone = get_model(name,pretrained)

	def forward(self,image):
		
		feat = self.backbone(image)
		#feat = self.conv_out(feat)
		
		return feat
