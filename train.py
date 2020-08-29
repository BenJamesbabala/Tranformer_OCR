#Train network
import torch
import torch.nn as nn
from torchvision import transforms

import argparse
import os
import pickle
import time

from Model import EncoderCNN, Decoder
from build_vocab import Vocabulary
from data_loader import get_loader
from Utils import create_masks

def main(args,encoder=None,decoder=None):
	#Model dir
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	
	print("[INFO]Loading data")
	print("[INFO] Reading file:{}".format(args.vocab_path))
	#Get dataloader	batch_size, shuffle, num_workers
	dataloader = get_loader(args.vocab_path, #'captions2.json',
							vocab=None,
							max_len=18,
							batch_size=args.batch_size,
							shuffle=False,
							num_workers=args.num_workers)
	
	print("[INFO]Creating models")
	#Models
	if encoder is None and decoder is None:
		encoder = EncoderCNN()
		decoder = Decoder(decoder_size=18)
	else:
		encoder = encoder.train()
		decoder = decoder.train()	
	
	#Loss and optimiser
	loss_func = nn.CrossEntropyLoss()
	#params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
	params = list(decoder.parameters()) + list(encoder.parameters())
	optimiser = torch.optim.Adam(params, lr=args.learning_rate)
	
	print("[INFO] Starting training loop")
	#Train the models
	start = time.time()
	savedmodel=False
	total_step = len(dataloader)
	for epoch in range(args.num_epochs):
		print("Epoch:{}/{}".format(epoch,args.num_epochs))
		prev_loss = 0
		for i, (images,captions,lengths) in enumerate(dataloader):
			
			#Feed forward, backwards and optimise
			features = encoder(images)
			features = features.long()
			outputs = decoder(features, captions, tgt_mask=None)
			loss = loss_func(outputs, captions)
			
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			optimiser.step()

			if loss == 0.000:
				prev_loss = prev_loss + 1
				if prev_loss == 5:
					print("Epoch: {}/{}--i:{}".format(epoch,args.num_epochs,i))
					print("Loss: {:.4f}".format(loss.item()))
					save_models(decoder,encoder)
					exit()
			
			if i % args.log_step == 0:
				print("Epoch: {}/{}--i:{}".format(epoch,args.num_epochs,i))
				print("Loss: {:.4f}".format(loss.item()))
				print("[INFO] Time elapsed: {}".format(time.time()-start))
			
			if (i+1) % args.save_step == 0:
				torch.save(decoder.state_dict(),os.path.join(
				args.model_path,'decoder-{}-{}.ckpt'.format(epoch+1,i+1)))
				torch.save(encoder.state_dict(),os.path.join(
				args.model_path,'encoder-{}-{}.ckpt'.format(epoch+1,i+1)))
	
	print("[INFO] Time elapsed: {}".format(time.time()-start))	
	print("[INFO] Exiting")

def save_models(decoder,encoder):
	
	torch.save(decoder.state_dict(),os.path.join(args.model_path,'decoder-final.pth'))
	torch.save(encoder.state_dict(),os.path.join(args.model_path,'encoder-final.pth'))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str,default = 'models/',
						help = 'path for saving models')
	parser.add_argument('--encoder_path',type=str,default='models/encoder-final.pth',
						help='path to trained encoder')
	parser.add_argument('--decoder_path',type=str,default='models/decoder-final.pth',
						help='path to trained decoder')
	parser.add_argument('--vocab_path', type=str,default='captions.json',
						help= 'path for saving vocabulary wrapper')
	parser.add_argument('--data',type=str,default='data',
						help='directory of data')
	parser.add_argument('--caption_path', type=str,default='captions.json' ,
						help = 'path for train annotation file')
	parser.add_argument('--log_step', type=int,default=20,
						help= 'step to print log info')
	parser.add_argument('--save_step',type=int,default=500,
						help='step size for saving trained models')
	
	#Model parameters
	parser.add_argument('--embed_size',type=int,default=256,
						help='dimension of word embedding vectors')
	parser.add_argument('--hidden_size',type=int,default=512,
						help='dimension of lstm hidden states')
	parser.add_argument('--num_layers',type=int,default=2,
						help='number of layers in lstm')
	
	parser.add_argument('--num_epochs',type=int,default=10,
						help='Total epochs')
	parser.add_argument('--batch_size',type=int,default=1,
						help='dataloader batch size')
	parser.add_argument('--num_workers',type=int,default=0,
						help='num of workers for model dataloader')
	parser.add_argument('--learning_rate',type=float,default=0.001,
						help='learning rate for models')
	args =parser.parse_args()
	
	if args.encoder_path is None and args.decoder_path is None:
		main(args)
	else:
		print("[INFO] Creating and Loading models")
		#Models
		encoder = EncoderCNN()
		decoder = Decoder(decoder_size=18)
		
		#Load trained models
		encoder.load_state_dict(torch.load(args.encoder_path))
		decoder.load_state_dict(torch.load(args.decoder_path))
		main(args,encoder,decoder)
