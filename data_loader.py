#data_loader
#Originally written by: Yunjey Choi
#Repo:https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
  
import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import os
import pickle
import numpy as np
import nltk

from PIL import Image
from build_vocab import Vocabulary,build_vocab,read_json


class ImageDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, json, vocab,max_len=15):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: annotation file path.
            vocab: vocabulary wrapper.
        """
        self.root = 'path/to/images/'
        self.images = list(sorted(os.listdir(self.root)))
        self.json_file = read_json(json)
        self.vocab = vocab if vocab is not None else build_vocab()
        self.max_len = max_len


    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        
        vocab = self.vocab
        
        img = self.images[index] 

        image = Image.open(os.path.join(self.root, img)).convert('RGB')
        image = self.transform(image)
        image = torch.Tensor(image)

        caption = self.json_file[self.images[index]]
        captions = []
        captions.append(vocab('<start>'))
        captions.extend([vocab(token) for token in caption])
        captions.append(vocab('<end>'))

        while len(captions) < self.max_len:
            captions.append(vocab('<pad>'))
            
        target = torch.Tensor(captions).long()
        #print("Target:{}".format(target))
        #exit()
        return image, target

    def __len__(self):
        return len(self.images)
	
    def transform(self,img):

        preprocess = transforms.Compose([
                        transforms.Resize(256),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
		])
        img = preprocess(img)

        return img
	

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(json, vocab, max_len,batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom dataset."""

    dataset = ImageDataset(json=json,vocab=vocab,max_len=max_len)
    
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=dataset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last=True)
    return data_loader
