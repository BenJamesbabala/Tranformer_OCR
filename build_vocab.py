#build_vocab
#Originally written by: Yunjey Choi
#Repo: https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning

import nltk
import pickle
import argparse
import json
from collections import Counter

class Vocabulary(object):
	'''Vocab wrapper'''
	def __init__(self):
		self.word2idx = {}
		self.idx2word = {}
		self.idx = 0
		
	def add_word(self,word):
		if not word in self.word2idx:
			self.word2idx[word] = self.idx
			self.idx2word[self.idx] = word
			self.idx += 1
			
	def __call__(self, word):
		#print("Word:{}".format(word))
		if not word in self.word2idx:
			return self.word2idx['<unk>']
		return self.word2idx[word]
	
	def __len__(self):
		return len(self.word2idx)

def read_json(filename):	#move to utils file
	# content ={}
	# counter = 0
	# with open(filename,'r') as F:
		# labels = F.read().split()
		
	# for l in labels:
		# content[counter] = l
		# counter += 1
	content  = json.load(open(filename,'r'))
	assert type(content) == dict, 'annotation file format {} not supported'.format(type(content))
	return content
		
def build_vocab(vocab_list=None):	#json_file,threshold):
	#json_content = read_json(json_file)
	counter = Counter()
	if vocab_list is None:
		print("[WARNING] Using in-built vocab list")
		vocab_list = ['0','1','2','3','4','5','6','7','8','9','*','-',',','.']
	
	#idx = len(vocab_list) #json_content.keys()
	
	#for i,id in enumerate(idx):
	for i in range(len(vocab_list)):
		caption = vocab_list[i]
		#print("Caption:{}".format(caption))
		tokens = nltk.tokenize.word_tokenize(caption)
		#print("Tokens:{}".format(tokens))
		counter.update(tokens)
	
	words = [word for word, cnt in counter.items()]
	
	vocab = Vocabulary()
	vocab.add_word('<pad>')
	vocab.add_word('<start>')
	vocab.add_word('<end>')
	vocab.add_word('<unk>')
	
	for i, word in enumerate(words):
		vocab.add_word(word)
		#print("Added:{}".format(word))
	#exit()
	return vocab

def main(args,vocab_list=None):
	vocab = build_vocab(vocab_list) #json = args.caption_path)
	vocab_path = args.vocab_path
	if vocab_path is None:
		return
	with open(vocab_path,'wb') as f:
		pickle.dump(vocab,f)
		
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--caption_path', type=str,required = True,
						help = 'path for train annotation file')
	parser.add_argument('--vocab_path', type=str,required=True,
						help= 'path for saving vocabulary wrapper')
	#parser.add_argument('--threshold',type=str,req)
	
	args =parser.parser_args()
	vocab_list = ['0','1','2','3','4','5','6','7','8','9','*','-',',','.']
	main(args,vocab_list)
