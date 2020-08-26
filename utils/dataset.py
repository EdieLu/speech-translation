# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import torch
import collections
import codecs
import numpy as np
import random
import torch.utils.data
from bpemb import BPEmb
from os.path import join

from utils.config import PAD, UNK, BOS, EOS, SPC

import logging
logging.basicConfig(level=logging.INFO)

class IterDataset(torch.utils.data.Dataset):

	"""
		load features from

		'src_word_ids':train_src_word_ids[i_start:i_end],
		'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
		'tgt_word_ids':train_tgt_word_ids[i_start:i_end],
		'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end],
		'acous_flis':train_acous_flis[i_start:i_end],
		'acous_spkids':train_acous_spkids[i_start:i_end],
		'acous_lengths':train_acous_lengths[i_start:i_end]
	"""

	def __init__(self, batches, max_seq_len_src, max_seq_len_tgt,
	 	acous_norm, acous_norm_path=None, mode='ST'):

		super(Dataset).__init__()

		self.mode = mode
		self.batches = batches
		self.acous_norm = acous_norm
		self.acous_norm_path = acous_norm_path
		self.max_seq_len_src = max_seq_len_src
		self.max_seq_len_tgt = max_seq_len_tgt

	def __len__(self):

		return len(self.batches)

	def __getitem__(self, index):

		if 'ST' in self.mode:
			# import pdb; pdb.set_trace()
			srcid = list(self.batches[index]['src_word_ids']) # lis
			srcid.append([BOS] * self.max_seq_len_src) # pad up to max_seq_len_src
			srcid = torch.nn.utils.rnn.pad_sequence(
				[torch.LongTensor(elem) for elem in srcid], batch_first=True) # tensor
			srcid = srcid[:-1]
			srclen = self.batches[index]['src_sentence_lengths'] # lis

			tgtid = list(self.batches[index]['tgt_word_ids']) # lis
			tgtid.append([BOS] * self.max_seq_len_tgt) # pad up to max_seq_len_tgt
			tgtid = torch.nn.utils.rnn.pad_sequence(
				[torch.LongTensor(elem) for elem in tgtid], batch_first=True) # tensor
			tgtid = tgtid[:-1]
			tgtlen = self.batches[index]['tgt_sentence_lengths'] # lis

			acous_feat = self.load_file(index) # tensor
			acouslen = self.batches[index]['acous_lengths'] # lis

			batch = {
				'srcid': srcid,
				'srclen': srclen,
				'tgtid': tgtid,
				'tgtlen': tgtlen,
				'acous_feat': acous_feat,
				'acouslen': acouslen,
			}

		elif 'MT' in self.mode:
			srcid = list(self.batches[index]['src_word_ids']) # lis
			srcid.append([BOS] * self.max_seq_len_src) # pad up to max_seq_len_src
			srcid = torch.nn.utils.rnn.pad_sequence(
				[torch.LongTensor(elem) for elem in srcid], batch_first=True) # tensor
			srcid = srcid[:-1]
			srclen = self.batches[index]['src_sentence_lengths'] # lis

			tgtid = list(self.batches[index]['tgt_word_ids']) # lis
			tgtid.append([BOS] * self.max_seq_len_tgt) # pad up to max_seq_len_tgt
			tgtid = torch.nn.utils.rnn.pad_sequence(
				[torch.LongTensor(elem) for elem in tgtid], batch_first=True) # tensor
			tgtid = tgtid[:-1]
			tgtlen = self.batches[index]['tgt_sentence_lengths'] # lis

			batch = {
				'srcid': srcid,
				'srclen': srclen,
				'tgtid': tgtid,
				'tgtlen': tgtlen,
			}

		elif 'ASR' in self.mode:

			srcid = list(self.batches[index]['src_word_ids']) # lis
			srcid.append([BOS] * self.max_seq_len_src) # pad up to max_seq_len_src
			srcid = torch.nn.utils.rnn.pad_sequence(
				[torch.LongTensor(elem) for elem in srcid], batch_first=True) # tensor
			srcid = srcid[:-1]
			srclen = self.batches[index]['src_sentence_lengths'] # lis

			acous_feat = self.load_file(index) # tensor
			acouslen = self.batches[index]['acous_lengths'] # lis

			batch = {
				'srcid': srcid,
				'srclen': srclen,
				'acous_feat': acous_feat,
				'acouslen': acouslen,
			}

		return batch


	def load_file(self, index):

		# import pdb; pdb.set_trace()
		if self.acous_norm:
			norm_param = self.load_mu_std(index)
		else:
			norm_param = None
		acous_feat = self.load_acous_from_flis(index, norm_param=norm_param)

		return acous_feat


	def load_mu_std(self, index):

		spkids = self.batches[index]['acous_spkids']
		norm_param = []
		mydict = {}
		base = self.acous_norm_path

		for idx in range(len(spkids)):
			spkid = spkids[idx]
			if spkid in mydict:
				pass
			else:
				f_mu = join(base, spkid+'.mu.npy')
				f_std = join(base, spkid+'.std.npy')
				mu = np.load(f_mu)
				std = np.load(f_std)
				mydict[spkid] = [mu, std]

			norm_param.append(mydict[spkid])

		return norm_param


	def load_acous_from_flis(self, index, norm_param=None):

		# import pdb; pdb.set_trace()
		flis = self.batches[index]['acous_flis']
		max_len = 0
		feat_lis = []
		for idx in range(len(flis)):
			f = flis[idx]
			featarr = np.load(f)
			acous_dim = featarr.shape[1]
			if type(norm_param) != type(None):
				mu, std = norm_param[idx]
				if mu.shape[0] != acous_dim:
					# get rid of training energy term
					mu = mu[:acous_dim]
					std = std[:acous_dim]
				featarr = 1. * (featarr - mu) / std
			feat = torch.FloatTensor(featarr) # np array (len x acous_dim)
			max_len = max(max_len, feat.size(0))
			feat_lis.append(feat)

		# import pdb; pdb.set_trace()
		divisible_eight = max_len + 8 - max_len % 8
		dummy = torch.ones(divisible_eight, acous_dim)
		feat_lis.append(dummy)
		feat_lis = torch.nn.utils.rnn.pad_sequence(feat_lis, batch_first=True)[:-1]

		return feat_lis


class Dataset(object):

	""" load from file """

	def __init__(self,
		# add params
		path_src=None,
		path_tgt=None,
		path_vocab_src=None,
		path_vocab_tgt=None,
		vocab_src_list=None,
		vocab_tgt_list=None,
		acous_path=None,
		acous_norm_path=None,
		#
		max_seq_len_src=32,
		max_seq_len_tgt=300,
		batch_size=64,
		use_gpu=True,
		logger=None,
		#
		seqrev=False,
		acous_norm=False,
		acous_max_len=1500,
		use_type='char',
		data_ratio=1.0,
		mode='ST'
		):

		super(Dataset, self).__init__()

		self.path_src = path_src
		self.path_tgt = path_tgt
		self.path_vocab_src = path_vocab_src
		self.path_vocab_tgt = path_vocab_tgt
		self.vocab_src_list = vocab_src_list
		self.vocab_tgt_list = vocab_tgt_list
		self.acous_path = acous_path
		self.acous_norm_path = acous_norm_path

		self.max_seq_len_src = max_seq_len_src
		self.max_seq_len_tgt = max_seq_len_tgt
		self.batch_size = batch_size
		self.use_gpu = use_gpu
		self.logger = logger

		self.seqrev = seqrev
		self.acous_norm=acous_norm
		self.use_type = use_type
		self.data_ratio = data_ratio
		self.mode = mode

		self.acous_max_len = acous_max_len # 1500 for train; 6000 for eval

		if type(self.logger) == type(None):
			self.logger = logging.getLogger(__name__)

		self.load_vocab()
		self.load_sentences()
		self.load_acous_flis()
		self.preprocess(mode=self.mode)


	def load_vocab(self):

		self.vocab_src = []
		self.vocab_tgt = []

		self.src_word2id = collections.OrderedDict()
		self.src_id2word = collections.OrderedDict()
		self.tgt_word2id = collections.OrderedDict()
		self.tgt_id2word = collections.OrderedDict()

		if type(self.path_vocab_src) != type(None) or type(self.path_vocab_tgt) != type(None):
			# load from path
			if type(self.path_vocab_src) != type(None):
				with codecs.open(self.path_vocab_src, encoding='UTF-8') as f:
					vocab_src_lines	= f.readlines()
				for i, word in enumerate(vocab_src_lines):
					if word == '\n':
						continue
					word = word.strip().split()[0] # remove \n
					self.vocab_src.append(word)
					self.src_word2id[word] = i
					self.src_id2word[i] = word
			if type(self.path_vocab_tgt) != type(None):
				with codecs.open(self.path_vocab_tgt, encoding='UTF-8') as f:
					vocab_tgt_lines	= f.readlines()
				for i, word in enumerate(vocab_tgt_lines):
					if word == '\n':
						continue
					word = word.strip().split()[0] # remove \n
					self.vocab_tgt.append(word)
					self.tgt_word2id[word] = i
					self.tgt_id2word[i] = word

		else:
			# load from saved vocab list
			assert type(self.vocab_src_list) != type(None)
			assert type(self.vocab_tgt_list) != type(None)

			for i in range(len(self.vocab_src_list)):
				word = self.vocab_src_list[i]
				self.vocab_src.append(word)
				self.src_word2id[word] = i
				self.src_id2word[i] = word
			for i in range(len(self.vocab_tgt_list)):
				word = self.vocab_tgt_list[i]
				self.vocab_tgt.append(word)
				self.tgt_word2id[word] = i
				self.tgt_id2word[i] = word


	def load_sentences(self):

		self.src_sentences = []
		self.tgt_sentences = []

		if type(self.path_src) != type(None):
			with codecs.open(self.path_src, encoding='UTF-8') as f:
				self.src_sentences = f.readlines()

		if type(self.path_tgt) != type(None):
			with codecs.open(self.path_tgt, encoding='UTF-8') as f:
				self.tgt_sentences = f.readlines()

			assert len(self.src_sentences) == len(self.tgt_sentences), \
				'Mismatch src:tgt - {}:{}'.format(len(self.src_sentences),len(self.tgt_sentences))

			if self.seqrev:
				for idx in range(len(self.tgt_sentences)):
					tgt_sent_rev = self.tgt_sentences[idx].strip().split()[::-1]
					self.tgt_sentences[idx] = ' '.join(tgt_sent_rev)


	def load_acous_flis(self):

		""" load acoustic npy file list """

		self.acous_flis = []
		self.acous_length_lis = []
		self.acous_spkids = []

		if type(self.acous_path) != type(None):
			f = open(self.acous_path, 'r')
			lines = f.readlines()
			for line in lines:
				elems = line.strip().split()
				fname = elems[0] # path to acoustic features
				length = int(elems[1])
				spkid = elems[2].split('.')[0] # BPL404-10184-20141107-212406-CIXXXXX
				self.acous_flis.append(fname)
				self.acous_length_lis.append(length)
				self.acous_spkids.append(spkid)


	def preprocess(self, mode='ST'):

		"""
			mode	data
			ST		acous/src/tgt
			MT		src/tgt
			ASR		acous/src
			(data not provided are filled with dummy lists)
			(src side default using word level target)
			(tgt side can be either word / char level target)

			SRC: 	BOS w1 w2 ... EOS PAD PAD
			TGT: 	BOS c1 c2 ... EOS PAD PAD
		"""

		# vocab
		self.vocab_size = {'src': len(self.src_word2id), 'tgt': len(self.tgt_word2id)}
		self.logger.info("num_vocab_src: {}".format(self.vocab_size['src']))
		self.logger.info("num_vocab_tgt: {}".format(self.vocab_size['tgt']))

		# declare temporary vars
		train_src_word_ids = []
		train_src_sentence_lengths = []
		train_tgt_word_ids = []
		train_tgt_sentence_lengths = []
		train_acous_flis = []
		train_acous_spkids = []
		train_acous_lengths = []

		partial_n_sent = int(len(self.src_sentences) * self.data_ratio)

		if 'ST' in mode:

			assert len(self.acous_flis) == len(self.src_sentences), \
				'mismatch acoustics and src sentences'

			for idx in range(partial_n_sent):
				# import pdb; pdb.set_trace()
				src_sentence = self.src_sentences[idx]
				tgt_sentence = self.tgt_sentences[idx]

				src_words = src_sentence.strip().split()
				if self.use_type == 'char':
					tgt_words = tgt_sentence.strip()
				elif self.use_type == 'word':
					tgt_words = tgt_sentence.strip().split()

				# ignore long seq of words
				if len(src_words) > self.max_seq_len_src - 2 \
					or len(tgt_words) > self.max_seq_len_tgt - 2:
						# BOS + src + EOS
						# BOS + tgt + EOS
						continue

				# ignore long seq of acoustic features
				if self.acous_length_lis[idx] > self.acous_max_len:
					continue
				else:
					train_acous_flis.append(self.acous_flis[idx])
					train_acous_spkids.append(self.acous_spkids[idx])
					train_acous_lengths.append(self.acous_length_lis[idx])

				# source - words
				src_ids = []
				src_ids.append(BOS)
				for i, word in enumerate(src_words):
					assert word != ' '
					if word in self.src_word2id:
						src_ids.append(self.src_word2id[word])
					else:
						src_ids.append(UNK)
				src_ids.append(EOS)

				# target - words/chars
				tgt_ids = []
				tgt_ids.append(BOS)
				for i, word in enumerate(tgt_words):
					if word == ' ':
						assert self.use_type == 'char'
						tgt_ids.append(SPC)
					elif word in self.tgt_word2id:
						tgt_ids.append(self.tgt_word2id[word])
					else:
						tgt_ids.append(UNK)
				tgt_ids.append(EOS)

				train_src_word_ids.append(src_ids)
				train_src_sentence_lengths.append(len(src_words)+2) # include BOS, EOS
				train_tgt_word_ids.append(tgt_ids)
				train_tgt_sentence_lengths.append(len(tgt_words)+2) # include BOS, EOS

			# import pdb; pdb.set_trace()
			assert (len(train_src_word_ids) == len(train_acous_flis)), \
				"train_src_word_ids != train_acous_flis"

		elif 'MT' in mode:

			assert len(self.src_sentences) == len(self.tgt_sentences), \
				'mismatch src and tgt sentences'

			for idx in range(partial_n_sent):
				# import pdb; pdb.set_trace()
				src_sentence = self.src_sentences[idx]
				tgt_sentence = self.tgt_sentences[idx]

				src_words = src_sentence.strip().split()
				if self.use_type == 'char':
					tgt_words = tgt_sentence.strip()
				elif self.use_type == 'word':
					tgt_words = tgt_sentence.strip().split()

				# ignore long seq of words
				if len(src_words) > self.max_seq_len_src - 2 \
					or len(tgt_words) > self.max_seq_len_tgt - 2:
						# BOS + src + EOS
						# BOS + tgt + EOS
						continue

				# dummy acoustics
				train_acous_flis.append('')
				train_acous_spkids.append('')
				train_acous_lengths.append(0)

				# source - words
				src_ids = []
				src_ids.append(BOS)
				for i, word in enumerate(src_words):
					assert word != ' '
					if word in self.src_word2id:
						src_ids.append(self.src_word2id[word])
					else:
						src_ids.append(UNK)
				src_ids.append(EOS)

				# target - words/chars
				tgt_ids = []
				tgt_ids.append(BOS)
				for i, word in enumerate(tgt_words):
					if word == ' ':
						assert self.use_type == 'char'
						tgt_ids.append(SPC)
					elif word in self.tgt_word2id:
						tgt_ids.append(self.tgt_word2id[word])
					else:
						tgt_ids.append(UNK)
				tgt_ids.append(EOS)

				train_src_word_ids.append(src_ids)
				train_src_sentence_lengths.append(len(src_words)+2) # include BOS, EOS
				train_tgt_word_ids.append(tgt_ids)
				train_tgt_sentence_lengths.append(len(tgt_words)+2) # include BOS, EOS

			# import pdb; pdb.set_trace()
			assert (len(train_src_word_ids) == len(train_acous_flis)), \
				"train_src_word_ids != train_acous_flis"

		elif 'ASR' in mode:

			assert len(self.acous_flis) == len(self.src_sentences), \
				'mismatch acoustics and src sentences'

			for idx in range(partial_n_sent):
				# import pdb; pdb.set_trace()
				src_sentence = self.src_sentences[idx]
				src_words = src_sentence.strip().split()

				# ignore long seq of words
				if len(src_words) > self.max_seq_len_src - 2:
						# BOS + src + EOS
						continue

				# ignore long seq of acoustic features
				if self.acous_length_lis[idx] > self.acous_max_len:
					continue
				else:
					train_acous_flis.append(self.acous_flis[idx])
					train_acous_spkids.append(self.acous_spkids[idx])
					train_acous_lengths.append(self.acous_length_lis[idx])

				# source - words
				src_ids = []
				src_ids.append(BOS)
				for i, word in enumerate(src_words):
					assert word != ' '
					if word in self.src_word2id:
						src_ids.append(self.src_word2id[word])
					else:
						src_ids.append(UNK)
				src_ids.append(EOS)

				train_src_word_ids.append(src_ids)
				train_src_sentence_lengths.append(len(src_words)+2) # include BOS, EOS
				# dummy tgt
				train_tgt_word_ids.append([BOS,EOS])
				train_tgt_sentence_lengths.append(2) # include BOS, EOS

			# import pdb; pdb.set_trace()
			assert (len(train_src_word_ids) == len(train_acous_flis)), \
				"train_src_word_ids != train_acous_flis"

		# only use partial data
		self.logger.info("data partition: {}".format(self.data_ratio))
		n_sent = len(train_src_word_ids)

		# set class var to be used in batchify
		self.train_src_word_ids = train_src_word_ids[:n_sent]
		self.train_src_sentence_lengths = train_src_sentence_lengths[:n_sent]
		self.train_tgt_word_ids = train_tgt_word_ids[:n_sent]
		self.train_tgt_sentence_lengths = train_tgt_sentence_lengths[:n_sent]
		self.train_acous_flis = train_acous_flis[:n_sent] # list of acous npy fnames
		self.train_acous_spkids = train_acous_spkids[:n_sent]
		self.train_acous_lengths = train_acous_lengths[:n_sent]

		# only those that are not too long
		self.num_training_sentences = len(self.train_src_word_ids)
		self.logger.info("num_sentences: {}".format(self.num_training_sentences))


	def construct_batches(self, is_train=False):

		"""
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src:
				if 'word' -
					a cat sat on the mat EOS PAD PAD ...
				if 'char' -
					a  SPC c a t SPC s a t SPC o n SPC t h e SPC m a t EOS PAD ...
		"""

		# organise by length
		_x = list(zip(self.train_src_word_ids, self.train_src_sentence_lengths,
			self.train_tgt_word_ids, self.train_tgt_sentence_lengths,
			self.train_acous_flis, self.train_acous_spkids, self.train_acous_lengths))
		if is_train:
			# _x = sorted(_x, key=lambda l:l[1])
			random.shuffle(_x)
		train_src_word_ids, train_src_sentence_lengths, \
			train_tgt_word_ids, train_tgt_sentence_lengths, \
			train_acous_flis, train_acous_spkids, train_acous_lengths = zip(*_x)

		# manual batching to allow shuffling by pt dataloader
		n_batches = int(self.num_training_sentences/self.batch_size +
			(self.num_training_sentences % self.batch_size > 0))
		batches = []
		for i in range(n_batches):
			i_start = i * self.batch_size
			i_end = min(i_start + self.batch_size, self.num_training_sentences)
			batch = {
				'src_word_ids':train_src_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'tgt_word_ids':train_tgt_word_ids[i_start:i_end],
				'tgt_sentence_lengths': train_tgt_sentence_lengths[i_start:i_end],
				'acous_flis':train_acous_flis[i_start:i_end],
				'acous_spkids':train_acous_spkids[i_start:i_end],
				'acous_lengths':train_acous_lengths[i_start:i_end]
			}
			batches.append(batch)

		# pt dataloader
		params = {'batch_size': 1,
					'shuffle': is_train,
					'num_workers': 0}

		self.iter_set = IterDataset(batches, self.max_seq_len_src, self.max_seq_len_tgt,
			self.acous_norm, acous_norm_path=self.acous_norm_path, mode=self.mode)
		self.iter_loader = torch.utils.data.DataLoader(self.iter_set, **params)
		# import pdb; pdb.set_trace()


	def my_collate(self, batch):

		""" srcid, srclen, acous_feat, acouslen """

		srcid = [torch.LongTensor(item[0]) for item in batch]
		srclen = [item[1] for item in batch]
		acous_feat = [torch.Tensor(item[2]) for item in batch]
		acouslen = [item[3] for item in batch]

		srcid = torch.nn.utils.rnn.pad_sequence(srcid, batch_first=True) # b x l
		acous_feat = torch.nn.utils.rnn.pad_sequence(acous_feat, batch_first=True)
					# b x l x 40

		return [srcid, srclen, acous_feat, acouslen]


def load_pretrained_embedding(word2id, embedding_matrix, embedding_path):

	""" assign value to src_word_embeddings and tgt_word_embeddings """

	counter = 0
	with codecs.open(embedding_path, encoding="UTF-8") as f:
		for line in f:
			items = line.strip().split()
			if len(items) <= 2:
				continue
			word = items[0].lower()
			if word in word2id:
				id = word2id[word]
				vector = np.array(items[1:])
				embedding_matrix[id] = vector
				counter += 1

	print('loaded pre-trained embedding:', embedding_path)
	print('embedding vectors found:', counter)

	return embedding_matrix
