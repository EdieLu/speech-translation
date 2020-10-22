import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.config import PAD, EOS, BOS
from utils.dataset import load_pretrained_embedding
from utils.misc import check_device

import warnings
warnings.filterwarnings("ignore")

class RNNLM(nn.Module):

	""" RNNLM """

	def __init__(self,
		vocab_size,
		embedding_size=200,
		embedding_dropout=0,
		hidden_size=200,
		dropout=0.0,
		batch_first=True,
		max_seq_len=32,
		load_embedding=None,
		word2id=None,
		id2word=None
		):

		super(RNNLM, self).__init__()

		# define embeddings
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size

		self.load_embedding = load_embedding
		self.word2id = word2id
		self.id2word = id2word

		# define model param
		self.hidden_size = hidden_size

		# define operations
		self.embedding_dropout = nn.Dropout(embedding_dropout)
		self.dropout = nn.Dropout(dropout)

		# load embeddings
		if self.load_embedding:
			# import pdb; pdb.set_trace()
			embedding_matrix = np.random.rand(self.vocab_size, self.embedding_size)
			embedding_matrix = torch.FloatTensor(load_pretrained_embedding(
				self.word2id, embedding_matrix, self.load_embedding))
			self.embedder = nn.Embedding.from_pretrained(embedding_matrix,
				freeze=False, sparse=False, padding_idx=PAD)
		else:
			self.embedder = nn.Embedding(self.vocab_size,
				self.embedding_size, sparse=False, padding_idx=PAD)

		# define lstm
		# embedding_size -> hidden_size
		self.lstm = torch.nn.LSTM(self.embedding_size, self.hidden_size,
			num_layers=1, batch_first=batch_first,
			bias=True, dropout=dropout, bidirectional=False)

		# define ffn
		self.out = nn.Linear(self.hidden_size, self.vocab_size, bias=False)


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None

			# set class attribute to default value
			setattr(self, var_name, var_val)


	def forward(self, src, src_lens, hidden=None, use_gpu=True):

		"""
			Args:
				src: list of src word_ids [batch_size, seq_len, word_ids]
		"""

		# import pdb; pdb.set_trace()

		out_dict = {}
		device = check_device(use_gpu)

		# src mask
		mask_src = src.data.eq(PAD)
		batch_size = src.size(0)
		seq_len = src.size(1)

		# convert id to embedding
		emb_src = self.embedding_dropout(self.embedder(src))

		# run lstm: packing + unpacking
		src_lens = torch.cat(src_lens)
		emb_src_pack = torch.nn.utils.rnn.pack_padded_sequence(emb_src,
			src_lens, batch_first=True, enforce_sorted=False)
		lstm_outputs_pack, lstm_hidden = self.lstm(emb_src_pack, hidden)
		lstm_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
			lstm_outputs_pack, batch_first=True)
		lstm_outputs = self.dropout(lstm_outputs)\
			.view(batch_size, seq_len, lstm_outputs.size(-1))

		# generate predictions
		logits = self.out(lstm_outputs)
		logps = F.log_softmax(logits, dim=2)
		symbols = logps.topk(1)[1]
		out_dict['sequence'] = symbols

		return logps, out_dict
