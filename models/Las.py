import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.misc import check_device
from utils.config import PAD, EOS, BOS
from .Enc import Enc
from .Dec import Dec

import warnings
warnings.filterwarnings("ignore")

class LAS(nn.Module):

	""" listen attend spell (LAS) model """

	def __init__(self,
		# params
		vocab_size,
		embedding_size=200,
		acous_dim=26,
		acous_hidden_size=256,
		acous_att_mode='bahdanau',
		hidden_size_dec=200,
		hidden_size_shared=200,
		num_unilstm_dec=4,
		#
		acous_norm=False,
		spec_aug=False,
		batch_norm=False,
		enc_mode='pyramid',
		#
		embedding_dropout=0,
		dropout=0.0,
		residual=True,
		batch_first=True,
		max_seq_len=32,
		embedder=None,
		word2id=None,
		id2word=None,
		hard_att=False
		):

		super(LAS, self).__init__()

		self.encoder = Enc(
			acous_dim=acous_dim,
			acous_hidden_size=acous_hidden_size,
			acous_norm=acous_norm,
			spec_aug=spec_aug,
			batch_norm=batch_norm,
			enc_mode=enc_mode,
			dropout=dropout,
			batch_first=batch_first
		)

		self.decoder = Dec(
			vocab_size=vocab_size,
			embedding_size=embedding_size,
			acous_hidden_size=acous_hidden_size,
			acous_att_mode=acous_att_mode,
			hidden_size_dec=hidden_size_dec,
			hidden_size_shared=hidden_size_shared,
			num_unilstm_dec=num_unilstm_dec,
			#
			embedding_dropout=embedding_dropout,
			dropout=dropout,
			residual=residual,
			batch_first=batch_first,
			max_seq_len=max_seq_len,
			embedder=embedder,
			word2id=word2id,
			id2word=id2word,
			hard_att=hard_att
		)


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None
			setattr(self, var_name, var_val)


	def forward(self, acous_feats, acous_lens=None, tgt=None,
		hidden=None, is_training=False, teacher_forcing_ratio=0.0,
		beam_width=1, use_gpu=False):

		"""
			Args:
				acous_feats: list of acoustic features 	[b x acous_len x acous_dim]
				tgt: list of word_ids 					[b x seq_len]
				hidden: initial hidden state
				is_training: whether or not use specaug
				teacher_forcing_ratio: default at 1 - always teacher forcing
			Returns:
				decoder_outputs: list of step_output -
					log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
		"""

		if teacher_forcing_ratio > 0.1: assert type(tgt) != type(None)

		# run encoder decoder
		acous_outputs = self.encoder(
			acous_feats,
			acous_lens=acous_lens,
			is_training=is_training,
			use_gpu=use_gpu)
		sequence_embs, sequence_logps, sequence_symbols, lengths = self.decoder(
			acous_outputs, acous_lens=acous_lens, tgt=tgt,
			is_training=is_training,
			teacher_forcing_ratio=teacher_forcing_ratio,
			beam_width=beam_width, use_gpu=use_gpu
		)

		return sequence_embs, sequence_logps, sequence_symbols, lengths
