import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.attention import AttentionLayer
from utils.config import PAD, EOS, BOS
from utils.misc import check_device

import warnings
warnings.filterwarnings("ignore")

class Enc(nn.Module):

	""" acoustic pyramidal LSTM """

	def __init__(self,
		# params
		acous_dim=26,
		acous_hidden_size=256,
		#
		acous_norm=False,
		spec_aug=False,
		batch_norm=False,
		enc_mode='pyramid',
		#
		dropout=0.0,
		batch_first=True,
		):

		super(Enc, self).__init__()

		# define model
		self.acous_dim = acous_dim
		self.acous_hidden_size = acous_hidden_size

		# tuning
		self.acous_norm = acous_norm
		self.spec_aug = spec_aug
		self.batch_norm = batch_norm
		self.enc_mode = enc_mode

		# define operations
		self.dropout = nn.Dropout(dropout)

		# ------ define acous enc -------
		if self.enc_mode == 'pyramid':
			self.acous_enc_l1 = torch.nn.LSTM(
				self.acous_dim, self.acous_hidden_size,
				num_layers=1, batch_first=batch_first,
				bias=True, dropout=dropout, bidirectional=True)
			self.acous_enc_l2 = torch.nn.LSTM(
				self.acous_hidden_size * 4, self.acous_hidden_size,
				num_layers=1, batch_first=batch_first,
				bias=True, dropout=dropout, bidirectional=True)
			self.acous_enc_l3 = torch.nn.LSTM(
				self.acous_hidden_size * 4, self.acous_hidden_size,
				num_layers=1, batch_first=batch_first,
				bias=True, dropout=dropout, bidirectional=True)
			self.acous_enc_l4 = torch.nn.LSTM(
				self.acous_hidden_size * 4, self.acous_hidden_size,
				num_layers=1, batch_first=batch_first,
				bias=True, dropout=dropout, bidirectional=True)
			if self.batch_norm:
				self.bn1 = nn.BatchNorm1d(self.acous_hidden_size * 2)
				self.bn2 = nn.BatchNorm1d(self.acous_hidden_size * 2)
				self.bn3 = nn.BatchNorm1d(self.acous_hidden_size * 2)
				self.bn4 = nn.BatchNorm1d(self.acous_hidden_size * 2)

		elif self.enc_mode == 'cnn':
			# todo
			pass


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None
			setattr(self, var_name, var_val)


	def pre_process_acous(self, acous_feats):

		"""
			acous_feats: b x max_time x max_channel
			spec-aug i.e. mask out certain time / channel
			time => t0 : t0 + t
			channel => f0 : f0 + f
		"""
		self.check_var('spec_aug', False)
		if not self.spec_aug:
			return acous_feats
		else:
			max_time = acous_feats.size(1)
			max_channel = acous_feats.size(2)

			CONST_MAXT_RATIO = 0.2
			CONST_T = int(min(40, CONST_MAXT_RATIO * max_time))
			CONST_F = int(7)
			REPEAT = 2

			for idx in range(REPEAT):

				t = random.randint(0, CONST_T)
				f = random.randint(0, CONST_F)
				t0 = random.randint(0, max_time-t-1)
				f0 = random.randint(0, max_channel-f-1)

				acous_feats[:,t0:t0+t,:] = 0
				acous_feats[:,:,f0:f0+f] = 0

			return acous_feats


	def forward(self, acous_feats, acous_lens=None,
		is_training=False, hidden=None, use_gpu=False):

		"""
			Args:
				acous_feats: list of acoustic features 	[b x acous_len x ?]
		"""

		# import pdb; pdb.set_trace()

		device = check_device(use_gpu)

		batch_size = acous_feats.size(0)
		acous_len = acous_feats.size(1)

		# pre-process acoustics
		if is_training: acous_feats = self.pre_process_acous(acous_feats)

		# pad to full length
		if type(acous_lens) == type(None):
			acous_lens = torch.tensor([acous_len]*batch_size).to(device=device)
		else:
			acous_lens = torch.cat([elem + 8 - elem % 8 for elem in acous_lens])

		# run acous enc - pyramidal
		acous_hidden_init = None
		if self.enc_mode == 'pyramid':
			# layer1
			# pack to rnn packed seq obj
			acous_lens_l1 = acous_lens
			acous_feats_pack = torch.nn.utils.rnn.pack_padded_sequence(acous_feats,
					acous_lens_l1, batch_first=True, enforce_sorted=False)
			# run lstm
			acous_outputs_l1_pack, acous_hidden_l1 = self.acous_enc_l1(
				acous_feats_pack, acous_hidden_init) # b x acous_len x 2dim
			# unpack
			acous_outputs_l1, _ = torch.nn.utils.rnn.pad_packed_sequence(
				acous_outputs_l1_pack,batch_first=True)
			# dropout
			acous_outputs_l1 = self.dropout(acous_outputs_l1)\
				.reshape(batch_size, acous_len, acous_outputs_l1.size(-1))
			# batch norm
			if self.batch_norm:
				acous_outputs_l1 = self.bn1(acous_outputs_l1.permute(0, 2, 1))\
					.permute(0, 2, 1)
			# reduce length
			acous_inputs_l2 = acous_outputs_l1\
				.reshape(batch_size, int(acous_len/2), 2*acous_outputs_l1.size(-1))
				# b x acous_len/2 x 4dim

			# layer2
			acous_lens_l2 = acous_lens_l1 / 2
			acous_inputs_l2_pack = torch.nn.utils.rnn.pack_padded_sequence(acous_inputs_l2,
					acous_lens_l2, batch_first=True, enforce_sorted=False)
			acous_outputs_l2_pack, acous_hidden_l2 = self.acous_enc_l2(
				acous_inputs_l2_pack, acous_hidden_init) # b x acous_len/2 x 2dim
			acous_outputs_l2, _ = torch.nn.utils.rnn.pad_packed_sequence(
				acous_outputs_l2_pack,batch_first=True)
			acous_outputs_l2 = self.dropout(acous_outputs_l2)\
				.reshape(batch_size, int(acous_len/2), acous_outputs_l2.size(-1))
			if self.batch_norm:
				acous_outputs_l2 = self.bn2(acous_outputs_l2.permute(0, 2, 1))\
					.permute(0, 2, 1)
			acous_inputs_l3 = acous_outputs_l2\
				.reshape(batch_size, int(acous_len/4), 2*acous_outputs_l2.size(-1))
				# b x acous_len/4 x 4dim

			# layer3
			acous_lens_l3 = acous_lens_l2 / 2
			acous_inputs_l3_pack = torch.nn.utils.rnn.pack_padded_sequence(acous_inputs_l3,
					acous_lens_l3, batch_first=True, enforce_sorted=False)
			acous_outputs_l3_pack, acous_hidden_l3 = self.acous_enc_l3(
				acous_inputs_l3_pack, acous_hidden_init) # b x acous_len/4 x 2dim
			acous_outputs_l3, _ = torch.nn.utils.rnn.pad_packed_sequence(
				acous_outputs_l3_pack,batch_first=True)
			acous_outputs_l3 = self.dropout(acous_outputs_l3)\
				.reshape(batch_size, int(acous_len/4), acous_outputs_l3.size(-1))
			if self.batch_norm:
				acous_outputs_l3 = self.bn3(acous_outputs_l3.permute(0, 2, 1))\
					.permute(0, 2, 1)
			acous_inputs_l4 = acous_outputs_l3\
				.reshape(batch_size, int(acous_len/8), 2*acous_outputs_l3.size(-1))
				# b x acous_len/8 x 4dim

			# layer4
			acous_lens_l4 = acous_lens_l3 / 2
			acous_inputs_l4_pack = torch.nn.utils.rnn.pack_padded_sequence(acous_inputs_l4,
					acous_lens_l4, batch_first=True, enforce_sorted=False)
			acous_outputs_l4_pack, acous_hidden_l4 = self.acous_enc_l4(
				acous_inputs_l4_pack, acous_hidden_init) # b x acous_len/8 x 2dim
			acous_outputs_l4, _ = torch.nn.utils.rnn.pad_packed_sequence(
				acous_outputs_l4_pack,batch_first=True)
			acous_outputs_l4 = self.dropout(acous_outputs_l4)\
				.reshape(batch_size, int(acous_len/8), acous_outputs_l4.size(-1))
			if self.batch_norm:
				acous_outputs_l4 = self.bn4(acous_outputs_l4.permute(0, 2, 1))\
					.permute(0, 2, 1)
			acous_outputs = acous_outputs_l4

		elif self.enc_mode == 'cnn':
			pass #todo

		# import pdb; pdb.set_trace()
		return acous_outputs
