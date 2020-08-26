import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.attention import AttentionLayer
from utils.config import PAD, EOS, BOS
from utils.dataset import load_pretrained_embedding
from utils.misc import check_device

import warnings
warnings.filterwarnings("ignore")

KEY_ATTN_SCORE = 'attention_score'
KEY_ATTN_OUT = 'attention_out'
KEY_LENGTH = 'length'
KEY_SEQUENCE = 'sequence'

class Dec(nn.Module):

	""" listen attend spell model + dd tag """

	def __init__(self,
		# params
		vocab_size,
		embedding_size=200,
		acous_hidden_size=256,
		acous_att_mode='bahdanau',
		hidden_size_dec=200,
		hidden_size_shared=200,
		num_unilstm_dec=4,
		#
		embedding_dropout=0,
		dropout=0.0,
		residual=True,
		batch_first=True,
		max_seq_len=32,
		embedder=None,
		word2id=None,
		id2word=None,
		hard_att=False,
		):

		super(Dec, self).__init__()

		# define model
		self.acous_hidden_size = acous_hidden_size
		self.acous_att_mode = acous_att_mode
		self.hidden_size_dec = hidden_size_dec
		self.hidden_size_shared = hidden_size_shared
		self.num_unilstm_dec = num_unilstm_dec

		# define var
		self.hard_att = hard_att
		self.residual = residual
		self.max_seq_len = max_seq_len

		# use shared embedding + vocab
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.word2id = word2id
		self.id2word = id2word

		# define operations
		self.embedding_dropout = nn.Dropout(embedding_dropout)
		self.dropout = nn.Dropout(dropout)

		if type(embedder) != type(None):
			self.embedder = embedder
		else:
			self.embedder = nn.Embedding(self.vocab_size, self.embedding_size,
				sparse=False, padding_idx=PAD)

		# ------ define acous att --------
		dropout_acous_att = dropout
		self.acous_hidden_size_att = 0 # ignored with bilinear

		self.acous_key_size = self.acous_hidden_size * 2 	# acous feats
		self.acous_value_size = self.acous_hidden_size * 2 	# acous feats
		self.acous_query_size = self.hidden_size_dec 		# use dec(words) as query
		self.acous_att = AttentionLayer(self.acous_query_size, self.acous_key_size,
									value_size=self.acous_value_size,
									mode=self.acous_att_mode,
									dropout=dropout_acous_att,
									query_transform=False,
									output_transform=False,
									hidden_size=self.acous_hidden_size_att,
									hard_att=False)

		# ------ define acous out --------
		self.acous_ffn = nn.Linear(self.acous_hidden_size * 2 + self.hidden_size_dec,
									self.hidden_size_shared, bias=False)
		self.acous_out = nn.Linear(self.hidden_size_shared, self.vocab_size, bias=True)


		# ------ define acous dec -------
		# embedding_size_dec + self.hidden_size_shared [200+200]-> hidden_size_dec [200]
		if not self.residual:
			self.dec = torch.nn.LSTM(
				self.embedding_size+self.hidden_size_shared, self.hidden_size_dec,
				num_layers=self.num_unilstm_dec, batch_first=batch_first,
				bias=True, dropout=dropout, bidirectional=False)
		else:
			self.dec = nn.Module()
			self.dec.add_module('l0', torch.nn.LSTM(
				self.embedding_size+self.hidden_size_shared, self.hidden_size_dec,
				num_layers=1, batch_first=batch_first,
				bias=True, dropout=dropout, bidirectional=False))

			for i in range(1, self.num_unilstm_dec):
				self.dec.add_module('l'+str(i), torch.nn.LSTM(self.hidden_size_dec,
					self.hidden_size_dec,num_layers=1, batch_first=batch_first,
					bias=True,dropout=dropout, bidirectional=False))


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None
			setattr(self, var_name, var_val)


	def forward(self, acous_outputs, acous_lens=None, tgt=None,
		hidden=None, is_training=False, teacher_forcing_ratio=0.0,
		beam_width=1, use_gpu=False):

		"""
			Args:
				enc_outputs: [batch_size, acous_len / 8, self.acous_hidden_size * 2]
				tgt: list of word_ids 			[b x seq_len]
				hidden: initial hidden state
				teacher_forcing_ratio: default at 1 - always teacher forcing
			Returns:
				decoder_outputs: list of step_output -
					log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
		"""

		# import pdb; pdb.set_trace()

		global device
		device = check_device(use_gpu)

		# 0. init var
		ret_dict = dict()
		ret_dict[KEY_ATTN_SCORE] = []

		decoder_outputs = []
		sequence_symbols = []
		batch_size = acous_outputs.size(0)

		if type(tgt) == type(None):
			tgt = torch.Tensor([BOS]).repeat(
				batch_size, self.max_seq_len).type(torch.LongTensor).to(device=device)

		max_seq_len = tgt.size(1)
		lengths = np.array([max_seq_len] * batch_size)

		# 1. convert id to embedding
		emb_tgt = self.embedding_dropout(self.embedder(tgt))

		# 2. att inputs: keys n values
		att_keys = acous_outputs
		att_vals = acous_outputs

		# generate acous mask: True for trailing 0's
		if type(acous_lens) != type(None):
			# reduce by 8
			lens = torch.cat([elem + 8 - elem % 8 for elem in acous_lens]) / 8
			max_acous_len = acous_outputs.size(1)
			# mask=True over trailing 0s
			mask = torch.arange(max_acous_len).to(device=device).expand(
				batch_size, max_acous_len) >= lens.unsqueeze(1).to(device=device)
		else:
			mask = None

		# 3. init hidden states
		dec_hidden = None

		# 4. run dec + att + shared + output
		"""
			teacher_forcing_ratio = 1.0 -> always teacher forcing
			E.g.:
				acous 	        = [acous_len/8]
				tgt_chunk in    = <bos> w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
				predicted       = w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
		"""

		# LAS under teacher forcing
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

		# no beam search decoding
		tgt_chunk = emb_tgt[:, 0].unsqueeze(1) # BOS
		cell_value = torch.FloatTensor([0]).repeat(
			batch_size, 1, self.hidden_size_shared).to(device=device)
		prev_c = torch.FloatTensor([0]).repeat(
			batch_size, 1, max_seq_len).to(device=device)
		sequence_embs = []
		for idx in range(max_seq_len - 1):
			# import pdb; pdb.set_trace()
			predicted_logsoftmax, dec_hidden, step_attn, c_out, cell_value, attn_output, logits = \
				self.forward_step(self.acous_att, self.acous_ffn, self.acous_out,
								att_keys, att_vals, tgt_chunk, cell_value,
								dec_hidden, mask, prev_c)
			predicted_logsoftmax = predicted_logsoftmax.squeeze(1) # [b, vocab_size]
			step_output = predicted_logsoftmax
			symbols, decoder_outputs, sequence_symbols, lengths = \
				self.decode(idx, step_output, decoder_outputs, sequence_symbols, lengths)
			prev_c = c_out
			if use_teacher_forcing:
				tgt_chunk = emb_tgt[:, idx+1].unsqueeze(1)
			else:
				tgt_chunk = self.embedder(symbols)
			sequence_embs.append(cell_value.squeeze(1)) # b x hidden_size_shared

		ret_dict[KEY_LENGTH] = lengths.tolist()
		sequence_embs = torch.stack(sequence_embs, dim=1) # b x l-1 x hidden_size_shared
		sequence_logps = torch.stack(decoder_outputs, dim=1) # b x l-1 x vocab_size
		sequence_symbols = torch.stack(sequence_symbols, dim=1) # b x l-1

		# import pdb; pdb.set_trace()

		return sequence_embs, sequence_logps, sequence_symbols, lengths


	def decode(self, step, step_output, decoder_outputs, sequence_symbols, lengths):

			"""
				Greedy decoding
				Args:
					step: step idx
					step_output: log predicted_softmax [batch_size, 1, vocab_size_dec]
				Returns:
					symbols: most probable symbol_id [batch_size, 1]
			"""
			decoder_outputs.append(step_output)
			symbols = decoder_outputs[-1].topk(1)[1]
			sequence_symbols.append(symbols)

			eos_batches = torch.max(symbols.data.eq(EOS), symbols.data.eq(PAD))
			# equivalent to logical OR
			# eos_batches = symbols.data.eq(PAD)
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				lengths[update_idx] = len(sequence_symbols)
			return symbols, decoder_outputs, sequence_symbols, lengths


	def forward_step(self, att_func, ffn_func, out_func,
		att_keys, att_vals, tgt_chunk, prev_cell_value,
		dec_hidden=None, mask_src=None, prev_c=None):

		"""
			manual unrolling

			Args:
				att_keys:   [batch_size, seq_len, acous_hidden_size * 2]
				att_vals:   [batch_size, seq_len, acous_hidden_size * 2]
				tgt_chunk:  tgt word embeddings
							no teacher forcing - [batch_size, 1, embedding_size_dec]
							(becomes 2d when indexed)
				prev_cell_value:
							previous cell value before prediction
							[batch_size, 1, self.state_size]
				dec_hidden:
							initial hidden state for dec layer
				mask_src:
							mask of PAD for src sequences
				prev_c:
							used in hybrid attention mechanism

			Returns:
				predicted_softmax: log probilities [batch_size, vocab_size_dec]
				dec_hidden: a list of hidden states of each dec layer
				attn: attention weights
				cell_value: transformed attention output
							[batch_size, 1, self.hidden_size_shared]
		"""

		# record sizes
		batch_size = tgt_chunk.size(0)
		tgt_chunk_etd = torch.cat([tgt_chunk, prev_cell_value], -1)
		tgt_chunk_etd = tgt_chunk_etd\
			.view(-1, 1, self.embedding_size + self.hidden_size_shared)

		# run dec
		# default dec_hidden: [h_0, c_0];
		# with h_0 [num_layers * num_directions(==1), batch, hidden_size]
		if not self.residual:
			dec_outputs, dec_hidden = self.dec(tgt_chunk, dec_hidden)
			dec_outputs = self.dropout(dec_outputs)
		else:
			# store states layer by layer -
			# num_layers * ([1, batch, hidden_size], [1, batch, hidden_size])
			dec_hidden_lis = []

			# layer0
			dec_func_first = getattr(self.dec, 'l0')
			if type(dec_hidden) == type(None):
				dec_outputs, dec_hidden_out = dec_func_first(tgt_chunk_etd, None)
			else:
				index = torch.tensor([0]).to(device=device) # choose the 0th layer
				dec_hidden_in = tuple(
					[h.index_select(dim=0, index=index) for h in dec_hidden])
				dec_outputs, dec_hidden_out = dec_func_first(tgt_chunk_etd, dec_hidden_in)
			dec_hidden_lis.append(dec_hidden_out)
			# no residual for 0th layer
			dec_outputs = self.dropout(dec_outputs)

			# layer1+
			for i in range(1, self.num_unilstm_dec):
				dec_inputs = dec_outputs
				dec_func = getattr(self.dec, 'l'+str(i))
				if type(dec_hidden) == type(None):
					dec_outputs, dec_hidden_out = dec_func(dec_inputs, None)
				else:
					index = torch.tensor([i]).to(device=device)
					dec_hidden_in = tuple(
						[h.index_select(dim=0, index=index) for h in dec_hidden])
					dec_outputs, dec_hidden_out = dec_func(dec_inputs, dec_hidden_in)
				dec_hidden_lis.append(dec_hidden_out)
				if i < self.num_unilstm_dec - 1:
					dec_outputs = dec_outputs + dec_inputs
				dec_outputs = self.dropout(dec_outputs)

			# convert to tuple
			h_0 = torch.cat([h[0] for h in dec_hidden_lis], 0)
			c_0 = torch.cat([h[1] for h in dec_hidden_lis], 0)
			dec_hidden = tuple([h_0, c_0])

		# run att
		att_func.set_mask(mask_src)
		att_outputs, attn, c_out = att_func(dec_outputs, att_keys, att_vals, prev_c=prev_c)
		att_outputs = self.dropout(att_outputs)

		# run ff + softmax
		ff_inputs = torch.cat((att_outputs, dec_outputs), dim=-1)
		ff_inputs_size = self.acous_hidden_size * 2 + self.hidden_size_dec
		cell_value = ffn_func(ff_inputs.view(-1, 1, ff_inputs_size))
		outputs = out_func(cell_value.contiguous().view(-1, self.hidden_size_shared))
		predicted_logsoftmax = F.log_softmax(outputs, dim=1).view(batch_size, 1, -1)

		return predicted_logsoftmax, dec_hidden, attn, c_out, cell_value, att_outputs, outputs
