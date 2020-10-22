import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.layers import TransformerDecoderLayer
from modules.layers import _get_pad_mask, _get_zero_mask, _get_subsequent_mask
from utils.config import PAD, EOS, BOS, UNK
from utils.dataset import load_pretrained_embedding
from utils.misc import check_device

from .Las import LAS
from .TFEnc import Encoder
from .TFDec import Decoder

import warnings
warnings.filterwarnings("ignore")


class Seq2seq(nn.Module):

	"""
		acous/en enc + en/de dec
		embedding passing
	"""

	def __init__(self,
		enc_vocab_size,
		dec_vocab_size,
		share_embedder,
		enc_embedding_size = 200,
		dec_embedding_size = 200,
		load_embedding_src = None,
		load_embedding_tgt = None,
		max_seq_len_src = 32,
		max_seq_len_tgt = 300,
		num_heads = 8,
		dim_model = 512,
		dim_feedforward = 1024,
		enc_layers = 6,
		dec_layers = 6,
		embedding_dropout=0.0,
		dropout=0.2,
		act=False,
		enc_word2id=None,
		enc_id2word=None,
		dec_word2id=None,
		dec_id2word=None,
		transformer_type='standard',
		enc_emb_proj=False,
		dec_emb_proj=False,
		# pyramidal lstm params
		acous_dim=40,
		acous_hidden_size=256,
		# mode to select params to init
		mode='ASR',
		load_mode='ASR', # useful for storing frozen var
		# other
		perturb_emb=False,
		perturb_emb_fraction=0.0,
		train_rnnlm=False
		):

		super(Seq2seq, self).__init__()

		# define var
		self.enc_vocab_size = enc_vocab_size
		self.dec_vocab_size = dec_vocab_size
		self.enc_embedding_size = enc_embedding_size
		self.dec_embedding_size = dec_embedding_size
		self.load_embedding_src = load_embedding_src
		self.load_embedding_tgt = load_embedding_tgt
		self.max_seq_len_src = max_seq_len_src
		self.max_seq_len_tgt = max_seq_len_tgt
		self.num_heads = num_heads
		self.dim_model = dim_model
		self.dim_feedforward = dim_feedforward

		self.enc_layers = enc_layers
		self.dec_layers = dec_layers

		self.embedding_dropout = nn.Dropout(embedding_dropout)
		self.dropout = nn.Dropout(dropout)
		self.act = act
		self.enc_emb_proj = enc_emb_proj
		self.dec_emb_proj = dec_emb_proj

		self.enc_word2id = enc_word2id
		self.enc_id2word = enc_id2word
		self.dec_word2id = dec_word2id
		self.dec_id2word = dec_id2word
		self.transformer_type = transformer_type
		self.mode = mode
		self.load_mode = load_mode

		self.perturb_emb = perturb_emb
		self.perturb_emb_fraction = perturb_emb_fraction

		# ------------- define embedders -------------
		if self.load_embedding_src:
			embedding_matrix = np.random.rand(self.enc_vocab_size, self.enc_embedding_size)
			embedding_matrix = torch.FloatTensor(load_pretrained_embedding(
				self.enc_word2id, embedding_matrix, self.load_embedding_src))
			self.enc_embedder = nn.Embedding.from_pretrained(embedding_matrix,
				freeze=False, sparse=False, padding_idx=PAD)
		else:
			self.enc_embedder = nn.Embedding(self.enc_vocab_size,
				self.enc_embedding_size, sparse=False, padding_idx=PAD)

		if self.load_embedding_tgt:
			embedding_matrix = np.random.rand(self.dec_vocab_size, self.dec_embedding_size)
			embedding_matrix = torch.FloatTensor(load_pretrained_embedding(
				self.dec_word2id, embedding_matrix, self.load_embedding_tgt))
			self.dec_embedder = nn.Embedding.from_pretrained(embedding_matrix,
				freeze=False, sparse=False, padding_idx=PAD)
		else:
			self.dec_embedder = nn.Embedding(self.dec_vocab_size,
				self.dec_embedding_size, sparse=False, padding_idx=PAD)

		if share_embedder:
			assert enc_vocab_size == dec_vocab_size
			self.enc_embedder = self.dec_embedder

		self.enc_emb_proj_flag = False
		if (self.enc_embedding_size != self.dim_model) or (self.enc_emb_proj == True):
			self.enc_emb_proj_flag = True
			self.enc_emb_proj = nn.Linear(self.enc_embedding_size,
				self.dim_model, bias=False) # embedding -> hidden

		self.dec_emb_proj_flag = False
		if (self.dec_embedding_size != self.dim_model) or (self.dec_emb_proj == True):
			self.dec_emb_proj_flag = True
			self.dec_emb_proj = nn.Linear(self.dec_embedding_size,
				self.dim_model, bias=False) # embedding -> hidden

		# ------------- construct enc, dec  -------------------
		# params
		self.acous_dim = acous_dim
		self.acous_hidden_size = acous_hidden_size
		enc_params = (self.dim_model, self.dim_feedforward, self.num_heads,
			self.enc_layers, self.act, dropout, self.transformer_type)
		dec_params = (self.dim_model, self.dim_feedforward, self.num_heads,
			self.dec_layers, self.act, dropout, self.transformer_type)

		# LAS
		comb_mode = '-'.join([self.mode,self.load_mode])
		if 'ASR' in comb_mode or 'ST' in comb_mode:
			self.las = LAS(
				self.enc_vocab_size,
				embedding_size=self.enc_embedding_size,
				acous_dim=self.acous_dim,
				acous_hidden_size=self.acous_hidden_size,
				acous_att_mode='bilinear',
				hidden_size_dec=self.dim_model,
				hidden_size_shared=self.dim_model,
				num_unilstm_dec=3,
				#
				acous_norm=True,
				spec_aug=True,
				batch_norm=False,
				enc_mode='pyramid',
				#
				embedding_dropout=embedding_dropout,
				dropout=dropout,
				residual=True,
				batch_first=True,
				max_seq_len=self.max_seq_len_src,
				embedder=None, # do not share embedder with text encoder
				word2id=self.enc_word2id,
				id2word=self.enc_id2word,
				hard_att=False,
				train_rnnlm=train_rnnlm
			)

		# En decode
		if 'AE' in comb_mode:
			self.out_src = self.las.decoder.acous_out # share with las out layer

		# En encode
		# De decode
		if 'ST' in comb_mode or 'MT' in comb_mode:
			self.enc_src = Encoder(*enc_params)
			self.dec_tgt = Decoder(*dec_params)
			self.out_tgt = nn.Linear(self.dim_model, self.dec_vocab_size, bias=False)


	def _perturb_emb(self, emb, device):

		""" add gaussian noice to embeddings """

		STD = 0.03 							# gaussian noise
		PORTION = self.perturb_emb_fraction	# perturb some of the words

		mask = (torch.rand(emb.size()[:2]) > (1-PORTION)).unsqueeze(2).to(device=device)
		pert = torch.empty(emb.size()).normal_(mean=0,std=STD).to(device=device)

		emb = emb + pert * mask.to(torch.float)
		return emb


	def _get_src_emb(self, src, device):
		# En mask
		src_mask_input = _get_pad_mask(src).to(device=device).type(torch.uint8)
		src_mask = ((_get_pad_mask(src).to(device=device).type(torch.uint8)
			& _get_subsequent_mask(src.size(-1)).type(torch.uint8).to(device=device)))
		if self.enc_emb_proj_flag:
			emb_src = self.enc_emb_proj(self.embedding_dropout(self.enc_embedder(src)))
		else:
			emb_src = self.embedding_dropout(self.enc_embedder(src))

		return src_mask, emb_src, src_mask_input


	def _get_tgt_emb(self, tgt, device):
		# De mask
		tgt_mask = ((_get_pad_mask(tgt).to(device=device).type(torch.uint8)
			& _get_subsequent_mask(tgt.size(-1)).type(torch.uint8).to(device=device)))
		if self.dec_emb_proj_flag:
			emb_tgt = self.dec_emb_proj(self.embedding_dropout(self.dec_embedder(tgt)))
		else:
			emb_tgt = self.embedding_dropout(self.dec_embedder(tgt))

		return tgt_mask, emb_tgt


	def _pre_proc_src(self, src, device):

		# remove initial BOS：to match with _encouder_acous output
		src_proc = src[:,1:]

		return src_proc


	def _encoder_acous(self, acous_feats, acous_lens, device, use_gpu, tgt=None,
		is_training=False, teacher_forcing_ratio=0.0, lm_mode='null', lm_model=None):
		# get acoustics - [batch_size, acous_len / 8, self.acous_hidden_size * 2]
		emb_src, logps_src, preds_src, lengths = self.las(acous_feats,
			acous_lens=acous_lens, tgt=tgt, is_training=is_training,
			teacher_forcing_ratio=teacher_forcing_ratio, use_gpu=use_gpu,
			lm_mode=lm_mode, lm_model=lm_model)

		return emb_src, logps_src, preds_src, lengths


	def _encoder_en(self, emb_src, src_mask=None):
		# En encoder
		enc_outputs, *_ = self.enc_src(emb_src, src_mask=src_mask)	# b x len x dim_model

		return enc_outputs


	def _decoder_en(self, emb_src):
		# En decoder
		logits_src = self.out_src(emb_src)	# b x len x vocab_size
		logps_src = torch.log_softmax(logits_src, dim=2)
		scores_src, preds_src = logps_src.data.topk(1)

		return logits_src, logps_src, preds_src, scores_src


	def _decoder_de(self, emb_tgt, enc_outputs,
		tgt_mask=None, src_mask=None, beam_width=1):
		# De decoder
		dec_outputs_tgt, *_ = self.dec_tgt(emb_tgt, enc_outputs, tgt_mask=tgt_mask, src_mask=src_mask)
		logits_tgt = self.out_tgt(dec_outputs_tgt)	# b x len x vocab_size
		logps_tgt = torch.log_softmax(logits_tgt, dim=2)
		scores_tgt, preds_tgt = logps_tgt.data.topk(beam_width)

		return dec_outputs_tgt, logits_tgt, logps_tgt, preds_tgt, scores_tgt


	def _prep_eval(self, batch, length_out, vocab_size, device):

		# eos
		eos_mask = torch.BoolTensor([False]).repeat(batch).to(device=device)
		# record
		logps = torch.Tensor([(1.0/vocab_size)]).log().repeat(batch,length_out,vocab_size).type(
			torch.FloatTensor).to(device=device)
		dec_outputs = torch.Tensor([0]).repeat(batch,length_out,self.dim_model).type(
			torch.FloatTensor).to(device=device)
		preds_save = torch.Tensor([PAD]).repeat(batch,length_out).type(
			torch.LongTensor).to(device=device) # used to update pred history

		# start from length = 1
		preds = torch.Tensor([BOS]).repeat(batch,1).type(
			torch.LongTensor).to(device=device)
		preds_save[:, 0] = preds[:, 0]

		return eos_mask, logps, dec_outputs, preds_save, preds, preds_save


	def _step_eval(self, i, eos_mask, dec_output, logp, pred,
		dec_outputs, logps, preds_save, preds, batch, length_out):

		# import pdb; pdb.set_trace
		eos_mask = ((pred[:, i-1].squeeze(1) == EOS).type(torch.uint8)
			+ eos_mask.type(torch.uint8)).type(torch.bool).type(torch.uint8) # >=pt1.1

		# b x len x dim_model - [:,0,:] is dummy 0's
		dec_outputs[:, i, :] = dec_output[:, i-1]
		# b x len x vocab_size - [:,0,:] is dummy - (1/vocab_size).log() # individual logps
		logps[:, i, :] = logp[:, i-1, :]
		# b x len - [:,0] is BOS
		preds_save[:, i] = pred[:, i-1].view(-1)

		# append current pred, length+1
		preds = torch.cat((preds,pred[:, i-1]),dim=1)
		flag = 0
		if sum(eos_mask.int()) == eos_mask.size(0):
			flag = 1
			if length_out != preds.size(1):
				dummy = torch.Tensor([PAD]).repeat(batch, length_out-preds.size(1)).type(
					torch.LongTensor).to(device=device)
				preds = torch.cat((preds,dummy),dim=1) # pad to max length

		return eos_mask, dec_outputs, logps, preds_save, preds, flag


	def _prep_translate(self, batch, beam_width, device, length_in, enc_outputs,
		src_mask_input=None):

		# prep
		eos_mask = torch.BoolTensor([False]).repeat(batch * beam_width).to(device=device)
		len_map = torch.Tensor([1]).repeat(batch * beam_width).to(device=device)
		preds = torch.Tensor([BOS]).repeat(batch, 1).type(
			torch.LongTensor).to(device=device)

		# repeat for beam_width times
		# a b c d -> aaa bbb ccc ddd

		# b x len x dim_model -> (b x beam_width) x len x dim_model
		enc_outputs_expand = enc_outputs.repeat(1, beam_width, 1).view(-1, length_in, self.dim_model)
		# (b x beam_width) x len
		preds_expand = preds.repeat(1, beam_width).view(-1, preds.size(-1))
		# (b x beam_width)
		scores_expand = torch.Tensor([0]).repeat(batch * beam_width).type(
			torch.FloatTensor).to(device=device)
		# b x 1 x len -> (b x beam_width) x 1 x len
		if type(src_mask_input) != type(None):
			src_mask_input_expand = src_mask_input.repeat(
				1, beam_width, 1).view(-1, 1, src_mask_input.size(-1))
		else:
			src_mask_input_expand = None

		return eos_mask, len_map, preds, enc_outputs_expand, preds_expand, \
			scores_expand, src_mask_input_expand


	def _step_translate(self, i, batch, beam_width, device,
		dec_output_expand, logp_expand, pred_expand, score_expand,
		preds_expand, scores_expand, eos_mask, len_map, penalty_factor):

		# select current slice
		dec_output = dec_output_expand[:, i-1]	# (b x beam_width) x dim_model - nouse
		logp = logp_expand[:, i-1, :] 	# (b x beam_width) x vocab_size - nouse
		pred = pred_expand[:, i-1] 		# (b x beam_width) x beam_width
		score = score_expand[:, i-1]		# (b x beam_width) x beam_width

		# select k candidates from k^2 candidates
		if i == 1:
			# inital state, keep first k candidates
			# b x (beam_width x beam_width) -> b x (beam_width) -> (b x beam_width) x 1
			score_select = scores_expand + score.reshape(batch, -1)[:,:beam_width]\
				.contiguous().view(-1)
			scores_expand = score_select
			pred_select = pred.reshape(batch, -1)[:, :beam_width].contiguous().view(-1)
			preds_expand = torch.cat((preds_expand,pred_select.unsqueeze(-1)),dim=1)

		else:
			# keep only 1 candidate when hitting eos
			# (b x beam_width) x beam_width
			eos_mask_expand = eos_mask.reshape(-1,1).repeat(1, beam_width)
			eos_mask_expand[:,0] = False
			# (b x beam_width) x beam_width
			score_temp = scores_expand.reshape(-1,1) + score.masked_fill(
				eos_mask.reshape(-1,1), 0).masked_fill(eos_mask_expand, -1e9)
			# length penalty
			score_temp = score_temp / (len_map.reshape(-1,1) ** penalty_factor)
			# select top k from k^2
			# (b x beam_width^2 -> b x beam_width)
			score_select, pos = score_temp.reshape(batch, -1).topk(beam_width)
			scores_expand = score_select.view(-1) * (len_map.reshape(-1,1) ** penalty_factor).view(-1)
			# select correct elements according to pos
			pos = (pos.float() + torch.range(0, (batch - 1) * (beam_width**2), (beam_width**2)).to(
				device=device).reshape(batch, 1)).long()
			r_idxs, c_idxs = pos // beam_width, pos % beam_width # b x beam_width
			pred_select = pred[r_idxs, c_idxs].view(-1) # b x beam_width -> (b x beam_width)
			# Copy the corresponding previous tokens.
			preds_expand[:, :i] = preds_expand[r_idxs.view(-1), :i] # (b x beam_width) x i
			# Set the best tokens in this beam search step
			preds_expand = torch.cat((preds_expand, pred_select.unsqueeze(-1)),dim=1)

		# locate the eos in the generated sequences
		# eos_mask = (pred_select == EOS) + eos_mask # >=pt1.3
		eos_mask = ((pred_select == EOS).type(torch.uint8)
			+ eos_mask.type(torch.uint8)).type(torch.bool).type(torch.uint8) # >=pt1.1
		len_map = len_map + torch.Tensor([1]).repeat(batch * beam_width).to(
			device=device).masked_fill(eos_mask.type(torch.uint8), 0)

		# early stop
		flag = 0
		if sum(eos_mask.int()) == eos_mask.size(0): flag = 1

		return scores_expand, preds_expand, eos_mask, len_map, flag


	def forward_train(self, src, tgt=None, acous_feats=None, acous_lens=None,
		mode='ST', use_gpu=True, lm_mode='null', lm_model=None):

		"""
			mode: 	ASR 		acous -> src
					AE 			src -> src
					ST			acous -> tgt
					MT 			src -> tgt
		"""

		# for backwrad compatibility
		self.check_var('perturb_emb', False)

		# import pdb; pdb.set_trace()
		# note: adding .type(torch.uint8) to be compatible with pytorch 1.1!
		out_dict={}

		# check gpu
		global device
		device = check_device(use_gpu)

		# check mode
		mode = mode.upper()
		assert type(src) != type(None)
		if 'ST' in mode or 'ASR' in mode:
			assert type(acous_feats) != type(None)
		if 'ST' in mode or 'MT' in mode:
			assert type(tgt) != type(None)

		if 'ASR' in mode:
			"""
				acous -> EN: RNN
				in : length reduced fbk features
				out: w1 w2 w3 <EOS> <PAD> <PAD> #=6
			"""
			emb_src, logps_src, preds_src, lengths = self._encoder_acous(acous_feats, acous_lens,
				device, use_gpu, tgt=src, is_training=True, teacher_forcing_ratio=1.0,
				lm_mode=lm_mode, lm_model=lm_model)

			# output dict
			out_dict['emb_asr'] = emb_src
			out_dict['preds_asr'] = preds_src
			out_dict['logps_asr'] = logps_src
			out_dict['lengths_asr'] = lengths

		if 'AE' in mode:
			"""
				EN -> EN: Embedder
				src: <BOS> w1 w2 w3 <EOS> <PAD> <PAD> #=7
				mid: w1 w2 w3 <EOS> <PAD> <PAD> #=6
				out: w1 w2 w3 <EOS> <PAD> <PAD> #=6
			"""

			if 'ASR' in mode:
				src_trim = out_dict['preds_asr'].squeeze(2)
			else:
				# run asr: by default
				_, _, preds_src, _ = self._encoder_acous(acous_feats, acous_lens,
					device, use_gpu, tgt=src, is_training=True, teacher_forcing_ratio=1.0,
					lm_mode=lm_mode, lm_model=lm_model)
				src_trim = preds_src.squeeze(2)

			# from src: use with NLL loss
			# src_trim = self._pre_proc_src(src, device)

			src_mask, emb_src, src_mask_input = self._get_src_emb(src_trim, device)
			logits_src, logps_src, preds_src, _ = self._decoder_en(emb_src)

			# output dict
			out_dict['emb_ae'] = emb_src
			out_dict['refs_ae'] = src_trim
			out_dict['preds_ae'] = preds_src
			out_dict['logps_ae'] = logps_src

		if 'MT' in mode:
			"""
				EN -> DE: Transformer
				src: <BOS> w1 w2 w3 <EOS> <PAD> <PAD> #=7
				mid: w1 w2 w3 <EOS> <PAD> <PAD> #=6
				out: c1 c2 c3 <EOS> <PAD> <PAD> [dummy] #=7

				optional: add perturbation to embeddings
			"""
			# get tgt emb
			tgt_mask, emb_tgt = self._get_tgt_emb(tgt, device)
			# get static src emb
			src_trim = self._pre_proc_src(src, device)
			src_mask, emb_src, src_mask_input = self._get_src_emb(src_trim, device)
			# perturb emb
			if self.perturb_emb: emb_src = self._perturb_emb(emb_src, device)
			# encode decode
			enc_outputs = self._encoder_en(emb_src, src_mask=src_mask_input) # b x len x dim_model
			# decode
			dec_outputs_tgt, logits_tgt, logps_tgt, preds_tgt, _ = \
				self._decoder_de(emb_tgt, enc_outputs, tgt_mask=tgt_mask, src_mask=src_mask_input)

			# output dict
			out_dict['emb_mt'] = emb_src
			out_dict['preds_mt'] = preds_tgt
			out_dict['logps_mt'] = logps_tgt

		if 'ST' in mode:
			"""
				acous -> DE: Transformer
				in : length reduced fbk features
				mid: w1 w2 w3 <EOS> <PAD> <PAD> #=6
				out: c1 c2 c3 <EOS> <PAD> <PAD> [dummy] #=7
			"""
			# get tgt emb
			tgt_mask, emb_tgt = self._get_tgt_emb(tgt, device)
			# get dynamic src emb
			if 'ASR' in mode:
				emb_src = out_dict['emb_asr']
				lengths = out_dict['lengths_asr']
			# else:
			# 	emb_src, _, _, lengths = self._encoder_acous(acous_feats, acous_lens,
			# 		device, use_gpu, tgt=src, is_training=True, teacher_forcing_ratio=1.0)
			else: # use free running if no 'ASR'
				emb_src, _, _, lengths = self._encoder_acous(acous_feats, acous_lens,
					device, use_gpu, is_training=False, teacher_forcing_ratio=0.0,
					lm_mode=lm_mode, lm_model=lm_model)
			# get mask
			max_len = emb_src.size(1)
			lengths = torch.LongTensor(lengths)
			src_mask_input = (torch.arange(max_len).expand(len(lengths), max_len)
				< lengths.unsqueeze(1)).unsqueeze(1).to(device=device)
			# encode
			enc_outputs = self._encoder_en(emb_src, src_mask=src_mask_input) # b x len x dim_model
			# decode
			dec_outputs_tgt, logits_tgt, logps_tgt, preds_tgt, _ = \
				self._decoder_de(emb_tgt, enc_outputs, tgt_mask=tgt_mask, src_mask=src_mask_input)

			# output dict
			out_dict['emb_st'] = emb_src
			out_dict['preds_st'] = preds_tgt
			out_dict['logps_st'] = logps_tgt

		return out_dict


	def forward_eval(self, src=None, acous_feats=None, acous_lens=None,
		mode='ST', use_gpu=True, lm_mode='null', lm_model=None):

		"""
			beam_width = 1
			note the output sequence different from training if using transformer model
		"""

		# import pdb; pdb.set_trace()
		out_dict={}

		# check gpu
		global device
		device = check_device(use_gpu)

		# check mode
		mode = mode.upper()
		if 'ST' in mode or 'ASR' in mode:
			assert type(acous_feats) != type(None)
			batch = acous_feats.size(0)
		if 'MT' in mode or 'AE' in mode:
			assert type(src) != type(None)
			batch = src.size(0)

		length_out_src = self.max_seq_len_src
		length_out_tgt = self.max_seq_len_tgt

		if 'ASR' in mode:
			"""
				acous -> EN: RNN
				in : length reduced fbk features
				out: w1 w2 w3 <EOS> <PAD> <PAD> #=6
			"""
			# run asr
			emb_src, logps_src, preds_src, lengths = self._encoder_acous(acous_feats, acous_lens,
				device, use_gpu, is_training=False, teacher_forcing_ratio=0.0,
				lm_mode=lm_mode, lm_model=lm_model)

			# output dict
			out_dict['emb_asr'] = emb_src
			out_dict['preds_asr'] = preds_src
			out_dict['logps_asr'] = logps_src
			out_dict['lengths_asr'] = lengths

		if 'AE' in mode:
			"""
				EN -> EN: Embedder
				in : <BOS> w1 w2 w3 <EOS> <PAD> <PAD> #=7
				mid: w1 w2 w3 <EOS> <PAD> <PAD> #=6
				out: w1 w2 w3 <EOS> <PAD> <PAD> #=6
			"""
			# txt encoder
			if 'ASR' in mode:
				src_trim = out_dict['preds_asr'].squeeze(2)
			else:
				# run asr: by default
				_, _, preds_src, _ = self._encoder_acous(acous_feats, acous_lens,
					device, use_gpu, is_training=False, teacher_forcing_ratio=0.0,
					lm_mode=lm_mode, lm_model=lm_model)
				src_trim = preds_src.squeeze(2)

			# from src
			# src_trim = self._pre_proc_src(src, device)

			src_mask, emb_src, src_mask_input = self._get_src_emb(src_trim, device)
			logits_src, logps_src, preds_src, _ = self._decoder_en(emb_src)

			# output dict
			out_dict['emb_ae'] = emb_src
			out_dict['refs_ae'] = src_trim
			out_dict['preds_ae'] = preds_src
			out_dict['logps_ae'] = logps_src

		if 'MT' in mode:
			"""
				EN -> DE: Transformer
				in : <BOS> w1 w2 w3 <EOS> <PAD> <PAD> #=7
				mid: w1 w2 w3 <EOS> <PAD> <PAD> <PAD> #=7
				out: <BOS> c1 c2 c3 <EOS> <PAD> <PAD> #=7
			"""
			# txt encoder
			src_trim = self._pre_proc_src(src, device)
			src_mask, emb_src, src_mask_input = self._get_src_emb(src_trim, device)
			enc_outputs = self._encoder_en(emb_src, src_mask=src_mask_input) # b x len x dim_model

			# prep
			eos_mask_tgt, logps_tgt, dec_outputs_tgt, preds_save_tgt, preds_tgt, preds_save_tgt = \
				self._prep_eval(batch, length_out_tgt, self.dec_vocab_size, device)

			for i in range(1, self.max_seq_len_tgt):

				tgt_mask, emb_tgt = self._get_tgt_emb(preds_tgt, device)
				dec_output_tgt, logit_tgt, logp_tgt, pred_tgt, _ = \
					self._decoder_de(emb_tgt, enc_outputs, tgt_mask=tgt_mask, src_mask=src_mask_input)

				eos_mask_tgt, dec_outputs_tgt, logps_tgt, preds_save_tgt, preds_tgt, flag \
					= self._step_eval(i, eos_mask_tgt, dec_output_tgt, logp_tgt, pred_tgt,
						dec_outputs_tgt, logps_tgt, preds_save_tgt, preds_tgt, batch, length_out_tgt)
				if flag == 1: break

			# output dict
			out_dict['emb_mt'] = emb_src
			out_dict['preds_mt'] = preds_tgt
			out_dict['logps_mt'] = logps_tgt

		if 'ST' in mode:
			"""
				acous -> DE: Transformer
				in : length reduced fbk features
				out: <BOS> c1 c2 c3 <EOS> <PAD> <PAD> #=7
			"""
			# get embedding
			if 'ASR' in mode:
				emb_src = out_dict['emb_asr']
				lengths = out_dict['lengths_asr']
			else:
				emb_src, _, _, lengths = self._encoder_acous(acous_feats, acous_lens, device,
					use_gpu, is_training=False, teacher_forcing_ratio=0.0,
					lm_mode=lm_mode, lm_model=lm_model)
			# get mask
			max_len = emb_src.size(1)
			lengths = torch.LongTensor(lengths)
			src_mask_input = (torch.arange(max_len).expand(len(lengths), max_len)
				< lengths.unsqueeze(1)).unsqueeze(1).to(device=device)
			# encode
			enc_outputs = self._encoder_en(emb_src, src_mask=src_mask_input) # b x len x dim_model

			# prep
			eos_mask_tgt, logps_tgt, dec_outputs_tgt, preds_save_tgt, preds_tgt, preds_save_tgt = \
				self._prep_eval(batch, length_out_tgt, self.dec_vocab_size, device)

			for i in range(1, self.max_seq_len_tgt):

				tgt_mask, emb_tgt = self._get_tgt_emb(preds_tgt, device)
				dec_output_tgt, logit_tgt, logp_tgt, pred_tgt, _ = \
					self._decoder_de(emb_tgt, enc_outputs, tgt_mask=tgt_mask, src_mask=src_mask_input)

				eos_mask_tgt, dec_outputs_tgt, logps_tgt, preds_save_tgt, preds_tgt, flag \
					= self._step_eval(i, eos_mask_tgt, dec_output_tgt, logp_tgt, pred_tgt,
						dec_outputs_tgt, logps_tgt, preds_save_tgt, preds_tgt, batch, length_out_tgt)
				if flag == 1: break

			# output dict
			out_dict['emb_st'] = emb_src
			out_dict['preds_st'] = preds_tgt
			out_dict['logps_st'] = logps_tgt

		return out_dict


	def forward_translate(self, acous_feats=None, acous_lens=None, src=None,
		beam_width=1, penalty_factor=1, use_gpu=True, max_seq_len=900, mode='ST',
		lm_mode='null', lm_model=None):

		"""
			run inference - with beam search (same output format as is in forward_eval)
		"""

		# import pdb; pdb.set_trace()

		# check gpu
		global device
		device = check_device(use_gpu)

		if mode == 'ASR':
			_, _, preds_src, _ = self._encoder_acous(acous_feats, acous_lens, device, use_gpu,
				is_training=False, teacher_forcing_ratio=0.0, lm_mode=lm_mode, lm_model=lm_model)
			preds = preds_src

		elif mode == 'AE':

			# run asr: by default
			_, _, preds_src, _ = self._encoder_acous(acous_feats, acous_lens, device, use_gpu,
				is_training=False, teacher_forcing_ratio=0.0, lm_mode=lm_mode, lm_model=lm_model)
			src_trim = preds_src.squeeze(2)

			# from src: use with NLL loss
			# src_trim = self._pre_proc_src(src, device)

			src_mask, emb_src, src_mask_input = self._get_src_emb(src_trim, device)
			logits_src, logps_src, preds_src, _ = self._decoder_en(emb_src)
			preds = preds_src

		elif mode == 'MT':
			batch = src.size(0)

			# txt encoder
			src_trim = self._pre_proc_src(src, device)
			src_mask, emb_src, src_mask_input = self._get_src_emb(src_trim, device)
			enc_outputs = self._encoder_en(emb_src, src_mask=src_mask_input)
			length_in = enc_outputs.size(1)

			# prep
			eos_mask, len_map, preds, enc_outputs_expand, preds_expand, \
				scores_expand, src_mask_input_expand = self._prep_translate(
				batch, beam_width, device, length_in, enc_outputs, src_mask_input)

			# loop over sequence length
			for i in range(1, max_seq_len):

				tgt_mask_expand, emb_tgt_expand = self._get_tgt_emb(preds_expand, device)
				dec_output_expand, logit_expand, logp_expand, pred_expand, score_expand = \
					self._decoder_de(emb_tgt_expand, enc_outputs_expand,
					tgt_mask=tgt_mask_expand, src_mask=src_mask_input_expand,
					beam_width=beam_width)

				scores_expand, preds_expand, eos_mask, len_map, flag = \
					self._step_translate(i, batch, beam_width, device,
						dec_output_expand, logp_expand, pred_expand, score_expand,
						preds_expand, scores_expand, eos_mask, len_map, penalty_factor)
				if flag == 1: break

			# select the best candidate
			preds = preds_expand.reshape(batch, -1)[:, :max_seq_len].contiguous() # b x len
			scores = scores_expand.reshape(batch, -1)[:, 0].contiguous() # b

		elif mode == 'ST':
			batch = acous_feats.size(0)

			# get embedding
			emb_src, _, _, lengths = self._encoder_acous(acous_feats, acous_lens, device, use_gpu,
				is_training=False, teacher_forcing_ratio=0.0, lm_mode=lm_mode, lm_model=lm_model)

			# get mask
			max_len = emb_src.size(1)
			lengths = torch.LongTensor(lengths)
			src_mask_input = (torch.arange(max_len).expand(len(lengths), max_len)
				< lengths.unsqueeze(1)).unsqueeze(1).to(device=device)
			# encode
			enc_outputs = self._encoder_en(emb_src, src_mask=src_mask_input) # b x len x dim_model
			length_in = enc_outputs.size(1)

			# prep
			eos_mask, len_map, preds, enc_outputs_expand, preds_expand, \
				scores_expand, src_mask_input_expand = self._prep_translate(
				batch, beam_width, device, length_in, enc_outputs, src_mask_input)

			# loop over sequence length
			for i in range(1, max_seq_len):

				# import pdb; pdb.set_trace()

				# Get k candidates for each beam, k^2 candidates in total (k=beam_width)
				tgt_mask_expand, emb_tgt_expand = self._get_tgt_emb(preds_expand, device)
				dec_output_expand, logit_expand, logp_expand, pred_expand, score_expand = \
					self._decoder_de(emb_tgt_expand, enc_outputs_expand,
					tgt_mask=tgt_mask_expand, src_mask=src_mask_input_expand,
					beam_width=beam_width)

				scores_expand, preds_expand, eos_mask, len_map, flag = \
					self._step_translate(i, batch, beam_width, device,
						dec_output_expand, logp_expand, pred_expand, score_expand,
						preds_expand, scores_expand, eos_mask, len_map, penalty_factor)
				if flag == 1: break

			# select the best candidate
			preds = preds_expand.reshape(batch, -1)[:, :max_seq_len].contiguous() # b x len
			scores = scores_expand.reshape(batch, -1)[:, 0].contiguous() # b

		return preds


	def forward_translate_refen(self, acous_feats=None, acous_lens=None, src=None,
		beam_width=1, penalty_factor=1, use_gpu=True, max_seq_len=900, mode='ST',
		lm_mode='null', lm_model=None):

		"""
			run inference - with beam search (same output format as is in forward_eval)
		"""

		# import pdb; pdb.set_trace()

		# check gpu
		global device
		device = check_device(use_gpu)

		if mode == 'ASR':
			_, _, preds_src, _ = self._encoder_acous(acous_feats, acous_lens,
				device, use_gpu, tgt=src, is_training=False, teacher_forcing_ratio=1.0,
				lm_mode=lm_mode, lm_model=lm_model)

			preds = preds_src

		elif mode == 'AE':

			# run asr: by default
			_, _, preds_src, _ = self._encoder_acous(acous_feats, acous_lens,
				device, use_gpu, tgt=src, is_training=False, teacher_forcing_ratio=1.0,
				lm_mode=lm_mode, lm_model=lm_model)
			src_trim = preds_src.squeeze(2)

			# from src: use with NLL loss
			# src_trim = self._pre_proc_src(src, device)

			src_mask, emb_src, src_mask_input = self._get_src_emb(src_trim, device)
			logits_src, logps_src, preds_src, _ = self._decoder_en(emb_src)
			preds = preds_src

		elif mode == 'ST':
			batch = acous_feats.size(0)

			# get embedding
			emb_src, _, _, lengths = self._encoder_acous(acous_feats, acous_lens,
				device, use_gpu, tgt=src, is_training=False, teacher_forcing_ratio=1.0,
				lm_mode=lm_mode, lm_model=lm_model)

			# get mask
			max_len = emb_src.size(1)
			lengths = torch.LongTensor(lengths)
			src_mask_input = (torch.arange(max_len).expand(len(lengths), max_len)
				< lengths.unsqueeze(1)).unsqueeze(1).to(device=device)
			# encode
			enc_outputs = self._encoder_en(emb_src, src_mask=src_mask_input) # b x len x dim_model
			length_in = enc_outputs.size(1)

			# prep
			eos_mask, len_map, preds, enc_outputs_expand, preds_expand, \
				scores_expand, src_mask_input_expand = self._prep_translate(
				batch, beam_width, device, length_in, enc_outputs, src_mask_input)

			# loop over sequence length
			for i in range(1, max_seq_len):

				# import pdb; pdb.set_trace()

				# Get k candidates for each beam, k^2 candidates in total (k=beam_width)
				tgt_mask_expand, emb_tgt_expand = self._get_tgt_emb(preds_expand, device)
				dec_output_expand, logit_expand, logp_expand, pred_expand, score_expand = \
					self._decoder_de(emb_tgt_expand, enc_outputs_expand,
					tgt_mask=tgt_mask_expand, src_mask=src_mask_input_expand,
					beam_width=beam_width)

				scores_expand, preds_expand, eos_mask, len_map, flag = \
					self._step_translate(i, batch, beam_width, device,
						dec_output_expand, logp_expand, pred_expand, score_expand,
						preds_expand, scores_expand, eos_mask, len_map, penalty_factor)
				if flag == 1: break

			# select the best candidate
			preds = preds_expand.reshape(batch, -1)[:, :max_seq_len].contiguous() # b x len
			scores = scores_expand.reshape(batch, -1)[:, 0].contiguous() # b

		return preds


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None

			# set class attribute to default value
			setattr(self, var_name, var_val)
