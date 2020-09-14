import torch
import torch.utils.tensorboard
import random
import time
import os
import logging
import argparse
import sys
import numpy as np
import torchtext

from utils.misc import get_memory_alloc, check_device, add2corpus
from utils.misc import _convert_to_words_batchfirst, _convert_to_words
from utils.config import PAD, EOS
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss, KLDivLoss, MSELoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from .trainer_base import Trainer

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


class Trainer_AE_ASR_MT(Trainer):

	def __init__(self,
		expt_dir='experiment',
		load_dir=None,
		load_mode='null',
		load_freeze=True,
		checkpoint_every=100,
		print_every=100,
		batch_size=256,
		use_gpu=False,
		gpu_id=0,
		learning_rate=0.00001,
		learning_rate_init=0.0005,
		lr_warmup_steps=16000,
		max_grad_norm=1.0,
		eval_with_mask=True,
		max_count_no_improve=2,
		max_count_num_rollback=2,
		keep_num=1,
		normalise_loss=True,
		loss_coeff=None,
		minibatch_partition=1
		):

		# inheritence
		super().__init__(
			expt_dir=expt_dir,
			load_dir=load_dir,
			load_mode=load_mode,
			load_freeze=load_freeze,
			checkpoint_every=checkpoint_every,
			print_every=print_every,
			batch_size=batch_size,
			use_gpu=use_gpu,
			gpu_id=gpu_id,
			learning_rate=learning_rate,
			learning_rate_init=learning_rate_init,
			lr_warmup_steps=lr_warmup_steps,
			max_grad_norm=max_grad_norm,
			eval_with_mask=eval_with_mask,
			max_count_no_improve=max_count_no_improve,
			max_count_num_rollback=max_count_num_rollback,
			keep_num=keep_num,
			normalise_loss=normalise_loss,
			loss_coeff=loss_coeff,
			minibatch_partition=minibatch_partition
		)


	def _evaluate_batches(self, model, dataset_asr, dataset_mt):

		# import pdb; pdb.set_trace()

		model.eval()

		resloss_asr = 0
		resloss_ae = 0
		resloss_kl = 0
		resloss_l2 = 0
		resloss_mt = 0
		resloss_norm = 0

		# accuracy
		match_asr = 0
		total_asr = 0
		match_ae = 0
		total_ae = 0
		match_mt = 0
		total_mt = 0

		# bleu
		hyp_corpus_asr = []
		ref_corpus_asr = []
		hyp_corpus_ae = []
		ref_corpus_ae = []
		hyp_corpus_mt = []
		ref_corpus_mt = []

		# ------- ASR,AE --------
		evaliter = iter(dataset_asr.iter_loader)
		out_count = 0

		with torch.no_grad():
			for idx in range(len(evaliter)):
				batch_items = evaliter.next()

				# import pdb; pdb.set_trace()

				# load data
				batch_src_ids = batch_items['srcid'][0]
				batch_src_lengths = batch_items['srclen']
				batch_acous_feats = batch_items['acous_feat'][0]
				batch_acous_lengths = batch_items['acouslen']

				# separate into minibatch
				batch_size = batch_src_ids.size(0)
				batch_seq_len = int(max(batch_src_lengths))

				n_minibatch = int(batch_size / self.minibatch_size)
				n_minibatch += int(batch_size % self.minibatch_size > 0)

				for bidx in range(n_minibatch):

					loss_asr = NLLLoss()
					loss_asr.reset()
					loss_ae = NLLLoss()
					loss_ae.reset()
					loss_kl = KLDivLoss()
					loss_kl.reset()
					loss_l2 = MSELoss()
					loss_l2.reset()

					i_start = bidx * self.minibatch_size
					i_end = min(i_start + self.minibatch_size, batch_size)
					src_ids = batch_src_ids[i_start:i_end]
					src_lengths = batch_src_lengths[i_start:i_end]
					acous_feats = batch_acous_feats[i_start:i_end]
					acous_lengths = batch_acous_lengths[i_start:i_end]

					src_len = max(src_lengths)
					acous_len = max(acous_lengths)
					acous_len = acous_len + 8 - acous_len % 8
					src_ids = src_ids.to(device=self.device)
					acous_feats = acous_feats[:,:acous_len].to(device=self.device)

					# debug oom
					# acous_feats, acous_lengths, src_ids, tgt_ids = self._debug_oom(acous_len, acous_feats)

					non_padding_mask_src = src_ids.data.ne(PAD)

					# [run-TF] to save time
					# out_dict =  model.forward_train(src_ids, acous_feats=acous_feats,
					#	acous_lens=acous_lengths, mode='ASR', use_gpu=self.use_gpu)

					# [run-FR] to get true stats
					out_dict = model.forward_eval(src=src_ids, acous_feats=acous_feats,
						acous_lens=acous_lengths, mode='AE_ASR', use_gpu=self.use_gpu)

					preds_asr = out_dict['preds_asr']
					logps_asr = out_dict['logps_asr']
					emb_asr = out_dict['emb_asr']
					preds_ae = out_dict['preds_ae']
					logps_ae = out_dict['logps_ae']
					emb_ae = out_dict['emb_ae']
					refs_ae = out_dict['refs_ae']
					logps_hyp_asr = logps_asr
					preds_hyp_asr = preds_asr
					emb_hyp_asr = emb_asr
					logps_hyp_ae = logps_ae
					preds_hyp_ae = preds_ae
					emb_hyp_ae = emb_ae

					# evaluation
					if not self.eval_with_mask:
						loss_asr.eval_batch(logps_hyp_asr.reshape(-1, logps_hyp_asr.size(-1)),
							src_ids[:, 1:].reshape(-1))
						loss_asr.norm_term = 1.0 * src_ids.size(0) * src_ids[:,1:].size(1)
						loss_ae.eval_batch(logps_hyp_ae.reshape(-1, logps_hyp_ae.size(-1)),
							refs_ae.reshape(-1))
						loss_ae.norm_term = 1.0 * src_ids.size(0) * src_ids[:,1:].size(1)
						loss_kl.eval_batch(logps_hyp_ae.reshape(-1, logps_hyp_ae.size(-1)),
							logps_hyp_asr.reshape(-1, logps_hyp_asr.size(-1)))
						loss_kl.norm_term = 1.0 * src_ids.size(0) * src_ids[:,1:].size(1)
						loss_l2.eval_batch(emb_hyp_asr.reshape(-1, emb_hyp_asr.size(-1)),
							emb_hyp_ae.reshape(-1, emb_hyp_ae.size(-1)))
						loss_l2.norm_term = 1.0 * src_ids.size(0) * src_ids[:,1:].size(1)

					else:
						loss_asr.eval_batch_with_mask(logps_hyp_asr.reshape(-1, logps_hyp_asr.size(-1)),
							src_ids[:,1:].reshape(-1), non_padding_mask_src[:,1:].reshape(-1))
						loss_asr.norm_term = 1.0 * torch.sum(non_padding_mask_src[:,1:])
						loss_ae.eval_batch_with_mask(logps_hyp_ae.reshape(-1, logps_hyp_ae.size(-1)),
							refs_ae.reshape(-1), non_padding_mask_src[:,1:].reshape(-1))
						loss_ae.norm_term = 1.0 * torch.sum(non_padding_mask_src[:,1:])
						loss_kl.eval_batch_with_mask(logps_hyp_ae.reshape(-1, logps_hyp_ae.size(-1)),
							logps_hyp_asr.reshape(-1, logps_hyp_asr.size(-1)),
							non_padding_mask_src[:,1:].reshape(-1))
						loss_kl.norm_term = 1.0 * torch.sum(non_padding_mask_src[:,1:])
						loss_l2.eval_batch_with_mask(emb_hyp_asr.reshape(-1, emb_hyp_asr.size(-1)),
							emb_hyp_ae.reshape(-1, emb_hyp_ae.size(-1)),
							non_padding_mask_src[:,1:].reshape(-1))
						loss_l2.norm_term = 1.0 * torch.sum(non_padding_mask_src[:,1:])

					if self.normalise_loss:
						loss_asr.normalise()
						loss_ae.normalise()
						loss_kl.normalise()
						loss_l2.normalise()

					resloss_asr += loss_asr.get_loss()
					resloss_ae += loss_ae.get_loss()
					resloss_kl += loss_kl.get_loss()
					resloss_l2 += loss_l2.get_loss()
					resloss_norm += 1

					# ----- debug -----
					# print('{}/{}, {}/{}'.format(bidx, n_minibatch, idx, len(evaliter)))
					# if loss_kl.get_loss() > 10 or (bidx==4 and idx==1):
					# 	import pdb; pdb.set_trace()
					# -----------------

					# accuracy
					seqres_asr = preds_hyp_asr
					correct_asr = seqres_asr.reshape(-1).eq(src_ids[:,1:].reshape(-1))\
						.masked_select(non_padding_mask_src[:,1:].reshape(-1)).sum().item()
					match_asr += correct_asr
					total_asr += non_padding_mask_src[:,1:].sum().item()

					seqres_ae = preds_hyp_ae
					correct_ae = seqres_ae.reshape(-1).eq(refs_ae.reshape(-1))\
						.masked_select(non_padding_mask_src[:,1:].reshape(-1)).sum().item()
					match_ae += correct_ae
					total_ae += non_padding_mask_src[:,1:].sum().item()

					# append to refs_ae
					dummy = torch.zeros(refs_ae.size(0),1).to(device=self.device).long()
					refs_ae_add = torch.cat((dummy, refs_ae),dim=1)

					# print
					out_count = self._print(out_count, src_ids,
						dataset_asr.src_id2word, seqres_asr, tail='-asr')
					out_count_dummy = self._print(out_count, refs_ae_add,
						dataset_asr.src_id2word, seqres_ae, tail='-ae ')

					# accumulate corpus
					hyp_corpus_asr, ref_corpus_asr = add2corpus(seqres_asr, src_ids,
						dataset_asr.src_id2word, hyp_corpus_asr, ref_corpus_asr, type='word')
					hyp_corpus_ae, ref_corpus_ae = add2corpus(seqres_ae, refs_ae_add,
						dataset_asr.src_id2word, hyp_corpus_ae, ref_corpus_ae, type='word')

		# ------- MT --------
		evaliter = iter(dataset_mt.iter_loader)
		out_count = 0

		with torch.no_grad():
			for idx in range(len(evaliter)):
				batch_items = evaliter.next()

				# import pdb; pdb.set_trace()

				# load data
				batch_src_ids = batch_items['srcid'][0]
				batch_src_lengths = batch_items['srclen']
				batch_tgt_ids = batch_items['tgtid'][0]
				batch_tgt_lengths = batch_items['tgtlen']

				# separate into minibatch
				batch_size = batch_src_ids.size(0)
				batch_seq_len = int(max(batch_src_lengths))

				n_minibatch = int(batch_size / self.minibatch_size)
				n_minibatch += int(batch_size % self.minibatch_size > 0)

				for bidx in range(n_minibatch):

					loss_mt = NLLLoss()
					loss_mt.reset()

					i_start = bidx * self.minibatch_size
					i_end = min(i_start + self.minibatch_size, batch_size)
					src_ids = batch_src_ids[i_start:i_end]
					src_lengths = batch_src_lengths[i_start:i_end]
					tgt_ids = batch_tgt_ids[i_start:i_end]
					tgt_lengths = batch_tgt_lengths[i_start:i_end]

					src_len = max(src_lengths)
					tgt_len = max(tgt_lengths)
					src_ids = src_ids.to(device=self.device)
					tgt_ids = tgt_ids.to(device=self.device)

					# debug oom
					# acous_feats, acous_lengths, src_ids, tgt_ids = self._debug_oom(acous_len, acous_feats)

					non_padding_mask_tgt = tgt_ids.data.ne(PAD)
					non_padding_mask_src = src_ids.data.ne(PAD)

					# [run-TF] to save time
					# out_dict = model.forward_train(src_ids, tgt=tgt_ids, mode='MT', use_gpu=self.use_gpu)
					# preds_mt = out_dict['preds_mt']
					# logps_mt = out_dict['logps_mt']
					# logps_hyp_mt = logps_mt[:,:-1,:]
					# preds_hyp_mt = preds_mt[:,:-1]

					# [run-FR] to get true stats
					out_dict = model.forward_eval(src=src_ids, mode='MT', use_gpu=self.use_gpu)
					preds_mt = out_dict['preds_mt']
					logps_mt = out_dict['logps_mt']
					logps_hyp_mt = logps_mt[:,1:,:]
					preds_hyp_mt = preds_mt[:,1:]

					# evaluation
					if not self.eval_with_mask:
						loss_mt.eval_batch(logps_hyp_mt.reshape(-1, logps_hyp_mt.size(-1)),
							tgt_ids[:, 1:].reshape(-1))
						loss_mt.norm_term = 1.0 * tgt_ids.size(0) * tgt_ids[:,1:].size(1)
					else:
						loss_mt.eval_batch_with_mask(logps_hyp_mt.reshape(-1, logps_hyp_mt.size(-1)),
							tgt_ids[:,1:].reshape(-1), non_padding_mask_tgt[:,1:].reshape(-1))
						loss_mt.norm_term = 1.0 * torch.sum(non_padding_mask_tgt[:,1:])

					if self.normalise_loss:
						loss_mt.normalise()

					resloss_mt += loss_mt.get_loss()
					resloss_norm += 1

					# accuracy
					seqres_mt = preds_hyp_mt
					correct_mt = seqres_mt.reshape(-1).eq(tgt_ids[:,1:].reshape(-1))\
						.masked_select(non_padding_mask_tgt[:,1:].reshape(-1)).sum().item()
					match_mt += correct_mt
					total_mt += non_padding_mask_tgt[:,1:].sum().item()

					# print
					out_count = self._print(out_count, tgt_ids,
						dataset_mt.tgt_id2word, seqres_mt, tail='-mt ')

					# accumulate corpus
					hyp_corpus_mt, ref_corpus_mt = add2corpus(seqres_mt, tgt_ids,
						dataset_mt.tgt_id2word, hyp_corpus_mt,
						ref_corpus_mt, type=dataset_mt.use_type)

		# import pdb; pdb.set_trace()
		bleu_asr = torchtext.data.metrics.bleu_score(hyp_corpus_asr, ref_corpus_asr)
		bleu_ae = torchtext.data.metrics.bleu_score(hyp_corpus_ae, ref_corpus_ae)
		bleu_mt = torchtext.data.metrics.bleu_score(hyp_corpus_mt, ref_corpus_mt)

		# torch.cuda.empty_cache()
		if total_asr == 0:
			accuracy_asr = float('nan')
		else:
			accuracy_asr = match_asr / total_asr
		if total_ae == 0:
			accuracy_ae = float('nan')
		else:
			accuracy_ae = match_ae / total_ae
		if total_mt == 0:
			accuracy_mt = float('nan')
		else:
			accuracy_mt = match_mt / total_mt

		resloss_asr *= self.loss_coeff['nll_asr']
		resloss_asr /= (1.0 * resloss_norm)
		resloss_ae *= self.loss_coeff['nll_ae']
		resloss_ae /= (1.0 * resloss_norm)
		resloss_mt *= self.loss_coeff['nll_mt']
		resloss_mt /= (1.0 * resloss_norm)
		resloss_kl *= self.loss_coeff['kl_en']
		resloss_kl /= (1.0 * resloss_norm)
		resloss_l2 *= self.loss_coeff['l2']
		resloss_l2 /= (1.0 * resloss_norm)

		losses = {}
		losses['l2_loss'] = resloss_l2
		losses['kl_loss'] = resloss_kl
		losses['nll_loss_asr'] = resloss_asr
		losses['nll_loss_ae'] = resloss_ae
		losses['nll_loss_mt'] = resloss_mt
		metrics = {}
		metrics['accuracy_asr'] = accuracy_asr
		metrics['bleu_asr'] = bleu_asr
		metrics['accuracy_ae'] = accuracy_ae
		metrics['bleu_ae'] = bleu_ae
		metrics['accuracy_mt'] = accuracy_mt
		metrics['bleu_mt'] = bleu_mt

		return losses, metrics


	def _train_batch(self,
		model, batch_items_asr, batch_items_mt, step, total_steps):

		# ---------- ASR,AE ----------
		# load data
		batch_src_ids = batch_items_asr['srcid'][0]
		batch_src_lengths = batch_items_asr['srclen']
		batch_acous_feats = batch_items_asr['acous_feat'][0]
		batch_acous_lengths = batch_items_asr['acouslen']

		# separate into minibatch
		batch_size = batch_src_ids.size(0)
		batch_seq_len = int(max(batch_src_lengths))
		n_minibatch = int(batch_size / self.minibatch_size)
		n_minibatch += int(batch_size % self.minibatch_size > 0)
		resloss_asr = 0
		resloss_ae = 0
		resloss_kl = 0
		resloss_l2 = 0

		for bidx in range(n_minibatch):

			# debug
			# print(bidx,n_minibatch)
			# import pdb; pdb.set_trace()

			# define loss
			loss_asr = NLLLoss()
			loss_asr.reset()
			loss_ae = NLLLoss()
			loss_ae.reset()
			loss_kl = KLDivLoss()
			loss_kl.reset()
			loss_l2 = MSELoss()
			loss_l2.reset()

			# load data
			i_start = bidx * self.minibatch_size
			i_end = min(i_start + self.minibatch_size, batch_size)
			src_ids = batch_src_ids[i_start:i_end]
			src_lengths = batch_src_lengths[i_start:i_end]
			acous_feats = batch_acous_feats[i_start:i_end]
			acous_lengths = batch_acous_lengths[i_start:i_end]

			src_len = max(src_lengths)
			acous_len = max(acous_lengths)
			acous_len = acous_len + 8 - acous_len % 8
			src_ids = src_ids.to(device=self.device)
			acous_feats = acous_feats[:,:acous_len].to(device=self.device)

			# debug oom
			# acous_feats, acous_lengths, src_ids, tgt_ids = self._debug_oom(acous_len, acous_feats)

			# get padding mask
			non_padding_mask_src = src_ids.data.ne(PAD)

			# Forward propagation
			out_dict = model.forward_train(src_ids, acous_feats=acous_feats,
				acous_lens=acous_lengths, mode='AE_ASR', use_gpu=self.use_gpu)

			logps_asr = out_dict['logps_asr']
			emb_asr = out_dict['emb_asr']
			logps_ae = out_dict['logps_ae']
			emb_ae = out_dict['emb_ae']
			refs_ae = out_dict['refs_ae']

			# import pdb; pdb.set_trace()
			# Get loss
			if not self.eval_with_mask:
				loss_asr.eval_batch(logps_asr.reshape(-1, logps_asr.size(-1)),
					src_ids[:, 1:].reshape(-1))
				loss_asr.norm_term = 1.0 * src_ids.size(0) * src_ids[:,1:].size(1)
				loss_ae.eval_batch(logps_ae.reshape(-1, logps_ae.size(-1)),
					refs_ae.reshape(-1))
				loss_ae.norm_term = 1.0 * src_ids.size(0) * src_ids[:,1:].size(1)
				loss_kl.eval_batch(logps_ae.reshape(-1, logps_ae.size(-1)),
					logps_asr.reshape(-1, logps_asr.size(-1)))
				loss_kl.norm_term = 1.0 * src_ids.size(0) * src_ids[:,1:].size(1)
				loss_l2.eval_batch(emb_asr.reshape(-1, emb_asr.size(-1)),
					emb_ae.reshape(-1, emb_ae.size(-1)))
				loss_l2.norm_term = 1.0 * src_ids.size(0) * src_ids[:,1:].size(1)
			else:
				loss_asr.eval_batch_with_mask(logps_asr.reshape(-1, logps_asr.size(-1)),
					src_ids[:,1:].reshape(-1), non_padding_mask_src[:,1:].reshape(-1))
				loss_asr.norm_term = 1.0 * torch.sum(non_padding_mask_src[:,1:])
				loss_ae.eval_batch_with_mask(logps_ae.reshape(-1, logps_ae.size(-1)),
					refs_ae.reshape(-1), non_padding_mask_src[:,1:].reshape(-1))
				loss_ae.norm_term = 1.0 * torch.sum(non_padding_mask_src[:,1:])
				loss_kl.eval_batch_with_mask(logps_ae.reshape(-1, logps_ae.size(-1)),
					logps_asr.reshape(-1, logps_asr.size(-1)),
					non_padding_mask_src[:,1:].reshape(-1))
				loss_kl.norm_term = 1.0 * torch.sum(non_padding_mask_src[:,1:])
				loss_l2.eval_batch_with_mask(emb_asr.reshape(-1, emb_asr.size(-1)),
					emb_ae.reshape(-1, emb_ae.size(-1)),
					non_padding_mask_src[:,1:].reshape(-1))
				loss_l2.norm_term = 1.0 * torch.sum(non_padding_mask_src[:,1:])

			# Backward propagation: accumulate gradient
			if self.normalise_loss:
				loss_asr.normalise()
				loss_ae.normalise()
				loss_kl.normalise()
				loss_l2.normalise()

			loss_asr.acc_loss *= self.loss_coeff['nll_asr']
			loss_asr.acc_loss /= n_minibatch
			resloss_asr += loss_asr.get_loss()

			loss_ae.acc_loss *= self.loss_coeff['nll_ae']
			loss_ae.acc_loss /= n_minibatch
			resloss_ae += loss_ae.get_loss()

			loss_kl.acc_loss *= self.loss_coeff['kl_en']
			loss_kl.acc_loss /= n_minibatch
			resloss_kl += loss_kl.get_loss()

			loss_l2.acc_loss *= self.loss_coeff['l2']
			loss_l2.acc_loss /= n_minibatch
			resloss_l2 += loss_l2.get_loss()

			loss_asr.add(loss_ae)
			loss_asr.add(loss_kl)
			loss_asr.add(loss_l2)
			loss_asr.backward()
			# torch.cuda.empty_cache()

		# ---------- MT ----------
		# load data
		batch_src_ids = batch_items_mt['srcid'][0]
		batch_src_lengths = batch_items_mt['srclen']
		batch_tgt_ids = batch_items_mt['tgtid'][0]
		batch_tgt_lengths = batch_items_mt['tgtlen']

		# separate into minibatch
		batch_size = batch_src_ids.size(0)
		batch_seq_len = int(max(batch_src_lengths))
		n_minibatch = int(batch_size / self.minibatch_size)
		n_minibatch += int(batch_size % self.minibatch_size > 0)
		resloss_mt = 0

		for bidx in range(n_minibatch):

			# debug
			# print(bidx,n_minibatch)
			# import pdb; pdb.set_trace()

			# define loss
			loss_mt = NLLLoss()
			loss_mt.reset()

			# load data
			i_start = bidx * self.minibatch_size
			i_end = min(i_start + self.minibatch_size, batch_size)
			src_ids = batch_src_ids[i_start:i_end]
			src_lengths = batch_src_lengths[i_start:i_end]
			tgt_ids = batch_tgt_ids[i_start:i_end]
			tgt_lengths = batch_tgt_lengths[i_start:i_end]

			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			src_ids = src_ids.to(device=self.device)
			tgt_ids = tgt_ids.to(device=self.device)

			# debug oom
			# acous_feats, acous_lengths, src_ids, tgt_ids = self._debug_oom(acous_len, acous_feats)

			# get padding mask
			non_padding_mask_src = src_ids.data.ne(PAD)
			non_padding_mask_tgt = tgt_ids.data.ne(PAD)

			# Forward propagation
			out_dict = model.forward_train(src_ids, tgt=tgt_ids,
				mode='MT', use_gpu=self.use_gpu)

			logps_mt = out_dict['logps_mt'][:,:-1,:]

			# Get loss
			if not self.eval_with_mask:
				loss_mt.eval_batch(logps_mt.reshape(-1, logps_mt.size(-1)),
					tgt_ids[:, 1:].reshape(-1))
				loss_mt.norm_term = 1.0 * tgt_ids.size(0) * tgt_ids[:,1:].size(1)
			else:
				loss_mt.eval_batch_with_mask(logps_mt.reshape(-1, logps_mt.size(-1)),
					tgt_ids[:,1:].reshape(-1), non_padding_mask_tgt[:,1:].reshape(-1))
				loss_mt.norm_term = 1.0 * torch.sum(non_padding_mask_tgt[:,1:])

			# import pdb; pdb.set_trace()
			# Backward propagation: accumulate gradient
			if self.normalise_loss:
				loss_mt.normalise()

			loss_mt.acc_loss *= self.loss_coeff['nll_mt']
			loss_mt.acc_loss /= n_minibatch
			resloss_mt += loss_mt.get_loss()
			loss_mt.backward()
			# torch.cuda.empty_cache()

		# ------ update weights --------
		self.optimizer.step()
		model.zero_grad()

		losses = {}
		losses['nll_loss_asr'] = resloss_asr
		losses['nll_loss_ae'] = resloss_ae
		losses['nll_loss_mt'] = resloss_mt
		losses['kl_loss'] = resloss_kl
		losses['l2_loss'] = resloss_l2

		return losses


	def _train_epoches(self,
		train_sets, model, n_epochs, start_epoch, start_step, dev_sets=None):

		# load datasets
		train_set_asr = train_sets['asr']
		dev_set_asr = dev_sets['asr']
		train_set_mt = train_sets['mt']
		dev_set_mt = dev_sets['mt']

		log = self.logger

		print_loss_ae_total = 0  # Reset every print_every
		print_loss_asr_total = 0
		print_loss_mt_total = 0
		print_loss_kl_total = 0
		print_loss_l2_total = 0

		step = start_step
		step_elapsed = 0
		prev_acc = 0.0
		prev_bleu = 0.0
		count_no_improve = 0
		count_num_rollback = 0
		ckpt = None

		# loop over epochs
		for epoch in range(start_epoch, n_epochs + 1):

			# update lr
			if self.lr_warmup_steps != 0:
				self.optimizer.optimizer = self.lr_scheduler(
					self.optimizer.optimizer, step, init_lr=self.learning_rate_init,
					peak_lr=self.learning_rate, warmup_steps=self.lr_warmup_steps)
			# print lr
			for param_group in self.optimizer.optimizer.param_groups:
				log.info('epoch:{} lr: {}'.format(epoch, param_group['lr']))
				lr_curr = param_group['lr']

			# construct batches - allow re-shuffling of data
			log.info('--- construct train set ---')
			train_set_asr.construct_batches(is_train=True)
			train_set_mt.construct_batches(is_train=True)
			if dev_set_asr is not None:
				log.info('--- construct dev set ---')
				dev_set_asr.construct_batches(is_train=False)
				dev_set_mt.construct_batches(is_train=False)

			# print info
			steps_per_epoch_asr = len(train_set_asr.iter_loader)
			steps_per_epoch_mt = len(train_set_mt.iter_loader)
			steps_per_epoch = min(steps_per_epoch_asr, steps_per_epoch_mt)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))

			log.info(" ---------- Epoch: %d, Step: %d ----------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			log.info('Memory used: {0:.2f} MB'.format(mem_mb))
			self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# loop over batches
			model.train(True)
			trainiter_asr = iter(train_set_asr.iter_loader)
			trainiter_mt = iter(train_set_mt.iter_loader)
			for idx in range(steps_per_epoch):

				# load batch items
				batch_items_asr = trainiter_asr.next()
				batch_items_mt = trainiter_mt.next()

				# update macro count
				step += 1
				step_elapsed += 1

				if self.lr_warmup_steps != 0:
					self.optimizer.optimizer = self.lr_scheduler(
						self.optimizer.optimizer, step, init_lr=self.learning_rate_init,
						peak_lr=self.learning_rate, warmup_steps=self.lr_warmup_steps)

				# Get loss
				losses = self._train_batch(model, batch_items_asr, batch_items_mt, step, total_steps)
				loss_ae = losses['nll_loss_ae']
				loss_asr = losses['nll_loss_asr']
				loss_mt = losses['nll_loss_mt']
				loss_kl = losses['kl_loss']
				loss_l2 = losses['l2_loss']

				print_loss_ae_total += loss_ae
				print_loss_asr_total += loss_asr
				print_loss_mt_total += loss_mt
				print_loss_kl_total += loss_kl
				print_loss_l2_total += loss_l2

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					print_loss_ae_avg = print_loss_ae_total / self.print_every
					print_loss_ae_total = 0
					print_loss_asr_avg = print_loss_asr_total / self.print_every
					print_loss_asr_total = 0
					print_loss_mt_avg = print_loss_mt_total / self.print_every
					print_loss_mt_total = 0
					print_loss_kl_avg = print_loss_kl_total / self.print_every
					print_loss_kl_total = 0
					print_loss_l2_avg = print_loss_l2_total / self.print_every
					print_loss_l2_total = 0

					log_msg = 'Progress: %d%%, Train nlll_ae: %.4f, nlll_asr: %.4f, ' % (
						step / total_steps * 100, print_loss_ae_avg, print_loss_asr_avg)
					log_msg += 'Train nlll_mt: %.4f, l2: %.4f, kl_en: %.4f' % (
						print_loss_mt_avg, print_loss_l2_avg, print_loss_kl_avg)
					log.info(log_msg)

					self.writer.add_scalar('train_loss_ae', print_loss_ae_avg, global_step=step)
					self.writer.add_scalar('train_loss_asr', print_loss_asr_avg, global_step=step)
					self.writer.add_scalar('train_loss_mt', print_loss_mt_avg, global_step=step)
					self.writer.add_scalar('train_loss_kl', print_loss_kl_avg, global_step=step)
					self.writer.add_scalar('train_loss_l2', print_loss_l2_avg, global_step=step)

				# Checkpoint
				if step % self.checkpoint_every == 0 or step == total_steps:

					# save criteria
					if dev_set_asr is not None:
						losses, metrics = self._evaluate_batches(model, dev_set_asr, dev_set_mt)

						loss_kl = losses['kl_loss']
						loss_l2 = losses['l2_loss']
						loss_ae = losses['nll_loss_ae']
						accuracy_ae = metrics['accuracy_ae']
						bleu_ae = metrics['bleu_ae']
						loss_asr = losses['nll_loss_asr']
						accuracy_asr = metrics['accuracy_asr']
						bleu_asr = metrics['bleu_asr']
						loss_mt = losses['nll_loss_mt']
						accuracy_mt = metrics['accuracy_mt']
						bleu_mt = metrics['bleu_mt']

						log_msg = 'Progress: %d%%, Dev AE loss: %.4f, accuracy: %.4f, bleu: %.4f' % (
							step / total_steps * 100, loss_ae, accuracy_ae, bleu_ae)
						log.info(log_msg)
						log_msg = 'Progress: %d%%, Dev ASR loss: %.4f, accuracy: %.4f, bleu: %.4f' % (
							step / total_steps * 100, loss_asr, accuracy_asr, bleu_asr)
						log.info(log_msg)
						log_msg = 'Progress: %d%%, Dev MT loss: %.4f, accuracy: %.4f, bleu: %.4f' % (
							step / total_steps * 100, loss_mt, accuracy_mt, bleu_mt)
						log.info(log_msg)
						log_msg = 'Progress: %d%%, Dev En KL loss: %.4f, L2 loss: %.4f' % (
							step / total_steps * 100, loss_kl, loss_l2)
						log.info(log_msg)

						self.writer.add_scalar('dev_loss_l2', loss_l2, global_step=step)
						self.writer.add_scalar('dev_loss_kl', loss_kl, global_step=step)
						self.writer.add_scalar('dev_loss_ae', loss_ae, global_step=step)
						self.writer.add_scalar('dev_acc_ae', accuracy_ae, global_step=step)
						self.writer.add_scalar('dev_bleu_ae', bleu_ae, global_step=step)
						self.writer.add_scalar('dev_loss_asr', loss_asr, global_step=step)
						self.writer.add_scalar('dev_acc_asr', accuracy_asr, global_step=step)
						self.writer.add_scalar('dev_bleu_asr', bleu_asr, global_step=step)
						self.writer.add_scalar('dev_loss_mt', loss_mt, global_step=step)
						self.writer.add_scalar('dev_acc_mt', accuracy_mt, global_step=step)
						self.writer.add_scalar('dev_bleu_mt', bleu_mt, global_step=step)

						# save - use ASR res
						accuracy_ave = (accuracy_asr/4.0 + accuracy_mt)/2.0
						bleu_ave = (bleu_asr/4.0 + bleu_mt)/2.0
						if ((prev_acc < accuracy_ave) and (bleu_ave < 0.1)) or prev_bleu < bleu_ave:

							# save best model - using bleu as metric
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=train_set_asr.vocab_src,
									   output_vocab=train_set_asr.vocab_tgt)

							saved_path = ckpt.save(self.expt_dir)
							log.info('saving at {} ... '.format(saved_path))
							# reset
							prev_acc = accuracy_ave
							prev_bleu = bleu_ave
							count_no_improve = 0
							count_num_rollback = 0
						else:
							count_no_improve += 1

						# roll back
						if count_no_improve > self.max_count_no_improve:
							# break after self.max_count_no_improve epochs
							if self.max_count_num_rollback == 0:
								break
							# resuming
							latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
							if type(latest_checkpoint_path) != type(None):
								resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
								log.info('epoch:{} step: {} - rolling back {} ...'.format(
									epoch, step, latest_checkpoint_path))
								model = resume_checkpoint.model
								self.optimizer = resume_checkpoint.optimizer
								# A walk around to set optimizing parameters properly
								resume_optim = self.optimizer.optimizer
								defaults = resume_optim.param_groups[0]
								defaults.pop('params', None)
								defaults.pop('initial_lr', None)
								self.optimizer.optimizer = resume_optim.__class__(
									model.parameters(), **defaults)

							# reset
							count_no_improve = 0
							count_num_rollback += 1

						# update learning rate
						if count_num_rollback > self.max_count_num_rollback:

							# roll back
							latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
							if type(latest_checkpoint_path) != type(None):
								resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
								log.info('epoch:{} step: {} - rolling back {} ...'.format(
									epoch, step, latest_checkpoint_path))
								model = resume_checkpoint.model
								self.optimizer = resume_checkpoint.optimizer
								# A walk around to set optimizing parameters properly
								resume_optim = self.optimizer.optimizer
								defaults = resume_optim.param_groups[0]
								defaults.pop('params', None)
								defaults.pop('initial_lr', None)
								self.optimizer.optimizer = resume_optim.__class__(
									model.parameters(), **defaults)

							# decrease lr
							for param_group in self.optimizer.optimizer.param_groups:
								param_group['lr'] *= 0.5
								lr_curr = param_group['lr']
								log.info('reducing lr ...')
								log.info('step:{} - lr: {}'.format(step, param_group['lr']))

							# check early stop
							if lr_curr <= 0.125 * self.learning_rate :
								log.info('early stop ...')
								break

							# reset
							count_no_improve = 0
							count_num_rollback = 0

						model.train(mode=True)
						if ckpt is not None:
							ckpt.rm_old(self.expt_dir, keep_num=self.keep_num)
						log.info('n_no_improve {}, num_rollback {}'.format(
							count_no_improve, count_num_rollback))

					sys.stdout.flush()

			else:
				if dev_set_asr is None:
					# save every epoch if no dev_set
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set_asr.vocab_src,
							   output_vocab=train_set_asr.vocab_tgt)
					saved_path = ckpt.save_epoch(self.expt_dir, epoch)
					log.info('saving at {} ... '.format(saved_path))
					continue

				else:
					continue

			# break nested for loop
			break
