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
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from .trainer_base import Trainer

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


class Trainer_ST(Trainer):

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


	def _evaluate_batches(self, model, dataset):

		# import pdb; pdb.set_trace()

		model.eval()

		resloss_de = 0
		resloss_norm = 0

		# accuracy
		match_de = 0
		total_de = 0

		# bleu
		hyp_corpus_de = []
		ref_corpus_de = []

		evaliter = iter(dataset.iter_loader)
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
				batch_acous_feats = batch_items['acous_feat'][0]
				batch_acous_lengths = batch_items['acouslen']

				# separate into minibatch
				batch_size = batch_src_ids.size(0)
				batch_seq_len = int(max(batch_src_lengths))

				n_minibatch = int(batch_size / self.minibatch_size)
				n_minibatch += int(batch_size % self.minibatch_size > 0)

				for bidx in range(n_minibatch):

					loss_de = NLLLoss()
					loss_de.reset()

					i_start = bidx * self.minibatch_size
					i_end = min(i_start + self.minibatch_size, batch_size)
					src_ids = batch_src_ids[i_start:i_end]
					src_lengths = batch_src_lengths[i_start:i_end]
					tgt_ids = batch_tgt_ids[i_start:i_end]
					tgt_lengths = batch_tgt_lengths[i_start:i_end]
					acous_feats = batch_acous_feats[i_start:i_end]
					acous_lengths = batch_acous_lengths[i_start:i_end]

					src_len = max(src_lengths)
					tgt_len = max(tgt_lengths)
					acous_len = max(acous_lengths)
					acous_len = acous_len + 8 - acous_len % 8
					src_ids = src_ids.to(device=self.device)
					tgt_ids = tgt_ids.to(device=self.device)
					acous_feats = acous_feats[:,:acous_len].to(device=self.device)

					# debug oom
					# acous_feats, acous_lengths, src_ids, tgt_ids = self._debug_oom(acous_len, acous_feats)

					non_padding_mask_tgt = tgt_ids.data.ne(PAD)
					non_padding_mask_src = src_ids.data.ne(PAD)

					# [run-TF] to save time
					# out_dict = model.forward_train(src_ids, tgt=tgt_ids, acous_feats=acous_feats,
					#	acous_lens=acous_lengths, mode='ST', use_gpu=self.use_gpu)
					# preds_de = out_dict['preds_st']
					# logps_de = out_dict['logps_st']
					# logps_hyp_de = logps_de[:,:-1,:]
					# preds_hyp_de = preds_de[:,:-1]

					# [run-FR] to get true stats
					out_dict = model.forward_eval(acous_feats=acous_feats,
						acous_lens=acous_lengths, mode='ST', use_gpu=self.use_gpu)
					preds_de = out_dict['preds_st']
					logps_de = out_dict['logps_st']
					logps_hyp_de = logps_de[:,1:,:]
					preds_hyp_de = preds_de[:,1:]

					# evaluation
					if not self.eval_with_mask:
						loss_de.eval_batch(logps_hyp_de.reshape(-1, logps_hyp_de.size(-1)),
							tgt_ids[:, 1:].reshape(-1))
						loss_de.norm_term = 1.0 * tgt_ids.size(0) * tgt_ids[:,1:].size(1)
					else:
						loss_de.eval_batch_with_mask(logps_hyp_de.reshape(-1, logps_hyp_de.size(-1)),
							tgt_ids[:,1:].reshape(-1), non_padding_mask_tgt[:,1:].reshape(-1))
						loss_de.norm_term = 1.0 * torch.sum(non_padding_mask_tgt[:,1:])

					if self.normalise_loss:
						loss_de.normalise()

					resloss_de += loss_de.get_loss()
					resloss_norm += 1

					# accuracy
					seqres_de = preds_hyp_de
					correct_de = seqres_de.reshape(-1).eq(tgt_ids[:,1:].reshape(-1))\
						.masked_select(non_padding_mask_tgt[:,1:].reshape(-1)).sum().item()
					match_de += correct_de
					total_de += non_padding_mask_tgt[:,1:].sum().item()

					# print
					out_count = self._print(out_count, tgt_ids,
						dataset.tgt_id2word, seqres_de, tail='-st ')

					# accumulate corpus
					hyp_corpus_de, ref_corpus_de = add2corpus(seqres_de, tgt_ids,
						dataset.tgt_id2word, hyp_corpus_de, ref_corpus_de, type=dataset.use_type)

		# import pdb; pdb.set_trace()
		bleu_de = torchtext.data.metrics.bleu_score(hyp_corpus_de, ref_corpus_de)

		# torch.cuda.empty_cache()
		if total_de == 0:
			accuracy_de = float('nan')
		else:
			accuracy_de = match_de / total_de

		resloss_de *= self.loss_coeff['nll_st']
		resloss_de /= (1.0 * resloss_norm)

		losses = {}
		losses['nll_loss_de'] = resloss_de
		metrics = {}
		metrics['accuracy_de'] = accuracy_de
		metrics['bleu_de'] = bleu_de

		return losses, metrics


	def _train_batch(self,
		model, batch_items, dataset, step, total_steps):

		# load data
		batch_src_ids = batch_items['srcid'][0]
		batch_src_lengths = batch_items['srclen']
		batch_tgt_ids = batch_items['tgtid'][0]
		batch_tgt_lengths = batch_items['tgtlen']
		batch_acous_feats = batch_items['acous_feat'][0]
		batch_acous_lengths = batch_items['acouslen']

		# separate into minibatch
		batch_size = batch_src_ids.size(0)
		batch_seq_len = int(max(batch_src_lengths))
		n_minibatch = int(batch_size / self.minibatch_size)
		n_minibatch += int(batch_size % self.minibatch_size > 0)
		resloss_de = 0

		for bidx in range(n_minibatch):

			# debug
			# print(bidx,n_minibatch)
			# import pdb; pdb.set_trace()

			# define loss
			loss_de = NLLLoss()
			loss_de.reset()

			# load data
			i_start = bidx * self.minibatch_size
			i_end = min(i_start + self.minibatch_size, batch_size)
			src_ids = batch_src_ids[i_start:i_end]
			src_lengths = batch_src_lengths[i_start:i_end]
			tgt_ids = batch_tgt_ids[i_start:i_end]
			tgt_lengths = batch_tgt_lengths[i_start:i_end]
			acous_feats = batch_acous_feats[i_start:i_end]
			acous_lengths = batch_acous_lengths[i_start:i_end]

			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			acous_len = max(acous_lengths)
			acous_len = acous_len + 8 - acous_len % 8
			src_ids = src_ids.to(device=self.device)
			tgt_ids = tgt_ids.to(device=self.device)
			acous_feats = acous_feats[:,:acous_len].to(device=self.device)

			# debug oom
			# acous_feats, acous_lengths, src_ids, tgt_ids = self._debug_oom(acous_len, acous_feats)

			# get padding mask
			non_padding_mask_src = src_ids.data.ne(PAD)
			non_padding_mask_tgt = tgt_ids.data.ne(PAD)

			# Forward propagation
			out_dict = model.forward_train(src_ids, tgt=tgt_ids, acous_feats=acous_feats,
				acous_lens=acous_lengths, mode='ST', use_gpu=self.use_gpu)

			logps_de = out_dict['logps_st'][:,:-1,:]

			# Get loss
			if not self.eval_with_mask:
				loss_de.eval_batch(logps_de.reshape(-1, logps_de.size(-1)),
					tgt_ids[:, 1:].reshape(-1))
				loss_de.norm_term = 1.0 * tgt_ids.size(0) * tgt_ids[:,1:].size(1)
			else:
				loss_de.eval_batch_with_mask(logps_de.reshape(-1, logps_de.size(-1)),
					tgt_ids[:,1:].reshape(-1), non_padding_mask_tgt[:,1:].reshape(-1))
				loss_de.norm_term = 1.0 * torch.sum(non_padding_mask_tgt[:,1:])

			# import pdb; pdb.set_trace()
			# Backward propagation: accumulate gradient
			if self.normalise_loss:
				loss_de.normalise()

			loss_de.acc_loss *= self.loss_coeff['nll_st']
			loss_de.acc_loss /= n_minibatch
			resloss_de += loss_de.get_loss()
			loss_de.backward()

			# torch.cuda.empty_cache()

		# update weights
		self.optimizer.step()
		model.zero_grad()

		losses = {}
		losses['nll_loss_de'] = resloss_de

		return losses


	def _train_epoches(self,
		train_sets, model, n_epochs, start_epoch, start_step, dev_sets=None):

		# load relevant dataset
		train_set = train_sets['st']
		dev_set = dev_sets['st']

		log = self.logger

		print_loss_de_total = 0  # Reset every print_every

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
			train_set.construct_batches(is_train=True)
			if dev_set is not None:
				log.info('--- construct dev set ---')
				dev_set.construct_batches(is_train=False)

			# print info
			steps_per_epoch = len(train_set.iter_loader)
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
			trainiter = iter(train_set.iter_loader)
			for idx in range(steps_per_epoch):

				# load batch items
				batch_items = trainiter.next()

				# update macro count
				step += 1
				step_elapsed += 1

				if self.lr_warmup_steps != 0:
					self.optimizer.optimizer = self.lr_scheduler(
						self.optimizer.optimizer, step, init_lr=self.learning_rate_init,
						peak_lr=self.learning_rate, warmup_steps=self.lr_warmup_steps)

				# Get loss
				losses = self._train_batch(model, batch_items, train_set, step, total_steps)
				loss_de = losses['nll_loss_de']
				print_loss_de_total += loss_de

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					print_loss_de_avg = print_loss_de_total / self.print_every
					print_loss_de_total = 0

					log_msg = 'Progress: %d%%, Train nlll_de: %.4f' % (
						step / total_steps * 100, print_loss_de_avg)
					log.info(log_msg)
					self.writer.add_scalar('train_loss_de', print_loss_de_avg, global_step=step)

				# Checkpoint
				if step % self.checkpoint_every == 0 or step == total_steps:

					# save criteria
					if dev_set is not None:
						losses, metrics = self._evaluate_batches(model, dev_set)

						loss_de = losses['nll_loss_de']
						accuracy_de = metrics['accuracy_de']
						bleu_de = metrics['bleu_de']

						log_msg = 'Progress: %d%%, Dev DE loss: %.4f, accuracy: %.4f, bleu: %.4f' % (
							step / total_steps * 100, loss_de, accuracy_de, bleu_de)
						log.info(log_msg)

						self.writer.add_scalar('dev_loss_de', loss_de, global_step=step)
						self.writer.add_scalar('dev_acc_de', accuracy_de, global_step=step)
						self.writer.add_scalar('dev_bleu_de', bleu_de, global_step=step)

						# save - use DE stats
						# if prev_acc < accuracy:
						if ((prev_acc < accuracy_de) and (bleu_de < 0.1)) or prev_bleu < bleu_de:

							# save best model - using bleu as metric
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=train_set.vocab_src,
									   output_vocab=train_set.vocab_tgt)

							saved_path = ckpt.save(self.expt_dir)
							log.info('saving at {} ... '.format(saved_path))
							# reset
							prev_acc = accuracy_de
							prev_bleu = bleu_de
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
						if ckpt is None:
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=train_set.vocab_src,
									   output_vocab=train_set.vocab_tgt)
						ckpt.rm_old(self.expt_dir, keep_num=self.keep_num)
						log.info('n_no_improve {}, num_rollback {}'.format(
							count_no_improve, count_num_rollback))

					sys.stdout.flush()

			else:
				if dev_set is None:
					# save every epoch if no dev_set
					ckpt = Checkpoint(model=model,
							   optimizer=self.optimizer,
							   epoch=epoch, step=step,
							   input_vocab=train_set.vocab_src,
							   output_vocab=train_set.vocab_tgt)
					saved_path = ckpt.save_epoch(self.expt_dir, epoch)
					log.info('saving at {} ... '.format(saved_path))
					continue

				else:
					continue

			# break nested for loop
			break
