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

from utils.misc import get_memory_alloc, check_device, add2corpus, reserve_memory
from utils.misc import _convert_to_words_batchfirst, _convert_to_words
from utils.config import PAD, EOS
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)


class Trainer(object):

	def __init__(self,
		expt_dir='experiment',
		load_dir=None,
		load_mode='null',
		load_freeze=False,
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

		self.use_gpu = use_gpu
		self.gpu_id = gpu_id
		self.device = check_device(self.use_gpu)

		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every
		self.loss_coeff = loss_coeff

		self.learning_rate = learning_rate
		self.learning_rate_init = learning_rate_init
		self.lr_warmup_steps = lr_warmup_steps
		if self.lr_warmup_steps == 0:
			assert self.learning_rate == self.learning_rate_init

		self.max_grad_norm = max_grad_norm
		self.eval_with_mask = eval_with_mask

		self.max_count_no_improve = max_count_no_improve
		self.max_count_num_rollback = max_count_num_rollback
		self.keep_num = keep_num
		self.normalise_loss = normalise_loss

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.load_dir = load_dir
		self.load_mode = load_mode
		self.load_freeze = load_freeze

		self.logger = logging.getLogger(__name__)
		self.writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.expt_dir)

		self.minibatch_partition = minibatch_partition
		self.batch_size = batch_size
		self.minibatch_size = int(self.batch_size / self.minibatch_partition) # to be changed if OOM


	def _print_all(self,
		out_count, src_ids, tgt_ids, src_id2word, tgt_id2word, seqlist_src, seqlist_tgt):

		if out_count < 3:
			srcwords = _convert_to_words_batchfirst(src_ids[:,1:], src_id2word)
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], tgt_id2word)
			seqwords_src = _convert_to_words_batchfirst(seqlist_src, src_id2word)
			seqwords_tgt = _convert_to_words_batchfirst(seqlist_tgt, tgt_id2word)
			outsrc = 'SRC: {}\n'.format(' '.join(srcwords[0])).encode('utf-8')
			outref = 'TGT: {}\n'.format(' '.join(refwords[0])).encode('utf-8')
			outline_en = 'EN:  {}\n'.format(' '.join(seqwords_src[0])).encode('utf-8')
			outline_de = 'DE:  {}\n'.format(' '.join(seqwords_tgt[0])).encode('utf-8')
			sys.stdout.buffer.write(outsrc)
			sys.stdout.buffer.write(outref)
			sys.stdout.buffer.write(outline_en)
			sys.stdout.buffer.write(outline_de)
			out_count += 1
		return out_count


	def _print(self, out_count, ids, id2word, seqlist, tail=''):

		if out_count < 3:
			words = _convert_to_words_batchfirst(ids[:,1:], id2word)
			seqwords = _convert_to_words_batchfirst(seqlist, id2word)
			outref = 'REF{}: {}\n'.format(tail, ' '.join(words[0])).encode('utf-8')
			outhyp = 'HYP{}: {}\n'.format(tail, ' '.join(seqwords[0])).encode('utf-8')
			sys.stdout.buffer.write(outref)
			sys.stdout.buffer.write(outhyp)
			out_count += 1
		return out_count


	def _debug_oom(self, acous_len, acous_feats):

		""" set to max size - try to avoid oom """

		pad_len = 3000-acous_len
		acous_pad = torch.zeros(acous_feats.size(0),pad_len,40).to(device=self.device)
		acous_feats = torch.cat((acous_feats,acous_pad),dim=1)
		acous_lengths = None
		src_ids = torch.ones(acous_feats.size(0),50).to(device=self.device).long()
		tgt_ids = torch.ones(acous_feats.size(0),300).to(device=self.device).long()

		return acous_feats, acous_lengths, src_ids, tgt_ids


	def lr_scheduler(self, optimizer, step,
		init_lr=0.00001, peak_lr=0.0005, warmup_steps=16000):

		""" Learning rate warmup + decay """

		# deactivate scheduler
		if warmup_steps <= 0:
			return optimizer

		# activate scheduler
		if step <= warmup_steps:
			lr = step * 1. * (peak_lr - init_lr) / warmup_steps + init_lr
		else:
			# lr = peak_lr * ((step - warmup_steps) ** (-0.5))
			lr = peak_lr * (step ** (-0.5)) * (warmup_steps ** 0.5)

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

		return optimizer


	def _evaluate_batches(self, *args):
		raise NotImplementedError


	def _train_batch(self, *args):
		raise NotImplementedError


	def _train_epoches(self, *args):
		raise NotImplementedError


	def train(self, train_sets, model, num_epochs=5, optimizer=None,
		dev_sets=None, grab_memory=True):

		"""
			Run training for a given model.
			Args:
				train_set: dataset
				dev_set: dataset, optional
				model: model to run training on
				optimizer (seq2seq.optim.Optimizer, optional): optimizer for training
			Returns:
				model (seq2seq.models): trained model.
		"""

		# torch.cuda.empty_cache()
		if self.load_mode == 'resume' or self.load_mode == 'restart':

			latest_checkpoint_path = self.load_dir
			self.logger.info('resuming {} ...'.format(latest_checkpoint_path))
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model = resume_checkpoint.model
			self.logger.info(model)
			self.optimizer = resume_checkpoint.optimizer

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			for name, param in model.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))
				if 'las.encoder' in name and self.load_freeze:
					log = self.logger.info('freezed')
					param.requires_grad = False

			if self.load_mode == 'resume':
				# start from prev
				start_epoch = resume_checkpoint.epoch # start from the saved epoch!
				step = resume_checkpoint.step# start from the saved step!
			elif self.load_mode == 'restart':
				# just for the sake of finetuning
				start_epoch = 1
				step = 0

		else:

			if self.load_mode == 'LAS':

				""" load LAS pyramidal LSTM from old dir """

				las_checkpoint_path = self.load_dir
				self.logger.info('loading Pyramidal lstm {} ...'.format(las_checkpoint_path))
				las_checkpoint = Checkpoint.load(las_checkpoint_path)
				las_model = las_checkpoint.model
				# assign param
				for name, param in model.named_parameters():
					loaded = False
					log = self.logger.info('{}:{}'.format(name, param.size()))
					for las_name, las_param in las_model.named_parameters():
						# las_name = encoder.acous_enc_l1.weight_ih_l0
						# name = las.enc.acous_enc_l1.weight_ih_l0
						name_init = '.'.join(name.split('.')[0:2])
						name_rest = '.'.join(name.split('.')[2:])
						las_name_rest = '.'.join(las_name.split('.')[1:])
						if name_init == 'las.encoder' and name_rest == las_name_rest:
							assert param.data.size() == las_param.data.size(), \
								'las_name {} {} : name {} {}'.format(las_name,
									las_param.data.size(), name, param.data.size())
							param.data = las_param.data
							self.logger.info('loading {}'.format(las_name))
							loaded = True
							if self.load_freeze: # freezing embedder too
								self.logger.info('freezed')
								param.requires_grad = False
							else:
								self.logger.info('not freezed')
					if not loaded:
						self.logger.info('not preloaded - {}'.format(name))
					# import pdb; pdb.set_trace()

			elif self.load_mode == 'ASR':

				""" load ASR model: compatible only with asr-v001 model """

				asr_checkpoint_path = self.load_dir
				self.logger.info('loading ASR {} ...'.format(asr_checkpoint_path))
				asr_checkpoint = Checkpoint.load(asr_checkpoint_path)
				asr_model = asr_checkpoint.model
				# assign param
				for name, param in model.named_parameters():
					loaded = False
					log = self.logger.info('{}:{}'.format(name, param.size()))
					# name = las.encoder.acous_enc_l1.weight_ih_l0
					name_init = '.'.join(name.split('.')[0:1])
					name_rest = '.'.join(name.split('.')[1:])

					for asr_name, asr_param in asr_model.named_parameters():
						if name_init == 'las' and name == asr_name:
							assert param.data.size() == asr_param.data.size()
							param.data = asr_param.data
							loaded = True
							self.logger.info('loading {}'.format(asr_name))
							if self.load_freeze: # freezing embedder too
								self.logger.info('freezed')
								param.requires_grad = False
							else:
								self.logger.info('not freezed')
					if not loaded:
						# make exception for las dec embedder
						if name == 'las.decoder.embedder.weight':
							model.las.decoder.embedder.weight.data = \
								asr_model.enc_embedder.weight.data
							self.logger.info('assigning {} with {}'.format(
								'las.decoder.embedder.weight', 'enc_embedder.weight'))
							if self.load_freeze:
								self.logger.info('freezed')
								param.requires_grad = False
							else:
								self.logger.info('not freezed')
						else:
							self.logger.info('not preloaded - {}'.format(name))
					# import pdb; pdb.set_trace()

			elif self.load_mode == 'AE-ASR':

				""" load AE-ASR model """

				aeasr_checkpoint_path = self.load_dir
				self.logger.info('loading AE-ASR model {} ...'.format(aeasr_checkpoint_path))
				aeasr_checkpoint = Checkpoint.load(aeasr_checkpoint_path)
				aeasr_model = aeasr_checkpoint.model
				# assign param
				for name, param in model.named_parameters():
					loaded = False
					log = self.logger.info('{}:{}'.format(name, param.size()))
					name_init = '.'.join(name.split('.')[0:1])
					for aeasr_name, aeasr_param in aeasr_model.named_parameters():
						if name == aeasr_name and (name_init == 'las' or name_init == 'enc_embedder'):
							assert param.data.size() == aeasr_param.data.size()
							param.data = aeasr_param.data
							self.logger.info('loading {}'.format(aeasr_name))
							loaded = True
							if self.load_freeze: # freezing embedder too
								self.logger.info('freezed')
								param.requires_grad = False
							else:
								self.logger.info('not freezed')
					if not loaded:
						self.logger.info('not preloaded - {}'.format(name))

			elif self.load_mode == 'AE-ASR-MT':

				""" load general models - lock LAS LSTM """

				checkpoint_path = self.load_dir
				self.logger.info('loading model {} ...'.format(checkpoint_path))
				checkpoint = Checkpoint.load(checkpoint_path)
				load_model = checkpoint.model
				# assign param
				for name, param in model.named_parameters():
					loaded = False
					log = self.logger.info('{}:{}'.format(name, param.size()))
					name_init = '.'.join(name.split('.')[:1])
					for load_name, load_param in load_model.named_parameters():
						if name == load_name:
							assert param.data.size() == load_param.data.size()
							param.data = load_param.data
							self.logger.info('loading {}'.format(load_name))
							loaded = True
							if self.load_freeze and name_init == 'las': # freeze las
								self.logger.info('freezed')
								param.requires_grad = False
							else:
								self.logger.info('not freezed')
					if not loaded:
						self.logger.info('not preloaded - {}'.format(name))

			elif type(self.load_dir) != type(None):

				""" load general models """

				checkpoint_path = self.load_dir
				self.logger.info('loading model {} ...'.format(checkpoint_path))
				checkpoint = Checkpoint.load(checkpoint_path)
				load_model = checkpoint.model
				# assign param
				for name, param in model.named_parameters():
					loaded = False
					log = self.logger.info('{}:{}'.format(name, param.size()))
					for load_name, load_param in load_model.named_parameters():
						if name == load_name:
							assert param.data.size() == load_param.data.size()
							param.data = load_param.data
							self.logger.info('loading {}'.format(load_name))
							loaded = True
							if self.load_freeze:
								self.logger.info('freezed')
								param.requires_grad = False
							else:
								self.logger.info('not freezed')
					if not loaded:
						self.logger.info('not preloaded - {}'.format(name))

			else:
				# not loading pre-trained model
				for name, param in model.named_parameters():
					log = self.logger.info('{}:{}'.format(name, param.size()))

			# init opt
			if optimizer is None:
				optimizer = Optimizer(torch.optim.Adam(
					model.parameters(),
					lr=self.learning_rate_init), max_grad_norm=self.max_grad_norm)
			self.optimizer = optimizer
			start_epoch = 1
			step = 0

		# train epochs
		self.logger.info("Optimizer: %s, Scheduler: %s" %
			(self.optimizer.optimizer, self.optimizer.scheduler))

		# reserve memory
		# import pdb; pdb.set_trace()
		if self.device == torch.device('cuda') and grab_memory:
			reserve_memory(device_id=self.gpu_id)

		# training
		self._train_epoches(
			train_sets, model, num_epochs, start_epoch, step, dev_sets=dev_sets)

		return model
