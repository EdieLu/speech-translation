import torch
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

from utils.dataset import Dataset
from utils.misc import save_config, validate_config, check_device
from utils.misc import get_memory_alloc, log_ckpts
from utils.misc import _convert_to_words_batchfirst, _convert_to_words
from utils.misc import _convert_to_tensor, _convert_to_tensor_pad
from utils.misc import plot_alignment, plot_attention, combine_weights
from utils.config import PAD, EOS
from modules.checkpoint import Checkpoint
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss, KLDivLoss, MSELoss
from torch.distributions import Categorical

logging.basicConfig(level=logging.INFO)


def load_arguments(parser):

	""" Seq2Seq eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--test_path_tgt', type=str, default='None', help='test tgt dir')
	parser.add_argument('--path_vocab_src', type=str, default='None', help='vocab src dir, no need')
	parser.add_argument('--path_vocab_tgt', type=str, default='None', help='vocab tgt dir, not needed')
	parser.add_argument('--use_type', type=str, default='char', help='use char | word level prediction')
	parser.add_argument('--acous_norm', type=str, default='False', help='input acoustic fbk normalisation')
	parser.add_argument('--acous_norm_path', type=str, default='None', help='acoustics norm')
	parser.add_argument('--test_acous_path', type=str, default='None', help='test set acoustics')

	parser.add_argument('--load', type=str, required=True, help='model load dir')
	parser.add_argument('--combine_path', type=str, default='None', help='combine multiple ckpts if given dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')

	# others
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--beam_width', type=int, default=0, help='beam width; set to 0 to disable beam search')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_mode', type=int, default=2, help='which evaluation mode to use')
	parser.add_argument('--gen_mode', type=str, default='ASR', help='AE|ASR|MT|ST')
	parser.add_argument('--seqrev', type=str, default=False, help='whether or not to reverse sequence')

	return parser


def translate(test_set, model, test_path_out, use_gpu,
	max_seq_len, beam_width, device, seqrev=False,
	gen_mode='ASR', history='HYP'):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	modes = '_'.join([model.mode, gen_mode])
	# reset max_len
	if 'ASR' in modes or 'ST' in modes:
		model.las.decoder.max_seq_len = 150
	if 'MT' in modes:
		model.enc_src.expand_time(150)
	if 'ST' in modes or 'MT' in modes:
		model.dec_tgt.expand_time(max_seq_len)

	print('max seq len {}'.format(max_seq_len))
	sys.stdout.flush()

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)

	print('num batches: {}'.format(len(evaliter)))
	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):

				print(idx+1, len(evaliter))
				batch_items = evaliter.next()

				# load data
				src_ids = batch_items['srcid'][0]
				src_lengths = batch_items['srclen']
				tgt_ids = batch_items['tgtid'][0]
				tgt_lengths = batch_items['tgtlen']
				acous_feats = batch_items['acous_feat'][0]
				acous_lengths = batch_items['acouslen']

				src_len = max(src_lengths)
				tgt_len = max(tgt_lengths)
				acous_len = max(acous_lengths)
				src_ids = src_ids[:,:src_len].to(device=device)
				tgt_ids = tgt_ids.to(device=device)
				acous_feats = acous_feats.to(device=device)

				n_minibatch = int(tgt_len / 100 + tgt_len % 100 > 0)
				minibatch_size = int(src_ids.size(0) / n_minibatch)
				n_minibatch = int(src_ids.size(0) / minibatch_size) + \
					(src_ids.size(0) % minibatch_size > 0)

				for j in range(n_minibatch):

					st = j * minibatch_size
					ed = min((j+1) * minibatch_size, src_ids.size(0))
					src_ids_sub = src_ids[st:ed,:]
					tgt_ids_sub = tgt_ids[st:ed,:]
					acous_feats_sub = acous_feats[st:ed,:]
					acous_lengths_sub = acous_lengths[st:ed]
					print('minibatch: ', st, ed, src_ids.size(0))

					time1 = time.time()
					if history == 'HYP':
						preds = model.forward_translate(acous_feats=acous_feats_sub,
							acous_lens=acous_lengths_sub, src=src_ids_sub,
							beam_width=beam_width, use_gpu=use_gpu,
							max_seq_len=max_seq_len, mode=gen_mode)
					elif history == 'REF':
						preds = model.forward_translate_refen(acous_feats=acous_feats_sub,
							acous_lens=acous_lengths_sub, src=src_ids_sub,
							beam_width=beam_width, use_gpu=use_gpu,
							max_seq_len=max_seq_len, mode=gen_mode)
					time2 = time.time()
					print('comp time: ', time2-time1)

					# ------ debug ------
					# import pdb; pdb.set_trace()
					# out_dict = model.forward_eval(acous_feats=acous_feats_sub,
					# 	acous_lens=acous_lengths_sub, src=src_ids_sub,
					# 	use_gpu=use_gpu, mode=gen_mode)
					# -------------------

					# write to file
					if gen_mode == 'MT' or gen_mode == 'ST':
						seqlist = preds[:,1:]
						seqwords =  _convert_to_words_batchfirst(seqlist, test_set.tgt_id2word)
						use_type = 'char'
					elif gen_mode == 'AE' or gen_mode == 'ASR':
						seqlist = preds
						seqwords =  _convert_to_words_batchfirst(seqlist, test_set.src_id2word)
						use_type = 'word'

					for i in range(len(seqwords)):
						words = []
						for word in seqwords[i]:
							if word == '<pad>':
								continue
							elif word == '<spc>':
								words.append(' ')
							elif word == '</s>':
								break
							else:
								words.append(word)
						if len(words) == 0:
							outline = ''
						else:
							if seqrev:
								words = words[::-1]
							if use_type == 'word':
								outline = ' '.join(words)
							elif use_type == 'char':
								outline = ''.join(words)
						f.write('{}\n'.format(outline))

						# import pdb; pdb.set_trace()
					sys.stdout.flush()


def plot_emb(test_set, model, test_path_out, use_gpu, max_seq_len, device):

	"""
		plot embedding spaces
	"""

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))
	path_out = os.path.join(test_path_out, 'embed.png')

	import torch.utils.tensorboard
	writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=test_path_out)

	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):

			print(idx+1, len(evaliter))
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['srcid'][0]
			src_lengths = batch_items['srclen']
			tgt_ids = batch_items['tgtid'][0]
			tgt_lengths = batch_items['tgtlen']
			acous_feats = batch_items['acous_feat'][0]
			acous_lengths = batch_items['acouslen']

			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			acous_len = max(acous_lengths)
			src_ids = src_ids[:,:src_len].to(device=device)
			tgt_ids = tgt_ids.to(device=device)
			acous_feats = acous_feats.to(device=device)

			n_minibatch = int(tgt_len / 100 + tgt_len % 100 > 0)
			minibatch_size = int(src_ids.size(0) / n_minibatch)
			n_minibatch = int(src_ids.size(0) / minibatch_size) + \
				(src_ids.size(0) % minibatch_size > 0)

			for j in range(n_minibatch):

				st = j * minibatch_size
				ed = min((j+1) * minibatch_size, src_ids.size(0))
				src_ids_sub = src_ids[st:ed,:]
				tgt_ids_sub = tgt_ids[st:ed,:]
				acous_feats_sub = acous_feats[st:ed,:]
				acous_lengths_sub = acous_lengths[st:ed]
				print('minibatch: ', st, ed, src_ids.size(0))

				# get dynamic
				dynamic_emb, logps, preds, lengths = model._encoder_acous(
					acous_feats_sub, acous_lengths_sub, device, use_gpu,
					is_training=False, teacher_forcing_ratio=0.0)

				# get static
				src = model._pre_proc_src(src_ids_sub, device)
				src_lengths = [elem - 1 for elem in src_lengths]
				_, static_emb, _ = model._get_src_emb(src, device)

				# prep plot
				commlen = min(dynamic_emb.size(1),static_emb.size(1))
				src_mask_input = (torch.arange(commlen).expand(len(src_lengths), commlen)
					< torch.LongTensor(src_lengths).unsqueeze(1)).to(device=device)
				dynamic_emb = dynamic_emb[:,:commlen][src_mask_input]
				static_emb = static_emb[:,:commlen][src_mask_input]
				hyp_ids = preds[:,:commlen][src_mask_input]
				ref_ids = src[:,:commlen][src_mask_input]
				hyp_words = [test_set.src_id2word[int(id)] for id in hyp_ids]
				ref_words = [test_set.src_id2word[int(id)] for id in ref_ids]

				# plot embeddings
				feats = torch.cat((dynamic_emb, static_emb), dim=0)
				hyp_words.extend(ref_words)
				meta = hyp_words
				color_dynamic = torch.Tensor([0,0,0]).repeat(dynamic_emb.size(0),1) #black
				color_static = torch.Tensor([1,0.5,0]).repeat(static_emb.size(0),1) #orange
				labels = torch.cat((color_dynamic, color_static),dim=0).view(-1,3,1,1)
				writer.add_embedding(feats,metadata=meta,label_img=labels)
				writer.close()

				import pdb; pdb.set_trace()


def gather_emb(test_set, model, test_path_out, use_gpu, max_seq_len, device):

	"""
		gather embedding statistics
	"""

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))
	path_out = os.path.join(test_path_out, 'emb.stats')

	hyp_dict = {}
	ref_dict = {}

	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):
		# for idx in range(2):

			print(idx+1, len(evaliter))
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['srcid'][0]
			src_lengths = batch_items['srclen']
			tgt_ids = batch_items['tgtid'][0]
			tgt_lengths = batch_items['tgtlen']
			acous_feats = batch_items['acous_feat'][0]
			acous_lengths = batch_items['acouslen']

			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			acous_len = max(acous_lengths)
			src_ids = src_ids[:,:src_len].to(device=device)
			tgt_ids = tgt_ids.to(device=device)
			acous_feats = acous_feats.to(device=device)

			n_minibatch = int(tgt_len / 100 + tgt_len % 100 > 0)
			minibatch_size = int(src_ids.size(0) / n_minibatch)
			n_minibatch = int(src_ids.size(0) / minibatch_size) + \
				(src_ids.size(0) % minibatch_size > 0)

			for j in range(n_minibatch):

				st = j * minibatch_size
				ed = min((j+1) * minibatch_size, src_ids.size(0))
				src_ids_sub = src_ids[st:ed,:]
				tgt_ids_sub = tgt_ids[st:ed,:]
				acous_feats_sub = acous_feats[st:ed,:]
				acous_lengths_sub = acous_lengths[st:ed]
				print('minibatch: ', st, ed, src_ids.size(0))

				# get dynamic
				dynamic_emb, logps, preds, lengths = model._encoder_acous(
					acous_feats_sub, acous_lengths_sub, device, use_gpu,
					is_training=False, teacher_forcing_ratio=0.0)

				# get static
				src = model._pre_proc_src(src_ids_sub, device)
				src_lengths = [elem - 1 for elem in src_lengths]
				_, static_emb, _ = model._get_src_emb(src, device)

				# prep
				commlen = min(dynamic_emb.size(1),static_emb.size(1))
				src_mask_input = (torch.arange(commlen).expand(len(src_lengths), commlen)
					< torch.LongTensor(src_lengths).unsqueeze(1)).to(device=device)
				dynamic_emb = dynamic_emb[:,:commlen][src_mask_input]
				static_emb = static_emb[:,:commlen][src_mask_input]
				hyp_ids = preds[:,:commlen][src_mask_input]
				ref_ids = src[:,:commlen][src_mask_input]
				hyp_words = [test_set.src_id2word[int(id)] for id in hyp_ids]
				ref_words = [test_set.src_id2word[int(id)] for id in ref_ids]

				# gather embeddings
				for idx in range(len(hyp_words)):
					word = hyp_words[idx]
					emb = dynamic_emb[idx]
					if word not in hyp_dict:
						hyp_dict[word] = [emb]
					else:
						hyp_dict[word].append(emb)

				for idx in range(len(ref_words)):
					word = ref_words[idx]
					emb = static_emb[idx]
					if word not in ref_dict:
						ref_dict[word] = [emb]
					else:
						ref_dict[word].append(emb)
	# analysis
	hyp_stats = {}
	ref_stats = {}
	for key, val in hyp_dict.items():
		mean = torch.mean(torch.stack(val,dim=0),dim=0)
		std = torch.std(torch.stack(val,dim=0),dim=0)
		hyp_stats[key] = [mean, std, len(val)]

	for key, val in ref_dict.items():
		mean = torch.mean(torch.stack(val,dim=0),dim=0)
		std = torch.std(torch.stack(val,dim=0),dim=0)
		ref_stats[key] = [mean, std]

	# print
	l2_lis = []
	std_lis = []
	fout = open(path_out, 'w')
	fout.write('{}\tl2\tstd_dyn\tcount\n'.format('word'.ljust(10)))
	fout.write('{}\n'.format('-'*50))
	for key, val in ref_stats.items():
		ref_mean = val[0]
		word = key
		word = word.ljust(10)
		if key in hyp_stats:
			hyp_mean = hyp_stats[key][0]
			hyp_std = torch.mean(hyp_stats[key][1])
			count = hyp_stats[key][2]
			l2 = torch.norm(ref_mean-hyp_mean,2)
			fout.write('{}\t{:0.2f}\t{:0.2f}\t{}\n'.format(word, l2, hyp_std, count))
			l2_lis.append(l2)
			if count > 1: std_lis.append(hyp_std)
		else:
			fout.write('{}\t{}\t{}\t{}\n'.format(word, '-', '-', '0'))

	fout.write('\n{}\n'.format('-'*50))
	fout.write('l2 mean: {}\n'.format(torch.mean(torch.Tensor(l2_lis))))
	fout.write('dynamic per word std mean: {}\n'.format(torch.mean(torch.Tensor(std_lis))))
	fout.write('static across words std mean: {}\n'.format(torch.mean(
		torch.std(model.enc_embedder.weight, dim=0))))
	fout.close()


def compute_kl(test_set, model, test_path_out, use_gpu, max_seq_len, device):

	"""
		compute KL divergence between ASR and AE output
	"""

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)
	print('num batches: {}'.format(len(evaliter)))
	path_out = os.path.join(test_path_out, 'kl.stats')

	# init losses
	resloss_asr = 0
	resloss_ae = 0
	resloss_kl = 0
	resloss_l2 = 0
	h_asr = 0
	h_ae = 0
	resloss_norm = 0

	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):
		# for idx in range(2):

			print(idx+1, len(evaliter))
			batch_items = evaliter.next()

			# load data
			src_ids = batch_items['srcid'][0]
			src_lengths = batch_items['srclen']
			tgt_ids = batch_items['tgtid'][0]
			tgt_lengths = batch_items['tgtlen']
			acous_feats = batch_items['acous_feat'][0]
			acous_lengths = batch_items['acouslen']

			src_len = max(src_lengths)
			tgt_len = max(tgt_lengths)
			acous_len = max(acous_lengths)
			src_ids = src_ids[:,:src_len].to(device=device)
			tgt_ids = tgt_ids.to(device=device)
			acous_feats = acous_feats.to(device=device)

			n_minibatch = int(tgt_len / 100 + tgt_len % 100 > 0)
			minibatch_size = int(src_ids.size(0) / n_minibatch)
			n_minibatch = int(src_ids.size(0) / minibatch_size) + \
				(src_ids.size(0) % minibatch_size > 0)

			for j in range(n_minibatch):

				st = j * minibatch_size
				ed = min((j+1) * minibatch_size, src_ids.size(0))
				src_ids_sub = src_ids[st:ed,:]
				tgt_ids_sub = tgt_ids[st:ed,:]
				acous_feats_sub = acous_feats[st:ed,:]
				acous_lengths_sub = acous_lengths[st:ed]
				print('minibatch: ', st, ed, src_ids_sub.size(0))

				# generate logp
				out_dict = model.forward_eval(src=src_ids_sub, acous_feats=acous_feats_sub,
					acous_lens=acous_lengths_sub, mode='AE_ASR', use_gpu=use_gpu)

				max_len = min(src_ids_sub.size(1), out_dict['preds_asr'].size(1))
				preds_hyp_asr = out_dict['preds_asr'][:,:max_len-1]
				preds_hyp_ae = out_dict['preds_ae'][:,:max_len-1]
				emb_hyp_asr = out_dict['emb_asr'][:,:max_len-1]
				emb_hyp_ae = out_dict['emb_ae'][:,:max_len-1]
				logps_hyp_asr = out_dict['logps_asr'][:,:max_len-1]
				logps_hyp_ae = out_dict['logps_ae'][:,:max_len-1]

				refs_ae = out_dict['refs_ae'][:,:max_len-1]
				src_ids_sub = src_ids_sub[:,:max_len]
				non_padding_mask_src = src_ids_sub.data.ne(PAD)
				import pdb; pdb.set_trace()

				# various losses
				loss_asr = NLLLoss()
				loss_asr.reset()
				loss_ae = NLLLoss()
				loss_ae.reset()
				loss_kl = KLDivLoss()
				loss_kl.reset()
				loss_l2 = MSELoss()
				loss_l2.reset()

				loss_asr.eval_batch_with_mask(logps_hyp_asr.reshape(-1, logps_hyp_asr.size(-1)),
					src_ids_sub[:,1:].reshape(-1), non_padding_mask_src[:,1:].reshape(-1))
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

				loss_asr.normalise()
				loss_ae.normalise()
				loss_kl.normalise()
				loss_l2.normalise()

				resloss_asr += loss_asr.get_loss()
				resloss_ae += loss_ae.get_loss()
				resloss_kl += loss_kl.get_loss()
				resloss_l2 += loss_l2.get_loss()
				resloss_norm += 1

				# compute per token entropy
				entropy_asr = torch.mean(Categorical(
					probs = torch.exp(logps_hyp_asr)).entropy())
				entropy_ae = torch.mean(Categorical(
					probs = torch.exp(logps_hyp_ae)).entropy())
				h_asr += entropy_asr.item()
				h_ae += entropy_ae.item()

				# import pdb; pdb.set_trace()

	fout = open(path_out, 'w')
	fout.write('Various stats (averaged over tokens)')
	fout.write('\n{}\n'.format('-'*50))
	fout.write('NLL ASR: {:0.2f}\n'.format(1. * resloss_asr / resloss_norm))
	fout.write('NLL AE: {:0.2f}\n'.format(1. * resloss_ae / resloss_norm))
	fout.write('KL between ASR, AE: {:0.2f}\n'.format(1. * resloss_kl / resloss_norm))
	fout.write('L2 between embeddings: {:0.2f}\n'.format(1. * resloss_l2 / resloss_norm))
	fout.write('Entropy ASR: {:0.2f}\n'.format(1. * h_asr / resloss_norm))
	fout.write('Entropy AE: {:0.2f}\n'.format(1. * h_ae / resloss_norm))

	fout.close()


def main():

	# load config
	parser = argparse.ArgumentParser(description='Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# load src-tgt pair
	test_path_src = config['test_path_src']
	test_path_tgt = config['test_path_tgt']
	if type(test_path_tgt) == type(None):
		test_path_tgt = test_path_src

	test_path_out = config['test_path_out']
	test_acous_path = config['test_acous_path']
	acous_norm_path = config['acous_norm_path']

	load_dir = config['load']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	seqrev = config['seqrev']
	use_type = config['use_type']

	# set test mode
	MODE = config['eval_mode']
	if MODE != 2:
		if not os.path.exists(test_path_out):
			os.makedirs(test_path_out)
		config_save_dir = os.path.join(test_path_out, 'eval.cfg')
		save_config(config, config_save_dir)

	# check device:
	device = check_device(use_gpu)
	print('device: {}'.format(device))

	# load model
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model = resume_checkpoint.model.to(device)
	vocab_src = resume_checkpoint.input_vocab
	vocab_tgt = resume_checkpoint.output_vocab
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# combine model
	if type(config['combine_path']) != type(None):
		model = combine_weights(config['combine_path'])
	# import pdb; pdb.set_trace()

	# load test_set
	test_set = Dataset(path_src=test_path_src, path_tgt=test_path_tgt,
						vocab_src_list=vocab_src, vocab_tgt_list=vocab_tgt,
						use_type=use_type,
						acous_path=test_acous_path,
						seqrev=seqrev,
						acous_norm=config['acous_norm'],
						acous_norm_path=config['acous_norm_path'],
						acous_max_len=6000,
						max_seq_len_src=900,
						max_seq_len_tgt=900,
						batch_size=batch_size,
						mode='ST',
						use_gpu=use_gpu)

	print('Test dir: {}'.format(test_path_src))
	print('Testset loaded')
	sys.stdout.flush()

	# '{AE|ASR|MT|ST}-{REF|HYP}'
	if len(config['gen_mode'].split('-')) == 2:
		gen_mode = config['gen_mode'].split('-')[0]
		history = config['gen_mode'].split('-')[1]
	elif len(config['gen_mode'].split('-')) == 1:
		gen_mode = config['gen_mode']
		history = 'HYP'

	# run eval:
	if MODE == 1:
		translate(test_set, model, test_path_out, use_gpu,
			max_seq_len, beam_width, device, seqrev=seqrev,
			gen_mode=gen_mode, history=history)

	elif MODE == 2: # save combined model
		ckpt = Checkpoint(model=model,
				   optimizer=None, epoch=0, step=0,
				   input_vocab=test_set.vocab_src,
				   output_vocab=test_set.vocab_tgt)
		saved_path = ckpt.save_customise(
			os.path.join(config['combine_path'].strip('/')+'-combine','combine'))
		log_ckpts(config['combine_path'], config['combine_path'].strip('/')+'-combine')
		print('saving at {} ... '.format(saved_path))

	elif MODE == 3:
		plot_emb(test_set, model, test_path_out, use_gpu, max_seq_len, device)

	elif MODE == 4:
		gather_emb(test_set, model, test_path_out, use_gpu, max_seq_len, device)

	elif MODE == 5:
		compute_kl(test_set, model, test_path_out, use_gpu, max_seq_len, device)



if __name__ == '__main__':
	main()
