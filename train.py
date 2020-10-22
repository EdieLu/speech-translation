import torch
import random
import time
import os
import argparse
import sys
import numpy as np

from utils.misc import set_global_seeds, save_config, validate_config, check_device
from utils.dataset import Dataset
from models.Seq2seq import Seq2seq

from trainer.trainer_asr import Trainer_ASR
from trainer.trainer_ae_asr import Trainer_AE_ASR
from trainer.trainer_mt import Trainer_MT
from trainer.trainer_ae_asr_mt import Trainer_AE_ASR_MT
from trainer.trainer_asr_st import Trainer_ASR_ST
from trainer.trainer_st import Trainer_ST

# from trainer.trainer_ae_mt import Trainer_AE_MT


def load_arguments(parser):

	""" Speech translation: sub tasks - AE/ASR/MT/ST """

	# acous params
	parser.add_argument('--las_acous_dim', type=int, default=40, help='acoustic feature dimension')
	parser.add_argument('--las_acous_hidden_size', type=int, default=256, help='acoustics hidden size')
	parser.add_argument('--las_acous_max_len', type=int, default=3000, help='maximum acous length')
	parser.add_argument('--las_acous_norm', type=str, default='True', help='input acoustic fbk normalisation')

	# data
	parser.add_argument('--loss_nll_asr_coeff', type=float, default=0.0, help='En nll loss coeff')
	parser.add_argument('--loss_nll_st_coeff', type=float, default=0.0, help='De nll loss coeff')
	parser.add_argument('--loss_nll_ae_coeff', type=float, default=0.0, help='En nll loss coeff')
	parser.add_argument('--loss_nll_mt_coeff', type=float, default=0.0, help='De nll loss coeff')
	parser.add_argument('--loss_kl_en_coeff', type=float, default=0.0, help='coefficient of KL loss of En')
	parser.add_argument('--loss_kl_de_coeff', type=float, default=0.0, help='coefficient of KL loss of De')
	parser.add_argument('--loss_l2_coeff', type=float, default=0.0, help='coefficient of l2 loss of ENC output')

	# paths-3way
	parser.add_argument('--st_data_ratio', type=float, default=1.0, help='data partition being used')
	parser.add_argument('--st_acous_norm_path', type=str, default=None, help='acoustics norm')
	parser.add_argument('--st_train_acous_path', type=str, default=None, help='train set acoustics')
	parser.add_argument('--st_dev_acous_path', type=str, default=None, help='dev set acoustics')
	parser.add_argument('--st_train_path_src', type=str, default=None, help='train src dir')
	parser.add_argument('--st_train_path_tgt', type=str, default=None, help='train src dir')
	parser.add_argument('--st_dev_path_src', type=str, default=None, help='dev src dir')
	parser.add_argument('--st_dev_path_tgt', type=str, default=None, help='dev src dir')

	# paths-asr
	parser.add_argument('--asr_data_ratio', type=float, default=1.0, help='data partition being used')
	parser.add_argument('--asr_train_acous_norm_path', type=str, default=None, help='asr train acoustics norm')
	parser.add_argument('--asr_train_acous_path', type=str, default=None, help='asr train set acoustics')
	parser.add_argument('--asr_train_path_src', type=str, default=None, help='asr train src dir')
	parser.add_argument('--asr_dev_acous_norm_path', type=str, default=None, help='asr dev acoustics norm')
	parser.add_argument('--asr_dev_acous_path', type=str, default=None, help='asr dev set acoustics')
	parser.add_argument('--asr_dev_path_src', type=str, default=None, help='asr dev src dir')

	# paths-mt
	parser.add_argument('--mt_data_ratio', type=float, default=1.0, help='data partition being used')
	parser.add_argument('--mt_train_path_src', type=str, default=None, help='mt train src dir')
	parser.add_argument('--mt_train_path_tgt', type=str, default=None, help='mt train src dir')
	parser.add_argument('--mt_dev_path_src', type=str, default=None, help='mt dev src dir')
	parser.add_argument('--mt_dev_path_tgt', type=str, default=None, help='mt dev src dir')

	# vocab
	parser.add_argument('--path_vocab_src', type=str, default=None, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, default=None, help='vocab src dir')
	parser.add_argument('--load_embedding_src', type=str, default=None, help='pretrained embedding src')
	parser.add_argument('--load_embedding_tgt', type=str, default=None, help='pretrained embedding tgt')

	parser.add_argument('--save', type=str, required=True, help='model save dir')
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--load_freeze', type=str, default=None, help='Freeze loaded parameters or not')
	parser.add_argument('--load_mode', type=str, default=None,
		help='null | resume | restart | LAS | ASR | AE-ASR | AE-ASR-MT')

	# model
	parser.add_argument('--use_type', type=str, default='char', help='use char level prediction for nmt')
	parser.add_argument('--share_embedder', type=str, default='False', help='share embedder or not')
	parser.add_argument('--embedding_size_enc', type=int, default=200, help='embedding size encoder')
	parser.add_argument('--embedding_size_dec', type=int, default=200, help='embedding size decoder')
	parser.add_argument('--enc_emb_proj', type=str, default='False', help='encoder embedding projection')
	parser.add_argument('--dec_emb_proj', type=str, default='False', help='decoder embedding projection')

	parser.add_argument('--num_heads', type=int, default=8, help='multi head attention')
	parser.add_argument('--dim_model', type=int, default=512, help='dim_model')
	parser.add_argument('--dim_feedforward', type=int, default=1024, help='dim_feedforward')
	parser.add_argument('--enc_layers', type=int, default=6, help='number of encoder layers')
	parser.add_argument('--dec_layers', type=int, default=6, help='number of decoder layers')
	parser.add_argument('--transformer_type', type=str, default='standard', help='universal | standard')
	parser.add_argument('--act', type=str, default='False', help='universal transformer, dynamic hault')

	# misc
	parser.add_argument('--eval_with_mask', type=str, default='True', help='calc loss excluding padded words')
	parser.add_argument('--embedding_dropout', type=float, default=0.0, help='embedding dropout')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
	parser.add_argument('--seqrev', type=str, default='False', help='reverse src, tgt sequence')
	parser.add_argument('--perturb_emb', type=str, default='False', help='add embed perturbation in mode MT')
	parser.add_argument('--perturb_emb_fraction', type=float, default=0.0, help='perturbation fraction')
	parser.add_argument('--train_rnnlm', type=str, default='False',
		help='whether or not using deep rnnlm fusion in LAS')

	# train
	parser.add_argument('--random_seed', type=int, default=333, help='random seed')
	parser.add_argument('--gpu_id', type=int, default=0, help='only used for memory reservation')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--num_epochs', type=int, default=10, help='number of training epoches')
	parser.add_argument('--max_seq_len_src', type=int, default=32, help='maximum src sequence length')
	parser.add_argument('--max_seq_len_tgt', type=int, default=32, help='maximum tgt sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--minibatch_partition', type=int, default=20, help='separate into minibatch - avoid OOM')
	parser.add_argument('--learning_rate', type=float, default=0.00001, help='learning rate')
	parser.add_argument('--learning_rate_init', type=float, default=0.0005, help='learning rate init')
	parser.add_argument('--lr_warmup_steps', type=int, default=12000, help='lr warmup steps')
	parser.add_argument('--normalise_loss', type=str, default='True', help='normalise loss or not')
	parser.add_argument('--max_grad_norm', type=float, default=1.0,
		help='optimiser gradient norm clipping: max grad norm')
	parser.add_argument('--mode', type=str, default='ASR',
		help='operating mode: combination of AE|ASR|MT|ST following this order')

	# save and print
	parser.add_argument('--grab_memory', type=str, default='True', help='grab full GPU memory')
	parser.add_argument('--max_count_no_improve', type=int, default=2,
		help='if meet max, operate roll back')
	parser.add_argument('--max_count_num_rollback', type=int, default=2,
		help='if meet max, reduce learning rate')
	parser.add_argument('--keep_num', type=int, default=1,
		help='number of models to keep')
	parser.add_argument('--checkpoint_every', type=int, default=10,
		help='save ckpt every n steps')
	parser.add_argument('--print_every', type=int, default=10,
		help='print every n steps')

	return parser


def main():

	# import pdb; pdb.set_trace()
	# load config
	parser = argparse.ArgumentParser(description='LAS + NMT Training')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# set random seed
	if config['random_seed'] is not None:
		set_global_seeds(config['random_seed'])

	# record config
	if not os.path.isabs(config['save']):
		config_save_dir = os.path.join(os.getcwd(), config['save'])
	if not os.path.exists(config['save']):
		os.makedirs(config['save'])

	# resume or not
	if type(config['load']) != type(None) and config['load_mode'] == 'resume':
		config_save_dir = os.path.join(config['save'], 'model-cont.cfg')
	else:
		config_save_dir = os.path.join(config['save'], 'model.cfg')
	save_config(config, config_save_dir)

	loss_coeff = {}
	loss_coeff['nll_ae'] = config['loss_nll_ae_coeff']
	loss_coeff['nll_asr'] = config['loss_nll_asr_coeff']
	loss_coeff['nll_mt'] = config['loss_nll_mt_coeff']
	loss_coeff['nll_st'] = config['loss_nll_st_coeff']
	loss_coeff['kl_en'] = config['loss_kl_en_coeff']
	loss_coeff['kl_de'] = config['loss_kl_de_coeff']
	loss_coeff['l2'] = config['loss_l2_coeff']

	# contruct trainer
	Trainer = globals()['Trainer_{}'.format(config['mode'])]
	t = Trainer(expt_dir=config['save'],
					load_dir=config['load'],
					load_mode=config['load_mode'],
					load_freeze=config['load_freeze'],
					batch_size=config['batch_size'],
					minibatch_partition=config['minibatch_partition'],
					checkpoint_every=config['checkpoint_every'],
					print_every=config['print_every'],
					learning_rate=config['learning_rate'],
					learning_rate_init=config['learning_rate_init'],
					lr_warmup_steps=config['lr_warmup_steps'],
					eval_with_mask=config['eval_with_mask'],
					use_gpu=config['use_gpu'],
					gpu_id=config['gpu_id'],
					max_grad_norm=config['max_grad_norm'],
					max_count_no_improve=config['max_count_no_improve'],
					max_count_num_rollback=config['max_count_num_rollback'],
					keep_num=config['keep_num'],
					normalise_loss=config['normalise_loss'],
					loss_coeff=loss_coeff)

	# vocab
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']

	# ----- 3WAY -----
	train_set = None
	dev_set = None
	mode = config['mode']
	if 'ST' in mode:
		# load train set
		if config['st_train_path_src']:
			t.logger.info(' -- load ST train set -- ')
			train_path_src = config['st_train_path_src']
			train_path_tgt = config['st_train_path_tgt']
			train_acous_path = config['st_train_acous_path']
			train_set = Dataset(path_src=train_path_src, path_tgt=train_path_tgt,
				path_vocab_src=path_vocab_src,
				path_vocab_tgt=path_vocab_tgt,
				use_type=config['use_type'],
				acous_path=train_acous_path,
				seqrev=config['seqrev'],
				acous_norm=config['las_acous_norm'],
				acous_norm_path=config['st_acous_norm_path'],
				acous_max_len=config['las_acous_max_len'],
				max_seq_len_src=config['max_seq_len_src'],
				max_seq_len_tgt=config['max_seq_len_tgt'],
				batch_size=config['batch_size'],
				data_ratio=config['st_data_ratio'],
				use_gpu=config['use_gpu'],
				mode='ST',
				logger=t.logger)

			vocab_size_enc = len(train_set.vocab_src)
			vocab_size_dec = len(train_set.vocab_tgt)
			src_word2id = train_set.src_word2id
			tgt_word2id = train_set.tgt_word2id
			src_id2word = train_set.src_id2word
			tgt_id2word = train_set.tgt_id2word

		# load dev set
		if config['st_dev_path_src']:
			t.logger.info(' -- load ST dev set -- ')
			dev_path_src = config['st_dev_path_src']
			dev_path_tgt = config['st_dev_path_tgt']
			dev_acous_path = config['st_dev_acous_path']
			dev_set = Dataset(path_src=dev_path_src, path_tgt=dev_path_tgt,
				path_vocab_src=path_vocab_src,
				path_vocab_tgt=path_vocab_tgt,
				use_type=config['use_type'],
				acous_path=dev_acous_path,
				acous_norm_path=config['st_acous_norm_path'],
				acous_max_len=config['las_acous_max_len'],
				seqrev=config['seqrev'],
				acous_norm=config['las_acous_norm'],
				max_seq_len_src=config['max_seq_len_src'],
				max_seq_len_tgt=config['max_seq_len_tgt'],
				batch_size=config['batch_size'],
				use_gpu=config['use_gpu'],
				mode='ST',
				logger=t.logger)
		else:
			dev_set = None

	# ----- ASR -----
	asr_train_set = None
	asr_dev_set = None
	if 'ASR' in mode:
		# load train set
		if config['asr_train_path_src']:
			t.logger.info(' -- load ASR train set -- ')
			asr_train_path_src = config['asr_train_path_src']
			asr_train_acous_path = config['asr_train_acous_path']
			asr_train_set = Dataset(path_src=asr_train_path_src, path_tgt=None,
				path_vocab_src=path_vocab_src,
				path_vocab_tgt=path_vocab_tgt,
				use_type=config['use_type'],
				acous_path=asr_train_acous_path,
				acous_norm_path=config['asr_train_acous_norm_path'],
				seqrev=config['seqrev'],
				acous_norm=config['las_acous_norm'],
				acous_max_len=config['las_acous_max_len'],
				max_seq_len_src=config['max_seq_len_src'],
				max_seq_len_tgt=config['max_seq_len_tgt'],
				batch_size=config['batch_size'],
				data_ratio=config['asr_data_ratio'],
				use_gpu=config['use_gpu'],
				mode='ASR',
				logger=t.logger)

			vocab_size_enc = len(asr_train_set.vocab_src)
			vocab_size_dec = len(asr_train_set.vocab_tgt)
			src_word2id = asr_train_set.src_word2id
			tgt_word2id = asr_train_set.tgt_word2id
			src_id2word = asr_train_set.src_id2word
			tgt_id2word = asr_train_set.tgt_id2word

		# load dev set
		if config['asr_dev_path_src']:
			t.logger.info(' -- load ASR dev set -- ')
			asr_dev_path_src = config['asr_dev_path_src']
			asr_dev_acous_path = config['asr_dev_acous_path']
			asr_dev_set = Dataset(path_src=asr_dev_path_src, path_tgt=None,
				path_vocab_src=path_vocab_src,
				path_vocab_tgt=path_vocab_tgt,
				use_type=config['use_type'],
				acous_path=asr_dev_acous_path,
				acous_norm_path=config['asr_dev_acous_norm_path'],
				acous_max_len=config['las_acous_max_len'],
				seqrev=config['seqrev'],
				acous_norm=config['las_acous_norm'],
				max_seq_len_src=config['max_seq_len_src'],
				max_seq_len_tgt=config['max_seq_len_tgt'],
				batch_size=config['batch_size'],
				use_gpu=config['use_gpu'],
				mode='ASR',
				logger=t.logger)
		else:
			asr_dev_set = None

	# ----- MT -----
	mt_train_set = None
	mt_dev_set = None
	if 'MT' in mode:
		# load train set
		if config['mt_train_path_src']:
			t.logger.info(' -- load MT train set -- ')
			mt_train_path_src = config['mt_train_path_src']
			mt_train_path_tgt = config['mt_train_path_tgt']
			mt_train_set = Dataset(path_src=mt_train_path_src, path_tgt=mt_train_path_tgt,
				path_vocab_src=path_vocab_src,
				path_vocab_tgt=path_vocab_tgt,
				use_type=config['use_type'],
				acous_path=None,
				acous_norm_path=None,
				seqrev=config['seqrev'],
				acous_norm=config['las_acous_norm'],
				acous_max_len=config['las_acous_max_len'],
				max_seq_len_src=config['max_seq_len_src'],
				max_seq_len_tgt=config['max_seq_len_tgt'],
				batch_size=config['batch_size'],
				data_ratio=config['mt_data_ratio'],
				use_gpu=config['use_gpu'],
				mode='MT',
				logger=t.logger)

			vocab_size_enc = len(mt_train_set.vocab_src)
			vocab_size_dec = len(mt_train_set.vocab_tgt)
			src_word2id = mt_train_set.src_word2id
			tgt_word2id = mt_train_set.tgt_word2id
			src_id2word = mt_train_set.src_id2word
			tgt_id2word = mt_train_set.tgt_id2word

		# load dev set
		if config['mt_dev_path_src']:
			t.logger.info(' -- load MT dev set -- ')
			mt_dev_path_src = config['mt_dev_path_src']
			mt_dev_path_tgt = config['mt_dev_path_tgt']
			mt_dev_set = Dataset(path_src=mt_dev_path_src, path_tgt=mt_dev_path_tgt,
				path_vocab_src=path_vocab_src,
				path_vocab_tgt=path_vocab_tgt,
				use_type=config['use_type'],
				acous_path=None,
				acous_norm_path=None,
				acous_max_len=config['las_acous_max_len'],
				seqrev=config['seqrev'],
				acous_norm=config['las_acous_norm'],
				max_seq_len_src=config['max_seq_len_src'],
				max_seq_len_tgt=config['max_seq_len_tgt'],
				batch_size=config['batch_size'],
				use_gpu=config['use_gpu'],
				mode='MT',
				logger=t.logger)
		else:
			mt_dev_set = None

	# collect all datasets
	train_sets = {}
	dev_sets = {}
	train_sets['st'] = train_set
	train_sets['asr'] = asr_train_set
	train_sets['mt'] = mt_train_set
	dev_sets['st'] = dev_set
	dev_sets['asr'] = asr_dev_set
	dev_sets['mt'] = mt_dev_set

	# device
	device = check_device(config['use_gpu'])
	t.logger.info('device:{}'.format(device))

	# construct nmt model
	seq2seq = Seq2seq(vocab_size_enc, vocab_size_dec,
					share_embedder=config['share_embedder'],
					enc_embedding_size=config['embedding_size_enc'],
					dec_embedding_size=config['embedding_size_dec'],
					load_embedding_src=config['load_embedding_src'],
					load_embedding_tgt=config['load_embedding_tgt'],
					num_heads=config['num_heads'],
					dim_model=config['dim_model'],
					dim_feedforward=config['dim_feedforward'],
					enc_layers=config['enc_layers'],
					dec_layers=config['dec_layers'],
					embedding_dropout=config['embedding_dropout'],
					dropout=config['dropout'],
					max_seq_len_src=config['max_seq_len_src'],
					max_seq_len_tgt=config['max_seq_len_tgt'],
					act=config['act'],
					enc_word2id=src_word2id,
					dec_word2id=tgt_word2id,
					enc_id2word=src_id2word,
					dec_id2word=tgt_id2word,
					transformer_type=config['transformer_type'],
					enc_emb_proj=config['enc_emb_proj'],
					dec_emb_proj=config['dec_emb_proj'],
					#
					acous_dim=config['las_acous_dim'],
					acous_hidden_size=config['las_acous_hidden_size'],
					#
					mode=config['mode'],
					load_mode=config['load_mode'],
					#
					perturb_emb=config['perturb_emb'],
					perturb_emb_fraction=config['perturb_emb_fraction'],
					#
					train_rnnlm=config['train_rnnlm']
					)
	seq2seq = seq2seq.to(device=device)

	# run training
	seq2seq = t.train(train_sets, seq2seq, num_epochs=config['num_epochs'],
		dev_sets=dev_sets, grab_memory=config['grab_memory'])


if __name__ == '__main__':
	main()
