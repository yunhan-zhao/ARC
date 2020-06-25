import os, time, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets, models, transforms
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from models.depth_generator_networks import _UNetGenerator, init_weights

from utils.metrics import *
from utils.image_pool import ImagePool

from training.base_model import set_requires_grad, base_model

try:    
	from apex import amp
except ImportError:
	print("\nPlease consider install apex from https://www.github.com/nvidia/apex to run with apex or set use_apex = False\n")

import warnings # ignore warnings
warnings.filterwarnings("ignore")

class train_initial_depth_predictor_D(base_model):
	def __init__(self, args, dataloaders_xLabels_joint, dataloaders_single):
		super(train_initial_depth_predictor_D, self).__init__(args)
		self._initialize_training()
		# self.KITTI_MAX_DEPTH_CLIP = 80.0
		# self.EVAL_DEPTH_MIN = 1.0
		# self.EVAL_DEPTH_MAX = 50.0

		self.NYU_MAX_DEPTH_CLIP = 10.0
		self.EVAL_DEPTH_MIN = 1.0
		self.EVAL_DEPTH_MAX = 8.0

		self.dataloaders_single = dataloaders_single
		self.dataloaders_xLabels_joint = dataloaders_xLabels_joint

		self.depthEstModel = _UNetGenerator(input_nc = 3, output_nc = 1)
		self.model_name = ['depthEstModel']
		self.L1loss = nn.L1Loss()

		if self.isTrain:
			self.depth_optimizer = optim.Adam(self.depthEstModel.parameters(), lr=self.task_lr, betas=(0.5, 0.999))
			self.optim_name = ['depth_optimizer']
			self._get_scheduler()
			self.loss_BCE = nn.BCEWithLogitsLoss()
			self._initialize_networks()

			# apex can only be applied to CUDA models
			if self.use_apex:
				self._init_apex(Num_losses=2)

		self.EVAL_best_loss = float('inf')
		self.EVAL_best_model_epoch = 0
		self.EVAL_all_results = {}

		self._check_parallel()

	def _get_project_name(self):
		return 'train_initial_depth_predictor_D'

	def _initialize_networks(self):
		for name in self.model_name:
			getattr(self, name).train().to(self.device)
			init_weights(getattr(self, name), net_name=name, init_type='normal', gain=0.02)

	def compute_depth_loss(self, input_rgb, depth_label, depthEstModel, valid_mask=None):

		prediction = depthEstModel(input_rgb)[-1]
		if valid_mask is not None:
			loss = self.L1loss(prediction[valid_mask], depth_label[valid_mask])
		else:
			assert valid_mask == None
			loss = self.L1loss(prediction, depth_label)
		
		return loss

	def train(self):
		phase = 'train'
		since = time.time()
		best_loss = float('inf')

		tensorboardX_iter_count = 0
		for epoch in range(self.total_epoch_num):
			print('\nEpoch {}/{}'.format(epoch+1, self.total_epoch_num))
			print('-' * 10)
			fn = open(self.train_log,'a')
			fn.write('\nEpoch {}/{}\n'.format(epoch+1, self.total_epoch_num))
			fn.write('--'*5+'\n')
			fn.close()

			self._set_models_train(['depthEstModel'])
			iterCount = 0

			for sample_dict in self.dataloaders_xLabels_joint:
				imageListReal, depthListReal = sample_dict['real']
				imageListSyn, depthListSyn = sample_dict['syn']

				imageListSyn = imageListSyn.to(self.device)
				depthListSyn = depthListSyn.to(self.device)
				imageListReal = imageListReal.to(self.device)
				depthListReal = depthListReal.to(self.device)
				valid_mask = (depthListReal > -1.) # remove undefined regions

				with torch.set_grad_enabled(phase=='train'):
					total_loss = 0.
					self.depth_optimizer.zero_grad()
					real_depth_loss = self.compute_depth_loss(imageListReal, depthListReal, self.depthEstModel, valid_mask)
					syn_depth_loss = self.compute_depth_loss(imageListSyn, depthListSyn, self.depthEstModel)
					total_loss += (real_depth_loss + syn_depth_loss)

					if self.use_apex:
						with amp.scale_loss(total_loss, self.depth_optimizer) as total_loss_scaled:
							total_loss_scaled.backward()
					else:
						total_loss.backward()

					self.depth_optimizer.step()

				iterCount += 1

				if self.use_tensorboardX:
					self.train_display_freq = len(self.dataloaders_xLabels_joint)
					nrow = imageListReal.size()[0]
					if tensorboardX_iter_count % self.train_display_freq == 0:
						pred_depth_real = self.depthEstModel(imageListReal)[-1]

						tensorboardX_grid_real_rgb = make_grid(imageListReal, nrow=nrow, normalize=True, range=(-1.0, 1.0))
						self.train_SummaryWriter.add_image('real rgb images', tensorboardX_grid_real_rgb, tensorboardX_iter_count)

						tensorboardX_depth_concat = torch.cat((depthListReal, pred_depth_real), dim=0)
						tensorboardX_grid_real_depth = make_grid(tensorboardX_depth_concat, nrow=nrow, normalize=True, range=(-1.0, 1.0))
						self.train_SummaryWriter.add_image('real depth and depth prediction', tensorboardX_grid_real_depth, tensorboardX_iter_count)

					# add loss values
					loss_val_list = [total_loss, real_depth_loss, syn_depth_loss]
					loss_name_list = ['total_loss', 'real_depth_loss', 'syn_depth_loss']
					self.write_2_tensorboardX(self.train_SummaryWriter, loss_val_list, name=loss_name_list, mode='scalar', count=tensorboardX_iter_count)

					tensorboardX_iter_count += 1

				if iterCount % 20 == 0:
					loss_summary = '\t{}/{} total_loss: {:.7f}, real_depth_loss: {:.7f}, syn_depth_loss: {:.7f}'.format(
						iterCount, len(self.dataloaders_xLabels_joint), total_loss, real_depth_loss, syn_depth_loss)

					print(loss_summary)
					fn = open(self.train_log,'a')
					fn.write(loss_summary)
					fn.close()

			# take step in optimizer
			for scheduler in self.scheduler_list:
				scheduler.step()
				for optim in self.optim_name:				
					lr = getattr(self, optim).param_groups[0]['lr']
					lr_update = 'Epoch {}/{} finished: {} learning rate = {:.7f}'.format(epoch+1, self.total_epoch_num, optim, lr)
					print(lr_update)
					
					fn = open(self.train_log,'a')
					fn.write(lr_update)
					fn.close()

			if (epoch+1) % self.save_steps == 0:
				self.save_models(self.model_name, mode=epoch+1)
				self.evaluate(epoch+1)

		time_elapsed = time.time() - since
		print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		
		fn = open(self.train_log,'a')
		fn.write('\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
		fn.close()

		best_model_summary = '\nOverall best model is epoch {}'.format(self.EVAL_best_model_epoch)
		print(best_model_summary)
		print(self.EVAL_all_results[str(self.EVAL_best_model_epoch)])
		fn = open(self.evaluate_log, 'a')
		fn.write(best_model_summary + '\n')
		fn.write(self.EVAL_all_results[str(self.EVAL_best_model_epoch)])
		fn.close()	

	def evaluate(self, mode):
		'''
			mode choose from <int> or best
			<int> is the number of epoch, represents the number of epoch, used for in training evaluation
			'best' is used for after training mode
		'''

		set_name = 'test'
		eval_model_list = ['depthEstModel']

		if isinstance(mode, int) and self.isTrain:
			self._set_models_eval(eval_model_list)
			if self.EVAL_best_loss == float('inf'):
				fn = open(self.evaluate_log, 'w')
			else:
				fn = open(self.evaluate_log, 'a')

			fn.write('Evaluating with mode: {}\n'.format(mode))
			fn.write('\tEvaluation range min: {} | max: {} \n'.format(self.EVAL_DEPTH_MIN, self.EVAL_DEPTH_MAX))
			fn.close()

		else:
			self._load_models(eval_model_list, mode)

		print('Evaluating with mode: {}'.format(mode))
		print('\tEvaluation range min: {} | max: {}'.format(self.EVAL_DEPTH_MIN, self.EVAL_DEPTH_MAX))

		total_loss, count = 0., 0
		predTensor = torch.zeros((1, 1, self.cropSize_h, self.cropSize_w)).to('cpu')
		grndTensor = torch.zeros((1, 1, self.cropSize_h, self.cropSize_w)).to('cpu')
		idx = 0

		tensorboardX_iter_count = 0
		for sample in self.dataloaders_single[set_name]:
			imageList, depthList = sample
			valid_mask = np.logical_and(depthList > self.EVAL_DEPTH_MIN, depthList < self.EVAL_DEPTH_MAX)

			idx += imageList.shape[0]
			print('epoch {}: have processed {} number samples in {} set'.format(mode, str(idx), set_name))
			imageList = imageList.to(self.device)
			depthList = depthList.to(self.device)	# real depth

			if self.isTrain and self.use_apex:
				with amp.disable_casts():
					predList = self.depthEstModel(imageList)[-1].detach().to('cpu')
			else:
				predList = self.depthEstModel(imageList)[-1].detach().to('cpu')

			# recover real depth
			predList = (predList + 1.0) * 0.5 * self.NYU_MAX_DEPTH_CLIP
			depthList = depthList.detach().to('cpu')
			predTensor = torch.cat((predTensor, predList), dim=0)
			grndTensor = torch.cat((grndTensor, depthList), dim=0)

			if self.use_tensorboardX:
				nrow = imageList.size()[0]
				if tensorboardX_iter_count % self.val_display_freq == 0:
					depth_concat = torch.cat((depthList, predList), dim=0)
					self.write_2_tensorboardX(self.eval_SummaryWriter, depth_concat, name='{}: ground truth and depth prediction'.format(set_name), 
						mode='image', count=tensorboardX_iter_count, nrow=nrow, value_range=(0.0, self.NYU_MAX_DEPTH_CLIP))

				tensorboardX_iter_count += 1

			if isinstance(mode, int) and self.isTrain:
				eval_depth_loss = self.L1loss(predList[valid_mask], depthList[valid_mask])
				total_loss += eval_depth_loss.detach().cpu()

			count += 1

		if isinstance(mode, int) and self.isTrain:
			validation_loss = (total_loss / count)
			print('validation loss is {:.7f}'.format(validation_loss))
			if self.use_tensorboardX:
				self.write_2_tensorboardX(self.eval_SummaryWriter, validation_loss, name='validation loss', mode='scalar', count=mode)

		results = Result(mask_min=self.EVAL_DEPTH_MIN, mask_max=self.EVAL_DEPTH_MAX)
		results.evaluate(predTensor[1:], grndTensor[1:])

		result1 = '\tabs_rel:{:.3f}, sq_rel:{:.3f}, rmse:{:.3f}, rmse_log:{:.3f}, mae:{:.3f} '.format(
				results.absrel,results.sqrel,results.rmse,results.rmselog,results.mae)
		result2 = '\t[<1.25]:{:.3f}, [<1.25^2]:{:.3f}, [<1.25^3]::{:.3f}'.format(results.delta1,results.delta2,results.delta3)

		print(result1)
		print(result2)

		if isinstance(mode, int) and self.isTrain:
			self.EVAL_all_results[str(mode)] = result1 + '\t' + result2

			if validation_loss.item() < self.EVAL_best_loss:
				self.EVAL_best_loss = validation_loss.item()
				self.EVAL_best_model_epoch = mode
				self.save_models(self.model_name, mode='best')

			best_model_summary = '\tCurrent best loss {:.7f}, current best model {}\n'.format(self.EVAL_best_loss, self.EVAL_best_model_epoch)
			print(best_model_summary)

			fn = open(self.evaluate_log, 'a')
			fn.write(result1 + '\n')
			fn.write(result2 + '\n')
			fn.write(best_model_summary + '\n')
			fn.close()