import os, time, sys
import random
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

from models.depth_generator_networks import _UNetGenerator, init_weights, _ResGenerator_Upsample
from models.discriminator_networks import Discriminator80x80InstNorm
from models.attention_networks import _Attention_FullRes

from utils.metrics import *
from utils.image_pool import ImagePool

from training.base_model import set_requires_grad, base_model

try:
	from apex import amp
except ImportError:
	print("\nPlease consider install apex from https://www.github.com/nvidia/apex to run with apex or set use_apex = False\n")

import warnings # ignore warnings
warnings.filterwarnings("ignore")

class jointly_train_depth_predictor_D_and_attention_module_A(base_model):
	def __init__(self, args, dataloaders_xLabels_joint, dataloaders_single):
		super(jointly_train_depth_predictor_D_and_attention_module_A, self).__init__(args)
		self._initialize_training()
		# self.KITTI_MAX_DEPTH_CLIP = 80.0
		# self.EVAL_DEPTH_MIN = 1.0
		# self.EVAL_DEPTH_MAX = 50.0

		self.NYU_MAX_DEPTH_CLIP = 10.0
		self.EVAL_DEPTH_MIN = 1.0
		self.EVAL_DEPTH_MAX = 8.0

		self.dataloaders_single = dataloaders_single
		self.dataloaders_xLabels_joint = dataloaders_xLabels_joint

		self.attModule = _Attention_FullRes(input_nc = 3, output_nc = 1)
		self.inpaintNet = _ResGenerator_Upsample(input_nc = 3, output_nc = 3)
		self.style_translator_T = _ResGenerator_Upsample(input_nc = 3, output_nc = 3)
		self.netD = Discriminator80x80InstNorm(input_nc = 3)
		self.depthEstModel = _UNetGenerator(input_nc = 3, output_nc = 1)

		self.tau_min = 0.05
		self.rho = 0.85
		self.KL_loss_weight = 1.0
		self.dis_weight = 1.0
		self.fake_loss_weight = 1e-3

		self.tensorboard_num_display_per_epoch = 1
		self.model_name = ['attModule', 'inpaintNet', 'style_translator_T', 'netD', 'depthEstModel']
		self.L1loss = nn.L1Loss()

		if self.isTrain:
			self.optim_netD = optim.Adam(self.netD.parameters(), lr=self.task_lr, betas=(0.5, 0.999))
			self.optim_depth = optim.Adam(list(self.depthEstModel.parameters()) + list(self.attModule.parameters()), lr=self.task_lr, betas=(0.5, 0.999))
			self.optim_name = ['optim_depth', 'optim_netD']
			self._get_scheduler()
			self.loss_BCE = nn.BCEWithLogitsLoss()

			self._initialize_networks(['netD'])

			# load the "best" depth predictor D (from step 1)
			preTrain_path = os.path.join(os.getcwd(), 'experiments', 'train_initial_depth_predictor_D')
			self._load_models(model_list=['depthEstModel'], mode='best', isTrain=True, model_path=preTrain_path)
			print('Successfully loaded pre-trained {} model from {}'.format('depthEstModel', preTrain_path))

			# load the "best" style translator T (from step 2)
			preTrain_path = os.path.join(os.getcwd(), 'experiments', 'train_style_translator_T')
			self._load_models(model_list=['style_translator_T'], mode=480, isTrain=True, model_path=preTrain_path)
			print('Successfully loaded pre-trained {} model from {}'.format('style_translator_T', preTrain_path))

			# load the "best" attention module A (from step 3)
			preTrain_path = os.path.join(os.getcwd(), 'experiments', 'train_initial_attention_module_A')
			self._load_models(model_list=['attModule'], mode=450, isTrain=True, model_path=preTrain_path)
			print('Successfully loaded pre-trained {} model from {}'.format('attModule', preTrain_path))

			# load the "best" inpainting module I (from step 4)
			preTrain_path = os.path.join(os.getcwd(), 'experiments', 'train_inpainting_module_I')
			self._load_models(model_list=['inpaintNet'], mode=450, isTrain=True, model_path=preTrain_path)
			print('Successfully loaded pre-trained {} model from {}'.format('inpaintNet', preTrain_path))

			# apex can only be applied to CUDA models
			if self.use_apex:
				self._init_apex(Num_losses=2)

		self.EVAL_best_loss = float('inf')
		self.EVAL_best_model_epoch = 0
		self.EVAL_all_results = {}

		self._check_parallel()

	def _get_project_name(self):
		return 'jointly_train_depth_predictor_D_and_attention_module_A'

	def _initialize_networks(self, model_name):
		for name in model_name:
			getattr(self, name).train().to(self.device)
			init_weights(getattr(self, name), net_name=name, init_type='normal', gain=0.02)

	def compute_D_loss(self, real_sample, fake_sample, netD):
		loss = 0
		syn_acc = 0
		real_acc = 0

		output = netD(fake_sample)
		label = torch.full((output.size()), self.syn_label, device=self.device)
		predSyn = (output > 0.5).to(self.device, dtype=torch.float32)
		total_num = torch.numel(output)
		syn_acc += (predSyn==label).type(torch.float32).sum().item()/total_num

		loss += self.loss_BCE(output, label)

		output = netD(real_sample)
		label = torch.full((output.size()), self.real_label, device=self.device)                    
		predReal = (output > 0.5).to(self.device, dtype=torch.float32)
		real_acc += (predReal==label).type(torch.float32).sum().item()/total_num

		loss += self.loss_BCE(output, label)

		return loss, syn_acc, real_acc

	def compute_depth_loss(self, input_rgb, depth_label, depthEstModel, valid_mask=None):

		prediction = depthEstModel(input_rgb)[-1]
		if valid_mask is not None:
			loss = self.L1loss(prediction[valid_mask], depth_label[valid_mask])
		else:
			assert valid_mask == None
			loss = self.L1loss(prediction, depth_label)
		
		return loss

	def compute_spare_attention(self, confident_score, t, isTrain=True):
		# t is the temperature --> scalar
		if isTrain:
			noise = torch.rand(confident_score.size(), requires_grad=False).to(self.device)
			noise = (noise + 0.00001) / 1.001
			noise = - torch.log(- torch.log(noise))

			confident_score = (confident_score + 0.00001) / 1.001
			confident_score = (confident_score + noise) / t
		else:
			confident_score = confident_score / t

		confident_score = F.sigmoid(confident_score)

		return confident_score

	def compute_KL_div(self, cf, target=0.5):
		g = cf.mean()
		g = (g + 0.00001) / 1.001 # prevent g = 0. or 1.
		y = target*torch.log(target/g) + (1-target)*torch.log((1-target)/(1-g))
		return y

	def compute_real_fake_loss(self, scores, loss_type, datasrc = 'real', loss_for='discr'):
		if loss_for == 'discr':
			if datasrc == 'real':
				if loss_type == 'lsgan':
					# The Loss for least-square gan
					d_loss = torch.pow(scores - 1., 2).mean()
				elif loss_type == 'hinge':
					# Hinge loss used in the spectral GAN paper
					d_loss = - torch.mean(torch.clamp(scores-1.,max=0.))
				elif loss_type == 'wgan':
					# The Loss for Wgan
					d_loss = - torch.mean(scores)
				else:
					scores = scores.view(scores.size(0),-1).mean(dim=1)
					d_loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores).detach())
			else:
				if loss_type == 'lsgan':
					# The Loss for least-square gan
					d_loss = torch.pow((scores),2).mean()
				elif loss_type == 'hinge':
					# Hinge loss used in the spectral GAN paper
					d_loss = -torch.mean(torch.clamp(-scores-1.,max=0.))
				elif loss_type == 'wgan':
					# The Loss for Wgan
					d_loss = torch.mean(scores)
				else:
					scores = scores.view(scores.size(0),-1).mean(dim=1)
					d_loss = F.binary_cross_entropy_with_logits(scores, torch.zeros_like(scores).detach())

			return d_loss
		else:
			if loss_type == 'lsgan':
				# The Loss for least-square gan
				g_loss = torch.pow(scores - 1., 2).mean()
			elif (loss_type == 'wgan') or (loss_type == 'hinge') :
				g_loss = - torch.mean(scores)
			else:
				scores = scores.view(scores.size(0),-1).mean(dim=1)
				g_loss = F.binary_cross_entropy_with_logits(scores, torch.ones_like(scores).detach())
			return g_loss

	def train(self):
		phase = 'train'
		since = time.time()
		best_loss = float('inf')

		set_requires_grad(self.style_translator_T, requires_grad=False) # freeze style translator T
		set_requires_grad(self.inpaintNet, requires_grad=False) # freeze inpainting module I

		tensorboardX_iter_count = 0
		for epoch in range(self.total_epoch_num):
			print('\nEpoch {}/{}'.format(epoch+1, self.total_epoch_num))
			print('-' * 10)
			fn = open(self.train_log,'a')
			fn.write('\nEpoch {}/{}\n'.format(epoch+1, self.total_epoch_num))
			fn.write('--'*5+'\n')
			fn.close()

			self._set_models_train(['attModule', 'inpaintNet', 'style_translator_T', 'depthEstModel'])
			iterCount = 0

			for sample_dict in self.dataloaders_xLabels_joint:
				imageListReal, depthListReal = sample_dict['real']
				imageListSyn, depthListSyn = sample_dict['syn']

				imageListSyn = imageListSyn.to(self.device)
				depthListSyn = depthListSyn.to(self.device)
				imageListReal = imageListReal.to(self.device)
				depthListReal = depthListReal.to(self.device)
				valid_mask = (depthListReal > -1.)

				B, C, H, W = imageListReal.size()[0], imageListReal.size()[1], imageListReal.size()[2], imageListReal.size()[3]

				with torch.set_grad_enabled(phase=='train'):
					r2s_img = self.style_translator_T(imageListReal)[-1]
					confident_score = self.attModule(imageListReal)[-1]
					# convert to sparse confident score
					confident_score = self.compute_spare_attention(confident_score, t=self.tau_min, isTrain=True)

					mod_r2s_img = r2s_img * confident_score
					inpainted_r2s = self.inpaintNet(mod_r2s_img)[-1]

					reconst_img = inpainted_r2s * (1. - confident_score) + confident_score * r2s_img

					# update depth predictor and attention module
					self.optim_depth.zero_grad()
					total_loss = 0.
					real_depth_loss = self.compute_depth_loss(reconst_img, depthListReal, self.depthEstModel, valid_mask)
					syn_depth_loss = self.compute_depth_loss(imageListSyn, depthListSyn, self.depthEstModel)
					KL_loss = self.compute_KL_div(confident_score, target=self.rho) * self.KL_loss_weight

					fake_pred = self.netD(inpainted_r2s)
					fake_label = torch.full(fake_pred.size(), self.real_label, device=self.device)
					fake_loss = self.loss_BCE(fake_pred, fake_label) * self.fake_loss_weight

					total_loss += (real_depth_loss + syn_depth_loss + KL_loss + fake_loss)
					if self.use_apex:
						with amp.scale_loss(total_loss, self.optim_depth, loss_id=0) as total_loss_scaled:
							total_loss_scaled.backward()
					else:
						total_loss.backward()

					self.optim_depth.step()

					# stop adding adversaial loss after stable
					if epoch <= 100:
						self.optim_netD.zero_grad()
						netD_loss = 0.
						netD_loss, _, _ = self.compute_D_loss(imageListSyn, inpainted_r2s.detach(), self.netD)

						if self.use_apex:
							with amp.scale_loss(netD_loss, self.optim_netD, loss_id=0) as netD_loss_scaled:
								netD_loss_scaled.backward()
						else:
							netD_loss.backward()

						self.optim_netD.step()
					else:
						netD_loss = 0.
						set_requires_grad(self.netD, requires_grad=False)

				iterCount += 1

				if self.use_tensorboardX:
					nrow = imageListReal.size()[0]
					self.train_display_freq = len(self.dataloaders_xLabels_joint) # feel free to adjust the display frequency
					if tensorboardX_iter_count % self.train_display_freq == 0:
						img_concat = torch.cat((imageListReal, r2s_img, mod_r2s_img, inpainted_r2s, reconst_img), dim=0)
						self.write_2_tensorboardX(self.train_SummaryWriter, img_concat, name='real, r2s, r2sMasked, inpaintedR2s, reconst', mode='image',
							count=tensorboardX_iter_count, nrow=nrow)

						self.write_2_tensorboardX(self.train_SummaryWriter, confident_score, name='Attention', mode='image',
							count=tensorboardX_iter_count, nrow=nrow, value_range=(0., 1.0))

					# add loss values
					loss_val_list = [total_loss, real_depth_loss, syn_depth_loss, KL_loss, fake_loss, netD_loss]
					loss_name_list = ['total_loss', 'real_depth_loss', 'syn_depth_loss', 'KL_loss', 'fake_loss', 'netD_loss']
					self.write_2_tensorboardX(self.train_SummaryWriter, loss_val_list, name=loss_name_list, mode='scalar', count=tensorboardX_iter_count)

					tensorboardX_iter_count += 1

				if iterCount % 20 == 0:
					loss_summary = '\t{}/{}, total_loss: {:.7f}, netD_loss: {:.7f}'.format(iterCount, len(self.dataloaders_xLabels_joint), total_loss, netD_loss)
					G_loss_summary = '\t\t G loss summary: real_depth_loss: {:.7f}, syn_depth_loss: {:.7f}, KL_loss: {:.7f} fake_loss: {:.7f}'.format(real_depth_loss, syn_depth_loss, KL_loss, fake_loss)

					print(loss_summary)
					print(G_loss_summary)

					fn = open(self.train_log,'a')
					fn.write(loss_summary + '\n')
					fn.write(G_loss_summary + '\n')
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
				self.save_models(['depthEstModel', 'attModule'], mode=epoch+1)
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
		eval_model_list = ['attModule', 'inpaintNet', 'style_translator_T', 'depthEstModel']

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
			depthList = depthList.to(self.device)

			if self.isTrain and self.use_apex:
				with amp.disable_casts():
					r2s_img = self.style_translator_T(imageList)[-1]
					confident_score = self.attModule(imageList)[-1]
					# convert to sparse confident score
					confident_score = self.compute_spare_attention(confident_score, t=self.tau_min, isTrain=False)
					# hard threshold
					confident_score[confident_score < 0.5] = 0.
					confident_score[confident_score >= 0.5] = 1.
					mod_r2s_img = r2s_img * confident_score
					inpainted_r2s = self.inpaintNet(mod_r2s_img)[-1]
					reconst_img = inpainted_r2s * (1. - confident_score) + confident_score * r2s_img
					predList = self.depthEstModel(reconst_img)[-1].detach().to('cpu') # [-1, 1]

			else:
				r2s_img = self.style_translator_T(imageList)[-1]
				confident_score = self.attModule(imageList)[-1]
				# convert to sparse confident score
				confident_score = self.compute_spare_attention(confident_score, t=self.tau_min, isTrain=False)
				# hard threshold
				confident_score[confident_score < 0.5] = 0.
				confident_score[confident_score >= 0.5] = 1.
				mod_r2s_img = r2s_img * confident_score
				inpainted_r2s = self.inpaintNet(mod_r2s_img)[-1]
				reconst_img = inpainted_r2s * (1. - confident_score) + confident_score * r2s_img
				predList = self.depthEstModel(reconst_img)[-1].detach().to('cpu') # [-1, 1]

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
				self.save_models(['depthEstModel', 'attModule'], mode='best')

			best_model_summary = '\tCurrent best loss {:.7f}, current best model {}\n'.format(self.EVAL_best_loss, self.EVAL_best_model_epoch)
			print(best_model_summary)

			fn = open(self.evaluate_log, 'a')
			fn.write(result1 + '\n')
			fn.write(result2 + '\n')
			fn.write(best_model_summary + '\n')
			fn.close()