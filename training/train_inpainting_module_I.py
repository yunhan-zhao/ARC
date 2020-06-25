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

from loss import PerceptualLoss, StyleLoss, VGG19
from models.depth_generator_networks import _UNetGenerator, init_weights, _ResGenerator_Upsample
from models.attention_networks import _Attention_FullRes
from models.discriminator_networks import Discriminator80x80InstNorm, DiscriminatorGlobalLocal

from utils.metrics import *
from utils.image_pool import ImagePool

from training.base_model import set_requires_grad, base_model

try:
	from apex import amp
except ImportError:
	print("\nPlease consider install apex from https://www.github.com/nvidia/apex to run with apex or set use_apex = False\n")

import warnings # ignore warnings
warnings.filterwarnings("ignore")

class Mask_Buffer():
	"""This class implements an image buffer that stores previously generated images.

	This buffer enables us to update discriminators using a history of generated images
	rather than the ones produced by the latest generators.
	"""

	def __init__(self, pool_size):
		"""Initialize the ImagePool class

		Parameters:
			pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
		"""
		self.pool_size = pool_size
		if self.pool_size > 0:  # create an empty pool
			self.num_imgs = 0
			self.images = []

	def query(self, images):
		"""Return an image from the pool.

		Parameters:
			images: the latest generated images from the generator

		Returns images from the buffer.

		By 50/100, the buffer will return input images.
		By 50/100, the buffer will return images previously stored in the buffer,
		and insert the current images to the buffer.
		"""
		if self.pool_size == 0:  # if the buffer size is 0, do nothing
			return images
		return_images = []
		for image in images:
			image = torch.unsqueeze(image.data, 0)
			if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
				self.num_imgs = self.num_imgs + 1
				self.images.append(image)
				return_images.append(image)
			else:
				# p = random.uniform(0, 1)
				# if p > 0.5:  # the buffer will always return a previously stored image, and insert the current image into the buffer
				random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
				tmp = self.images[random_id].clone()
				self.images[random_id] = image
				return_images.append(tmp)
				# else:       # by another 50% chance, the buffer will return the current image
					# return_images.append(image)
		return_images = torch.cat(return_images, 0)   # collect all the images and return
		return return_images

class train_inpainting_module_I(base_model):
	def __init__(self, args, dataloaders_xLabels_joint, dataloaders_single):
		super(train_inpainting_module_I, self).__init__(args)
		self._initialize_training()

		self.dataloaders_single = dataloaders_single
		self.dataloaders_xLabels_joint = dataloaders_xLabels_joint

		self.use_apex = False # use apex might cause style loss to be 0

		self.mask_buffer = Mask_Buffer(500)

		self.attModule = _Attention_FullRes(input_nc = 3, output_nc = 1) # logits, no tanh()
		self.inpaintNet = _ResGenerator_Upsample(input_nc = 3, output_nc = 3)
		self.style_translator_T = _ResGenerator_Upsample(input_nc = 3, output_nc = 3)
		self.netD = DiscriminatorGlobalLocal(image_size=240)

		self.tau_min = 0.05
		self.use_perceptual_loss = True

		self.p_vgg = VGG19()
		self.s_vgg = VGG19()

		self.perceptual_loss = PerceptualLoss(vgg19=self.p_vgg)
		self.style_loss = StyleLoss(vgg19=self.s_vgg)

		self.reconst_loss_weight = 1.0
		self.perceptual_loss_weight = 1.0
		self.style_loss_weight = 1.0
		self.fake_loss_weight = 0.01

		self.model_name = ['attModule', 'inpaintNet', 'style_translator_T', 'netD', 'p_vgg', 's_vgg']
		self.L1loss = nn.L1Loss()

		if self.isTrain:
			self.optim_inpaintNet = optim.Adam(self.inpaintNet.parameters(), lr=self.task_lr, betas=(0.5, 0.999))
			self.optim_netD = optim.Adam(self.netD.parameters(), lr=self.task_lr, betas=(0.5, 0.999))
			self.optim_name = ['optim_inpaintNet', 'optim_netD']
			self._get_scheduler()
			self.loss_BCE = nn.BCEWithLogitsLoss()
			self._initialize_networks(['inpaintNet', 'netD'])

			# load the "best" style translator T (from step 2)
			preTrain_path = os.path.join(os.getcwd(), 'experiments', 'train_style_translator_T')
			self._load_models(model_list=['style_translator_T'], mode=480, isTrain=True, model_path=preTrain_path)
			print('Successfully loaded pre-trained {} model from {}'.format('style_translator_T', preTrain_path))

			# load the "best" attention module A (from step 3)
			preTrain_path = os.path.join(os.getcwd(), 'experiments', 'train_initial_attention_module_A')
			self._load_models(model_list=['attModule'], mode=450, isTrain=True, model_path=preTrain_path)
			print('Successfully loaded pre-trained {} model from {}'.format('attModule', preTrain_path))

			# apex can only be applied to CUDA models
			if self.use_apex:
				self._init_apex(Num_losses=2)

		self._check_parallel()

	def _get_project_name(self):
		return 'train_inpainting_module_I'

	def _initialize_networks(self, model_name):
		for name in model_name:
			getattr(self, name).train().to(self.device)
			init_weights(getattr(self, name), net_name=name, init_type='normal', gain=0.02)

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

		set_requires_grad(self.attModule, requires_grad=False) # freeze attention module
		set_requires_grad(self.style_translator_T, requires_grad=False) # freeze sytle translator

		tensorboardX_iter_count = 0
		for epoch in range(self.total_epoch_num):
			print('\nEpoch {}/{}'.format(epoch+1, self.total_epoch_num))
			print('-' * 10)
			fn = open(self.train_log,'a')
			fn.write('\nEpoch {}/{}\n'.format(epoch+1, self.total_epoch_num))
			fn.write('--'*5+'\n')
			fn.close()

			iterCount = 0

			for sample_dict in self.dataloaders_xLabels_joint:
				imageListReal, depthListReal = sample_dict['real']
				imageListSyn, depthListSyn = sample_dict['syn']

				imageListSyn = imageListSyn.to(self.device)
				depthListSyn = depthListSyn.to(self.device)
				imageListReal = imageListReal.to(self.device)
				depthListReal = depthListReal.to(self.device)

				B, C, H, W = imageListReal.size()[0], imageListReal.size()[1], imageListReal.size()[2], imageListReal.size()[3]

				with torch.set_grad_enabled(phase=='train'):
					r2s_img = self.style_translator_T(imageListReal)[-1]
					confident_score = self.attModule(imageListReal)[-1]
					# convert to sparse confident score
					confident_score = self.compute_spare_attention(confident_score, t=self.tau_min, isTrain=False)
					# hard threshold
					confident_score[confident_score < 0.5] = 0.
					confident_score[confident_score >= 0.5] = 1.

					confident_score = self.mask_buffer.query(confident_score)

					mod_r2s_img = r2s_img * confident_score
					inpainted_r2s = self.inpaintNet(mod_r2s_img)[-1]

					reconst_img = inpainted_r2s * (1. - confident_score) + confident_score * r2s_img

					# update generators
					self.optim_inpaintNet.zero_grad()
					total_loss = 0.
					reconst_loss = self.L1loss(inpainted_r2s, r2s_img) * self.reconst_loss_weight
					if self.use_perceptual_loss:
						perceptual_loss = self.perceptual_loss(inpainted_r2s, r2s_img) * self.perceptual_loss_weight
						style_loss = self.style_loss(inpainted_r2s * (1.-confident_score), r2s_img * (1.-confident_score)) * self.style_loss_weight
						total_loss += (perceptual_loss + style_loss)
						
					d_score, _, _ = self.netD(inpainted_r2s, boxImg=confident_score.expand(B, 3, H, W))
					fake_loss = self.compute_real_fake_loss(d_score, loss_type='lsgan', loss_for='generator') * self.fake_loss_weight

					total_loss += (reconst_loss + fake_loss)
					if self.use_apex:
						with amp.scale_loss(total_loss, self.optim_inpaintNet, loss_id=0) as total_loss_scaled:
							total_loss_scaled.backward()
					else:
						total_loss.backward()

					self.optim_inpaintNet.step()

					# update discriminator
					self.optim_netD.zero_grad()

					real_d_score, _, _ = self.netD(r2s_img, boxImg=confident_score.expand(B, 3, H, W))
					real_d_loss = self.compute_real_fake_loss(real_d_score, loss_type='lsgan', datasrc='real')

					fake_d_score, _, _ = self.netD(inpainted_r2s.detach(), boxImg=confident_score.expand(B, 3, H, W))
					fake_d_loss = self.compute_real_fake_loss(fake_d_score, loss_type='lsgan', datasrc='fake')

					total_d_loss = (real_d_loss + fake_d_loss)

					if self.use_apex:
						with amp.scale_loss(total_d_loss, self.optim_netD, loss_id=1) as total_d_loss_scaled:
							total_d_loss_scaled.backward()
					else:
						total_d_loss.backward()

					self.optim_netD.step()

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
					loss_val_list = [total_loss, total_d_loss]
					loss_name_list = ['total_loss', 'total_d_loss']
					self.write_2_tensorboardX(self.train_SummaryWriter, loss_val_list, name=loss_name_list, mode='scalar', count=tensorboardX_iter_count)

					tensorboardX_iter_count += 1

				if iterCount % 20 == 0:
					loss_summary = '\t{}/{}, total_loss: {:.7f}, total_d_loss: {:.7f}'.format(
						iterCount, len(self.dataloaders_xLabels_joint), total_loss, total_d_loss)
					G_loss_summary = '\t\t G loss summary: reconst_loss: {:.7f}, fake_loss: {:.7f}, perceptual_loss: {:.7f} style_loss: {:.7f}'.format(
						reconst_loss, fake_loss, perceptual_loss, style_loss)
					D_loss_summary = '\t\t D loss summary: real_d_loss: {:.7f}, fake_d_loss: {:.7f}'.format(real_d_loss, fake_d_loss)

					print(loss_summary)
					print(G_loss_summary)
					print(D_loss_summary)

					fn = open(self.train_log,'a')
					fn.write(loss_summary + '\n')
					fn.write(G_loss_summary + '\n')
					fn.write(D_loss_summary + '\n')
					fn.close()

			if (epoch+1) % self.save_steps == 0:
				self.save_models(['inpaintNet'], mode=epoch+1)

			# take step in optimizer
			for scheduler in self.scheduler_list:
				scheduler.step()
				# print learning rate
				for optim in self.optim_name:
					lr = getattr(self, optim).param_groups[0]['lr']
					lr_update = 'Epoch {}/{} finished: {} learning rate = {:.7f}'.format(epoch+1, self.total_epoch_num, optim, lr)
					print(lr_update)

					fn = open(self.train_log,'a')
					fn.write(lr_update + '\n')
					fn.close()

		time_elapsed = time.time() - since
		print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

		fn = open(self.train_log,'a')
		fn.write('\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
		fn.close()

	def evaluate(self, mode):
		pass