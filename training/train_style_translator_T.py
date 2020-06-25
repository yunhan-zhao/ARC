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

from models.depth_generator_networks import _UNetGenerator, init_weights, _ResGenerator_Upsample
from models.discriminator_networks import Discriminator80x80InstNorm

from utils.metrics import *
from utils.image_pool import ImagePool

from training.base_model import set_requires_grad, base_model

try:    
	from apex import amp
except ImportError:
	print("\nPlease consider install apex from https://www.github.com/nvidia/apex to run with apex or set use_apex = False\n")

import warnings # ignore warnings
warnings.filterwarnings("ignore")

class train_style_translator_T(base_model):
	def __init__(self, args, dataloaders_xLabels_joint, dataloaders_single):
		super(train_style_translator_T, self).__init__(args)
		self._initialize_training()

		self.dataloaders_single = dataloaders_single
		self.dataloaders_xLabels_joint = dataloaders_xLabels_joint

		# define loss weights
		self.lambda_identity = 0.5 # coefficient of identity mapping score
		self.lambda_real = 10.0
		self.lambda_synthetic = 10.0
		self.lambda_GAN = 1.0

		# define pool size in adversarial loss
		self.pool_size = 50
		self.generated_syn_pool = ImagePool(self.pool_size)
		self.generated_real_pool = ImagePool(self.pool_size)

		self.netD_s = Discriminator80x80InstNorm(input_nc = 3)
		self.netD_r = Discriminator80x80InstNorm(input_nc = 3)
		self.netG_s2r = _ResGenerator_Upsample(input_nc = 3, output_nc = 3)
		self.netG_r2s = _ResGenerator_Upsample(input_nc = 3, output_nc = 3)
		self.model_name = ['netD_s', 'netD_r', 'netG_s2r', 'netG_r2s']
		self.L1loss = nn.L1Loss()

		if self.isTrain:
			self.netD_optimizer = optim.Adam(list(self.netD_s.parameters()) + list(self.netD_r.parameters()), lr=self.D_lr, betas=(0.5, 0.999))
			self.netG_optimizer = optim.Adam(list(self.netG_r2s.parameters()) + list(self.netG_s2r.parameters()), lr=self.G_lr, betas=(0.5, 0.999))
			self.optim_name = ['netD_optimizer', 'netG_optimizer']
			self._get_scheduler()
			self.loss_BCE = nn.BCEWithLogitsLoss()
			self._initialize_networks()

			# apex can only be applied to CUDA models
			if self.use_apex:
				self._init_apex(Num_losses=3)

		self._check_parallel()

	def _get_project_name(self):
		return 'train_style_translator_T'

	def _initialize_networks(self):
		for name in self.model_name:
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

	def compute_G_loss(self, real_sample, synthetic_sample, r2s_rgb, s2r_rgb, reconstruct_real, reconstruct_syn):
		'''
		real_sample: [batch_size, 4, 240, 320] real rgb
		synthetic_sample: [batch_size, 4, 240, 320] synthetic rgb
		r2s_rgb: netG_r2s(real)
		s2r_rgb: netG_s2r(synthetic)
		'''
		loss = 0

		# identity loss if applicable
		if self.lambda_identity > 0:
			idt_real = self.netG_s2r(real_sample)[-1]
			idt_synthetic = self.netG_r2s(synthetic_sample)[-1]
			idt_loss = (self.L1loss(idt_real, real_sample) * self.lambda_real + 
				self.L1loss(idt_synthetic, synthetic_sample) * self.lambda_synthetic) * self.lambda_identity
		else:
			idt_loss = 0

		# GAN loss
		real_pred = self.netD_r(s2r_rgb)
		real_label = torch.full(real_pred.size(), self.real_label, device=self.device)
		GAN_loss_real = self.loss_BCE(real_pred, real_label)

		syn_pred = self.netD_s(r2s_rgb)
		syn_label = torch.full(syn_pred.size(), self.real_label, device=self.device)
		GAN_loss_syn = self.loss_BCE(syn_pred, syn_label)

		GAN_loss = (GAN_loss_real + GAN_loss_syn) * self.lambda_GAN

		# cycle consistency loss
		rec_real_loss = self.L1loss(reconstruct_real, real_sample) * self.lambda_real
		rec_syn_loss = self.L1loss(reconstruct_syn, synthetic_sample) * self.lambda_synthetic
		rec_loss = rec_real_loss + rec_syn_loss

		loss += (idt_loss + GAN_loss + rec_loss)

		return loss, idt_loss, GAN_loss, rec_loss

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

			iterCount = 0

			for sample_dict in self.dataloaders_xLabels_joint:
				imageListReal, depthListReal = sample_dict['real']
				imageListSyn, depthListSyn = sample_dict['syn']

				imageListSyn = imageListSyn.to(self.device)
				depthListSyn = depthListSyn.to(self.device)
				imageListReal = imageListReal.to(self.device)
				depthListReal = depthListReal.to(self.device)

				with torch.set_grad_enabled(phase=='train'):
					s2r_rgb = self.netG_s2r(imageListSyn)[-1]
					reconstruct_syn = self.netG_r2s(s2r_rgb)[-1]

					r2s_rgb = self.netG_r2s(imageListReal)[-1]
					reconstruct_real = self.netG_s2r(r2s_rgb)[-1]

					#############  update generator
					set_requires_grad([self.netD_r, self.netD_s], False)

					netG_loss = 0.
					self.netG_optimizer.zero_grad()
					netG_loss, G_idt_loss, G_GAN_loss, G_rec_loss = self.compute_G_loss(imageListReal, imageListSyn,
						r2s_rgb, s2r_rgb, reconstruct_real, reconstruct_syn)

					if self.use_apex:
						with amp.scale_loss(netG_loss, self.netG_optimizer, loss_id=0) as netG_loss_scaled:
							netG_loss_scaled.backward()
					else:
						netG_loss.backward()

					self.netG_optimizer.step()

					#############  update discriminator
					set_requires_grad([self.netD_r, self.netD_s], True)

					self.netD_optimizer.zero_grad()
					r2s_rgb_pool = self.generated_syn_pool.query(r2s_rgb)
					netD_s_loss, netD_s_syn_acc, netD_s_real_acc  = self.compute_D_loss(imageListSyn, r2s_rgb.detach(), self.netD_s)
					s2r_rgb_pool = self.generated_real_pool.query(s2r_rgb)
					netD_r_loss, netD_r_syn_acc, netD_r_real_acc = self.compute_D_loss(imageListReal, s2r_rgb.detach(), self.netD_r)	

					netD_loss = netD_s_loss + netD_r_loss

					if self.use_apex:
						with amp.scale_loss(netD_loss, self.netD_optimizer, loss_id=1) as netD_loss_scaled:
							netD_loss_scaled.backward()
					else:
						netD_loss.backward()
					self.netD_optimizer.step()

				iterCount += 1

				if self.use_tensorboardX:
					self.train_display_freq = len(self.dataloaders_xLabels_joint) # feel free to adjust the display frequency
					nrow = imageListReal.size()[0]
					if tensorboardX_iter_count % self.train_display_freq == 0:
						s2r_rgb_concat = torch.cat((imageListSyn, s2r_rgb, imageListReal, reconstruct_syn), dim=0)
						self.write_2_tensorboardX(self.train_SummaryWriter, s2r_rgb_concat, name='RGB: syn, s2r, real, reconstruct syn', mode='image', 
							count=tensorboardX_iter_count, nrow=nrow)

						r2s_rgb_concat = torch.cat((imageListReal, r2s_rgb, imageListSyn, reconstruct_real), dim=0)
						self.write_2_tensorboardX(self.train_SummaryWriter, r2s_rgb_concat, name='RGB: real, r2s, synthetic, reconstruct real', mode='image', 
							count=tensorboardX_iter_count, nrow=nrow)

					loss_val_list = [netD_loss, netG_loss]
					loss_name_list = ['netD_loss', 'netG_loss']
					self.write_2_tensorboardX(self.train_SummaryWriter, loss_val_list, name=loss_name_list, mode='scalar', count=tensorboardX_iter_count)

					tensorboardX_iter_count += 1

				if iterCount % 20 == 0:
					loss_summary = '\t{}/{} netD: {:.7f}, netG: {:.7f}'.format(iterCount, len(self.dataloaders_xLabels_joint), netD_loss, netG_loss)
					G_loss_summary = '\t\tG loss summary: netG: {:.7f}, idt_loss: {:.7f}, GAN_loss: {:.7f}, rec_loss: {:.7f}'.format(
						netG_loss, G_idt_loss, G_GAN_loss, G_rec_loss)

					print(loss_summary)
					print(G_loss_summary)

					fn = open(self.train_log,'a')
					fn.write(loss_summary + '\n')
					fn.write(G_loss_summary + '\n')
					fn.close()
			
			if (epoch+1) % self.save_steps == 0:
				self.save_models(['netG_r2s'], mode=epoch+1, save_list=['style_translator_T'])

			# take step in optimizer
			for scheduler in self.scheduler_list:
				scheduler.step()
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