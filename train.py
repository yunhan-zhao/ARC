import os, sys
import random, time, copy
import argparse
import torch
from torch.utils.data import Dataset, DataLoader

from dataloader.NYUv2_dataLoader import NYUv2_dataLoader
from dataloader.Joint_xLabel_dataLoader import Joint_xLabel_train_dataLoader

# step 1: Train Initial Depth Predictor D
# from training.train_initial_depth_predictor_D import train_initial_depth_predictor_D as train_model

# step 2: Train Style Translator T (pre-train T)
# from training.train_style_translator_T import train_style_translator_T as train_model

# step 3: Train Initial Attention Module A
# from training.train_initial_attention_module_A import train_initial_attention_module_A as train_model

# step 4: Train Inpainting Module I (pre-train I)
# from training.train_inpainting_module_I import train_inpainting_module_I as train_model

# step 5: Jointly Train Depth Predictor D and Attention Module A (pre-train A, D)
# from training.jointly_train_depth_predictor_D_and_attention_module_A import jointly_train_depth_predictor_D_and_attention_module_A as train_model

# step 6: Finetune the Whole System with Depth Loss (Modular Coordinate Descent)
from training.finetune_the_whole_system_with_depth_loss import finetune_the_whole_system_with_depth_loss as train_model

import warnings # ignore warnings
warnings.filterwarnings("ignore")

print(sys.version)
print(torch.__version__)

################## set attributes for this project/experiment ##################

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', type=str, default=os.path.join(os.getcwd(), 'experiments'),
							 	help='place to store all experiments')
parser.add_argument('--project_name', type=str, help='Test Project')
parser.add_argument('--path_to_PBRS', type=str, default='/home/yunhaz5/project/Syn2Real/dataset/pbrs',
							 	help='absolute dir of pbrs dataset')
parser.add_argument('--path_to_NYUv2', type=str, default='/home/yunhaz5/project/Syn2Real/dataset/nyu_v2',
								 help='absolute dir of NYUV2 dataset')
parser.add_argument('--isTrain', action='store_true', help='whether this is training phase')
parser.add_argument('--originalDepth', action='store_true', help='whether to use original depth instead of inpainted in training')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=1, help='batch size')
parser.add_argument('--cropSize', type=list, default=[240, 320] , help='size of samples in experiments')
parser.add_argument('--total_epoch_num', type=int, default=50, help='total number of epoch')
parser.add_argument('--device', type=str, default='cpu', help='whether running on gpu')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers in dataLoaders')
args = parser.parse_args()

if torch.cuda.is_available(): 
	args.device='cuda'
	torch.cuda.empty_cache()

# here only for evaluation purpose
datasets_nyuv2 = {set_name: NYUv2_dataLoader(root_dir=args.path_to_NYUv2, set_name=set_name, size=args.cropSize, rgb=True)
				 for set_name in ['train', 'test']}
dataloaders_nyuv2 = {set_name: DataLoader(datasets_nyuv2[set_name], 
									batch_size=args.batch_size if set_name=='train' else args.eval_batch_size,
									shuffle=set_name=='train',
									drop_last=set_name=='train',
									num_workers=args.num_workers)
					 for set_name in ['train', 'test']}

# for training purpose
datasets_xLabels_joint = Joint_xLabel_train_dataLoader(real_root_dir=args.path_to_NYUv2, syn_root_dir=args.path_to_PBRS, paired_data=False)
dataloaders_xLabels_joint = DataLoader(datasets_xLabels_joint,
								 batch_size=args.batch_size,
								 shuffle=True,
								 drop_last=True,
								 num_workers=args.num_workers)

model = train_model(args, dataloaders_xLabels_joint, dataloaders_nyuv2)

if args.isTrain:
	model.train()
	model.evaluate(mode='best')
else:
	model.evaluate(mode='best')