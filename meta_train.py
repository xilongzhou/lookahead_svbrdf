# from livelossplot import PlotLosses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict

from torchmeta.utils.gradient_based import gradient_update_parameters

from torch.utils.data import DataLoader
import random
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data as data
from os import listdir
from PIL import Image

from models.renderer import *
from models.networks import VGGLoss,Net_Des19

import argparse

from torch.utils.tensorboard import SummaryWriter

from util.util import EPSILON
from util.descriptor import TDLoss_2

from skimage.transform import resize

from collections import OrderedDict

from meta_model import *

from meta_utils import *



def meta_learn(opt, model, coords, train_data, val_data, device, FeaExtractor=None):

	print(opt.resume_name)

	load_model='pretrain' if opt.load_pretrain else opt.model

	max_iters = opt.train_iter#500000 #@param {type:"integer"}

	outer_lr = opt.outer_lr #@param
	inner_lr = opt.inner_lr #@param
	inner_steps = opt.inner_step #@param {type:"integer"}
	val_steps = opt.val_step #@param {type:"integer"}
	first_order = opt.first_order #@param {type:"boolean"}

	if opt.no_inner:
		val_step=1
		inner_steps=0

	print('######################### inner lr is %.8f #############################'%(inner_lr))
	print('######################### outer lr is %.8f, decay_lr %d #############################'%(outer_lr, opt.decay_lr))
	print('######################### use first order %s #############################'%(first_order))

	model.train()

	if opt.losstype=='L2':
		print('######################### using L2 loss #######################')
		criterion = torch.nn.MSELoss()
	else:
		print('######################### using L1 loss #######################')
		criterion = torch.nn.L1Loss()

	params = OrderedDict(model.meta_named_parameters())

	if opt.use_TD_loss:
		criterionTD = TDLoss_2(device, 0)
	criterionVGG = VGGLoss(opt)

	weights = [0.1, 0.1, 0.1, 0.1]
	keys = [1, 2, 3,4]
	if opt.netloss=='Des19Net':
		criterionNet = MyLoss(opt, keys, weights, device)

	if opt.optim=='adam':
		if opt.netloss=='Des19Net' and not opt.no_learn_loss:
			print('------------------ learn loss %.8f-----------------------' % opt.lossnet_lr)
			if FeaExtractor is None:
				print('------------------ FeaExtractor in inner loop or no FeaExtractor -----------------------')
				opt_outer = torch.optim.Adam([
					{'params':model.parameters(),'lr':outer_lr, 'betas':[opt.beta1,opt.beta2], 'eps':opt.eps_adam},
					{'params':criterionNet.parameters(),'lr':opt.lossnet_lr},
					])
			else:
				print('------------------ FeaExtractor in outer loop %.8f-----------------------'% opt.lossnet_lr)
				opt_outer = torch.optim.Adam([
					{'params':model.parameters(),'lr':outer_lr, 'betas':[opt.beta1,opt.beta2], 'eps':opt.eps_adam},
					{'params':list(criterionNet.parameters())+list(FeaExtractor.parameters()),'lr':opt.lossnet_lr}
					])

		else:
			print('------------------ no learn loss -----------------------')
			if FeaExtractor is None:
				opt_outer = torch.optim.Adam(model.parameters(), lr=outer_lr, betas=[opt.beta1,opt.beta2], eps=opt.eps_adam)
			else:
				print('------------------ FeaExtractor in outer loop %.8f-----------------------'% opt.lossnet_lr)
				opt_outer = torch.optim.Adam([
					{'params':model.parameters(),'lr':outer_lr, 'betas':[opt.beta1,opt.beta2], 'eps':opt.eps_adam},
					{'params':FeaExtractor.parameters(),'lr':opt.lossnet_lr}
					])

		# print('# ------------------- using adam ----------------------- #')
	elif opt.optim=='rprop':
		opt_outer = torch.optim.Rprop(model.parameters(), lr=outer_lr, etas=[opt.eta1,opt.eta2], step_sizes=[opt.step_min,opt.step_max])
		print('# ------------------- using rprop ----------------------- #')

	save_train_img_path = '%s/%s/train_imgs'%(opt.model, opt.name)
	save_val_img_path = '%s/%s/val_imgs'%(opt.model, opt.name)
	save_model_path = '%s/%s/models'%(opt.model, opt.name)
	save_loss_path = '%s/%s/loss'%(opt.model, opt.name)
	if not os.path.exists(save_train_img_path):         
		os.makedirs(save_train_img_path)
	if not os.path.exists(save_val_img_path):         
		os.makedirs(save_val_img_path)
	if not os.path.exists(save_model_path):         
		os.makedirs(save_model_path)
	if not os.path.exists(save_loss_path):         
		os.makedirs(save_loss_path)

	writer = SummaryWriter(save_loss_path)

	if opt.resume_name!='':	
		print('resuming from ', opt.resume_name)
		if opt.model=='OutConSiren' and opt.load_pretrain:
			load_network_pretrain(model, FeaExtractor, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), cond_type=opt.cond_type, name='model')
		else:
			load_network(model,opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='model')
			# load_network(opt_outer,opt.load_iter,'%s/%s/models'%(opt.model, opt.resume_name), name='optimizer')
			if FeaExtractor is not None and not opt.load_pretrain:
				load_network(FeaExtractor, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='FeaExtractor')

			if opt.netloss=='Des19Net' and not opt.load_pretrain:
				load_network(criterionNet.lossnet, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='des19net')

	log_step = []
	val_log_ren = OrderedDict()
	val_log_fea = OrderedDict()
	trainfea_log = []
	trainren_log = []
	weight_list = []
	step = 1

	Position_map=PositionMap(opt.res,opt.res,3).to(device)
	lightinten =16

	in_lirange=0.01

	if opt.model=='Siren':
		myinput = coords[0]
		# print('input shape', myinput.shape)

	LightPos = torch.tensor([[0,0,4.0]], device=device)

	while step < max_iters:

		for i,example in enumerate(train_data):

			if step > max_iters:
				break

			if opt.vary_light:
				LightPos = NLight_VaryH(opt.N_input, in_lirange, device) # 0.03 is good to make light around center

			# hard-code light intensity
			lightinten = LightPos[0][-1].item()**2 if not opt.fixed_inten else 16

			if opt.netloss=='Des19Net':
				criterionNet.set_gradient(True)

			tr_rens = OrderedDict()
			tr_feas = OrderedDict()

			model.zero_grad()
			image = example.to(device)

			VG_ren_loss = torch.tensor(0., device=device)
			L_ren_loss = torch.tensor(0., device=device)
			TD_ren_loss = torch.tensor(0., device=device)
			outer_fea_loss = torch.tensor(0., device=device)
			outer_feavg_loss = torch.tensor(0., device=device)

			for task_id in range(opt.batch_size):
				
				image_tr = image[task_id].permute(1,2,0) #[H,W,C]
				gt_ren_tr = SingleRender_NumberPointLight_FixedCamera(2*image_tr[:,:,0:3]-1, image_tr[:,:,3:6],image_tr[:,:,6:7], image_tr[:,:,9:12], LightPos, Position_map, device, CamLi_co=True, lightinten=lightinten, no_spec=opt.no_spec) #[B,N, W,H,C]
				gt_ren_tr = (gt_ren_tr.clamp(0,1)+opt.EPSILON)**(1/2.2)

				# if opt.N_input>1:
				# gt_ren_tr = gt_ren_tr.clone().reshape(1, gt_ren_tr.shape[1], gt_ren_tr.shape[2], opt.N_input*3) #[1, H, W, 3*N]
				# print('gt_ren_tr ', gt_ren_tr.shape)

				if opt.model=='UNet':
					myinput = 2*gt_ren_tr.permute(0,3,1,2)-1 # [-1,1] #[N,3,H,W]
				elif opt.model=='InConSiren':
					myinput = 2*gt_ren_tr.permute(0,3,1,2)-1 #[N,3,H,W] [-1,1]
					myinput = [coords[0],myinput] # coords:[H,W,2] myinput: [N,3,H,W]
				elif opt.model=='OutConSiren':
					myfea_in = FeaExtractor(2*gt_ren_tr.permute(0,3,1,2)-1).squeeze(0).permute(1,2,0) #[N,3,H,W] --> [H,W,32]
					myinput = torch.cat([coords[0], myfea_in],dim=-1)

				params_cp = params

				## inner loop
				for in_step in range(inner_steps):

					g_tr = model(myinput, params=params_cp) # [0,1] 

					if g_tr.dim()==4:
						g_tr = g_tr.squeeze(0).permute(1,2,0) # [B,C,H,W] --> [H,W,C]

					if 'N' in opt.fea and 'all' not in opt.fea: 
						g_tr = ProcessNormal(opt, g_tr)
					else:
						g_tr = paramize_out(opt,g_tr)

					if in_step==0:
						tr_feas[str(in_step)] = g_tr
					elif in_step==1:
						tr_feas[str(in_step)] = g_tr
					elif in_step==2:
						tr_feas[str(in_step)] = g_tr

					out_ren_tr = SingleRender_NumberPointLight_FixedCamera(2*g_tr[:,:,0:3]-1, g_tr[:,:,3:6], g_tr[:,:,6:7], g_tr[:,:,9:12], LightPos, Position_map, device, CamLi_co=True, lightinten=lightinten, no_spec=opt.no_spec) #[B,N, W,H,C]
					out_ren_tr = (out_ren_tr.clamp(0,1)+opt.EPSILON)**(1/2.2) 
					ren_loss = criterion(gt_ren_tr, out_ren_tr)*opt.Winner_ren

					if opt.netloss=='Des19Net':
						inner_loss, des19_svbrdf_tr = criterionNet(g_tr, 2*gt_ren_tr.permute(0,3,1,2)-1)
						inner_loss += ren_loss
					elif opt.netloss=='GT':
						inner_loss = criterion(g_tr, image_tr)
						inner_loss += ren_loss						
					else:
						inner_loss = ren_loss

					# if opt.Wvgg_inner>0:
					# 	out_ren_tr_vg = VGGpreprocess(out_ren_tr.permute(0,3,1,2))	
					# 	gt_ren_tr_vg = VGGpreprocess(gt_ren_tr.permute(0,3,1,2))
					# 	inner_ren_vgg_loss = criterionVGG(out_ren_tr_vg, gt_ren_tr_vg)*opt.Wvgg_inner
					# 	inner_loss += inner_ren_vgg_loss

					if opt.Wtd_inner>0:
						out_ren_tr_td = out_ren_tr.permute(0,3,1,2)
						gt_ren_tr_td = gt_ren_tr.permute(0,3,1,2)
						inner_ren_td = criterionTD(out_ren_tr_td, gt_ren_tr_td)*opt.Wtd_inner
						inner_loss += inner_ren_td

					# downL2_loss=0
					# if opt.WdownL2_inner>0:
					# 	gt16 = F.interpolate(gt_ren_tr, size=(16,16), mode='bilinear', align_corners=True)
					# 	out16 = F.interpolate(out_ren_tr, size=(16,16), mode='bilinear', align_corners=True)
					# 	downL2_loss = criterion(gt16, out16)*opt.WdownL2_inner
					# 	inner_loss += downL2_loss

					if in_step==0:
						tr_rens[str(in_step)] = out_ren_tr
					elif in_step==1:
						tr_rens[str(in_step)]=out_ren_tr
					elif in_step==2:
						tr_rens[str(in_step)]=out_ren_tr


					model.zero_grad()
					params_cp = gradient_update_parameters(model, inner_loss, params=params_cp, 
													step_size=inner_lr, first_order=first_order)

					if in_step==0 and task_id==0:
						params_cp1=params_cp

					if in_step==1 and task_id==0:
						params_cp2=params_cp

				## outer loop
				pred_tr = model(myinput, params=params_cp)

				if pred_tr.dim()==4:
					pred_tr = pred_tr.squeeze(0).permute(1,2,0)

				if 'N' in opt.fea and 'all' not in opt.fea: 
					pred_tr = ProcessNormal(opt, pred_tr)
				else:
					pred_tr = paramize_out(opt,pred_tr)

				tr_feas[str(inner_steps)] = pred_tr

				pred_ren_tr = SingleRender_NumberPointLight_FixedCamera(2*pred_tr[:,:,0:3]-1, pred_tr[:,:,3:6], pred_tr[:,:,6:7], pred_tr[:,:,9:12], LightPos, Position_map, device, CamLi_co=True, lightinten=lightinten, no_spec=opt.no_spec) #[B,N, W,H,C]
				pred_ren_tr = (pred_ren_tr.clamp(0,1)+opt.EPSILON)**(1/2.2)					
				tr_rens[str(inner_steps)] = pred_ren_tr

				if not opt.no_fea_outer:
					outer_fea_loss += 0.25*(criterion(pred_tr[:,:,0:3], image_tr[:,:,0:3])*opt.wN_outer 
									+ criterion(pred_tr[:,:,3:6], image_tr[:,:,3:6])*opt.wD_outer  
									+ criterion(pred_tr[:,:,6:9], image_tr[:,:,6:9])*opt.wR_outer
									+ criterion(pred_tr[:,:,9:12], image_tr[:,:,9:12])*opt.wS_outer)

					# add VGG loss to features
					if opt.Wfea_vgg>0:
						D_vg_pred = VGGpreprocess(pred_tr[:,:,3:6].unsqueeze(0).permute(0,3,1,2))	
						D_vg_gt = VGGpreprocess(image_tr[:,:,3:6].unsqueeze(0).permute(0,3,1,2))
						D_vg = criterionVGG(D_vg_pred, D_vg_gt)*opt.Wfea_vgg

						S_vg_pred = VGGpreprocess(pred_tr[:,:,9:12].unsqueeze(0).permute(0,3,1,2))	
						S_vg_gt = VGGpreprocess(image_tr[:,:,9:12].unsqueeze(0).permute(0,3,1,2))
						S_vg = criterionVGG(S_vg_pred, S_vg_gt)*opt.Wfea_vgg

						outer_feavg_loss += (D_vg + S_vg)*0.5

				if opt.Nren_outerloss>0:

					if opt.no_fea_outer:
						rand_light_pos = LightPos
					else:
						rand_light_pos = NLight_VaryH(opt.Nren_outerloss, 0.05, device)
					# lightinten_rand = rand_light_pos[0][-1].item()**2 if opt.rand_inlight=='jitter' else 16
					lightinten_rand = rand_light_pos[0][-1].item()**2
					gt_ren_sampled = SingleRender_NumberPointLight_FixedCamera(2*image_tr[:,:,0:3]-1, image_tr[:,:,3:6],image_tr[:,:,6:7], image_tr[:,:,9:12], rand_light_pos, Position_map, device, CamLi_co=True, lightinten=lightinten_rand, no_spec=opt.no_spec) #[B,N, W,H,C]
					gt_ren_sampled = (gt_ren_sampled.clamp(0,1)+opt.EPSILON)**(1/2.2) 
					pred_ren_sampled = SingleRender_NumberPointLight_FixedCamera(2*pred_tr[:,:,0:3]-1, pred_tr[:,:,3:6], pred_tr[:,:,6:7], pred_tr[:,:,9:12], rand_light_pos, Position_map, device, CamLi_co=True, lightinten=lightinten_rand, no_spec=opt.no_spec) #[B,N, W,H,C]
					pred_ren_sampled = (pred_ren_sampled.clamp(0,1)+opt.EPSILON)**(1/2.2)	

					if opt.use_TD_loss:
						out_ren = pred_ren_sampled[0:1,...].permute(0,3,1,2)
						gt_ren = gt_ren_sampled[0:1,...].permute(0,3,1,2)
						out_16 = nn.functional.interpolate(out_ren, size = opt.dres, mode='bilinear', align_corners=True)
						gt_16 = nn.functional.interpolate(gt_ren, size = opt.dres, mode='bilinear', align_corners=True)
						dren_loss = criterion(out_16, gt_16)*opt.Wdren_outer
						TDren_loss = criterionTD(out_ren, gt_ren)*opt.WTDren_outer
						L_ren_loss+=dren_loss
						TD_ren_loss+=TDren_loss
					else:
						L_ren_loss += criterion(pred_ren_sampled, gt_ren_sampled)
						if opt.Wvgg_outer>0:
							out_ren_te_vg = VGGpreprocess(pred_ren_sampled[0:1,...].permute(0,3,1,2))	
							gt_ren_te_vg = VGGpreprocess(gt_ren_sampled[0:1,...].permute(0,3,1,2))
							VG_ren_loss += criterionVGG(out_ren_te_vg, gt_ren_te_vg)*opt.Wvgg_outer


			outer_fea_loss.div_(opt.batch_size)
			outer_feavg_loss.div_(opt.batch_size)
			# outer_ren_loss.div_(opt.batch_size)
			L_ren_loss.div_(opt.batch_size)
			TD_ren_loss.div_(opt.batch_size)
			VG_ren_loss.div_(opt.batch_size)

			outer_ren_loss = L_ren_loss + VG_ren_loss + TD_ren_loss

			outer_loss = outer_fea_loss + outer_feavg_loss + outer_ren_loss

			opt_outer.zero_grad()
			outer_loss.backward()
			opt_outer.step()
			
			if step % opt.save_freq == 0 or step==1:
				if opt.netloss=='Des19Net':
					criterionNet.set_gradient(False)
				if not opt.no_val:
					val_rens = OrderedDict()
					val_feas = OrderedDict()
					val_rens_loss = OrderedDict()
					val_feas_loss = OrderedDict()
					val_N_loss = OrderedDict()
					val_D_loss = OrderedDict()
					val_R_loss = OrderedDict()
					val_S_loss = OrderedDict()

					model.eval()
					g_test = None
					out_val = None
					valfea_loss = 0
					valren_loss = 0

					for i,val_example in enumerate(val_data):

						val_example, _ = val_example

						val_img = val_example.to(device)

						image_te = val_img[0].permute(1,2,0) # [256, 256, C]
						gt_ren_te = SingleRender_NumberPointLight_FixedCamera(2*image_te[:,:,0:3]-1, image_te[:,:,3:6],image_te[:,:,6:7], image_te[:,:,9:12], LightPos, Position_map, device, CamLi_co=True, lightinten=lightinten, no_spec=opt.no_spec) #[B,N, W,H,C]
						gt_ren_te = (gt_ren_te.clamp(0,1)+EPSILON)**(1/2.2)

						if opt.model=='UNet':
							myinput = 2*gt_ren_te.permute(0,3,1,2)-1 # #[B,C,H,W] [-1,1]
						elif opt.model=='InConSiren':
							myinput = 2*gt_ren_te.permute(0,3,1,2)-1 #[B,3,H,W] [-1,1]
							myinput = [coords[0],myinput] # coords:[H,W,2] myinput: [B,3,H,W]
						elif opt.model=='OutConSiren':
							myfea_in = FeaExtractor(2*gt_ren_te.permute(0,3,1,2)-1).squeeze(0).permute(1,2,0) #[B,3,H,W] --> [H,W,32]
							myinput = torch.cat([coords[0], myfea_in],dim=-1)

						params_test = params

						for in_step in range(val_steps+1):
							g_te = model(myinput, params=params_test)
							if g_te.dim()==4:
								g_te = g_te.squeeze(0).permute(1,2,0)

							if opt.fea=='all_N1' or opt.fea=='D+R' or opt.fea=='all_N2' or opt.fea=='all_N3':
								g_te = paramize_out(opt,g_te)
							elif opt.fea=='N' or opt.fea=='N2' or opt.fea=='N3':
								g_te = ProcessNormal(opt, g_te)
							elif opt.fea=='D':
								g_te = g_te
							elif opt.fea=='R':
								g_te = g_te.repeat(1,1,3)	

							if in_step==0:
								val_feas[str(in_step)] = g_te
							elif in_step==1:
								val_feas[str(in_step)] = g_te
							elif in_step==2:
								val_feas[str(in_step)] = g_te

							if not opt.no_inner:

								out_ren_te = SingleRender_NumberPointLight_FixedCamera(2*g_te[:,:,0:3]-1, g_te[:,:,3:6], g_te[:,:,6:7], g_te[:,:,9:12], LightPos, Position_map, device, CamLi_co=True, lightinten=lightinten, no_spec=opt.no_spec) #[B,N, W,H,C]
								out_ren_te = (out_ren_te.clamp(0,1)+EPSILON)**(1/2.2)
								ren_loss = criterion(gt_ren_te, out_ren_te)*opt.Winner_ren

								if opt.netloss=='Des19Net':
									# inner_loss, des19_svbrdf = criterionNet(2*out_ren_tr.permute(0,3,1,2)-1, 2*gt_ren_tr.permute(0,3,1,2)-1)
									inner_loss, des19_svbrdf_te = criterionNet(g_te, 2*gt_ren_te.permute(0,3,1,2)-1)
									inner_loss += ren_loss
								elif opt.netloss=='GT':
									inner_loss = criterion(g_te, image_te)
									inner_loss += ren_loss	
								else:
									inner_loss = ren_loss


								if opt.Wtd_inner>0:
									out_ren_te_td = out_ren_te.permute(0,3,1,2)
									gt_ren_te_td = gt_ren_te.permute(0,3,1,2)
									inner_loss += criterionTD(out_ren_te_td, gt_ren_te_td)*opt.Wtd_inner

								if in_step==0:
									val_rens[str(in_step)] = out_ren_te
								elif in_step==1:
									val_rens[str(in_step)] = out_ren_te
								elif in_step==2:
									val_rens[str(in_step)] = out_ren_te

								model.zero_grad()
								params_test = gradient_update_parameters(model, inner_loss, params=params_test, 
																step_size=inner_lr, first_order=True)
					
						# compute losses
						for ((k1,fea), (k2,ren)) in zip(val_feas.items(), val_rens.items()):
							val_feas_loss[k1] = val_feas_loss[k1] + criterion(fea, image_te) if k1 in val_feas_loss else criterion(fea, image_te)
							val_rens_loss[k1] = val_rens_loss[k1] + criterion(ren, gt_ren_te) if k1 in val_rens_loss else criterion(ren, gt_ren_te)

							val_N_loss[k1] = val_N_loss[k1] + criterion(fea[:,:,0:3], image_te[:,:,0:3]) if k1 in val_N_loss else criterion(fea[:,:,0:3], image_te[:,:,0:3])
							val_D_loss[k1] = val_D_loss[k1] + criterion(fea[:,:,3:6], image_te[:,:,3:6]) if k1 in val_D_loss else criterion(fea[:,:,3:6], image_te[:,:,3:6])
							val_R_loss[k1] = val_R_loss[k1] + criterion(fea[:,:,6:9], image_te[:,:,6:9]) if k1 in val_R_loss else criterion(fea[:,:,6:9], image_te[:,:,6:9])
							val_S_loss[k1] = val_S_loss[k1] + criterion(fea[:,:,9:12], image_te[:,:,9:12]) if k1 in val_S_loss else criterion(fea[:,:,9:12], image_te[:,:,9:12])

					################# save imgs ################33
					final_vis_tr = OrderedDict()
					final_vis_te = OrderedDict()
					if 'all' in opt.fea:
						for key_fea, fea in val_feas.items():
							final_vis_te[key_fea] = torch.cat((fea[:,:,0:3],(fea[:,:,3:6]+EPSILON)**(1/2.2),fea[:,:,6:9],(fea[:,:,9:12]+EPSILON)**(1/2.2)),dim=1)
						final_vis_te['gt'] = torch.cat((image_te[:,:,0:3],image_te[:,:,3:6]**(1/2.2),image_te[:,:,6:9],image_te[:,:,9:12]**(1/2.2)),dim=1)
						for key_fea, fea in tr_feas.items():
							final_vis_tr[key_fea] = torch.cat((fea[:,:,0:3],(fea[:,:,3:6]+EPSILON)**(1/2.2),fea[:,:,6:9],(fea[:,:,9:12]+EPSILON)**(1/2.2)),dim=1)
						final_vis_tr['gt'] = torch.cat((image_tr[:,:,0:3],image_tr[:,:,3:6]**(1/2.2),image_tr[:,:,6:9],image_tr[:,:,9:12]**(1/2.2)),dim=1)

					if opt.netloss=='Des19Net':
						final_vis_tr['des19'] = torch.cat((des19_svbrdf_tr[:,:,0:3],des19_svbrdf_tr[:,:,3:6]**(1/2.2),des19_svbrdf_tr[:,:,6:9],des19_svbrdf_tr[:,:,9:12]**(1/2.2),des19_svbrdf_tr[:,:,3:6]**(1/2.2)),dim=1)						
						final_vis_te['des19'] = torch.cat((des19_svbrdf_te[:,:,0:3],des19_svbrdf_te[:,:,3:6]**(1/2.2),des19_svbrdf_te[:,:,6:9],des19_svbrdf_te[:,:,9:12]**(1/2.2),des19_svbrdf_te[:,:,3:6]**(1/2.2)),dim=1)						
					
					for key_ren, val_ren in val_rens.items():
						# final_vis_te[key_ren] = torch.cat((final_vis_te[key_ren], val_ren[save_index,...]),dim=1)
						for j in range(opt.N_input):
							final_vis_te[key_ren] = torch.cat((final_vis_te[key_ren], val_ren[j,...]),dim=1)
					for j in range(opt.N_input):
						final_vis_te['gt'] = torch.cat((final_vis_te['gt'], gt_ren_te[j,...]),dim=1)

					for key_ren, tr_ren in tr_rens.items():
						for j in range(opt.N_input):
							final_vis_tr[key_ren] = torch.cat((final_vis_tr[key_ren], tr_ren[j,...]),dim=1)
					for j in range(opt.N_input):
						final_vis_tr['gt'] = torch.cat((final_vis_tr['gt'], gt_ren_tr[j,...]),dim=1)

					if opt.Nren_outerloss>0:
						final_vis_tr['gt'] = torch.cat((final_vis_tr['gt'], gt_ren_sampled[0,...]),dim=1)
						final_vis_tr[str(inner_steps)] = torch.cat((final_vis_tr[str(inner_steps)], pred_ren_sampled[0,...]),dim=1)

					# save images
					fig = plt.figure(figsize=(30,10))
					for i, key in enumerate(final_vis_te):
						print(i, key)
						plt.subplot(len(final_vis_te), 1, int(i)+1)
						plt.imshow((torch.clamp(final_vis_te[key], 0, 1).cpu().detach().numpy()*255.0).astype(np.uint8))
					fig.savefig(os.path.join(save_val_img_path,'%d.png'%(step)))
					plt.close()

					# save images
					fig = plt.figure(figsize=(30,10))
					for i, key in enumerate(final_vis_tr):
						print(i, key)
						plt.subplot(len(final_vis_tr), 1, int(i)+1)
						plt.imshow((torch.clamp(final_vis_tr[key], 0, 1).cpu().detach().numpy()*255.0).astype(np.uint8))
					fig.savefig(os.path.join(save_train_img_path,'%d.png'%(step)))
					plt.close()

					del final_vis_tr
					del final_vis_te
					del val_feas
					del val_rens

					################# save logs ################
					log_step.append(step)
					# weight_list.append(weight_fea)

					for k,v in val_rens_loss.items():
						val_N_loss[k] = val_N_loss[k].div_(len(val_data))
						val_D_loss[k] = val_D_loss[k].div_(len(val_data))
						val_R_loss[k] = val_R_loss[k].div_(len(val_data))
						val_S_loss[k] = val_S_loss[k].div_(len(val_data))
						val_feas_loss[k] = val_feas_loss[k].div_(len(val_data))
						val_rens_loss[k] = val_rens_loss[k].div_(len(val_data))

						if k in val_log_fea:
							val_log_fea[k].append(val_feas_loss[k].data.cpu().numpy())
						else:
							val_log_fea[k] = [val_feas_loss[k].data.cpu().numpy()]

						if k in val_log_ren:
							val_log_ren[k].append(val_rens_loss[k].data.cpu().numpy())
						else:
							val_log_ren[k] = [val_rens_loss[k].data.cpu().numpy()]

					# trainfea_log.append(outer_loss.data.cpu().numpy())
					# trainren_log.append(outer_ren_loss.data.cpu().numpy())

					# save_loss(weight_list, save_loss_path, log_step, save_name='weight_fea')
					save_loss(val_log_fea, save_loss_path, log_step, save_name='val_fea')
					save_loss(val_log_ren, save_loss_path, log_step, save_name='val_ren')
					# save_loss(trainfea_log, save_loss_path, log_step, save_name='train_fea')
					# save_loss(trainren_log, save_loss_path, log_step, save_name='train_ren')

					writer.add_scalar(f'valfea_1step_loss',val_feas_loss[f'{opt.inner_step}'].item(), step)
					writer.add_scalar(f'valren_1step_loss',val_rens_loss[f'{opt.inner_step}'].item(), step)

					writer.add_scalars(f'val_1step_loss',{'N': val_N_loss[f'{opt.inner_step}'].item()}, step)
					writer.add_scalars(f'val_1step_loss',{'D': val_D_loss[f'{opt.inner_step}'].item()}, step)
					writer.add_scalars(f'val_1step_loss',{'R':val_R_loss[f'{opt.inner_step}'].item()}, step)
					writer.add_scalars(f'val_1step_loss',{'S':val_S_loss[f'{opt.inner_step}'].item()}, step)

					writer.add_scalar('trainfe_loss',outer_loss.item(), step)
					writer.add_scalar('trainren_loss',outer_ren_loss.item(), step)

					temp_lr = opt_outer.param_groups[0]['lr']
					print(f'step: {step:d}, outer_lr: {temp_lr:.8f}, wN: {opt.wN_outer:f}, wS: {opt.wS_outer:f}, wD: {opt.wD_outer:f}, wR: {opt.wR_outer:f},trfea: {outer_fea_loss:.5f},trfeavg: {outer_feavg_loss:.5f},trren: {outer_ren_loss:.5f}, valfea: {val_feas_loss[str(inner_steps)]:.5f}, valren: {val_rens_loss[str(inner_steps)]:.5f}')

					print(f'Ren downsampled {opt.dres} L2 loss: {L_ren_loss:.5f}, TD loss:{TD_ren_loss:.5f}, VG loss:{VG_ren_loss:.5f}')

					model.train()
					# for p in criterionNet.parameters():
					# 	p.requires_grad = True
				# ------------------------------ for debuging ------------------------------------
				else:
					print(step)
				# ------------------------------ debuging ------------------------------------

			if step%10000==0 or step==1:

				model_dict = {}
				model_dict['model']=model.cpu().state_dict()
				model.cuda()
				if opt.netloss=='Des19Net':
					model_dict['des19net']=criterionNet.lossnet.cpu().state_dict()
					criterionNet.lossnet.cuda()
				if opt.model=='OutConSiren':
					model_dict['FeaExtractor']=FeaExtractor.cpu().state_dict()
					FeaExtractor.cuda()
				
				# model_dict['optimizer'] = opt_outer.state_dict()

				save_all_models(model_dict, save_model_path, step)
				del model_dict

			if step%opt.decay_lr==0 and step!=0:
				opt_outer.param_groups[0]['lr'] *= 0.5 

			step += 1

def pretrain(opt, model, coords, train_data, val_data, device, FeaExtractor=None):

	coords = coords.repeat(opt.batch_size,1,1,1)

	max_iters = opt.train_iter#500000 #@param {type:"integer"}

	outer_lr = opt.outer_lr #@param
	inner_lr = opt.inner_lr #@param
	inner_steps = opt.inner_step #@param {type:"integer"}
	val_steps = opt.val_step #@param {type:"integer"}
	first_order = opt.first_order #@param {type:"boolean"}

	model.train()

	if opt.losstype=='L2':
		print('######################### using L2 loss #######################')
		criterion = torch.nn.MSELoss().to(device)
	else:
		print('######################### using L1 loss #######################')
		criterion = torch.nn.L1Loss().to(device)

	criterionTD = TDLoss_2(device, 0)


	params = OrderedDict(model.named_parameters())
	criterionVGG = VGGLoss(opt)

	weights = [0.1, 0.1, 0.1, 0.1]
	keys = [1, 2, 3,4]

	print('------------------ no learn loss -----------------------')
	opt_outer = torch.optim.Adam(model.parameters(), lr=outer_lr, betas=[opt.beta1,opt.beta2], eps=opt.eps_adam)
	
	load_model='pretrain' if opt.load_pretrain else opt.model

	train_img_dir = '%s/%s/train_imgs'%(opt.model, opt.name)
	val_img_dir = '%s/%s/val_imgs'%(opt.model, opt.name)
	ckpt_dir = '%s/%s/models'%(opt.model, opt.name)
	loss_path = '%s/%s/loss'%(opt.model, opt.name)
	if not os.path.exists(train_img_dir):         
		os.makedirs(train_img_dir)
	if not os.path.exists(val_img_dir):         
		os.makedirs(val_img_dir)
	if not os.path.exists(ckpt_dir):         
		os.makedirs(ckpt_dir)
	if not os.path.exists(loss_path):         
		os.makedirs(loss_path)


	if opt.resume_name!='':	
		print('resuming from ', opt.resume_name)

		load_network(model,opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='model')
		# load_network(opt_outer,opt.load_iter,'%s/%s/models'%(opt.model, opt.resume_name), name='optimizer')
		if FeaExtractor is not None and not opt.load_pretrain:
			load_network(FeaExtractor, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='FeaExtractor')

		if opt.netloss=='Des19Net' and not opt.load_pretrain:
			load_network(criterionNet.lossnet, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='des19net')


	writer = SummaryWriter(loss_path)
	step = 1
	Position_map=PositionMap(opt.res,opt.res,3).to(device)
	resize = transforms.Resize((opt.res,opt.res))

	LightPos = torch.tensor([[0,0,4.0]], device=device)

	if opt.model=='Siren':
		myinput = coords

	while step < max_iters:

		for i,example in enumerate(train_data):
			if step > max_iters:
				break

			params = OrderedDict(model.named_parameters())

			model.zero_grad()
			image = example.to(device)

			outer_ren_loss = torch.tensor(0., device=device)
			outer_fea_loss = torch.tensor(0., device=device)

			image_tr = image.permute(0,2,3,1) #[B,H,W,C]
			gt_ren_tr = SingleRender_NumberPointLight_FixedCamera(2*image_tr[...,0:3]-1, image_tr[...,3:6],image_tr[...,6:7], image_tr[...,9:12], LightPos, Position_map, device, CamLi_co=True) #[B, W,H,C]
			gt_ren_tr = (gt_ren_tr.clamp(0,1)+opt.EPSILON)**(1/2.2)

			myinput = 2*gt_ren_tr.permute(0,3,1,2)-1 #[B,3,H,W] [-1,1]
			myinput = [coords,myinput] # coords:[H,W,2] myinput: [B,3,H,W]

			## outer loop
			pred_tr = model(myinput, params = params)
			pred_tr = paramize_out(opt,pred_tr)

			pred_ren_tr = SingleRender_NumberPointLight_FixedCamera(2*pred_tr[...,0:3]-1, pred_tr[...,3:6], pred_tr[...,6:7], pred_tr[...,9:12], LightPos, Position_map, device, CamLi_co=True) #[B,N, W,H,C]
			pred_ren_tr = (pred_ren_tr.clamp(0,1)+opt.EPSILON)**(1/2.2)					

			outer_fea_loss = 0.25*(criterion(pred_tr[...,0:3], image_tr[...,0:3])*opt.wN_outer 
							+ criterion(pred_tr[...,3:6], image_tr[...,3:6]) 
							+ criterion(pred_tr[...,6:9], image_tr[...,6:9])*opt.wR_outer
							+ criterion(pred_tr[...,9:12], image_tr[...,9:12])*opt.wS_outer)	

			# add VGG loss to features
			if opt.Wfea_vgg>0:
				D_vg_pred = VGGpreprocess(pred_tr[...,3:6].permute(0,3,1,2))	
				D_vg_gt = VGGpreprocess(image_tr[...,3:6].permute(0,3,1,2))
				D_vg = criterionVGG(D_vg_pred, D_vg_gt)*opt.Wfea_vgg

				S_vg_pred = VGGpreprocess(pred_tr[...,9:12].permute(0,3,1,2))	
				S_vg_gt = VGGpreprocess(image_tr[...,9:12].permute(0,3,1,2))
				S_vg = criterionVGG(S_vg_pred, S_vg_gt)*opt.Wfea_vgg

				outer_feavg_loss = (D_vg + S_vg)*0.5


			if opt.Nren_outerloss>0:

				rand_light_pos = NLight_VaryH(opt.Nren_outerloss, 0.05, device)
				lightinten_rand = rand_light_pos[0][-1].item()**2
				gt_ren_sampled = SingleRender_NumberPointLight_FixedCamera(2*image_tr[...,0:3]-1, image_tr[...,3:6],image_tr[...,6:7], image_tr[...,9:12], rand_light_pos, Position_map, device, CamLi_co=True, no_spec=opt.no_spec) #[B,N, W,H,C]
				gt_ren_sampled = (gt_ren_sampled.clamp(0,1)+opt.EPSILON)**(1/2.2) 
				pred_ren_sampled = SingleRender_NumberPointLight_FixedCamera(2*pred_tr[...,0:3]-1, pred_tr[...,3:6], pred_tr[...,6:7], pred_tr[...,9:12], rand_light_pos, Position_map, device, CamLi_co=True, no_spec=opt.no_spec) #[B,N, W,H,C]
				pred_ren_sampled = (pred_ren_sampled.clamp(0,1)+opt.EPSILON)**(1/2.2)	

				out_ren = pred_ren_sampled[0:1,...].permute(0,3,1,2)
				gt_ren = gt_ren_sampled[0:1,...].permute(0,3,1,2)
				out_16 = nn.functional.interpolate(out_ren, size = opt.dres, mode='bilinear', align_corners=True)
				gt_16 = nn.functional.interpolate(gt_ren, size = opt.dres, mode='bilinear', align_corners=True)
				L_ren_loss = criterion(out_16, gt_16)*opt.Wdren_outer
				TD_ren_loss = criterionTD(out_ren, gt_ren)*opt.WTDren_outer

				outer_ren_loss = L_ren_loss + TD_ren_loss

			outer_loss = outer_fea_loss + outer_ren_loss + outer_feavg_loss
			opt_outer.zero_grad()
			outer_loss.backward()
			opt_outer.step()

			if step % opt.save_freq == 0 or step==1:
				if not opt.no_val:
					model.eval()
					val_ren_loss = 0
					val_fea_loss = 0
					for i,val_example in enumerate(val_data):
						val_example,_ = val_example
						val_img = val_example.to(device)

						gt_fea_te = val_img.permute(0,2,3,1) # [B, 256, 256, C]
						gt_ren_te = SingleRender_NumberPointLight_FixedCamera(2*gt_fea_te[...,0:3]-1, gt_fea_te[...,3:6],gt_fea_te[...,6:7], gt_fea_te[...,9:12], LightPos, Position_map, device, CamLi_co=True) #[B,N, W,H,C]
						gt_ren_te = (gt_ren_te.clamp(0,1)+EPSILON)**(1/2.2)
						myinput = 2*gt_ren_te.permute(0,3,1,2)-1 #[B,3,H,W] [-1,1]
						myinput = [coords[0:1], myinput] # coords:[H,W,2] myinput: [B,3,H,W]
						out_fea_te = model(myinput, params=params)
						out_fea_te = paramize_out(opt,out_fea_te)

						out_ren_te = SingleRender_NumberPointLight_FixedCamera(2*out_fea_te[...,0:3]-1, out_fea_te[...,3:6], out_fea_te[...,6:7], out_fea_te[...,9:12], LightPos, Position_map, device, CamLi_co=True) #[B,N, W,H,C]
						out_ren_te = (out_ren_te.clamp(0,1)+EPSILON)**(1/2.2)
						val_ren_loss += criterion(gt_ren_te, out_ren_te).data

						# compute losses
						val_fea_loss += criterion(gt_fea_te, out_fea_te).data

					################# save imgs ################33
					fea_tr = torch.cat([image_tr[0,:,:,0:3],image_tr[0,:,:,3:6]**(1/2.2),image_tr[0,:,:,6:9],image_tr[0,:,:,9:12]**(1/2.2)], dim=1)
					pred_tr = torch.cat([pred_tr[0,:,:,0:3],pred_tr[0,:,:,3:6]**(1/2.2),pred_tr[0,:,:,6:9],pred_tr[0,:,:,9:12]**(1/2.2)], dim=1)
					save_image(fea_tr.detach(), os.path.join(train_img_dir,f'{step}gt_fea_tr.png'))
					save_image(pred_tr.detach(), os.path.join(train_img_dir,f'{step}out_fea_tr.png'))
					save_image(gt_ren_sampled[0,...].detach(), os.path.join(train_img_dir,f'{step}gt_ren_tr.png'))
					save_image(pred_ren_sampled[0,...].detach(), os.path.join(train_img_dir,f'{step}out_ren_tr.png'))

					fea_tr = torch.cat([gt_fea_te[0,:,:,0:3],gt_fea_te[0,:,:,3:6]**(1/2.2),gt_fea_te[0,:,:,6:9],gt_fea_te[0,:,:,9:12]**(1/2.2)], dim=1)
					pred_tr = torch.cat([out_fea_te[0,:,:,0:3],out_fea_te[0,:,:,3:6]**(1/2.2),out_fea_te[0,:,:,6:9],out_fea_te[0,:,:,9:12]**(1/2.2)], dim=1)
					save_image(fea_tr.detach(), os.path.join(val_img_dir,f'{step}gt_fea_te.png'))
					save_image(pred_tr.detach(), os.path.join(val_img_dir,f'{step}out_fea_te.png'))
					save_image(gt_ren_te[0,...].detach(), os.path.join(val_img_dir,f'{step}gt_ren_te.png'))
					save_image(out_ren_te[0,...].detach(), os.path.join(val_img_dir,f'{step}out_ren_te.png'))

					################# save logs ################

					val_fea_loss = val_fea_loss.div_(len(val_data))
					val_ren_loss = val_ren_loss.div_(len(val_data))

					writer.add_scalar('val_fea',val_fea_loss.item(), step)
					writer.add_scalar('val_ren',val_ren_loss.item(), step)
					writer.add_scalar('val_total',val_ren_loss.item()+val_fea_loss.item(), step)
					writer.add_scalar('trainfea',outer_fea_loss.item(), step)
					writer.add_scalar('trainren',outer_ren_loss.item(), step)

					print(f'step: {step:d}, trfea: {outer_fea_loss:.5f},trren: {outer_ren_loss:.5f}, valfea: {val_fea_loss:.5f}, valren: {val_ren_loss:.5f}')

					model.train()
					# for p in criterionNet.parameters():
					# 	p.requires_grad = True
				# ------------------------------ for debuging ------------------------------------
				else:
					print(f'step: {step:d},tr_outer_fea: {outer_loss.item():.4f},tr_outer_ren: {outer_ren_loss.item():.4f}')
				# ------------------------------ debuging ------------------------------------

			if step%50000==0 or step==1:
				model_dict = {}
				model_dict['model']=model.cpu().state_dict()
				model.cuda()
				save_all_models(model_dict, ckpt_dir, step)
				del model_dict

			step += 1


def embed(opt, model, coords, device, FeaExtractor=None):

	if opt.test_img=='OurReal':
		# opt.realdata_path='../Dataset/MyRealData_All9Re_Total'
		opt.realdata_path='../MG2/data/out_tmp/my_egsr1'
		opt.light_path='../Dataset/Light/MyReal9_Total'
	elif opt.test_img=='OurReal2':
		opt.realdata_path='../MG2/data/mydata3'	
	elif opt.test_img=='Real2':
		opt.realdata_path='../Dataset/MGRealData_All9'
		# opt.realdata_path='../MG2/data/out_tmp/egsr1'
		opt.light_path='../Dataset/Light/MGReal9'

	load_model='pretrain' if opt.load_pretrain else opt.model

	if opt.test_img=='Real2':
		scenes = []
		with open(os.path.join(opt.realdata_path,'{}.txt'.format(opt.file)), 'r') as files:
			lines = files.readlines() 
			
			for line in lines:
				scenes.append(line.strip())
	else:
		scenes = os.listdir(opt.realdata_path)

	txt_dir = f'{load_model}/{opt.resume_name}/test/{opt.name}'
	if not os.path.exists(txt_dir):         
		os.makedirs(txt_dir)

	RenRMSE = open(os.path.join(txt_dir,'RenRMSE.txt'),'w')
	RenLPIPS = open(os.path.join(txt_dir,'RenLPIPS.txt'),'w')

	scene_N = 0
	TotalRMSE_ren = 0
	TotalLPIPS_ren = 0

	print(len(scenes))

	for scene in scenes:

		scene_N += 1

		print('....................now training scene %s .................'% scene )

		# if scene != 'nima_wood_1':
		# 	continue

		img_dir = f'{load_model}/{opt.resume_name}/test/{opt.name}/{scene}'
		if not os.path.exists(img_dir):         
			os.makedirs(img_dir)

		val_steps = opt.val_step #@param {type:"integer"}
		first_order = opt.first_order #@param {type:"boolean"}

		L1 = torch.nn.L1Loss()
		mse = torch.nn.MSELoss().to(device)

		criterionVGG = VGGLoss(opt)

		weights = [0.1, 0.1, 0.1, 0.1]
		keys = [1, 2, 3,4]
		if opt.netloss=='Des19Net':
			criterionNet = MyLoss(opt, keys, weights, device)

		if opt.resume_name!='':	
			if opt.model=='OutConSiren' and opt.load_pretrain:
				load_network_pretrain(model, FeaExtractor, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), cond_type=opt.cond_type, name='model')
			else:
				# if opt.load_old:
				# 	load_network2(model, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='model')
				# else:
				load_network(model, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='model')
				if opt.netloss=='Des19Net':
					load_network(criterionNet.lossnet, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='des19net')
				if FeaExtractor is not None:
					load_network(FeaExtractor, opt.load_iter,'%s/%s/models'%(load_model, opt.resume_name), name='FeaExtractor')


		val_log_ren = OrderedDict()
		val_log_fea = OrderedDict()
		flag=False


		val_rens = OrderedDict()
		val_feas = OrderedDict()
		val_rens_loss = OrderedDict()
		val_feas_loss = OrderedDict()

		model.eval()
		g_test = None
		out_val = None
		valfea_loss = 0
		valren_loss = 0

		# url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
		# with open_url(url) as f:
		# 	vgg16 = torch.jit.load(f).eval().to(device)

		#################################### load input images ##############################################
		for i in range(9):
			if opt.test_img=='Real2':
				image_i = os.path.join(opt.realdata_path,'{}_{}.png'.format(scene, i)) 
				image_i = Image.open(image_i).convert('RGB')				
			else:
				image_i = os.path.join(opt.realdata_path,scene,'0{}.{}'.format(i, opt.img_format)) 
				image_i = Image.open(image_i).convert('RGB')
				if not image_i.width == 256:
					image_i = image_i.resize((256, 256), Image.LANCZOS)
			image_i = transforms.ToTensor()(image_i).permute(1,2,0) 
			if i==0:
				val_example = image_i.unsqueeze(0)
			else:
				val_example = torch.cat((val_example,image_i.unsqueeze(0)),dim=0)

		image_te = val_example[0:opt.N_input,...].to(device)
		val_test = val_example[opt.N_input:,...].to(device)


		#################################### load MG images for embedding ##############################################
		if opt.test_img=='Real2':
			fea_gt = os.path.join('/home/grads/z/zhouxilong199213/Projects/NERF/MG2/data/out/{}.png'.format(scene)) 
			fea_gt = Image.open(fea_gt).convert('RGB')				
		else:
			fea_gt = os.path.join(opt.realdata_path,scene,'0{}.{}'.format(i, opt.img_format)) 
			fea_gt = Image.open(fea_gt).convert('RGB')
			# if not image_i.width == 256:
			# 	image_i = image_i.resize((256, 256), Image.LANCZOS)

		fea_gt = transforms.ToTensor()(fea_gt).permute(1,2,0) #[H,W,C] 
		fea_gt = torch.cat([fea_gt[:,0:256,:], fea_gt[:,256:2*256,:], fea_gt[:,2*256:3*256,:], fea_gt[:,3*256:4*256,:]], dim=-1)

		MG_loss = FeatureLoss('vgg_conv.pt', [0.125, 0.125, 0.125, 0.125])
		for p in MG_loss.parameters():
			p.requires_grad = False

		params_test = OrderedDict(model.meta_named_parameters())
		inner_lr = opt.inner_lr #@param

		flag_lr = True
		inner_loss = 0.0
		TD_loss = 0.0
		TD16Loss = 0.0

		opt_outer = torch.optim.Adam(model.parameters(), lr=opt.after_innerlr, betas=[opt.beta1, opt.beta2])


		for in_step in range(2000):

			if opt.model=='UNet':
				myinput = 2*image_te.permute(0,3,1,2)-1 # [-1,1] # [B,3,H,W]
			elif opt.model=='InConSiren':
				myinput = 2*image_te.permute(0,3,1,2)-1 # [B,3,H,W] [-1,1]
				myinput = [coords[0],myinput] # coords:[H,W,2] myinput: [B,3,H,W]
			elif opt.model=='OutConSiren':
				myfea_in = FeaExtractor(2*image_te.permute(0,3,1,2)-1).squeeze(0).permute(1,2,0) #[B,3,H,W] --> [H,W,32]
				myinput = torch.cat([coords[0], myfea_in],dim=-1)

			g_te = model(myinput, params=params_test)

			if opt.fea=='all_N1' or opt.fea=='D+R' or opt.fea=='all_N2' or opt.fea=='all_N3':
				g_te = paramize_out(opt,g_te)
			elif opt.fea=='N' or opt.fea=='N2' or opt.fea=='N3':
				g_te = ProcessNormal(opt, g_te)
			elif opt.fea=='D':
				g_te = g_te
			elif opt.fea=='R':
				g_te = g_te.repeat(1,1,3)	

			fea_loss = criterion(g_te[:,:,0:3], fea_gt[:,:,0:3])+ criterion(g_te[:,:,3:6], fea_gt[:,:,3:6])+ criterion(g_te[:,:,6:9], fea_gt[:,:,6:9])+ criterion(g_te[:,:,9:12], fea_gt[:,:,9:12])


			# if opt.loss_after1 =='mse':
			# 	temp_loss = ren_loss
			# elif opt.loss_after1=='vgg':
			# 	out_ren_tr_vg = VGGpreprocess(out_ren_te.permute(0,3,1,2))	
			# 	gt_ren_tr_vg = VGGpreprocess(gt_ren_te.permute(0,3,1,2))
			# 	temp_loss = criterionVGG(out_ren_tr_vg, gt_ren_tr_vg) * 0.005 + ren_loss
			# elif opt.loss_after1=='MG':
			# 	pixel_loss = mse(out_ren_te, gt_ren_te)          
			# 	fea_loss = mse(MG_loss(normalize_vgg19(out_ren_te.permute(0,3,1,2).clone(), False)), gt_fea)*0.001          
			# 	temp_loss = pixel_loss + fea_loss

			# print('out_ren_te', out_ren_te.shape)

			if in_step%100==0:
				print(f'step: {in_step:d}, fea_loss: {fea_loss:.5f}')
				# save_image(svbrdf_vis, os.path.join(img_dir,'fea.png'))
				# save_image(svbrdf_vis0, os.path.join(img_dir,'fea0.png'))
			
			opt_outer.zero_grad()
			fea_loss.backward()
			opt_outer.step()


def save_args(opt):
	args = vars(opt)
	save_opt_path = '%s/%s'%(opt.model, opt.name)

	if not os.path.exists(save_opt_path):         
		os.makedirs(save_opt_path)
	file_name = os.path.join(save_opt_path, 'opt.txt')
	with open(file_name, 'wt') as opt_file:
		opt_file.write('------------ Options -------------\n')
		for k, v in sorted(args.items()):
			opt_file.write('%s: %s\n' % (str(k), str(v)))
		opt_file.write('-------------- End ----------------\n')
	return 


def filter_args(args):
	if args.Wvgg_outer>0:
		args.Nren_outerloss = 1

	if args.mode=='pretrain':
		args.model='InConSiren'
		args.save_freq=1000
		# args.outer_lr=1e-5
		args.decay_lr=300000

	return args

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--res', type=int, default=256, help='resolution')        
	parser.add_argument('--batch_size', type=int, default=3, help='batch_size')        
	parser.add_argument('--n_threads', type=int, default=0, help='number of n_threads')        
	parser.add_argument('--inner_step', type=int, default=1, help='# of inner step')        
	parser.add_argument('--val_step', type=int, default=1, help='# of inner step')        
	parser.add_argument('--load_iter', type=int, default=500000, help='# of iteration')        
	parser.add_argument('--train_iter', type=int, default=1000000, help='# of training iteration')        
	parser.add_argument('--save_freq', type=int, default=2000, help='frequency of saving images')        
	# parser.add_argument('--Light_range', type=float, default=0.0, help='range of input light')        
	parser.add_argument('--EPSILON', type=float, default=1e-6, help='eps value for rendering')        
	parser.add_argument('--lambda_decay', type=float, default=1.0, help='weight_decay of feature weight or stage decay')        
	parser.add_argument('--clip', type=float, default=0, help='gradient clipping')        
	parser.add_argument('--clip_hook', action='store_true', help='gradient clipping using hook')        
	parser.add_argument('--first_order', action='store_true', help='use the first order gradient')        

	parser.add_argument('--no_val', action='store_true', help='no validation dataset')        
	parser.add_argument('--inner_as_outer_loss', action='store_true', help='outer loss same as inner loss')        
	parser.add_argument('--Nren_outerloss', type=int, default=-1, help='# of rendered loss as outer loss')        
	parser.add_argument('--reverse_outer', action='store_true', help='reverse the feature lambda to render lambda')        
	parser.add_argument('--no_learn_loss', action='store_true', help='DO NOT learn loss function')        
	parser.add_argument('--no_spec', action='store_true', help='DO NOT use specular map') 
	parser.add_argument('--gamma', action='store_true', help='gamma output D and S') 

	# for training       
	parser.add_argument('--inner_lr', type=float, default=0.001, help='the inner lr')        
	parser.add_argument('--vary_light', action='store_true', help='vary_light')        
	parser.add_argument('--wN_outer', type=float, default=1, help='weight of normal in the outer loss')        
	parser.add_argument('--wD_outer', type=float, default=1, help='weight of diffuse in the outer loss')        
	parser.add_argument('--wR_outer', type=float, default=1, help='weight of rougness in the outer loss')        
	parser.add_argument('--wS_outer', type=float, default=1, help='weight of specular in the outer loss')        
	parser.add_argument('--HN_factor', type=float, default=10.0, help='height to normal scale factor')        
	parser.add_argument('--lossnet_lr', type=float, default=1e-5, help='the lr of loss net')        
	parser.add_argument('--outer_lr', type=float, default=1e-6, help='the outer lr')        
	parser.add_argument('--decay_lr', type=int, default=200000, help='decay outer learning rate every')        
	parser.add_argument('--eps_adam', type=float, default=1e-8, help='eps of adam optimzer') 
	parser.add_argument('--losstype', type=str, default='L2', help='use L2 or L1 loss')        
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of Adam optimizer')        
	parser.add_argument('--beta2', type=float, default=0.5, help='beta2 of Adam optimizer')
	parser.add_argument('--eta1', type=float, default=0.5, help='eta1 of rprop optimizer')        
	parser.add_argument('--eta2', type=float, default=1.2, help='eta2 of rprop optimizer')	 
	parser.add_argument('--step_min', type=float, default=1e-6, help='minimum step of rprop')        
	parser.add_argument('--step_max', type=float, default=50, help='maximum step of rprop')	 
	parser.add_argument('--Wren', type=float, default=1.0, help='weight of rendering loss')	 
	parser.add_argument('--N_input', type=int, default=1, help='number of input images')        
	parser.add_argument('--Wvgg_outer', type=float, default=-1, help='add vgg for rendering outer')
	parser.add_argument('--WdownL2_inner', type=float, default=-1, help='weight of downL2 inner')	 
	parser.add_argument('--Wvgg_inner', type=float, default=-1, help='add vgg for rendering inner')
	parser.add_argument('--Wtd_inner', type=float, default=-1, help='add td for rendering inner')
	parser.add_argument('--Winner_ren', type=float, default=1, help='weight of inner rendering loss')
	parser.add_argument('--no_inner', action='store_true', help='not use inner step')   
	parser.add_argument('--Wfea_vgg', type=float, default=-1, help='add vgg of feature maps in outer loop')
	parser.add_argument('--use_TD_loss', action='store_true', help='use TD loss and downsampled L1 loss')   
	parser.add_argument('--WTDren_outer', type=float, default=1, help='weight of TD for rendering outer')
	parser.add_argument('--Wdren_outer', type=float, default=1, help='weight of downsampled L2 for rendering outer')
	parser.add_argument('--dres', type=int, default=16, help='downsampled res for L1')        

	# for dataset
	parser.add_argument('--const_weightfea', type=float, default=-1, help='const weight feature: -1: not use const | >=0: use const')	 	       
	parser.add_argument('--train_root', type=str, default='D:/XilongZhou/Research/Research/dataset/DeepMaterialsData/ImageData/SynTrainData', help='input image name')        
	parser.add_argument('--val_root', type=str, default='D:/XilongZhou/Research/Research/dataset/DeepMaterialsData/ImageData/SynTestData2', help='input image name')  
	parser.add_argument('--exp', type=str, default='SVBRDF', help='input image name')        
	parser.add_argument('--name', type=str, default='SVBRDF', help='name of folder')
	parser.add_argument('--mode', type=str, default='meta', help='meta || pretrain || test')
	parser.add_argument('--optim', type=str, default='adam', help='which optimzer to use? adam || sgd || rprop')
	parser.add_argument('--fea', type=str,default='all_N1', help='use all feature maps or just albedo')
	parser.add_argument('--netloss', type=str, default='Des19Net',help='train network loss function')
	parser.add_argument('--Des19Net_npy_path', type=str, default='../Dataset/Des19Net/Des19.npy', help='npy path for des19 ')

	parser.add_argument('--resume_name', type=str, default='', help='resume training from folder wth name')
	parser.add_argument('--load_pretrain', action='store_true', help='load the pretrained model')   
	parser.add_argument('--add_16L2', action='store_true', help='load the pretrained model')   
	parser.add_argument('--fixed_inten', action='store_true', help='fix light intensity or not')   
	parser.add_argument('--sc_des19', action='store_true', help='scale for des19 network')   
	parser.add_argument('--no_fea_outer', action='store_true', help='not using feature loss in outer loop')   

	# network architecture
	parser.add_argument('--num_layers', type=int, default=5, help='# of layers')        
	parser.add_argument('--dim_c', type=int, default=8, help='# of hidden channels in UNet')        
	parser.add_argument('--extfea_c', type=int, default=32, help='# of output channels in feature extractor')        
	parser.add_argument('--model', type=str, default='InConSiren', help='use Siren or Fourier Feature')
	parser.add_argument('--cond_type', type=str, default='unet', help='unet or conv')
	parser.add_argument('--n_layer_unet', type=int, default=4, help='the # of layer in UNet')   
	parser.add_argument('--branch', action='store_true', help='use branch for network')   
	parser.add_argument('--no_coords', action='store_true', help='no_coords input')   
	parser.add_argument('--branch_len', type=int, default=2, help='use branch length for network')   

	# for debug
	parser.add_argument('--file', type=str, default='filesall', help='name of folder')

	parser.add_argument('--test_img', type=str,default='Real2', help='test images')
	parser.add_argument('--meta_debug', type=str,default='', help='debugging')
	opt = parser.parse_args()

	opt = filter_args(opt)

	save_args(opt)

	torch.set_num_threads(8)

	num_val_exs = 5
	CROP_SIZE = opt.res

	train_data_root = 'data/celeba'
	val_data_root = 'data/celeba_val'

	mytraindata_root = opt.train_root
	myvaldata_root = opt.val_root

	Mytrain_set = DataLoaderHelper(mytraindata_root, opt)
	Mytrain_data = DataLoader(dataset=Mytrain_set, batch_size=opt.batch_size, num_workers=opt.n_threads, shuffle=True, drop_last=True)

	Myval_set = DataLoaderHelper_test(myvaldata_root, opt)
	Myval_data = DataLoader(dataset=Myval_set, batch_size=1, num_workers=0, shuffle=True, drop_last=False)

	if opt.exp=='SVBRDF':
		train_dataloader = Mytrain_data
		val_dataloader = Myval_data

	##########################################################################################################
	##########################################################################################################

	device = torch.device('cuda')

	x1 = torch.linspace(0, 1, opt.res+1)[:-1]
	coords = torch.stack(torch.meshgrid([x1,x1]), dim=-1)[None,...]
	coords = coords.to(device)


	if opt.fea=='all_N1':
		out_nc = 8
	elif opt.fea=='all_N2':
		out_nc = 9

	if opt.no_spec:
		out_nc = out_nc - 3

	FeaExtractNet = None
	if opt.model=='Siren':
		model = SirenNet(2, 256, out_nc, opt.num_layers, w0_initial=200., w0=200., 
						 final_activation=lambda x: x + .5).to(device)
	# Featuer Extractor not in inner loop
	elif opt.model=='OutConSiren':
		FeaExtractNet = FeatureExtractor(opt.extfea_c, cond_type=opt.cond_type).to(device)
		model = SirenNet(2+opt.extfea_c, 256, out_nc, opt.num_layers, w0_initial=200., w0=200., 
						 final_activation=lambda x: x + .5).to(device)
	# Featuer Extractor in inner loop
	elif opt.model=='InConSiren':
		print('InConSiren model...')
		model = ConSirenNet(opt, 2, 256, out_nc, opt.num_layers, w0_initial=200., w0=200., 
						 final_activation=lambda x: x + .5, cond_type=opt.cond_type, N_in=opt.N_input, n_layer_unet=opt.n_layer_unet).to(device)		

	elif opt.model=='UNet':
		model = MaterialMetaUNet(layer_n=opt.n_layer_unet).to(device)	


	print(model)
	if FeaExtractNet is not None:
		print(FeaExtractNet)
	print('output nc:', out_nc)
	print('model N params', sum(p.numel() for n,p in model.meta_named_parameters() if p.requires_grad))

	if opt.mode=='meta':
		meta_learn(opt, model, coords, Mytrain_data, Myval_data, device, FeaExtractNet)
	elif opt.mode=='pretrain':
		pretrain(opt, model, coords, Mytrain_data, Myval_data, device, FeaExtractNet)
	elif opt.mode=='embed':
		embed(opt, model, coords, device, FeaExtractNet)