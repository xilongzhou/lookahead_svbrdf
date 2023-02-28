# from livelossplot import PlotLosses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from tqdm.notebook import tqdm as tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
							   MetaBatchNorm2d, MetaLinear)

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
from models.networks import VGGLoss,Net_Des19,FeatureLoss

import argparse

from torch.utils.tensorboard import SummaryWriter

from util.util import EPSILON, open_url
from util.descriptor import TDLoss_2

from skimage.transform import resize

from collections import OrderedDict
from torchvision.transforms import Normalize

from meta_model import *

from meta_utils import *

from pandas import DataFrame
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores


def normalize_vgg19(input, isGram):
	
	if isGram:
		transform = Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.255]
		)
	else:
		transform = Normalize(
			mean=[0.48501961, 0.45795686, 0.40760392],
			std=[1./255, 1./255, 1./255]
		)
	return transform(input.cpu()).cuda()

## assign the grad of source to target
def write_params(model, param_dict):

	for name, param in model.named_parameters():
		param.data.copy_(param_dict[name])


def N_light_sample(Number, r_max, mydevice):

	# u_1= torch.rand((Number,1),device=mydevice)*r_max #+ r_max	# rmax: 0.95 (default)
	# u_2= torch.rand((Number,1),device=mydevice)

	u_1= torch.rand((Number,1),device=mydevice)*0.0 + r_max	# gif
	u_2= 1.0/Number*torch.arange(1,Number+1,dtype=torch.float32).unsqueeze(-1).to(mydevice) # gif
	# print('u_2:', u_2)
	# print('u_22:', u_22)

	r = torch.sqrt(u_1)
	theta = 2*PI*u_2

	x = r*torch.cos(theta)
	y = r*torch.sin(theta)
	z = torch.sqrt(1-r*r)

	temp_out = torch.cat([x,y,z],1)

	return temp_out


def N_Light_demo(Near_Number,r_max, mydevice):

	rand_light = N_light_sample(Near_Number, r_max, device)

	# dist = torch.exp(torch.normal(torch.tensor([1.0]), torch.tensor([0.2]))).cuda()
	Light_po = rand_light *4.0

	return Light_po


def test(opt, model, coords, val_data, device, FeaExtractor=None):

	val_steps = opt.val_step #@param {type:"integer"}
	first_order = opt.first_order #@param {type:"boolean"}

	L1 = torch.nn.L1Loss()
	mse = torch.nn.MSELoss().to(device)
	params = OrderedDict(model.meta_named_parameters())
	for key in params:
		print(key)
	criterionVGG = VGGLoss(opt)

	weights = [0.1, 0.1, 0.1, 0.1]
	keys = [1, 2, 3,4]
	if opt.netloss=='Des19Net':
		criterionNet = MyLoss(opt, keys, weights, device)

	log_step = []
	val_log_ren = OrderedDict()
	val_log_fea = OrderedDict()
	step = 1
	flag=False

	# LightPos = torch.tensor([[0,0,2.14]], device=device)
	LightPos = torch.tensor([[0,0,4]], device=device)
	Position_map=PositionMap(opt.res,opt.res,3).to(device)

	if opt.netloss=='Des19Net':
		criterionNet.set_gradient(False)

	# Plot_iter=[1,3,5,7,10,12,15,17,20,25,30,35,45,50,60,70,80,90,100,120,140,150]
	Plot_iter=[t for t in range(0,val_steps+1)]
	print(Plot_iter)
	# Plot_iter=[]

	val_rens = OrderedDict()
	val_feas = OrderedDict()

	val_feas_loss = OrderedDict()
	val_total_loss = OrderedDict()
	val_ren_loss = OrderedDict()

	if opt.excel:
		Excel_dict = {}
		Fea_Excel_dict = {}
		Ren_Excel_dict = {}


	model.eval()
	g_test = None
	out_val = None
	valfea_loss = 0
	valren_loss = 0

	load_model = 'pretrain' if opt.load_pretrain else opt.model

	# create directory
	save_valfea_path = f'test/{opt.name}/fea'
	save_in_path = f'test/{opt.name}/in'
	save_valren_path = f'test/{opt.name}/ren'
	save_pro_path = f'test/{opt.name}/pro'

	if not os.path.exists(save_in_path):         
		os.makedirs(save_in_path)

	if not os.path.exists(save_valfea_path):         
		os.makedirs(save_valfea_path)
	if not os.path.exists(save_valren_path):         
		os.makedirs(save_valren_path)
	if not os.path.exists(save_pro_path):         
		os.makedirs(save_pro_path)

	Num = 40

	def delogTensor(tensor):
		temp = torch.exp(tensor*(np.log(1.01)-np.log(0.01)) + np.log(0.01))-0.01

		return temp**(1/2.2)


	if opt.test_img=='SynPlot':
		# rand_light_pos_loss = NLight_VaryH(Num, 0.03, device)
		# np.save('40_light.npy',rand_light_pos_loss.cpu().numpy())
		rand_light_pos_loss  =torch.from_numpy(np.load('40_light_3.npy')).to(device)
		print(rand_light_pos_loss.shape)

	if opt.loss_after1=='TD':
		criterionTD = TDLoss_2(device, 0)

	for img_id, val_example in enumerate(val_data):


		val_example, name=val_example
	
		val_img = val_example.to(device)
		image_gt = val_img[0].permute(1,2,0) # [256, 256, C]

		# load model
		load_network(model, "final", 'ckpt/', name='model')
		if opt.netloss=='Des19Net':
			load_network(criterionNet.lossnet, "final", 'ckpt/', name='des19net')

		params_test = OrderedDict(model.meta_named_parameters())
		inner_lr = opt.inner_lr #@param
		flag_lr = True
		inner_loss = 0.0
		TD_loss = 0.0
		TD16Loss = 0.0

		if not opt.no_use_adam:
			optimAdam = torch.optim.Adam(model.parameters(), lr=opt.after_innerlr, betas=[0.5, 0.5])

		if image_gt.shape[-1]>3:
			image_te = SingleRender_NumberPointLight_FixedCamera(2*image_gt[:,:,0:3]-1, image_gt[:,:,3:6],image_gt[:,:,6:7], image_gt[:,:,9:12], LightPos, Position_map, device, True, lightinten=opt.light_inten) #[B,N, W,H,C]
			image_te = (image_te.clamp(0,1)+EPSILON)**(1/2.2)
		else:
			image_te = image_gt.unsqueeze(0)


		for in_step in range(val_steps+1):

			if opt.model=='UNet':
				myinput = 2*image_te.permute(0,3,1,2)-1 # [-1,1] # [B,3,H,W]
			elif opt.model=='InConSiren':
				gt_ren_te = image_te
				myinput = 2*image_te.permute(0,3,1,2)-1 # [B,3,H,W] [-1,1]
				myinput = [coords[0],myinput] # coords:[H,W,2] myinput: [B,3,H,W]
			elif opt.model=='OutConSiren':
				print(image_te.shape)
				myfea_in = FeaExtractor(2*image_te.permute(0,3,1,2)-1).squeeze(0).permute(1,2,0) #[B,3,H,W] --> [H,W,32]
				myinput = torch.cat([coords[0], myfea_in],dim=-1)

			g_te = model(myinput, params=params_test)

			# print('g_te ',g_te.shape)
			if opt.fea=='all_N1' or opt.fea=='D+R' or opt.fea=='all_N2' or opt.fea=='all_N3':
				g_te = paramize_out(opt,g_te)
			elif opt.fea=='N' or opt.fea=='N2' or opt.fea=='N3':
				g_te = ProcessNormal(opt, g_te)

			if in_step==0:
				val_feas[str(in_step)] = g_te
			elif in_step==1:
				val_feas[str(in_step)] = g_te
			elif in_step==2:
				val_feas[str(in_step)] = g_te
			elif in_step==5:
				val_feas[str(in_step)] = g_te					
			elif in_step==opt.val_step:
				val_feas[str(in_step)] = g_te

			out_ren_te = SingleRender_NumberPointLight_FixedCamera(2*g_te[:,:,0:3]-1, g_te[:,:,3:6], g_te[:,:,6:7], g_te[:,:,9:12], LightPos, Position_map, device, True, lightinten=opt.light_inten) #[B,N, W,H,C]
			out_ren_te = (out_ren_te.clamp(0,1)+EPSILON)**(1/2.2)

			ren_loss = mse(gt_ren_te, out_ren_te)*opt.Winner_ren

			if in_step==0:
				if opt.netloss=='Des19Net':
					# inner_loss, des19_svbrdf = criterionNet(2*out_ren_tr.permute(0,3,1,2)-1, 2*gt_ren_tr.permute(0,3,1,2)-1)
					# inner_loss, des19_svbrdf_te = criterionNet(g_te, 2*gt_ren_te.permute(0,3,1,2)-1)
					inner_loss, des19_svbrdf_te = criterionNet(g_te, 2*gt_ren_te.permute(0,3,1,2)-1)
					inner_loss += ren_loss
				elif opt.netloss=='GT':
					# print(g_tr.shape)
					# print(image_tr.shape)
					inner_loss = mse(g_te, image_gt)
					inner_loss += ren_loss			
				else:
					inner_loss = ren_loss

				if opt.Wvgg_inner>0:
					out_ren_te_vg = VGGpreprocess(out_ren_te.permute(0,3,1,2))	
					gt_ren_te_vg = VGGpreprocess(gt_ren_te.permute(0,3,1,2))
					inner_ren_vgg_loss = criterionVGG(out_ren_te_vg, gt_ren_te_vg)*opt.Wvgg_inner
					inner_loss += inner_ren_vgg_loss

				if opt.Wtd_inner>0:
					out_ren_te_td = out_ren_te.permute(0,3,1,2)
					gt_ren_te_td = gt_ren_te.permute(0,3,1,2)
					inner_ren_td = criterionTD(out_ren_te_td, gt_ren_te_td)*opt.Wtd_inner
					inner_loss += inner_ren_td

			if opt.loss_after1 =='mse':
				temp_loss = ren_loss
			elif opt.loss_after1=='vgg':
				out_ren_tr_vg = VGGpreprocess(out_ren_te.permute(0,3,1,2))	
				gt_ren_tr_vg = VGGpreprocess(gt_ren_te.permute(0,3,1,2))
				temp_loss = criterionVGG(out_ren_tr_vg, gt_ren_tr_vg) * 0.005 + ren_loss

			elif opt.loss_after1=='TD':
				out_ren = out_ren_te[0:1,...].permute(0,3,1,2)
				gt_ren = gt_ren_te[0:1,...].permute(0,3,1,2)
				out_16 = nn.functional.interpolate(out_ren, size = opt.dres, mode='bilinear', align_corners=True)
				gt_16 = nn.functional.interpolate(gt_ren, size = opt.dres, mode='bilinear', align_corners=True)
				dren_loss = mse(out_16, gt_16)*opt.Wdren_outer
				TDren_loss = criterionTD(out_ren, gt_ren)*opt.WTDren_outer
				
				temp_loss = dren_loss + TDren_loss


			if in_step>0 or opt.no_inner_step:
				inner_loss = temp_loss
				vis_loss = temp_loss
			else:
				vis_loss = temp_loss


			# after 1 step
			if flag_lr and in_step>0:
				inner_lr = opt.after_innerlr 

			if 'MG' in opt.loss_after1 and in_step>0:
				# print(in_step)
				flag_lr = False
				if in_step%opt.decay_interval==0:
					inner_lr = inner_lr*0.5

				if not opt.no_use_adam:
					for g in optimAdam.param_groups:
						g['lr'] = inner_lr

			if in_step==0:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==1:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==2:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==5:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==opt.val_step:
				val_rens[str(in_step)] = out_ren_te

			if in_step%10==0:
				print(f'step: {in_step:d}, inner lr: {inner_lr:.8f} vis loss: {vis_loss:.5f} inner loss: {inner_loss:.5f} TD_loss: {TD_loss:.5f} TD16Loss: {TD16Loss:.5f}')

			if (in_step==0 or opt.no_use_adam) and not opt.no_inner_step:
				print('1st step')
				params_test = gradient_update_parameters(model, inner_loss, params=params_test, 
												step_size=inner_lr, first_order=True)
				write_params(model, params_test)
			else:
				# print('use_adam')
				if in_step==0:
					write_params(model, params_test)
				optimAdam.zero_grad()
				inner_loss.backward()
				optimAdam.step()
				params_test = OrderedDict(model.named_parameters())

			# save loss
			if in_step in Plot_iter and opt.test_img=='SynPlot':

				g_te_d = g_te.detach()

				# print(image_gt.shape, g_te_d.shape)
				outer_fea_loss = 0.25*(mse(image_gt[:,:,0:3], g_te_d[:,:,0:3])*opt.wN_outer 
								+ mse(image_gt[:,:,3:6], g_te_d[:,:,3:6]) *opt.wD_outer
								+ mse(image_gt[:,:,6:9], g_te_d[:,:,6:9])*opt.wR_outer
								+ mse(image_gt[:,:,9:12], g_te_d[:,:,9:12]))*opt.wS_outer

				# add VGG loss to features
				if opt.Wfea_vgg>0:
					D_vg_pred = VGGpreprocess(g_te_d[:,:,3:6].unsqueeze(0).permute(0,3,1,2))	
					D_vg_gt = VGGpreprocess(image_gt[:,:,3:6].unsqueeze(0).permute(0,3,1,2))
					D_vg = criterionVGG(D_vg_pred, D_vg_gt)*opt.Wfea_vgg

					S_vg_pred = VGGpreprocess(g_te_d[:,:,9:12].unsqueeze(0).permute(0,3,1,2))	
					S_vg_gt = VGGpreprocess(image_gt[:,:,9:12].unsqueeze(0).permute(0,3,1,2))
					S_vg = criterionVGG(S_vg_pred, S_vg_gt)*opt.Wfea_vgg

					outer_fea_loss += (D_vg + S_vg)*0.5

				# compute loss of 20 renderings
				outer_ren_loss=0
				for i in range(20):
					gt_ren_sampled = SingleRender_NumberPointLight_FixedCamera(2*image_gt[:,:,0:3]-1, image_gt[:,:,3:6],image_gt[:,:,6:7], image_gt[:,:,9:12], rand_light_pos_loss[i:i+1,...], Position_map, device, CamLi_co=True) #[B,N, W,H,C]
					gt_ren_sampled = (gt_ren_sampled.clamp(0,1)+opt.EPSILON)**(1/2.2) 
					pred_ren_sampled = SingleRender_NumberPointLight_FixedCamera(2*g_te_d[:,:,0:3]-1, g_te_d[:,:,3:6], g_te_d[:,:,6:7], g_te_d[:,:,9:12], rand_light_pos_loss[i:i+1,...], Position_map, device, CamLi_co=True) #[B,N, W,H,C]
					pred_ren_sampled = (pred_ren_sampled.clamp(0,1)+opt.EPSILON)**(1/2.2)					
					
					if opt.loss_after1=='TD':
						out_ren = pred_ren_sampled[0:1,...].permute(0,3,1,2)
						gt_ren = gt_ren_sampled[0:1,...].permute(0,3,1,2)
						out_16 = nn.functional.interpolate(out_ren, size = opt.dres, mode='bilinear', align_corners=True)
						gt_16 = nn.functional.interpolate(gt_ren, size = opt.dres, mode='bilinear', align_corners=True)
						outer_ren_loss += mse(out_16, gt_16)*opt.Wdren_outer
						outer_ren_loss += criterionTD(out_ren, gt_ren)*opt.WTDren_outer
					else:
						outer_ren_loss += mse(pred_ren_sampled, gt_ren_sampled)
						out_ren_te_vg = VGGpreprocess(pred_ren_sampled[0:1,...].permute(0,3,1,2))	
						gt_ren_te_vg = VGGpreprocess(gt_ren_sampled[0:1,...].permute(0,3,1,2))
						outer_ren_vgg_loss = criterionVGG(out_ren_te_vg, gt_ren_te_vg)*opt.Wvgg_outer
						outer_ren_loss += outer_ren_vgg_loss

				total_outer = outer_fea_loss + outer_ren_loss/20.0

				excel_total = total_outer
				excel_fea = outer_fea_loss
				excel_ren = outer_ren_loss/20.0

				val_ren_loss[str(in_step)] = outer_ren_loss/20.0 if str(in_step) not in val_ren_loss else val_ren_loss[str(in_step)] + outer_ren_loss/20.0
				val_total_loss[str(in_step)] = total_outer if str(in_step) not in val_total_loss else val_total_loss[str(in_step)] + total_outer
				val_feas_loss[str(in_step)] = outer_fea_loss if str(in_step) not in val_feas_loss else val_feas_loss[str(in_step)] + outer_fea_loss


				if opt.excel:
					if name[0] in Excel_dict:
						Excel_dict[name[0]].append(excel_total.cpu().detach().numpy())
					else:
						Excel_dict[name[0]] = [excel_total.cpu().detach().numpy()]

					if name[0] in Fea_Excel_dict:
						Fea_Excel_dict[name[0]].append(excel_fea.cpu().detach().numpy())
					else:
						Fea_Excel_dict[name[0]] = [excel_fea.cpu().detach().numpy()]

					if name[0] in Ren_Excel_dict:
						Ren_Excel_dict[name[0]].append(excel_ren.cpu().detach().numpy())
					else:
						Ren_Excel_dict[name[0]] = [excel_ren.cpu().detach().numpy()]


		################# save imgs ################33
		final_vis_tr = OrderedDict()
		final_vis_te = OrderedDict()
		if 'all' in opt.fea:
			for key_fea, fea in val_feas.items():
				final_vis_te[key_fea] = torch.cat((fea[:,:,0:3],(fea[:,:,3:6]+EPSILON)**(1/2.2),fea[:,:,6:9],(fea[:,:,9:12]+EPSILON)**(1/2.2)),dim=1)
			if image_gt.shape[-1]>3:
				final_vis_te['gt'] = torch.cat((image_gt[:,:,0:3],image_gt[:,:,3:6]**(1/2.2),image_gt[:,:,6:9],image_gt[:,:,9:12]**(1/2.2)),dim=1)
			else:
				final_vis_te['gt'] = torch.cat((image_gt,image_gt,image_gt,image_gt),dim=1)

		svbrdf_vis = final_vis_te[f'{val_steps}'].detach()

		if opt.netloss=='Des19Net':
			final_vis_te['des19'] = torch.cat((des19_svbrdf_te[:,:,0:3],des19_svbrdf_te[:,:,3:6]**(1/2.2),des19_svbrdf_te[:,:,6:9],des19_svbrdf_te[:,:,9:12]**(1/2.2),des19_svbrdf_te[:,:,3:6]**(1/2.2)),dim=1)						

		for key_ren, val_ren in val_rens.items():
			for temp in range(val_ren.shape[0]):
				final_vis_te[key_ren] = torch.cat((final_vis_te[key_ren], val_ren[temp,...]),dim=1)

		for temp in range(gt_ren_te.shape[0]):
			final_vis_te['gt'] = torch.cat((final_vis_te['gt'], gt_ren_te[temp,...]),dim=1)


		# save progressive images
		fig = plt.figure(figsize=(40,20))
		for i, key in enumerate(final_vis_te):
			print(i, key)
			plt.subplot(len(final_vis_te), 1, int(i)+1)
			plt.imshow((torch.clamp(final_vis_te[key], 0, 1).cpu().detach().numpy()*255.0).astype(np.uint8))
		fig.savefig(os.path.join(save_pro_path,'progressive_%s.png'%(name[0])))
		plt.close()

			
		save_image(svbrdf_vis, os.path.join(save_valfea_path,'%s.png'%(name[0])))

		print(image_te.shape)
		save_image(image_te[0,...], os.path.join(save_in_path,'%s.png'%(name[0])))


		# save illustration images
		if opt.demo:
			print(final_vis_te['0'].shape)
			save_image(final_vis_te['0'][:,0:256,:], os.path.join(save_valfea_path,'%s_init_N.png'%(name[0])))
			save_image(final_vis_te['0'][:,256:512,:], os.path.join(save_valfea_path,'%s_init_D.png'%(name[0])))
			save_image(final_vis_te['0'][:,512:768,:], os.path.join(save_valfea_path,'%s_init_R.png'%(name[0])))
			save_image(final_vis_te['0'][:,768:1024,:], os.path.join(save_valfea_path,'%s_init_S.png'%(name[0])))

			save_image(final_vis_te['1'][:,0:256,:], os.path.join(save_valfea_path,'%s_1_N.png'%(name[0])))
			save_image(final_vis_te['1'][:,256:512,:], os.path.join(save_valfea_path,'%s_1_D.png'%(name[0])))
			save_image(final_vis_te['1'][:,512:768,:], os.path.join(save_valfea_path,'%s_1_R.png'%(name[0])))
			save_image(final_vis_te['1'][:,768:1024,:], os.path.join(save_valfea_path,'%s_1_S.png'%(name[0])))


			save_image(final_vis_te[f'{val_steps}'][:,0:256,:], os.path.join(save_valfea_path,'%s_final_N.png'%(name[0])))
			save_image(final_vis_te[f'{val_steps}'][:,256:512,:], os.path.join(save_valfea_path,'%s_final_D.png'%(name[0])))
			save_image(final_vis_te[f'{val_steps}'][:,512:768,:], os.path.join(save_valfea_path,'%s_final_R.png'%(name[0])))
			save_image(final_vis_te[f'{val_steps}'][:,768:1024,:], os.path.join(save_valfea_path,'%s_final_S.png'%(name[0])))


			save_image(final_vis_te['gt'][:,0:256,:], os.path.join(save_valfea_path,'%s_gt_N.png'%(name[0])))
			save_image(final_vis_te['gt'][:,256:512,:], os.path.join(save_valfea_path,'%s_gt_D.png'%(name[0])))
			save_image(final_vis_te['gt'][:,512:768,:], os.path.join(save_valfea_path,'%s_gt_R.png'%(name[0])))
			save_image(final_vis_te['gt'][:,768:1024,:], os.path.join(save_valfea_path,'%s_gt_S.png'%(name[0])))

			save_image(final_vis_te['des19'][:,0:256,:], os.path.join(save_valfea_path,'%s_pseudo_N.png'%(name[0])))
			save_image(final_vis_te['des19'][:,256:512,:], os.path.join(save_valfea_path,'%s_pseudo_D.png'%(name[0])))
			save_image(final_vis_te['des19'][:,512:768,:], os.path.join(save_valfea_path,'%s_pseudo_R.png'%(name[0])))
			save_image(final_vis_te['des19'][:,768:1024,:], os.path.join(save_valfea_path,'%s_pseudo_S.png'%(name[0])))

		g_te_d = g_te.detach()
		N = g_te_d[...,0:3]	
		D = g_te_d[...,3:6]	
		R = g_te_d[...,6:7]	
		S = g_te_d[...,9:12]	


	if opt.excel:
		df = DataFrame(Excel_dict)
		df.to_excel(f'{load_model}/{opt.resume_name}/test/{opt.name}/test.xlsx', sheet_name='sheet1', index=True)

		df = DataFrame(Fea_Excel_dict)
		df.to_excel(f'{load_model}/{opt.resume_name}/test/{opt.name}/fea_test.xlsx', sheet_name='sheet1', index=True)
		df = DataFrame(Ren_Excel_dict)
		df.to_excel(f'{load_model}/{opt.resume_name}/test/{opt.name}/ren_test.xlsx', sheet_name='sheet1', index=True)


	# plot loss
	if opt.test_img=='SynPlot':
		fealoss_log = []
		for p in val_feas_loss:
			if p=='0':
				continue
			val_feas_loss[p] = val_feas_loss[p].detach().cpu()/len(val_data)
			fealoss_log.append(val_feas_loss[p])

		totalloss_log = []
		for p in val_total_loss:
			if p=='0':
				continue			
			val_total_loss[p] = val_total_loss[p].detach().cpu()/len(val_data)
			totalloss_log.append(val_total_loss[p])

		renloss_log = []
		for p in val_ren_loss:
			if p=='0':
				continue			
			val_ren_loss[p] = val_ren_loss[p].detach().cpu()/len(val_data)
			renloss_log.append(val_ren_loss[p])

		save_loss(totalloss_log, save_valfea_path, Plot_iter, save_name='total')
		save_loss(fealoss_log, save_valfea_path, Plot_iter, save_name='fea')
		save_loss(renloss_log, save_valfea_path, Plot_iter, save_name='ren')



def output_ff(opt, model, coords, val_data, device, FeaExtractor=None):

	val_steps = opt.val_step #@param {type:"integer"}
	first_order = opt.first_order #@param {type:"boolean"}

	L1 = torch.nn.L1Loss()
	mse = torch.nn.MSELoss().to(device)
	params = OrderedDict(model.meta_named_parameters())
	for key in params:
		print(key)
	criterionVGG = VGGLoss(opt)

	weights = [0.1, 0.1, 0.1, 0.1]
	keys = [1, 2, 3,4]
	if opt.netloss=='Des19Net':
		criterionNet = MyLoss(opt, keys, weights, device)

	log_step = []
	val_log_ren = OrderedDict()
	val_log_fea = OrderedDict()
	step = 1
	flag=False

	# LightPos = torch.tensor([[0,0,2.14]], device=device)
	LightPos = torch.tensor([[0,0,4]], device=device)
	Position_map=PositionMap(opt.res,opt.res,3).to(device)

	# resize = transforms.Resize((opt.res,opt.res))
	# resize = transforms.Compose([transforms.ToPILImage(), transforms.Resize((opt.res,opt.res)), transforms.ToTensor()])
	# FeaExtractNet.set_gradient(False)
	if opt.netloss=='Des19Net':
		criterionNet.set_gradient(False)

	# Plot_iter=[1,3,5,7,10,12,15,17,20,25,30,35,45,50,60,70,80,90,100,120,140,150]
	Plot_iter=[t for t in range(0,val_steps+1)]
	print(Plot_iter)
	# Plot_iter=[]

	val_rens = OrderedDict()
	val_feas = OrderedDict()

	val_feas_loss = OrderedDict()
	val_total_loss = OrderedDict()
	val_ren_loss = OrderedDict()


	model.eval()
	g_test = None
	out_val = None
	valfea_loss = 0
	valren_loss = 0

	load_model = 'pretrain' if opt.load_pretrain else opt.model

	# create directory
	save_valfea_path = f'{load_model}/{opt.resume_name}/test/{opt.name}/fea'
	save_in_path = f'{load_model}/{opt.resume_name}/test/{opt.name}/in'
	save_valren_path = f'{load_model}/{opt.resume_name}/test/{opt.name}/ren'
	save_pro_path = f'{load_model}/{opt.resume_name}/test/{opt.name}/pro'

	if not os.path.exists(save_in_path):         
		os.makedirs(save_in_path)

	if not os.path.exists(save_valfea_path):         
		os.makedirs(save_valfea_path)
	if not os.path.exists(save_valren_path):         
		os.makedirs(save_valren_path)
	if not os.path.exists(save_pro_path):         
		os.makedirs(save_pro_path)

	Num = 40

	def delogTensor(tensor):
		temp = torch.exp(tensor*(np.log(1.01)-np.log(0.01)) + np.log(0.01))-0.01

		return temp**(1/2.2)
		# return  (tf.log(tf.add(tensor,0.01)) - tf.log(0.01)) / (tf.log(1.01)-tf.log(0.01))




	if opt.loss_after1=='TD':
		criterionTD = TDLoss_2(device, 0)

	for img_id, val_example in enumerate(val_data):

		# if opt.demo:
		# 	LightPos = NLight_VaryH(1, 0.03, device)

		# if img_id > 20:
		# if img_id > 40 or img_id<=20:
		# if img_id > 60 or img_id<=40:
		# if img_id <= 60:
		# 	continue

		val_example, name=val_example
	
		# if name[0] != '0000027;leather_tilesXbrick_uneven_stones;defaultX2':
		# 	continue

		val_img = val_example.to(device)
		image_gt = val_img[0].permute(1,2,0) # [256, 256, C]

		# image_gt = delogTensor(image_gt)

		if opt.resume_name!='':	
			if opt.load_pretrain:
				load_network(model, opt.load_iter,'pretrain/%s/models'%(opt.resume_name), name='model')
			else:
				load_network(model,opt.load_iter,'%s/%s/models'%(opt.model, opt.resume_name), name='model')
			if opt.netloss=='Des19Net':
				load_network(criterionNet.lossnet, opt.load_iter,'%s/%s/models'%(opt.model, opt.resume_name), name='des19net')
			if FeaExtractor is not None:
				load_network(FeaExtractor, opt.load_iter,'%s/%s/models'%(opt.model, opt.resume_name), name='FeaExtractor')

		params_test = OrderedDict(model.meta_named_parameters())
		inner_lr = opt.inner_lr #@param
		flag_lr = True
		inner_loss = 0.0
		TD_loss = 0.0
		TD16Loss = 0.0

		if not opt.no_use_adam:
			optimAdam = torch.optim.Adam(model.parameters(), lr=opt.after_innerlr, betas=[0.5, 0.5])

		if image_gt.shape[-1]>3:
			image_te = SingleRender_NumberPointLight_FixedCamera(2*image_gt[:,:,0:3]-1, image_gt[:,:,3:6],image_gt[:,:,6:7], image_gt[:,:,9:12], LightPos, Position_map, device, True, lightinten=opt.light_inten) #[B,N, W,H,C]
			image_te = (image_te.clamp(0,1)+EPSILON)**(1/2.2)
		else:
			image_te = image_gt.unsqueeze(0)


		for in_step in range(val_steps+1):

			if opt.model=='UNet':
				myinput = 2*image_te.permute(0,3,1,2)-1 # [-1,1] # [B,3,H,W]
			elif opt.model=='InConSiren':
				gt_ren_te = image_te
				myinput = 2*image_te.permute(0,3,1,2)-1 # [B,3,H,W] [-1,1]
				myinput = [coords[0],myinput] # coords:[H,W,2] myinput: [B,3,H,W]
			elif opt.model=='OutConSiren':
				print(image_te.shape)
				myfea_in = FeaExtractor(2*image_te.permute(0,3,1,2)-1).squeeze(0).permute(1,2,0) #[B,3,H,W] --> [H,W,32]
				myinput = torch.cat([coords[0], myfea_in],dim=-1)

			g_te = model(myinput, params=params_test)

			# print('g_te ',g_te.shape)
			if opt.fea=='all_N1' or opt.fea=='D+R' or opt.fea=='all_N2' or opt.fea=='all_N3':
				g_te = paramize_out(opt,g_te)
			elif opt.fea=='N' or opt.fea=='N2' or opt.fea=='N3':
				g_te = ProcessNormal(opt, g_te)

			if in_step==0:
				val_feas[str(in_step)] = g_te
			elif in_step==1:
				val_feas[str(in_step)] = g_te
			elif in_step==2:
				val_feas[str(in_step)] = g_te
			elif in_step==5:
				val_feas[str(in_step)] = g_te					
			elif in_step==opt.val_step:
				val_feas[str(in_step)] = g_te

			out_ren_te = SingleRender_NumberPointLight_FixedCamera(2*g_te[:,:,0:3]-1, g_te[:,:,3:6], g_te[:,:,6:7], g_te[:,:,9:12], LightPos, Position_map, device, CamLi_co=True, lightinten=opt.light_inten) #[B,N, W,H,C]
			out_ren_te = (out_ren_te.clamp(0,1)+EPSILON)**(1/2.2)

			ren_loss = mse(gt_ren_te, out_ren_te)*opt.Winner_ren

			if in_step==0:
				if opt.netloss=='Des19Net':
					# inner_loss, des19_svbrdf = criterionNet(2*out_ren_tr.permute(0,3,1,2)-1, 2*gt_ren_tr.permute(0,3,1,2)-1)
					# inner_loss, des19_svbrdf_te = criterionNet(g_te, 2*gt_ren_te.permute(0,3,1,2)-1)
					inner_loss, des19_svbrdf_te = criterionNet(g_te, 2*gt_ren_te.permute(0,3,1,2)-1)
					inner_loss += ren_loss
				elif opt.netloss=='GT':
					# print(g_tr.shape)
					# print(image_tr.shape)
					inner_loss = mse(g_te, image_gt)
					inner_loss += ren_loss			
				else:
					inner_loss = ren_loss

				if opt.Wvgg_inner>0:
					out_ren_te_vg = VGGpreprocess(out_ren_te.permute(0,3,1,2))	
					gt_ren_te_vg = VGGpreprocess(gt_ren_te.permute(0,3,1,2))
					inner_ren_vgg_loss = criterionVGG(out_ren_te_vg, gt_ren_te_vg)*opt.Wvgg_inner
					inner_loss += inner_ren_vgg_loss

				if opt.Wtd_inner>0:
					out_ren_te_td = out_ren_te.permute(0,3,1,2)
					gt_ren_te_td = gt_ren_te.permute(0,3,1,2)
					inner_ren_td = criterionTD(out_ren_te_td, gt_ren_te_td)*opt.Wtd_inner
					inner_loss += inner_ren_td

			if opt.loss_after1 =='mse':
				temp_loss = ren_loss
			elif opt.loss_after1=='vgg':
				out_ren_tr_vg = VGGpreprocess(out_ren_te.permute(0,3,1,2))	
				gt_ren_tr_vg = VGGpreprocess(gt_ren_te.permute(0,3,1,2))
				temp_loss = criterionVGG(out_ren_tr_vg, gt_ren_tr_vg) * 0.005 + ren_loss

			elif opt.loss_after1=='TD':
				out_ren = out_ren_te[0:1,...].permute(0,3,1,2)
				gt_ren = gt_ren_te[0:1,...].permute(0,3,1,2)
				out_16 = nn.functional.interpolate(out_ren, size = opt.dres, mode='bilinear', align_corners=True)
				gt_16 = nn.functional.interpolate(gt_ren, size = opt.dres, mode='bilinear', align_corners=True)
				dren_loss = mse(out_16, gt_16)*opt.Wdren_outer
				TDren_loss = criterionTD(out_ren, gt_ren)*opt.WTDren_outer
				
				temp_loss = dren_loss + TDren_loss


			if in_step>0 or opt.no_inner_step:
				inner_loss = temp_loss
				vis_loss = temp_loss
			else:
				vis_loss = temp_loss


			# after 1 step
			if flag_lr and in_step>0:
				inner_lr = opt.after_innerlr 

			if 'MG' in opt.loss_after1 and in_step>0:
				# print(in_step)
				flag_lr = False
				if in_step%opt.decay_interval==0:
					inner_lr = inner_lr*0.5

				if not opt.no_use_adam:
					for g in optimAdam.param_groups:
						g['lr'] = inner_lr

			if in_step==0:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==1:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==2:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==5:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==opt.val_step:
				val_rens[str(in_step)] = out_ren_te

			if in_step%10==0:
				print(f'step: {in_step:d}, inner lr: {inner_lr:.8f} vis loss: {vis_loss:.5f} inner loss: {inner_loss:.5f} TD_loss: {TD_loss:.5f} TD16Loss: {TD16Loss:.5f}')

			if (in_step==0 or opt.no_use_adam) and not opt.no_inner_step:
				print('1st step')
				params_test = gradient_update_parameters(model, inner_loss, params=params_test, 
												step_size=inner_lr, first_order=True)
				write_params(model, params_test)
			else:
				# print('use_adam')
				if in_step==0:
					write_params(model, params_test)
				optimAdam.zero_grad()
				inner_loss.backward()
				optimAdam.step()
				params_test = OrderedDict(model.named_parameters())

			# save loss
			if in_step in Plot_iter and opt.test_img=='SynPlot':

				g_te_d = g_te.detach()

				# print(image_gt.shape, g_te_d.shape)
				outer_fea_loss = 0.25*(mse(image_gt[:,:,0:3], g_te_d[:,:,0:3])*opt.wN_outer 
								+ mse(image_gt[:,:,3:6], g_te_d[:,:,3:6]) *opt.wD_outer
								+ mse(image_gt[:,:,6:9], g_te_d[:,:,6:9])*opt.wR_outer
								+ mse(image_gt[:,:,9:12], g_te_d[:,:,9:12]))*opt.wS_outer

				# add VGG loss to features
				if opt.Wfea_vgg>0:
					D_vg_pred = VGGpreprocess(g_te_d[:,:,3:6].unsqueeze(0).permute(0,3,1,2))	
					D_vg_gt = VGGpreprocess(image_gt[:,:,3:6].unsqueeze(0).permute(0,3,1,2))
					D_vg = criterionVGG(D_vg_pred, D_vg_gt)*opt.Wfea_vgg

					S_vg_pred = VGGpreprocess(g_te_d[:,:,9:12].unsqueeze(0).permute(0,3,1,2))	
					S_vg_gt = VGGpreprocess(image_gt[:,:,9:12].unsqueeze(0).permute(0,3,1,2))
					S_vg = criterionVGG(S_vg_pred, S_vg_gt)*opt.Wfea_vgg

					outer_fea_loss += (D_vg + S_vg)*0.5

				# compute loss of 20 renderings
				outer_ren_loss=0
				for i in range(20):
					gt_ren_sampled = SingleRender_NumberPointLight_FixedCamera(2*image_gt[:,:,0:3]-1, image_gt[:,:,3:6],image_gt[:,:,6:7], image_gt[:,:,9:12], rand_light_pos_loss[i:i+1,...], Position_map, device, CamLi_co=True) #[B,N, W,H,C]
					gt_ren_sampled = (gt_ren_sampled.clamp(0,1)+opt.EPSILON)**(1/2.2) 
					pred_ren_sampled = SingleRender_NumberPointLight_FixedCamera(2*g_te_d[:,:,0:3]-1, g_te_d[:,:,3:6], g_te_d[:,:,6:7], g_te_d[:,:,9:12], rand_light_pos_loss[i:i+1,...], Position_map, device, CamLi_co=True) #[B,N, W,H,C]
					pred_ren_sampled = (pred_ren_sampled.clamp(0,1)+opt.EPSILON)**(1/2.2)					
					
					if opt.loss_after1=='TD':
						out_ren = pred_ren_sampled[0:1,...].permute(0,3,1,2)
						gt_ren = gt_ren_sampled[0:1,...].permute(0,3,1,2)
						out_16 = nn.functional.interpolate(out_ren, size = opt.dres, mode='bilinear', align_corners=True)
						gt_16 = nn.functional.interpolate(gt_ren, size = opt.dres, mode='bilinear', align_corners=True)
						outer_ren_loss += mse(out_16, gt_16)*opt.Wdren_outer
						outer_ren_loss += criterionTD(out_ren, gt_ren)*opt.WTDren_outer
					else:
						outer_ren_loss += mse(pred_ren_sampled, gt_ren_sampled)
						out_ren_te_vg = VGGpreprocess(pred_ren_sampled[0:1,...].permute(0,3,1,2))	
						gt_ren_te_vg = VGGpreprocess(gt_ren_sampled[0:1,...].permute(0,3,1,2))
						outer_ren_vgg_loss = criterionVGG(out_ren_te_vg, gt_ren_te_vg)*opt.Wvgg_outer
						outer_ren_loss += outer_ren_vgg_loss

				total_outer = outer_fea_loss + outer_ren_loss/20.0

				excel_total = total_outer
				excel_fea = outer_fea_loss
				excel_ren = outer_ren_loss/20.0

				val_ren_loss[str(in_step)] = outer_ren_loss/20.0 if str(in_step) not in val_ren_loss else val_ren_loss[str(in_step)] + outer_ren_loss/20.0
				val_total_loss[str(in_step)] = total_outer if str(in_step) not in val_total_loss else val_total_loss[str(in_step)] + total_outer
				val_feas_loss[str(in_step)] = outer_fea_loss if str(in_step) not in val_feas_loss else val_feas_loss[str(in_step)] + outer_fea_loss


		################# save imgs ################33
		final_vis_tr = OrderedDict()
		final_vis_te = OrderedDict()
		if 'all' in opt.fea:
			for key_fea, fea in val_feas.items():
				final_vis_te[key_fea] = torch.cat((fea[:,:,0:3],(fea[:,:,3:6]+EPSILON)**(1/2.2),fea[:,:,6:9],(fea[:,:,9:12]+EPSILON)**(1/2.2)),dim=1)
			if image_gt.shape[-1]>3:
				final_vis_te['gt'] = torch.cat((image_gt[:,:,0:3],image_gt[:,:,3:6]**(1/2.2),image_gt[:,:,6:9],image_gt[:,:,9:12]**(1/2.2)),dim=1)
			else:
				final_vis_te['gt'] = torch.cat((image_gt,image_gt,image_gt,image_gt),dim=1)

		svbrdf_vis = final_vis_te[f'{val_steps}'].detach()

		if opt.netloss=="Des19Net":
			top = torch.cat((des19_svbrdf_te[:,:,3:6]**(1/2.2),des19_svbrdf_te[:,:,0:3]), dim=1)
			bot = torch.cat((des19_svbrdf_te[:,:,6:9],des19_svbrdf_te[:,:,9:12]**(1/2.2)), dim=1)
			final_vis_te['des19'] = torch.cat((top, bot), dim=0)
			save_image(final_vis_te['des19'], os.path.join(save_valfea_path,'%s_pseudo_gt.png'%(name[0])))


		h = 256

		# save gt
		N = final_vis_te['gt'][:,0:h,:]
		D = final_vis_te['gt'][:,h:2*h,:]
		R = final_vis_te['gt'][:,2*h:3*h,:]
		S = final_vis_te['gt'][:,3*h:4*h,:]
		top = torch.cat((D,N), dim=1)
		bot = torch.cat((R,S), dim=1)
		maps = torch.cat((top, bot), dim=0)
		save_image(maps, os.path.join(save_valfea_path,'%s_gt.png'%(name[0])))

		save_image(out_ren_te[0,...], os.path.join(save_valren_path,'%s_ren.png'%(name[0])))

		# save fea
		N = svbrdf_vis[:,0:h,:]
		D = svbrdf_vis[:,h:2*h,:]
		R = svbrdf_vis[:,2*h:3*h,:]
		S = svbrdf_vis[:,3*h:4*h,:]
		top = torch.cat((D,N), dim=1)
		bot = torch.cat((R,S), dim=1)
		maps = torch.cat((top, bot), dim=0)
		save_image(maps, os.path.join(save_valfea_path,'%s.png'%(name[0])))

		# save init
		N = final_vis_te['0'][:,0:h,:]
		D = final_vis_te['0'][:,h:2*h,:]
		R = final_vis_te['0'][:,2*h:3*h,:]
		S = final_vis_te['0'][:,3*h:4*h,:]
		top = torch.cat((D,N), dim=1)
		bot = torch.cat((R,S), dim=1)
		maps = torch.cat((top, bot), dim=0)
		save_image(maps, os.path.join(save_pro_path,'%s_init.png'%(name[0])))

		# save input

		save_image(image_te[0,...], os.path.join(save_in_path,'%s.png'%(name[0])))



		# save_image(final_vis_te['gt'], os.path.join(save_valfea_path,'%s_gt.png'%(name[0])))
		# save_image(final_vis_te['0'], os.path.join(save_pro_path,'%s_init.png'%(name[0])))
		# save_image(svbrdf_vis, os.path.join(save_valfea_path,'%s.png'%(name[0])))
		# save_image(image_te[0,...], os.path.join(save_in_path,'%s.png'%(name[0])))


def test_real(opt, model, coords, device, FeaExtractor=None):


	if opt.test_img=='OurReal2':
		realdata_path = "./dataset/OurReal/OurRealData_scaled"
	elif opt.test_img=='MGReal2':
		realdata_path = "./dataset/MGReal/MGRealData_scaled"

	load_model='pretrain' if opt.load_pretrain else opt.model

	scenes = os.listdir(realdata_path)

	txt_dir = f'test/{opt.name}'
	if not os.path.exists(txt_dir):         
		os.makedirs(txt_dir)

	RenRMSE = open(os.path.join(txt_dir,'RenRMSE.txt'),'w')
	RenLPIPS = open(os.path.join(txt_dir,'RenLPIPS.txt'),'w')

	scene_N = 0
	TotalRMSE_ren = 0
	TotalLPIPS_ren = 0

	if opt.loss_after1=='TD':
		criterionTD = TDLoss_2(device, 0)

	for scene in scenes:

		scene_N += 1

		print('....................now training scene %s .................'% scene )

		# if scene != 'nima_wood_1':
		# 	continue

		img_dir = f'test/{opt.name}/{scene}'
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

		# load model
		load_network(model, "final", 'ckpt/', name='model')
		if opt.netloss=='Des19Net':
			load_network(criterionNet.lossnet, "final", 'ckpt/', name='des19net')


		if opt.no_load:
			model = SirenNet(2, 256, out_nc, opt.num_layers, w0_initial=200., w0=200., 
							 final_activation=lambda x: x + .5).to(device)

		val_log_ren = OrderedDict()
		val_log_fea = OrderedDict()
		flag=False

		if opt.netloss=='Des19Net':
			criterionNet.set_gradient(False)

		val_rens = OrderedDict()
		val_feas = OrderedDict()
		val_rens_loss = OrderedDict()
		val_feas_loss = OrderedDict()

		model.eval()
		g_test = None
		out_val = None
		valfea_loss = 0
		valren_loss = 0

		#################################### load input images ##############################################
		## for training Data
		for i in range(9):
			image_i = os.path.join(realdata_path,scene,'0{}.{}'.format(i, opt.img_format)) 
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

		#################################### load Light ##############################################
		## load all 7 lights && get first No_Input lights
		LightPos=torch.from_numpy(load_light_txt(os.path.join(realdata_path,f'{scene}/camera_pos.txt'))).to(device).float()

		### reorganize train & test light pos
		LightPos_train = LightPos[:opt.N_input,:]
		LightPos_test = LightPos[opt.N_input:,:]
		# LightPos_test = torch.cat((LightPos[opt.No_Input:,:],LightPos_test),dim=0)

		Position_map=PositionMap(opt.res,opt.res,3).to(device)

		LightPos_train = torch.tensor([[0,0,4.0]], device=device, dtype=torch.float32)

		params_test = OrderedDict(model.meta_named_parameters())
		inner_lr = opt.inner_lr #@param

		flag_lr = True
		inner_loss = 0.0
		TD_loss = 0.0
		TD16Loss = 0.0

		log_step = []
		loss_log = []

		if not opt.no_use_adam:
			optimAdam = torch.optim.Adam(model.parameters(), lr=opt.after_innerlr, betas=[opt.beta1, opt.beta2])

		LightInten = 16
		LightInten_test = 16
		if opt.adjust_light:
			LightInten = LightPos_train[-1][2]**2
			LightInten_test = LightPos_test[-1][2]**2

		# centered Dataset
		shift=torch.from_numpy(load_shift_txt(os.path.join(realdata_path,scene,'info.txt'))).to(device)
		shift_tensor = torch.tensor([shift[2], -shift[1],0], device='cuda',dtype=torch.float32).repeat(256,256,1)
		Position_map = Position_map*shift[0] + shift_tensor/512.

		for in_step in range(val_steps+1):

			if opt.model=='UNet':
				myinput = 2*image_te.permute(0,3,1,2)-1 # [-1,1] # [B,3,H,W]
			elif opt.model=='InConSiren':
				myinput = 2*image_te.permute(0,3,1,2)-1 # [B,3,H,W] [-1,1]
				myinput = [coords[0],myinput] # coords:[H,W,2] myinput: [B,3,H,W]

			g_te = model(myinput, params=params_test)

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
			elif in_step==5:
				val_feas[str(in_step)] = g_te					
			elif in_step==opt.val_step:
				val_feas[str(in_step)] = g_te

			gt_ren_te = image_te

			out_ren_te = SingleRender_NumberPointLight_FixedCamera(2*g_te[:,:,0:3]-1, g_te[:,:,3:6], g_te[:,:,6:7], g_te[:,:,9:12], LightPos_train, Position_map, device, True, lightinten=LightInten) #[B,N, W,H,C]
			out_ren_te = (out_ren_te.clamp(0,1)+EPSILON)**(1/2.2)

			ren_loss = mse(gt_ren_te, out_ren_te)*opt.Winner_ren

			if in_step==0:
				if opt.netloss=='Des19Net':
					# inner_loss, des19_svbrdf = criterionNet(2*out_ren_tr.permute(0,3,1,2)-1, 2*gt_ren_tr.permute(0,3,1,2)-1)
					# inner_loss, des19_svbrdf_te = criterionNet(g_te, 2*gt_ren_te.permute(0,3,1,2)-1)
					inner_loss, des19_svbrdf_te = criterionNet(g_te, 2*gt_ren_te.permute(0,3,1,2)-1)
					inner_loss += ren_loss
				else:
					inner_loss = ren_loss
			
				if opt.Wvgg_inner>0:
					out_ren_te_vg = VGGpreprocess(out_ren_te.permute(0,3,1,2))	
					gt_ren_te_vg = VGGpreprocess(gt_ren_te.permute(0,3,1,2))
					inner_ren_vgg_loss = criterionVGG(out_ren_te_vg, gt_ren_te_vg)*opt.Wvgg_inner
					inner_loss += inner_ren_vgg_loss

				if opt.Wtd_inner>0:
					out_ren_te_td = out_ren_te.permute(0,3,1,2)
					gt_ren_te_td = gt_ren_te.permute(0,3,1,2)
					inner_ren_td = criterionTD(out_ren_te_td, gt_ren_te_td)*opt.Wtd_inner
					inner_loss += inner_ren_td

			if opt.loss_after1 =='mse':
				temp_loss = ren_loss

			elif opt.loss_after1=='TD':
				out_ren = out_ren_te[0:1,...].permute(0,3,1,2)
				gt_ren = gt_ren_te[0:1,...].permute(0,3,1,2)
				out_16 = nn.functional.interpolate(out_ren, size = opt.dres, mode='bilinear', align_corners=True)
				gt_16 = nn.functional.interpolate(gt_ren, size = opt.dres, mode='bilinear', align_corners=True)
				dren_loss = mse(out_16, gt_16)
				TDren_loss = criterionTD(out_ren, gt_ren)
				
				temp_loss = dren_loss + TDren_loss

			if in_step>0:
				inner_loss = temp_loss
				vis_loss = temp_loss
			else:
				vis_loss = temp_loss

			# after 1 step
			if flag_lr and in_step>0:
				# print(in_step)
				inner_lr = opt.after_innerlr 


			if in_step==0:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==1:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==2:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==5:
				val_rens[str(in_step)] = out_ren_te
			elif in_step==opt.val_step:
				val_rens[str(in_step)] = out_ren_te

			if in_step%10==0:
				print(f'step: {in_step:d}, inner lr: {inner_lr:.8f} vis loss: {vis_loss:.5f} inner loss: {inner_loss:.5f} TD_loss: {TD_loss:.5f} TD16Loss: {TD16Loss:.5f}')

			log_step.append(in_step)
			loss_log.append(vis_loss.data.cpu().numpy())
			

			if (in_step==0 or opt.no_use_adam) and not opt.no_inner_step:
				params_test = gradient_update_parameters(model, inner_loss, params=params_test, 
												step_size=inner_lr, first_order=True)
				write_params(model, params_test)
			else:
				if in_step==0:
					write_params(model, params_test)
				optimAdam.zero_grad()
				inner_loss.backward()
				optimAdam.step()
				params_test = OrderedDict(model.named_parameters())

			g_te = g_te.detach()

		out_ren_test = SingleRender_NumberPointLight_FixedCamera(2*g_te[:,:,0:3]-1, g_te[:,:,3:6], g_te[:,:,6:7], g_te[:,:,9:12], LightPos_test, Position_map, device, True, lightinten=LightInten_test) #[B,N, W,H,C]
		out_ren_test = (out_ren_test.clamp(0,1)+EPSILON)**(1/2.2)

		# L2loss = mse(out_ren_test, val_test)
		################# save imgs and compute loss ################33
		final_vis_tr = OrderedDict()
		final_vis_te = OrderedDict()
		if 'all' in opt.fea:
			for key_fea, fea in val_feas.items():
				final_vis_te[key_fea] = torch.cat((fea[:,:,0:3],(fea[:,:,3:6]+EPSILON)**(1/2.2),fea[:,:,6:9],(fea[:,:,9:12]+EPSILON)**(1/2.2)),dim=1)
			final_vis_te['gt'] = torch.cat((image_te,image_te,image_te,image_te),dim=2)[0,...]
		svbrdf_vis = final_vis_te[f'{val_steps}'].detach()
		svbrdf_vis0 = final_vis_te['0']

		if opt.netloss=='Des19Net':
			final_vis_te['des19'] = torch.cat((des19_svbrdf_te[:,:,0:3],des19_svbrdf_te[:,:,3:6]**(1/2.2),des19_svbrdf_te[:,:,6:9],des19_svbrdf_te[:,:,9:12]**(1/2.2),des19_svbrdf_te[:,:,3:6]**(1/2.2)),dim=1)						

		for key_ren, val_ren in val_rens.items():
			for temp in range(val_ren.shape[0]):
				final_vis_te[key_ren] = torch.cat((final_vis_te[key_ren], val_ren[temp,...]),dim=1)

		for temp in range(gt_ren_te.shape[0]):
			# print(final_vis_te['gt'].shape)
			# print(gt_ren_te[temp,...].shape)
			final_vis_te['gt'] = torch.cat((final_vis_te['gt'], gt_ren_te[temp,...]),dim=1)

		################# save logs ################
		save_loss(loss_log, img_dir, log_step, save_name='')


		# save progressive images
		fig = plt.figure(figsize=(40,20))
		for i, key in enumerate(final_vis_te):
			print(i, key)
			plt.subplot(len(final_vis_te), 1, int(i)+1)
			plt.imshow((torch.clamp(final_vis_te[key], 0, 1).cpu().detach().numpy()*255.0).astype(np.uint8))
		fig.savefig(os.path.join(img_dir,'progressive_img.png'))
		plt.close()

		# save final images
		for j in range(out_ren_te.shape[0]):
			save_image(gt_ren_te[j,...].detach(), os.path.join(img_dir,f'render_t{j}.png'))
			save_image(out_ren_te[j,...].detach(), os.path.join(img_dir,f'render_o{j}.png'))
		save_image(svbrdf_vis, os.path.join(img_dir,'fea.png'))
		save_image(svbrdf_vis0, os.path.join(img_dir,'fea0.png'))
		for i in range(out_ren_test.shape[0]):
			# print(out_ren_test[i,...].shape)
			save_image(out_ren_test[i,...].detach(), os.path.join(img_dir,f'render_{i}.png'))

		# compute final loss
		# lpips
		EachLPIPS_ren = 0
		fake_ren_lpips = out_ren_test*2-1
		gt_ren_lpips = val_test*2-1
		for i in range(out_ren_test.shape[0]):
			LPIPS_ren = loss_fn_alex(gt_ren_lpips[i:i+1,...].permute(0,3,1,2), fake_ren_lpips[i:i+1,...].permute(0,3,1,2))
			LPIPS_ren = LPIPS_ren.squeeze().squeeze().squeeze().squeeze().detach().cpu().numpy()
			EachLPIPS_ren += LPIPS_ren

		EachLPIPS_ren = EachLPIPS_ren/out_ren_test.shape[0]
		EachRMSE_ren = torch.sqrt(mse(out_ren_test,val_test))

		RenRMSE.write('{} {:.4f}\n'.format(scene,EachRMSE_ren))
		RenLPIPS.write('{} {:.4f}\n'.format(scene,EachLPIPS_ren))

		TotalRMSE_ren += EachRMSE_ren
		TotalLPIPS_ren += EachLPIPS_ren

		print(f'for image {scene:s}, RMSE Loss: {EachRMSE_ren:.4f}, LPIPS Loss: : {EachLPIPS_ren:.4f}' )

	RenRMSE.write('Total loss: {:.4f}\n'.format(TotalRMSE_ren/scene_N))
	RenLPIPS.write('Total loss: {:.4f}\n'.format(TotalLPIPS_ren/scene_N))

	RenRMSE.close()
	RenLPIPS.close()



if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--res', type=int, default=256, help='resolution')        
	parser.add_argument('--batch_size', type=int, default=3, help='batch_size')        
	parser.add_argument('--n_threads', type=int, default=0, help='number of n_threads')        
	parser.add_argument('--val_step', type=int, default=1, help='# of inner step')        
	parser.add_argument('--decay_interval', type=int, default=500, help='# of iteration')        
	parser.add_argument('--load_iter', type=int, default=500000, help='# of iteration')        
	parser.add_argument('--num_layers', type=int, default=5, help='# of layers')        
	parser.add_argument('--Light_range', type=float, default=0.0, help='range of input light')        
	parser.add_argument('--EPSILON', type=float, default=1e-6, help='eps value for rendering')        
	parser.add_argument('--lambda_decay', type=float, default=1.0, help='weight_decay of feature weight or stage decay')        
	parser.add_argument('--first_order', action='store_true', help='use the first order gradient')        
	parser.add_argument('--no_use_adam', action='store_true', help='not use_adam after 1 inner step')   
	# parser.add_argument('--fixed_light', action='store_true', help='use_adam after 1 inner step')   
	parser.add_argument('--no_load', action='store_true', help='use_adam after 1 inner step')   
	parser.add_argument('--no_inner_step', action='store_true', help='use_adam after 1 inner step')   
	parser.add_argument('--no_spec', action='store_true', help='use_adam after 1 inner step')   
	parser.add_argument('--demo', action='store_true', help='use_adam after 1 inner step')   
	parser.add_argument('--HN_factor', type=float, default=10.0, help='height to normal scale factor')        
	parser.add_argument('--model', type=str, default='InConSiren', help='use Siren or Fourier Feature')
	parser.add_argument('--N_input', type=int, default=1, help='number of input images')        

	parser.add_argument('--beta1', type=float, default=0.5, help='beta1')        
	parser.add_argument('--beta2', type=float, default=0.5, help='beta2')        

	parser.add_argument('--inner_lr', type=float, default=0.001, help='the inner lr')        
	parser.add_argument('--after_innerlr', type=float, default=2e-7, help='the inner lr')        
	parser.add_argument('--const_weightfea', type=float, default=-1, help='const weight feature: -1: not use const | >=0: use const')	 	       
	parser.add_argument('--val_root', type=str, default='F:/LoganZhou/Research/dataset/DeepMaterialsData/ImageData/DebugTest', help='input image name')        
	parser.add_argument('--name', type=str, default='SVBRDF', help='name of folder')
	parser.add_argument('--img_format', type=str, default='png', help='img format')
	parser.add_argument('--losstype', type=str, default='L2', help='name of folder')
	parser.add_argument('--file', type=str, default='filesall', help='name of folder')
	parser.add_argument('--optim', type=str, default='adam', help='which optimzer to use? adam || sgd || rprop')
	parser.add_argument('--fea', type=str,default='all_N2', help='use all feature maps or just albedo')
	parser.add_argument('--netloss', type=str, default='Des19Net',help='train network loss function')
	parser.add_argument('--loss_after1', type=str, default='TD',help='train network loss function')
	parser.add_argument('--Des19Net_npy_path', type=str, default='../Dataset/Des19Net/Des19.npy', help='npy path for des19 ')
	parser.add_argument('--resume_name', type=str, default='', help='resume training from folder wth name')
	parser.add_argument('--test_img', type=str, default='Real1', help='resume training from folder wth name')
	parser.add_argument('--Wvgg_inner', type=float, default=-1, help='add vgg for rendering inner')
	parser.add_argument('--Wtd_inner', type=float, default=-1, help='add vgg for rendering inner')
	parser.add_argument('--Wcolor_loss', type=float, default=-1, help='add vgg for rendering inner')
	parser.add_argument('--load_pretrain', action='store_true', help='load the pretrained model')   
	parser.add_argument('--load_old', action='store_true', help='load the pretrained model')   
	parser.add_argument('--fix_extra', action='store_true', help='fix featuer extractor after 1 step')   
	parser.add_argument('--gamma', action='store_true', help='gamma output D and S') 
	parser.add_argument('--excel', action='store_true', help='output excel') 

	parser.add_argument('--cuda', action='store_true', help='gamma output D and S') 
	parser.add_argument('--cond_type', type=str, default='unet', help='unet or conv')
	parser.add_argument('--n_layer_unet', type=int, default=4, help='the # of layer in UNet')   
	parser.add_argument('--Winner_ren', type=float, default=1, help='weight of inner rendering loss')

	parser.add_argument('--extfea_c', type=int, default=32, help='# of output channels in feature extractor')        
	parser.add_argument('--branch', action='store_true', help='use branch for network')   
	parser.add_argument('--branch_len', type=int, default=2, help='use branch length for network')   
	parser.add_argument('--no_coords', action='store_true', help='no_coords input')   

	# light information
	parser.add_argument('--Wvgg_outer', type=float, default=5e-3, help='weight of vgg for outer loss')        
	parser.add_argument('--wN_outer', type=float, default=1.0, help='weight of N for outer loss')        
	parser.add_argument('--wD_outer', type=float, default=1.0, help='weight of D for outer loss')        
	parser.add_argument('--wR_outer', type=float, default=1.0, help='weight of R for outer loss')        
	parser.add_argument('--wS_outer', type=float, default=1.0, help='weight of S for outer loss')        
	parser.add_argument('--light_inten', type=float, default=16.0, help='the inner lr')        
	parser.add_argument('--light_height', type=float, default=4.0, help='the inner lr')        
	parser.add_argument('--optim_light', action='store_true', help='load the pretrained model')   
	parser.add_argument('--adjust_light', action='store_true', help='load the pretrained model')   
	parser.add_argument('--sc_des19', action='store_true', help='sc_des19')   
	parser.add_argument('--Wfea_vgg', type=float, default=-1, help='add vgg of feature maps in outer loop')
	parser.add_argument('--WTDren_outer', type=float, default=1, help='weight of TD for rendering outer')
	parser.add_argument('--Wdren_outer', type=float, default=1, help='weight of downsampled L2 for rendering outer')
	parser.add_argument('--dres', type=int, default=16, help='downsampled res for L1')        

	# for debug
	opt = parser.parse_args()

	# save_args(opt)


	torch.set_num_threads(8)

	num_val_exs = 5
	CROP_SIZE = opt.res


	myvaldata_root = opt.val_root


	##########################################################################################################
	##########################################################################################################
	if opt.cuda:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	x1 = torch.linspace(0, 1, opt.res+1)[:-1]
	coords = torch.stack(torch.meshgrid([x1,x1]), dim=-1)[None,...]
	coords = coords.to(device)

	if opt.fea=='all_N1':
		out_nc = 8
	elif opt.fea=='all_N2':
		out_nc = 9
	elif opt.fea=='all_N3':
		out_nc = 10

	if opt.no_spec:
		out_nc = out_nc - 3
	FeaExtractNet=None


	if opt.model=='Des19Net':
		model = Des19Net(opt, device)
		Des19_Test(opt, model, coords, Myval_data, device, FeaExtractNet)
	else:
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
			if opt.load_old:
				model = OldConSirenNet(opt, 2, 256, out_nc, opt.num_layers, w0_initial=200., w0=200., 
								 final_activation=lambda x: x + .5, cond_type=opt.cond_type).to(device)			
			else:
				model = ConSirenNet(opt, 2, 256, out_nc, opt.num_layers, w0_initial=200., w0=200., 
								 final_activation=lambda x: x + .5, cond_type=opt.cond_type, test=True, N_in=opt.N_input, n_layer_unet=opt.n_layer_unet).to(device)


		print(model)
		print('output nc:', out_nc)

		nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print('Number of learnable parameters: %d' %(nparams))

		if opt.test_img=='Syn' or opt.test_img=='Real':
			Myval_set = DataLoaderHelper_test(myvaldata_root, opt)
			Myval_data = DataLoader(dataset=Myval_set, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

			test(opt, model, coords, Myval_data, device, FeaExtractNet)
			# output_ff(opt, model, coords, Myval_data, device, FeaExtractNet)

		elif opt.test_img=='SynPlot':
			Myval_set = DataLoaderHelper_test(myvaldata_root, opt)
			Myval_data = DataLoader(dataset=Myval_set, batch_size=1, num_workers=0, shuffle=False, drop_last=False)

			test(opt, model, coords, Myval_data, device, FeaExtractNet)
		elif opt.test_img=='MGReal2' or opt.test_img=='OurReal2':
			test_real(opt, model, coords, device, FeaExtractNet)
			# output_video(opt, model, coords, device, FeaExtractNet)


