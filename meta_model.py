

from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d,
                               MetaBatchNorm2d, MetaLinear)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from models.networks import Net_Des19
from models.renderer import PositionMap_Des19

import torchvision.transforms as transforms


###################################################################################################
# --------------------------------------- define networks -----------------------------------------
###################################################################################################

def exists(val):
	return val is not None

def cast_tuple(val, repeat = 1):
	return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation
class Sine(MetaModule):
	def __init__(self, w0 = 1.):
		super().__init__()
		self.w0 = w0
	def forward(self, x):
		return torch.sin(self.w0 * x)

# siren layer
class Siren(MetaModule):
	def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
		super().__init__()
		self.dim_in = dim_in
		self.is_first = is_first

		weight = torch.zeros(dim_out, dim_in)
		bias = torch.zeros(dim_out) if use_bias else None
		self.init_(weight, bias, c = c, w0 = w0)

		self.weight = nn.Parameter(weight)
		self.bias = nn.Parameter(bias) if use_bias else None
		# self.linear = MetaLinear(dim_in, dim_out, bias=use_bias)
		self.activation = Sine(w0) if activation is None else activation

		self.flag = activation

	def init_(self, weight, bias, c, w0):
		dim = self.dim_in

		w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
		weight.uniform_(-w_std, w_std)

		if exists(bias):
			bias.uniform_(-w_std, w_std)

	def forward(self, x, params=None):

		out =  F.linear(x, params['weight'], params['bias'])
		# if self.flag is not None:
		# 	print('out:', out[0:10,0,...])
		out = self.activation(out)
		# if self.flag is not None:
		# 	print('final:', out[0:10,0,...])
		return out

# siren network
class SirenNet(MetaModule):
	def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
		super().__init__()
		self.num_layers = num_layers
		self.dim_hidden = dim_hidden

		self.layers = nn.ModuleList([])
		for ind in range(num_layers):
			is_first = ind == 0
			layer_w0 = w0_initial if is_first else w0
			layer_dim_in = dim_in if is_first else dim_hidden

			self.layers.append(Siren(
				dim_in = layer_dim_in,
				dim_out = dim_hidden,
				w0 = layer_w0,
				use_bias = use_bias,
				is_first = is_first
			))

		self.final_activation = nn.Identity() if not exists(final_activation) else final_activation
		self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

	def forward(self, x, mods = None, params = None):
		x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
		mods = cast_tuple(mods, self.num_layers)

		for i, (layer, mod) in enumerate(zip(self.layers, mods)):
			# print('subdict:', self.get_subdict(params, 'layers.{}'.format(i)))
			x = layer(x, params=self.get_subdict(params, 'layers.{}'.format(i)))

			if exists(mod):
				x *= rearrange(mod, 'd -> () d')

		x = self.last_layer(x, params=self.get_subdict(params, 'last_layer'))
		return x

class DeConv2d(nn.Module):
	def __init__(self,in_c, out_c):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='nearest')
		self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)

	def forward(self, x, params=None):
		x = self.conv1(self.up(x))
		return x

class FeatureExtractor(nn.Module):
	def __init__(self, dim_hidden, layer_n=3, cond_type='conv'):
		super().__init__()
		self.layer_n = layer_n
		self.dim_hidden = dim_hidden
		self.cond_type=cond_type

		self.Tanh = nn.Tanh()
		self.LeakyReLU = nn.LeakyReLU()

		if self.cond_type=='conv':
			fea_c = 32
			self.net = nn.Sequential(
				nn.Conv2d(3, fea_c, 3, 1, 1),
				nn.LeakyReLU(),
				nn.Conv2d(fea_c, fea_c, 3, 1, 1),
				nn.LeakyReLU(),
				nn.Conv2d(fea_c, fea_c, 3, 1, 1),
				nn.LeakyReLU(),
				nn.Conv2d(fea_c, fea_c, 3, 1, 1),
				nn.Tanh()
				)

		elif self.cond_type=='unet':
			self.convs = nn.ModuleList([])
			for i in range(self.layer_n):
				in_c = 3 if i==0 else out_c
				out_c = self.dim_hidden*2**i
				self.convs.append(nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1))

			self.deconvs = nn.ModuleList([])
			for i in range(self.layer_n):
				in_c = self.dim_hidden*2**(self.layer_n-i-1) if i==0 else self.dim_hidden*2**(self.layer_n-i)
				out_c = self.dim_hidden*2**(self.layer_n-i-2) if i!=self.layer_n-1 else self.dim_hidden
				self.deconvs.append(DeConv2d(in_c, out_c))


	def forward(self, x):
		x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

		if self.cond_type=='conv':
			return self.net(x)
		elif self.cond_type=='unet':
			convs=[]
			for i in range(self.layer_n):
				x = self.LeakyReLU(self.convs[i](x))
				if i!=self.layer_n-1:
					convs.append(x)

			for i in range(self.layer_n):
				if i==0:
					x = self.LeakyReLU(self.deconvs[i](x))
				elif i==self.layer_n-1:
					temp = torch.cat([x, convs[self.layer_n-1-i]], dim=1)
					x = self.Tanh(self.deconvs[i](temp))
				else:
					temp = torch.cat([x, convs[self.layer_n-1-i]], dim=1)
					x = self.LeakyReLU(self.deconvs[i](temp))
			return x


# con-siren network
class ConSirenNet(MetaModule):
	def __init__(self, opt, dim_in, dim_hidden, dim_out, num_layers, fea_c=32, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, cond_type='conv', N_in=1, n_layer_unet=3, test=False):
		super().__init__()
		self.opt = opt
		self.test = test
		self.N_in = N_in
		self.num_layers = num_layers
		self.dim_hidden = dim_hidden
		self.cond_type = cond_type
		self.n_layer_unet = n_layer_unet
		if self.cond_type=='conv':
			self.con_layer = self._conv_layers(fea_c)
		elif self.cond_type=='unet':
			self.con_layer = MetaUNet(32, layer_n=self.n_layer_unet, N_in=self.N_in, test=test)
		elif self.cond_type=='dir':
			fea_c = 3

		self.layers = nn.ModuleList([])
		for ind in range(num_layers):
			is_first = ind == 0
			layer_w0 = w0_initial if is_first else w0
			if is_first:
				if opt.no_coords:
					print('............................ no_coords ...........................')
					layer_dim_in = fea_c
				else:
					print('............................ coords ...........................')
					layer_dim_in = dim_in+fea_c
			else:
				layer_dim_in = dim_hidden


			self.layers.append(Siren(
				dim_in = layer_dim_in,
				dim_out = dim_hidden,
				w0 = layer_w0,
				use_bias = use_bias,
				is_first = is_first
			))

		self.final_activation = nn.Identity() if not exists(final_activation) else final_activation


		if self.opt.branch:

			# # Diffuse branch
			# self.D_branch = nn.ModuleList([])
			# for temp in range(self.opt.branch_len):
			# 	is_last = temp==1
			# 	dim_out_branch = 3 if is_last else dim_hidden
			# 	self.D_branch.append(Siren(dim_in = dim_hidden, dim_out = dim_out_branch, w0 = w0, use_bias = use_bias, activation = final_activation))

			self.D_branch = MetaSequential(
				Siren(dim_in = dim_hidden, dim_out = dim_hidden, w0 = w0, use_bias = use_bias),
				Siren(dim_in = dim_hidden, dim_out = 3, w0 = w0, use_bias = use_bias, activation = final_activation)
				)

			# # Height branch
			# self.H_branch = nn.ModuleList([])
			# for temp in range(self.opt.branch_len):
			# 	is_last = temp==1
			# 	dim_out_branch = 1 if is_last else dim_hidden
			# 	self.H_branch.append(Siren(dim_in = dim_hidden, dim_out = dim_out_branch, w0 = w0, use_bias = use_bias, activation = final_activation))

			self.H_branch = MetaSequential(
				Siren(dim_in = dim_hidden, dim_out = dim_hidden, w0 = w0, use_bias = use_bias),
				Siren(dim_in = dim_hidden, dim_out = 1, w0 = w0, use_bias = use_bias, activation = final_activation)
				)


			# # Rough branch
			# self.R_branch = nn.ModuleList([])
			# for temp in range(self.opt.branch_len):
			# 	is_last = temp==1
			# 	dim_out_branch = 1 if is_last else dim_hidden
			# 	self.R_branch.append(Siren(dim_in = dim_hidden, dim_out = dim_out_branch, w0 = w0, use_bias = use_bias, activation = final_activation))

			self.R_branch = MetaSequential(
				Siren(dim_in = dim_hidden, dim_out = dim_hidden, w0 = w0, use_bias = use_bias),
				Siren(dim_in = dim_hidden, dim_out = 1, w0 = w0, use_bias = use_bias, activation = final_activation)
				)

			# # Spec branch
			# self.S_branch = nn.ModuleList([])
			# for temp in range(self.opt.branch_len):
			# 	is_last = temp==1
			# 	dim_out_branch = 3 if is_last else dim_hidden
			# 	self.S_branch.append(Siren(dim_in = dim_hidden, dim_out = dim_out_branch, w0 = w0, use_bias = use_bias, activation = final_activation))

			self.S_branch = MetaSequential(
				Siren(dim_in = dim_hidden, dim_out = dim_hidden, w0 = w0, use_bias = use_bias),
				Siren(dim_in = dim_hidden, dim_out = 3, w0 = w0, use_bias = use_bias, activation = final_activation)
				)

		else:
			self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

	def _conv_layers(self, feature_c):
		net = MetaSequential(
			MetaConv(self.N_in*3, feature_c, ksize=3, s=1, p=1),
			nn.LeakyReLU(),
			MetaConv(feature_c, feature_c, ksize=3, s=1, p=1),
			nn.LeakyReLU(),
			MetaConv(feature_c, feature_c, ksize=3, s=1, p=1),
			nn.LeakyReLU(),
			MetaConv(feature_c, feature_c, ksize=3, s=1, p=1),
			nn.Tanh(),
			)
		return net

	def forward(self, x, mods = None, params = None):

		if not self.test:
			coor = x[0].clone().detach().requires_grad_(True) # [H,W,2] allows to take derivative w.r.t. input
			con_img = x[1].clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
		else:
			coor = x[0]
			con_img = x[1]

		if self.cond_type =='dir':
			imgFea = con_img.reshape(1,self.N_in*3,256,256) if self.N_in>1 else con_img
		else:
			imgFea = self.con_layer(con_img.reshape(1,self.N_in*3,256,256) if self.N_in>1 else con_img, self.get_subdict(params, 'con_layer'))

		if coor.dim()==3:
			imgFea = imgFea.squeeze(0).permute(1,2,0) #[H,W,fea_c+2]
		else:
			imgFea = imgFea.permute(0,2,3,1) #[H,W,fea_c+2]

		if self.opt.no_coords:
			x = imgFea
		else:
			x = torch.cat([coor,imgFea],dim=-1)

		mods = cast_tuple(mods, self.num_layers)

		for i, (layer, mod) in enumerate(zip(self.layers, mods)):
			# print('subdict:', self.get_subdict(params, 'layers.{}'.format(i)))
			x = layer(x, params=self.get_subdict(params, 'layers.{}'.format(i)))

			if exists(mod):
				x *= rearrange(mod, 'd -> () d')

		if self.opt.branch:
			D = self.D_branch(x, params=self.get_subdict(params, 'D_branch'))
			H = self.H_branch(x, params=self.get_subdict(params, 'H_branch'))
			R = self.R_branch(x, params=self.get_subdict(params, 'R_branch'))
			S = self.S_branch(x, params=self.get_subdict(params, 'S_branch')) 
			# print('ranch S:', S.shape)

			x = torch.cat([D,H,R,S], dim=-1) # [H,W,C]
			# print('x:', x.shape)
		else:

			x = self.last_layer(x, params=self.get_subdict(params, 'last_layer'))


		return x


class MetaConv(MetaModule):
	def __init__(self,in_c, out_c, ksize=4, s=2, p=1):
		super().__init__()
		self.metaconv = MetaConv2d(in_c, out_c,  kernel_size=ksize, stride=s, padding=p)

	def forward(self,x, params=None):
		x = self.metaconv(x, params=self.get_subdict(params, 'metaconv'))
		return x

class MetaDeConv(MetaModule):
	def __init__(self,in_c, out_c):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='nearest')
		self.metaconv1 = MetaConv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)

	def forward(self, x, params=None):
		x = self.metaconv1(self.up(x), params=self.get_subdict(params, 'metaconv1'))
		return x

# siren network
class MetaUNet(MetaModule):
	def __init__(self, dim_hidden, layer_n=3, N_in=1, test=False):
		super().__init__()
		self.test=test
		self.layer_n = layer_n
		self.dim_hidden = dim_hidden
		self.convs = nn.ModuleList([])

		conv_channels={   0: 3*N_in,
						  1: self.dim_hidden,
						  2: self.dim_hidden*2,
						  3: self.dim_hidden*4,
						  4: self.dim_hidden*4
					}

		for i in range(self.layer_n):
			in_c = conv_channels[i]
			out_c = conv_channels[i+1]

			self.convs.append(MetaConv(in_c, out_c, ksize=4, s=2, p=1))

		self.deconvs = nn.ModuleList([])
		for i in range(self.layer_n):
			in_c = conv_channels[self.layer_n-i] if i==0 else conv_channels[self.layer_n-i]*2 
			out_c = conv_channels[self.layer_n-i-1] if i!=self.layer_n-1 else self.dim_hidden
			self.deconvs.append(MetaDeConv(in_c, out_c))

		# self.lastlayer = nn.Conv2d(out_c, final_c, kernel_size=3, stride=1, padding=1)
		self.Tanh = nn.Tanh()
		self.LeakyReLU = nn.LeakyReLU()

	def forward(self, x, params):
		if not self.test:
			x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

		# vis image
		# import matplotlib
		# matplotlib.use('Agg')
		# import matplotlib.pyplot as plt
		# from meta_utils import save_image

		# save_image((1+x[0,0:3,:,:].permute(1,2,0))*0.5, '0.png')
		# save_image((1+x[0,3:6,:,:].permute(1,2,0))*0.5, '1.png')


		convs=[]
		for i in range(self.layer_n):
			x = self.LeakyReLU(self.convs[i](x, params=self.get_subdict(params, f'convs.{i}')))
			if i!=self.layer_n-1:
				convs.append(x)

		for i in range(self.layer_n):
			if i==0:
				x = self.LeakyReLU(self.deconvs[i](x, params=self.get_subdict(params, f'deconvs.{i}')))
			elif i==self.layer_n-1:
				temp = torch.cat([x, convs[self.layer_n-1-i]], dim=1)
				x = self.Tanh(self.deconvs[i](temp, params=self.get_subdict(params, f'deconvs.{i}')))
			else:
				temp = torch.cat([x, convs[self.layer_n-1-i]], dim=1)
				x = self.LeakyReLU(self.deconvs[i](temp, params=self.get_subdict(params, f'deconvs.{i}')))

		return x


# Meta_Material

class MaterialMetaDeConv(MetaModule):
	def __init__(self,in_c, out_c):
		super().__init__()
		self.up = nn.Upsample(scale_factor=2, mode='bilinear')
		self.metaconv1 = MetaConv2d(in_c, out_c, kernel_size=3, stride=1, padding=1)

	def forward(self, x, params=None):
		x = self.metaconv1(self.up(x), params=self.get_subdict(params, 'metaconv1'))
		return x

class MaterialMetaUNet(MetaModule):
	def __init__(self, dim_hidden=32, layer_n=5, N_in=1, test=False):
		super().__init__()
		self.test=test
		self.layer_n = layer_n
		self.dim_hidden = dim_hidden
		self.convs = nn.ModuleList([])

		conv_channels={   0: 3*N_in, # [256, 256]
						  1: self.dim_hidden, # [128, 128]
						  2: self.dim_hidden*2, # [64, 64]
						  3: self.dim_hidden*4, # [32, 32]
						  4: self.dim_hidden*8, # [16, 16]
						  5: self.dim_hidden*16, # [8, 8]
						  6: self.dim_hidden*16, # [4, 4]
					}

		for i in range(self.layer_n):
			in_c = conv_channels[i]
			out_c = conv_channels[i+1]

			self.convs.append(MetaConv(in_c, out_c, ksize=4, s=2, p=1))

		self.deconvs = nn.ModuleList([])
		for i in range(self.layer_n):
			in_c = conv_channels[self.layer_n-i] if i==0 else conv_channels[self.layer_n-i]*2 
			out_c = conv_channels[self.layer_n-i-1] if i!=self.layer_n-1 else 8
			self.deconvs.append(MaterialMetaDeConv(in_c, out_c))

		# self.lastlayer = nn.Conv2d(out_c, final_c, kernel_size=3, stride=1, padding=1)
		self.Sigmoid = nn.Sigmoid()
		self.LeakyReLU = nn.LeakyReLU()



	def forward(self, x, params):
		if not self.test:
			x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input

		# vis image
		# import matplotlib
		# matplotlib.use('Agg')
		# import matplotlib.pyplot as plt
		# from meta_utils import save_image

		# save_image((1+x[0,0:3,:,:].permute(1,2,0))*0.5, '0.png')
		# save_image((1+x[0,3:6,:,:].permute(1,2,0))*0.5, '1.png')


		convs=[]
		for i in range(self.layer_n):
			x = self.LeakyReLU(self.convs[i](x, params=self.get_subdict(params, f'convs.{i}')))
			if i!=self.layer_n-1:
				convs.append(x)

		for i in range(self.layer_n):
			if i==0:
				x = self.LeakyReLU(self.deconvs[i](x, params=self.get_subdict(params, f'deconvs.{i}')))
			elif i==self.layer_n-1:
				temp = torch.cat([x, convs[self.layer_n-1-i]], dim=1)
				x = self.Sigmoid(self.deconvs[i](temp, params=self.get_subdict(params, f'deconvs.{i}')))
			else:
				temp = torch.cat([x, convs[self.layer_n-1-i]], dim=1)
				x = self.LeakyReLU(self.deconvs[i](temp, params=self.get_subdict(params, f'deconvs.{i}')))

		return x


# loss network
class LossNet(nn.Module):
	def __init__(self, layer_n=6, dim_c=64):
		super().__init__()
		self.layer_n = layer_n
		self.LeakyReLU = nn.LeakyReLU()

		for i in range(layer_n):
			in_c = 3 if i==0 else dim_c*i
			out_c = dim_c*(i+1)
			out_c = 512 if out_c > 512 else out_c
			in_c = 512 if in_c > 512 else in_c
			layer = nn.Conv2d(in_c, out_c, 4, 2, 1)
			setattr(self, f'layer{i}',layer)


	def forward(self, in_image, keys):
		out = []

		for index in range(self.layer_n):
			layer = getattr(self, f'layer{index}')
			in_image = in_image if index==0 else out[-1]
			out.append(self.LeakyReLU(layer(in_image)))

		return [out[key] for key in keys]

from util.util import Process_des19normal

def paramize_out_des19(vec): 

	if vec.dim()==4:
		vec = vec[0,...]

	# print(vec[:,:,0:2])
	N = Process_des19normal(vec[:,:,0:2])
	D = vec[:,:,2:5]
	R = vec[:,:,5:6].repeat(1,1,3)
	S = vec[:,:,6:9]
	return torch.cat((N,D,R,S), dim=-1)




class MyLoss(nn.Module):
	def __init__(self, opt, keys, weights, device):
		super().__init__()
		self.opt = opt
		self.keys = keys
		self.weights = weights
		self.use_svbrdf = False
		if opt.netloss=='Des19Net':
			self.lossnet = Net_Des19(opt.N_input).to(device)
			self.use_svbrdf = True
			self.PositionMap_Des19 = PositionMap_Des19(256, 256).cuda().unsqueeze(0).repeat(1,1,1,1).permute(0,3,1,2)

			self.LoadDes19Net(self.lossnet,np.load(opt.Des19Net_npy_path,allow_pickle=True).item())

		else:
			self.lossnet = LossNet().to(device)

		self.N_in = opt.N_input
		self.no_spec = opt.no_spec
		if opt.losstype=='L2':
			self.criterion = nn.MSELoss()
		else:
			self.criterion = nn.L1Loss()

		self.resize = transforms.Resize((opt.res,opt.res))


	def set_gradient(self, grad):
		for param in self.lossnet.parameters():
			param.requires_grad = grad

	def LoadDes19Net(self, netG, des19_npy):
		params=netG.state_dict()
		for key in params:
			if 'instance' in key:
				params[key].copy_(torch.from_numpy(des19_npy[key]).squeeze(0).squeeze(0).squeeze(0))
			elif 'global' in key:
				if 'weight' in key:
					params[key].copy_(torch.from_numpy(des19_npy[key]).permute(1,0))
				else:
					params[key].copy_(torch.from_numpy(des19_npy[key]))
			elif 'lastconv3.bias' in key:
				params[key].copy_(torch.from_numpy(des19_npy[key]).squeeze(0).squeeze(0).squeeze(0))
			else:
				params[key].copy_(torch.from_numpy(des19_npy[key]).permute(3,2,0,1))

		print('finish loading Des19 npy file')

	def forward(self, in_image, gt_256):

		if not self.use_svbrdf:
			self.out_fea = self.lossnet(in_image, self.keys)
			self.gt_fea = self.lossnet(gt, self.keys)
			sum_loss = 0
			for index, (out, gt) in enumerate(zip(self.out_fea, self.gt_fea)):
				sum_loss += self.L1(out, gt)*self.weights[index]
		else:
			# out_svbbrdf = self.lossnet(torch.cat((in_image, self.PositionMap_Des19), dim=1))

			if gt_256.shape[0]==1:
				des_in = torch.cat((gt_256, self.PositionMap_Des19), dim=1) #[1,5,H,W]
			else:
				des_in = torch.cat((gt_256, self.PositionMap_Des19.repeat(gt_256.shape[0],1,1,1)), dim=1) #[N,5,H,W]
				# des_in = des_in.reshape(self.N_in,-1,256,256)
				# gt_256 = gt_256.reshape(1,gt_256.shape[0]*3,256,256)

			gt_svbrdf = self.lossnet(des_in) #[N,9,256,256] [-1,1]

			assert gt_svbrdf.shape[0]==1, 'should have batch sizez 1 for each sample'
			gt_svbrdf = paramize_out_des19((gt_svbrdf.permute(0,2,3,1)+1)*0.5 ) #[N, W,H,12] [0,1]

			# resize images
			gt_svbrdf = self.resize(gt_svbrdf.unsqueeze(0).permute(0,3,1,2)).permute(0,2,3,1).squeeze(0)

			assert gt_svbrdf.shape[0]==in_image.shape[0], 'dim not match'

			# print(in_image.shape)
			# print(gt_svbrdf.shape)
			if self.no_spec:
				sum_loss = self.criterion(in_image[:,:,:9],gt_svbrdf[:,:,:9])
			else:
				if self.opt.sc_des19:
					# print('scale')
					sum_loss = self.criterion(in_image,gt_svbrdf*10-5)
				else:
					sum_loss = self.criterion(in_image,gt_svbrdf)

		return sum_loss, gt_svbrdf


# con-siren network
class OldConSirenNet(MetaModule):
	def __init__(self, dim_in, dim_hidden, dim_out, num_layers, fea_c=32, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None, cond_type='conv'):
		super().__init__()
		self.num_layers = num_layers
		self.dim_hidden = dim_hidden
		self.cond_type = cond_type
		if self.cond_type=='conv':
			self.con_layer = self._conv_layers(fea_c)
		elif self.cond_type=='unet':
			self.con_layer = MetaUNet(32)

		self.layers = nn.ModuleList([])
		for ind in range(num_layers):
			is_first = ind == 0
			layer_w0 = w0_initial if is_first else w0
			layer_dim_in = dim_in+fea_c if is_first else dim_hidden

			self.layers.append(Siren(
				dim_in = layer_dim_in,
				dim_out = dim_hidden,
				w0 = layer_w0,
				use_bias = use_bias,
				is_first = is_first
			))

		self.final_activation = nn.Identity() if not exists(final_activation) else final_activation
		self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

	def _conv_layers(self, feature_c):
		net = MetaSequential(
			OldMetaConv(3, feature_c, ksize=3, s=1, p=1),
			OldMetaConv(feature_c, feature_c, ksize=3, s=1, p=1),
			OldMetaConv(feature_c, feature_c, ksize=3, s=1, p=1),
			OldMetaConv(feature_c, feature_c, ksize=3, s=1, p=1, activation=nn.Tanh())
			)
		return net

	def forward(self, x, mods = None, params = None):
		coor = x[0]#.clone().detach().requires_grad_(True) # [H,W,2] allows to take derivative w.r.t. input
		con_img = x[1]#.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
		
		imgFea = self.con_layer(con_img, self.get_subdict(params, 'con_layer')).squeeze(0).permute(1,2,0) #[H,W,fea_c+2]
		# print('concat shape', coor.shape)
		# print('concat shape', imgFea.shape)
		x = torch.cat([coor,imgFea],dim=-1)
		# print('concat shape', x.shape)

		mods = cast_tuple(mods, self.num_layers)

		for i, (layer, mod) in enumerate(zip(self.layers, mods)):
			# print('subdict:', self.get_subdict(params, 'layers.{}'.format(i)))
			x = layer(x, params=self.get_subdict(params, 'layers.{}'.format(i)))

			if exists(mod):
				x *= rearrange(mod, 'd -> () d')

		x = self.last_layer(x, params=self.get_subdict(params, 'last_layer'))
		return x


class OldMetaConv(MetaModule):
	def __init__(self,in_c, out_c, ksize=4, s=2, p=1, activation=None):
		super().__init__()
		self.metaconv = MetaConv2d(in_c, out_c,  kernel_size=ksize, stride=s, padding=p)
		self.activation = nn.LeakyReLU() if activation is None else activation

	def forward(self,x, params=None):
		# for p in self.get_subdict(params, 'metaconv'):
		# 	print('metaconv ',p, self.get_subdict(params, 'metaconv')[p][0,0,0,0] if self.get_subdict(params, 'metaconv')[p].dim()==4 else self.get_subdict(params, 'metaconv')[p][0])
		x = self.metaconv(x, params=self.get_subdict(params, 'metaconv'))
		x = self.activation(x)

		return x



class Des19Net(nn.Module):
	def __init__(self, opt,device):
		super().__init__()
		self.opt = opt

		self.lossnet = Net_Des19(opt.N_input).to(device)
		self.PositionMap_Des19 = PositionMap_Des19(256, 256).cuda().unsqueeze(0).repeat(1,1,1,1).permute(0,3,1,2)
		self.LoadDes19Net(self.lossnet,np.load(opt.Des19Net_npy_path,allow_pickle=True).item())



	def LoadDes19Net(self, netG, des19_npy):
		params=netG.state_dict()
		for key in params:
			if 'instance' in key:
				params[key].copy_(torch.from_numpy(des19_npy[key]).squeeze(0).squeeze(0).squeeze(0))
			elif 'global' in key:
				if 'weight' in key:
					params[key].copy_(torch.from_numpy(des19_npy[key]).permute(1,0))
				else:
					params[key].copy_(torch.from_numpy(des19_npy[key]))
			elif 'lastconv3.bias' in key:
				params[key].copy_(torch.from_numpy(des19_npy[key]).squeeze(0).squeeze(0).squeeze(0))
			else:
				params[key].copy_(torch.from_numpy(des19_npy[key]).permute(3,2,0,1))

		print('finish loading Des19 npy file')

	def forward(self, gt_256):

		if gt_256.shape[0]==1:
			des_in = torch.cat((gt_256, self.PositionMap_Des19), dim=1) #[1,5,H,W]
		else:
			des_in = torch.cat((gt_256, self.PositionMap_Des19.repeat(gt_256.shape[0],1,1,1)), dim=1) #[N,5,H,W]
			# des_in = des_in.reshape(self.N_in,-1,256,256)
			# gt_256 = gt_256.reshape(1,gt_256.shape[0]*3,256,256)

		gt_svbrdf = self.lossnet(des_in) #[N,9,256,256] [-1,1]

		assert gt_svbrdf.shape[0]==1, 'should have batch sizez 1 for each sample'
		gt_svbrdf = paramize_out_des19((gt_svbrdf.permute(0,2,3,1)+1)*0.5 ) #[N, W,H,12] [0,1]

		return gt_svbrdf

