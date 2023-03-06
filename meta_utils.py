
# from livelossplot import PlotLosses
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torch.utils.data import DataLoader
import random
from torchvision import datasets

import torchvision
import torchvision.transforms as transforms


import torch.utils.data as data
from os import listdir
from PIL import Image

from models.renderer import *



from util.util import EPSILON

from skimage.transform import resize

from collections import OrderedDict



def save_image(image, image_path):
    ndarr = image.mul(255).clamp(0, 255).byte().cpu().numpy()
    image_pil = Image.fromarray(ndarr)
    image_pil.save(image_path)

def get_params(res, size):
	w, h = size
	new_h = h
	new_w = w

	x = random.randint(0, np.maximum(0, new_w - res))
	y = random.randint(0, np.maximum(0, new_h - res))
	
	# flip = random.random() > 0.5
	return {'crop_pos': (x, y), 'flip': 0}


class DataLoaderHelper(data.Dataset):
	def __init__(self, image_dir, opt):
		super(DataLoaderHelper, self).__init__()
		self.path = image_dir
		# self.image_filenames = glob('{:s}/*.png'.format(image_dir),recursive=True)
		# self.image_filenames = [x for x in listdir(image_dir) if 'output_iter200000' in x]
		self.image_filenames = [x for x in listdir(image_dir)]
		self.fea = opt.fea
		self.res=opt.res
		self.no_spec = opt.no_spec
		# self.resize = False if opt.netloss=='Des19Net' else True
		self.resize = False

	def __crop(self, img, pos, size):
		ow, oh = img.size
		x1, y1 = pos
		tw = th = size
		if (ow > tw or oh > th):        
			return img.crop((x1, y1, x1 + tw, y1 + th))
		return img

	def __getitem__(self, index):
		#load the image
		fullimage = Image.open(os.path.join(self.path,self.image_filenames[index])).convert('RGB') 
		# fullimage = mpimg.imread(join(self.path,self.image_filenames[index]))
		w, h = fullimage.size 

		w5 = int(w / 5)
		I = fullimage.crop((0, 0, w5, h))    
		N = fullimage.crop((w5, 0, 2*w5, h))    
		D = fullimage.crop((2*w5, 0, 3*w5, h))    
		R = fullimage.crop((3*w5, 0, 4*w5, h))  
		S = fullimage.crop((4*w5, 0, 5*w5, h))  

		params = get_params(256, N.size)
		crop = transforms.Compose([transforms.Lambda(lambda img: self.__crop(img, params['crop_pos'], 256)), transforms.ToTensor()])

		resize = transforms.Compose( [transforms.Resize((256,256)),transforms.ToTensor()])

		if self.resize:
			N=resize(N)
			D=resize(D)
			R=resize(R)
			S=resize(S)
		else:
			N=crop(N)
			D=crop(D)
			R=crop(R)
			S=crop(S)			

		if self.no_spec:
			S = S*0 + 0.04

		if 'all' in self.fea or self.meta_debug != '':
			return torch.cat((N,D,R,S), dim=0)

	def __len__(self):
		return len(self.image_filenames)


class DataLoaderHelper_test(data.Dataset):
	def __init__(self, image_dir, opt):
		super(DataLoaderHelper_test, self).__init__()
		self.path = image_dir
		# self.image_filenames = glob('{:s}/*.png'.format(image_dir),recursive=True)
		# self.image_filenames = [x for x in listdir(image_dir) if 'output_iter200000' in x]
		self.image_filenames = [x for x in listdir(image_dir)]
		self.fea = opt.fea
		# self.CROP_SIZE = 256 if opt.netloss=='Des19Net' else  opt.res
		self.CROP_SIZE = 256
		self.no_spec = opt.no_spec


	def __crop(self, img, pos, size):
		ow, oh = img.size
		x1, y1 = pos
		tw = th = size
		if (ow > tw or oh > th):        
			return img.crop((x1, y1, x1 + tw, y1 + th))
		return img

	def __getitem__(self, index):
		#load the image
		fullimage = Image.open(os.path.join(self.path,self.image_filenames[index])).convert('RGB') 
		name = self.image_filenames[index].split('.')[0]
		w, h = fullimage.size 

		if fullimage.size[1]!=256:
			fullimage = fullimage.resize((256,256))

		if w==5*h:
			# print('5555555555555555')

			w5 = int(w / 5)
			I = fullimage.crop((0, 0, w5, h))    
			N = fullimage.crop((w5, 0, 2*w5, h))    
			D = fullimage.crop((2*w5, 0, 3*w5, h))    
			R = fullimage.crop((3*w5, 0, 4*w5, h))  
			S = fullimage.crop((4*w5, 0, 5*w5, h))  

			resize = transforms.Compose( [transforms.Resize((self.CROP_SIZE,self.CROP_SIZE)),transforms.ToTensor()])
			N=resize(N)
			D=resize(D)
			R=resize(R)
			S=resize(S)

			if self.no_spec:
				S = S*0 + 0.04

			if 'all' in self.fea or self.meta_debug != '':
				return torch.cat((N,D,R,S), dim=0), name


		elif w==4*h:
			# print('44444444444444444444444444444')
			w5 = int(w / 4)
			D = fullimage.crop((0, 0, w5, h))    
			N = fullimage.crop((w5, 0, 2*w5, h))    
			R = fullimage.crop((2*w5, 0, 3*w5, h))    
			S = fullimage.crop((3*w5, 0, 4*w5, h))  

			resize = transforms.Compose( [transforms.Resize((self.CROP_SIZE,self.CROP_SIZE)),transforms.ToTensor()])
			N=resize(N)
			D=resize(D)**2.2+EPSILON
			R=resize(R)+EPSILON
			S=resize(S)**2.2+EPSILON

			if self.no_spec:
				S = S*0 + 0.04


			if 'all' in self.fea or self.meta_debug != '':
				return torch.cat((N,D,R,S), dim=0), name


		elif w==h:
			ToTensor = transforms.ToTensor()

			return ToTensor(fullimage), name


	def __len__(self):
		return len(self.image_filenames)

def process_example(example, RES):
	# image = torch.tensor(example) / 255.0
	# image = example / 255.0
	image = example 

	return image[:,image.shape[1]//2-RES//2:image.shape[1]//2+RES//2,image.shape[2]//2-RES//2:image.shape[2]//2+RES//2]


# ----------------------------------------------------------------------------

# helper saving function that can be used by subclasses
def save_all_models(model_dict, path,iteration):
    save_filename = '%s_net.pth' % str(iteration)
    save_path = os.path.join(path, save_filename)
    torch.save(model_dict, save_path)


# helper saving function that can be used by subclasses
def save_network(net, path,iteration, name=''):
    save_filename = '%s_%snet.pth' % (str(iteration),name)
    save_path = os.path.join(path, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        net.cuda()

# helper loading function that can be used by subclasses
def load_network_pretrain(network, feaext, iter_label, save_dir, cond_type, name=''):        
	save_filename = '%s_net.pth' % str(iter_label)
	save_path = os.path.join(save_dir, save_filename)        
	if not os.path.isfile(save_path):
		print('%s not exists yet!' % save_path)
	else:
		print('................loading {} network{}..........'.format(name, iter_label))
		try:
			network.load_state_dict(torch.load(save_path)[name])

		except:   
			pretrained_dict = torch.load(save_path)[name]               
			model_dict = network.state_dict()

			for i,j in model_dict.items():
				print('model_dict ',i)

			for i,j in pretrained_dict.items():
				print('pretrained_dict ',i)

			feaext_dict = feaext.state_dict()

			try:
				# print('try.........')
				if cond_type=='conv':
					# load feature extractor
					pretrained_feaext_dict={}
					for k,v in pretrained_dict.items():
						if 'con_layer' in k:
							idx = k.split('.')[1]
							w_or_b = k.split('.')[-1]
							new_name = 'net.'+str(int(idx)*2)+'.'+w_or_b
							pretrained_feaext_dict[new_name] = v
					feaext.load_state_dict(pretrained_feaext_dict)
				elif cond_type=='unet':
					pretrained_feaext_dict={}
					for k,v in pretrained_dict.items():
						if 'con_layer' in k:
							conv = k.split('.')[1]
							idx = k.split('.')[2]
							w_or_b = k.split('.')[-1]
							if conv=='convs':
								new_name = conv+'.'+idx+'.'+w_or_b
							elif conv=='deconvs':
								new_name = conv+'.'+idx+'.conv1.'+w_or_b
							pretrained_feaext_dict[new_name] = v
					feaext.load_state_dict(pretrained_feaext_dict)					

				# load model
				pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
				network.load_state_dict(pretrained_dict)
				print('end try.........')

			except:
				print('Pretrained %s %s has fewer layers; The following are not initialized:' % (name, iter_label))
				for k, v in pretrained_dict.items():                      
					if v.size() == model_dict[k].size():
						model_dict[k] = v

				if sys.version_info >= (3,0):
					not_initialized = set()
				else:
					from sets import Set
					not_initialized = Set()                    

				for k, v in model_dict.items():
					if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
						not_initialized.add(k.split('.')[0])
				
				print(sorted(not_initialized))
				network.load_state_dict(model_dict)  

# helper loading function that can be used by subclasses
def load_network2(network, iter_label, save_dir, name=''):        
	save_filename = '%s_net.pth' % str(iter_label)
	save_path = os.path.join(save_dir, save_filename)        
	if not os.path.isfile(save_path):
		print('%s not exists yet!' % save_path)
	else:
		print('................loading {} network{}..........'.format(name, iter_label))
		# net = torch.load(save_path)

		# network.load_state_dict(torch.load(save_path))
		try:
			network.load_state_dict(torch.load(save_path)[name])
		except:   
			pretrained_dict = torch.load(save_path)[name]              
			model_dict = network.state_dict()

			try:

				pretrained_dict['con_layer.6.metaconv.weight'] = pretrained_dict['con_layer.3.metaconv.weight']
				pretrained_dict['con_layer.6.metaconv.bias'] = pretrained_dict['con_layer.3.metaconv.bias']
				pretrained_dict['con_layer.4.metaconv.weight'] = pretrained_dict['con_layer.2.metaconv.weight']
				pretrained_dict['con_layer.4.metaconv.bias'] = pretrained_dict['con_layer.2.metaconv.bias']
				pretrained_dict['con_layer.2.metaconv.weight'] = pretrained_dict['con_layer.1.metaconv.weight']
				pretrained_dict['con_layer.2.metaconv.bias'] = pretrained_dict['con_layer.1.metaconv.bias']
				pretrained_dict['con_layer.0.metaconv.weight'] = pretrained_dict['con_layer.0.metaconv.weight']
				pretrained_dict['con_layer.0.metaconv.bias'] = pretrained_dict['con_layer.0.metaconv.bias']

				del pretrained_dict['con_layer.1.metaconv.weight']
				del pretrained_dict['con_layer.1.metaconv.bias']
				del pretrained_dict['con_layer.3.metaconv.weight']
				del pretrained_dict['con_layer.3.metaconv.bias']

				for parameters,_ in pretrained_dict.items():
				    print('pretrained_dict ',parameters)

				for parameters in model_dict:
				    print('model_dict ', parameters)


				pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
				network.load_state_dict(pretrained_dict)

			except:

				# model_dict['con_layer.0.metaconv.weight'] = pretrained_dict['con_layer.0.metaconv.weight']
				# model_dict['con_layer.0.metaconv.bias'] = pretrained_dict['con_layer.0.metaconv.weight']
				# model_dict['con_layer.2.metaconv.weight'] = pretrained_dict['con_layer.1.metaconv.weight']
				# model_dict['con_layer.2.metaconv.bias'] = pretrained_dict['con_layer.1.metaconv.weight']
				# model_dict['con_layer.4.metaconv.weight'] = pretrained_dict['con_layer.2.metaconv.weight']
				# model_dict['con_layer.4.metaconv.bias'] = pretrained_dict['con_layer.2.metaconv.weight']
				# model_dict['con_layer.6.metaconv.weight'] = pretrained_dict['con_layer.3.metaconv.weight']
				# model_dict['con_layer.6.metaconv.bias'] = pretrained_dict['con_layer.3.metaconv.weight']

				# network.load_state_dict(model_dict)


				print('Pretrained %s %s has fewer layers; The following are not initialized:' % (name, iter_label))
				for k, v in pretrained_dict.items():                      
					if v.size() == model_dict[k].size():
						model_dict[k] = v

				if sys.version_info >= (3,0):
					not_initialized = set()
				else:
					from sets import Set
					not_initialized = Set()                    

				for k, v in model_dict.items():
					if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
						not_initialized.add(k.split('.')[0])
				
				print(sorted(not_initialized))
				network.load_state_dict(model_dict)  



# helper loading function that can be used by subclasses
def load_network(network, iter_label, save_dir, name=''):     

    save_filename = '%s_net.pth' % str(iter_label)
    # print(save_dir)
    save_path = os.path.join(save_dir, save_filename)        
    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
    else:
        print('................loading {} network{}..........'.format(name, iter_label))
        # net = torch.load(save_path)
        # for parameters in net:
        #     print(parameters, 'shape: ', net[parameters].shape)
        # network.load_state_dict(torch.load(save_path))
        try:
            network.load_state_dict(torch.load(save_path)[name])
        except:   
            pretrained_dict = torch.load(save_path)                
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                network.load_state_dict(pretrained_dict)

            except:
                print('Pretrained %s %s has fewer layers; The following are not initialized:' % (name, iter_label))
                for k, v in pretrained_dict.items():                      
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                if sys.version_info >= (3,0):
                    not_initialized = set()
                else:
                    from sets import Set
                    not_initialized = Set()                    

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])
                
                print(sorted(not_initialized))
                network.load_state_dict(model_dict)  

# def laod_
def save_loss(loss, save_dir, step, save_name=''):

	if isinstance(loss,dict):
		for key, val in loss.items():
		    plt.figure()
		    plt.plot(step, val)
		    plt.savefig(save_dir+'/%sstep_%sloss.png'%(key,save_name))
		    plt.close() 
	else:
	    plt.figure()
	    plt.plot(step, loss)
	    plt.savefig(save_dir+'/%sloss.png'%save_name)
	    plt.close() 

# input shape [B,H,W,C] or [H,W,C]
def paramize_out(opt, vec):
	# assert len(vec.shape)==3, "error"
	if opt.fea=='all_N1':
		N = ProcessNormal(opt,vec[...,0:1])
		D = vec[...,1:4]
		R = vec[...,4:5].repeat(1,1,3) if vec.dim()==3 else vec[...,4:5].repeat(1,1,1,3)

		if opt.no_spec:
			if opt.gamma:
				D = torch.max(D,torch.tensor([0.], device='cuda'))**2.2
			S = D*0.0+0.04
		else:
			S = vec[...,5:8]
			if opt.gamma:
				D = torch.max(D,torch.tensor([0.], device='cuda'))**2.2
				S = torch.max(S,torch.tensor([0.], device='cuda'))**2.2

		return torch.cat((N,D,R,S), dim=-1)

	elif opt.fea=='all_N2':
		N = ProcessNormal(opt,vec[...,0:2])
		D = vec[...,2:5]
		R = vec[...,5:6].repeat(1,1,3) if vec.dim()==3 else vec[...,5:6].repeat(1,1,1,3)
		S = vec[...,6:9]

		if opt.gamma:
			D = torch.max(D,torch.tensor([0.], device='cuda'))**2.2
			S = torch.max(S,torch.tensor([0.], device='cuda'))**2.2

		return torch.cat((N,D,R,S), dim=-1)

	elif opt.fea=='all_N3':
		N = ProcessNormal(opt,vec[...,0:3])
		D = vec[...,3:6]
		R = vec[...,6:7].repeat(1,1,3) if vec.dim()==3 else vec[...,6:7].repeat(1,1,1,3)
		S = vec[...,7:10]
		return torch.cat((N,D,R,S), dim=-1)
