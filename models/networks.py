import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from .base_model import BaseModel
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm2d') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)
	elif classname.find('Linear') != -1:
		n = m.in_features
		y = np.sqrt(1/float(n))
		# print('input features: ',n)
		m.weight.data.normal_(0.0, y)
		if m.bias is not None:
			m.weight.data.normal_(0.0, y)   
			m.bias.data.normal_(0.0, 0.02) 


def define_G(input_nc, opt, gpu_ids=[]):    

	if opt.Net_Option=='MLP':
		netG = MLP(input_nc, output_nc)
	elif opt.Net_Option=='UNetS':
		if opt.Meta_train or opt.Meta_test or opt.Finetune_des19:
			netG = MetaUNet(input_nc,opt)
		else:
			if opt.maxpool:
				netG=UNet_Skip_Maxpool(input_nc, opt)
			else:
				netG=UNet_Skip_test(input_nc, opt)
	elif opt.Net_Option=='Des19Net':
		if opt.Meta_train or opt.Meta_test or opt.Finetune_des19:
			netG = MetaNet_Des19(opt)
		else:
			netG = Net_Des19(opt)
	elif opt.Net_Option=='Siren':
		netG = SirenModel(opt)

	if len(gpu_ids) > 0:
		assert(torch.cuda.is_available())   
		netG.cuda(gpu_ids[0])

	netG.apply(weights_init)

	return netG


def print_network(net):
	if isinstance(net, list):
		net = net[0]
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)


# define VA paper nework
class FC(nn.Module):
	def __init__(self,input_channel,output_channel,BIAS):
		super(FC,self).__init__()
		self.fc_layer=nn.Linear(input_channel,output_channel,bias=BIAS)

	def forward(self,input):

		out=self.fc_layer(input)

		return out



class MLP(nn.Module):
	def __init__(self,input_channel,output_channel,D=6,Hidden=512):
		super(MLP,self).__init__()
		model=[]
		activation=nn.ReLU()
		for i in range(D):
			if i==0:
				model += [FC(input_channel,Hidden,True),activation]
			else:
				model += [FC(Hidden,Hidden,True),activation]

		# model += [FC(Hidden,output_channel,True), nn.Sigmoid()]
		model += [FC(Hidden,output_channel,True), nn.Tanh()]
		self.mlp = nn.Sequential(*model)


	def forward(self, input):

		# input [B,2] -> [B,4]
		output=self.mlp(input)

		return output


class Deconv(nn.Module):

	def __init__(self,input_channel,output_channel,trans_conv):
		super(Deconv,self).__init__()
		## upsampling method (non-deterministic in pytorch)
		self.leaky_relu = nn.LeakyReLU(0.2)
		self.transconv=trans_conv

		if trans_conv:
			self.trans_conv1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=4, stride=2, padding=1)
			# self.conv1 = nn.Conv2d(input_channel,output_channel,kernel_size=3, stride=1, padding=1)
			self.conv2 = nn.Conv2d(output_channel,output_channel,kernel_size=3, stride=1, padding=1)
		else:
			self.upsampling=nn.Upsample(scale_factor=2, mode='nearest')
			self.conv1 = nn.Conv2d(input_channel,output_channel,kernel_size=3, stride=1, padding=1)
			self.conv2 = nn.Conv2d(output_channel,output_channel,kernel_size=3, stride=1, padding=1)

		# realize same padding in tensorflow
		# self.padding=nn.ConstantPad2d((1, 2, 1, 2), 0)
	def forward(self,input):

		# print('Deco input shape,',input.shape[1])
		## hack upsampling method to make is deterministic
		# Upsamp = input[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(input.size(0), input.size(1), input.size(2)*2, input.size(3)*2)
		# print(Upsamp.shape)
		if self.transconv:
			out1=self.trans_conv1(input)
			out=self.conv2(self.leaky_relu(out1))
		else:
			out1=self.conv1(self.upsampling(input))
			out=self.conv2(self.leaky_relu(out1))

		# print('output shape,',out.shape)
		return out

class UNet(nn.Module):
	def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
		super(UNet, self).__init__()        

		model = [nn.Conv2d(input_nc, ngf, 3,1,1), nn.LeakyReLU(0.2)]   

		### downsample
		for i in range(n_downsampling):
			mult = 2**i
			model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2)]

		### upsample         
		for i in range(n_downsampling):
			mult = 2**(n_downsampling - i)
			model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
					   norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        
			# model += [nn.Upsample(scale_factor = 2, mode='bilinear'),nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2)]

		model += [nn.Conv2d(ngf, output_nc, 3,1,1), nn.Tanh()]
		self.model = nn.Sequential(*model) 

	def forward(self, input):
		outputs = self.model(input)
		return outputs

class UNet_Skip_test(nn.Module):
	def __init__(self, input_nc, opt):
		super(UNet_Skip_test,self).__init__()

		## define local networks
		#encoder and downsampling
		# conv layer
		self.skip = opt.SkipCon
		self.IN = opt.InNorm
		self.layers_n= opt.layers_n
		self.output_nc = opt.output_nc
		self.trans_conv = opt.transconv
		self.ngf = opt.ngf
		self.drop_value = opt.dropout

		### skip connection or not
		skip_coefficient = 1
		if self.skip:
			skip_coefficient = 2

		######## conv layers
		for i in range(self.layers_n):
			multi = 2**(i-1)
			if i==0:
				setattr(self,'conv'+str(i+1), nn.Conv2d(input_nc, self.ngf, 4, 2, 1))
			else:
				setattr(self,'conv'+str(i+1), nn.Conv2d(self.ngf*multi, self.ngf*multi*2, 4, 2, 1))

		##### deconv layer
		for i in range(self.layers_n):
			multi = 2**(self.layers_n-i-1)
			if i==0:
				setattr(self,'deconv'+str(i+1), Deconv(self.ngf*multi, int(self.ngf*multi*0.5), self.trans_conv))
			elif i==self.layers_n-1:
				setattr(self,'deconv'+str(i+1), Deconv(self.ngf*multi*skip_coefficient, self.output_nc, self.trans_conv))
			else:
				setattr(self,'deconv'+str(i+1), Deconv(self.ngf*multi*skip_coefficient, int(self.ngf*multi*0.5),self.trans_conv))

		self.tan = nn.Tanh()

		self.leaky_relu = nn.LeakyReLU(0.2)
		self.drop = False

		if self.drop_value != 0.0:
			self.drop=True
			self.dropout = nn.Dropout(self.dropout)

		#### if Instance norm
		if self.IN:
			for i in range(self.layers_n):
				multi = 2**i
				setattr(self, 'instance_normal'+str(i+1), nn.InstanceNorm2d(int(self.ngf*multi),affine=True))
				if i != self.layers_n-1:
					multi = 2**(self.layers_n-i-2)
					setattr(self, 'instance_normal_de'+str(i+1), nn.InstanceNorm2d(int(self.ngf*multi),affine=True))

	def forward(self, input):
		### encoder 
		encoder=[]
		for i in range(self.layers_n):
			conv=getattr(self,'conv'+str(i+1))

			if self.IN:
				IN=getattr(self,'instance_normal'+str(i+1))

			if i==0:
				encoder_in = IN(conv(input)) if self.IN else conv(input)
				encoder.append(self.leaky_relu(encoder_in))
			else:
				temp = self.dropout(encoder[i-1]) if self.drop else encoder[i-1]
				temp = IN(conv(temp)) if self.IN else conv(temp)
				encoder.append(self.leaky_relu(temp))

		### decoder
		decoder=[]
		for i in range(self.layers_n):
			deconv=getattr(self,'deconv'+str(i+1))

			if self.IN and i != self.layers_n-1:
				IN=getattr(self,'instance_normal_de'+str(i+1))

			## last decoder layer
			if i==self.layers_n-1:
				decoder.append(deconv(decoder[i-1]))
			## other decoder layer
			else:
				decoder_in = encoder[self.layers_n-i-1] if i==0 else decoder[i-1]
				temp = IN(deconv(decoder_in)) if self.IN else deconv(decoder_in)
				if self.skip:
					temp = torch.cat((self.leaky_relu(temp), encoder[self.layers_n-i-2]),1)
				else:
					temp = self.leaky_relu(temp)
				temp = self.dropout(temp) if self.drop else temp
				decoder.append(temp)

		output = self.tan(decoder[self.layers_n-1])

		return output

class UNet_Skip_Maxpool(nn.Module):
	def __init__(self, input_nc, opt):
		super(UNet_Skip_Maxpool,self).__init__()

		## define local networks
		#encoder and downsampling
		# conv layer
		self.skip = opt.SkipCon
		self.IN = opt.InNorm
		self.layers_n= opt.layers_n
		self.output_nc = opt.output_nc
		self.trans_conv = opt.transconv
		self.ngf = opt.ngf
		self.drop_value = opt.dropout

		### skip connection or not
		skip_coefficient = 1
		if self.skip:
			skip_coefficient = 2

		######## conv layers
		for i in range(self.layers_n):
			multi = 2**(i-1)
			if i==0:
				setattr(self,'conv'+str(i+1), nn.Conv2d(input_nc, self.ngf, 4, 2, 1))
			else:
				setattr(self,'conv'+str(i+1), nn.Conv2d(self.ngf*multi, self.ngf*multi*2, 4, 2, 1))

		##### deconv layer
		for i in range(self.layers_n):
			multi = 2**(self.layers_n-i-1)
			if i==0:
				setattr(self,'deconv'+str(i+1), Deconv(self.ngf*multi, int(self.ngf*multi*0.5), self.trans_conv))
			elif i==self.layers_n-1:
				setattr(self,'deconv'+str(i+1), Deconv(self.ngf*multi*skip_coefficient, self.output_nc, self.trans_conv))
			else:
				setattr(self,'deconv'+str(i+1), Deconv(self.ngf*multi*skip_coefficient, int(self.ngf*multi*0.5),self.trans_conv))

		self.tan = nn.Tanh()

		self.leaky_relu = nn.LeakyReLU(0.2)
		self.drop = False

		if self.drop_value != 0.0:
			self.drop=True
			self.dropout = nn.Dropout(self.dropout)

		#### if Instance norm
		if self.IN:
			for i in range(self.layers_n):
				multi = 2**i
				setattr(self, 'instance_normal'+str(i+1), nn.InstanceNorm2d(int(self.ngf*multi),affine=True))
				if i != self.layers_n-1:
					multi = 2**(self.layers_n-i-2)
					setattr(self, 'instance_normal_de'+str(i+1), nn.InstanceNorm2d(int(self.ngf*multi),affine=True))
			

	def forward(self, input):

		### encoder 
		# print('input_shape: ', input.shape) [b,n,c,w,h]
		N = input.shape[1]
		for j in range(N):
			encoder=[]
			for i in range(self.layers_n):
				conv=getattr(self,'conv'+str(i+1))

				if self.IN:
					IN=getattr(self,'instance_normal'+str(i+1))

				if i==0:
					encoder_in = IN(conv(input[:,j,...])) if self.IN else conv(input[:,j,...])
					encoder.append(self.leaky_relu(encoder_in))
				else:
					temp = self.dropout(encoder[i-1]) if self.drop else encoder[i-1]
					temp = IN(conv(temp)) if self.IN else conv(temp)
					encoder.append(self.leaky_relu(temp))

			if j==0:
				latent = encoder[self.layers_n-1].unsqueeze(1)
			else:
				latent = torch.cat((latent,encoder[self.layers_n-1].unsqueeze(1)),dim=1)		

		## max pool:
		maxlatent = torch.max(latent,dim=1)[0]

		### decoder
		decoder=[]
		for i in range(self.layers_n):
			deconv=getattr(self,'deconv'+str(i+1))

			if self.IN and i != self.layers_n-1:
				IN=getattr(self,'instance_normal_de'+str(i+1))

			## last decoder layer
			if i==self.layers_n-1:
				decoder.append(deconv(decoder[i-1]))
			## other decoder layer
			else:
				decoder_in = maxlatent if i==0 else decoder[i-1]
				temp = IN(deconv(decoder_in)) if self.IN else deconv(decoder_in)

				if self.skip:
					temp = torch.cat((self.leaky_relu(temp), encoder[self.layers_n-i-2]),1)
				else:
					temp = self.leaky_relu(temp)

				temp = self.dropout(temp) if self.drop else temp
				decoder.append(temp)

		output = self.tan(decoder[self.layers_n-1])






		return output

class UNet_Skip(nn.Module):
	def __init__(self,input_nc,output_nc,ngf=32):
		super(UNet_Skip,self).__init__()

		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_nc, ngf, 3, 1, 1)
		self.conv2 = nn.Conv2d(ngf, ngf*2, 4, 2, 1)
		self.conv3 = nn.Conv2d(ngf*2, ngf*4, 4, 2, 1)
		self.conv4 = nn.Conv2d(ngf*4, ngf*8, 4, 2, 1)
		self.conv5 = nn.Conv2d(ngf*8, ngf*16,4,2,1)

		#decoder
		self.Upsample=nn.Upsample(scale_factor = 2, mode='bilinear')

		self.deconv1 = nn.Conv2d(ngf * 16, int(ngf * 8), kernel_size=3, stride=1, padding=1)
		self.deconv2 = nn.Conv2d(ngf * 16, int(ngf * 4), kernel_size=3, stride=1, padding=1)
		self.deconv3 = nn.Conv2d(ngf * 8, int(ngf * 2), kernel_size=3, stride=1, padding=1)
		self.deconv4 = nn.Conv2d(ngf * 4, int(ngf), kernel_size=3, stride=1, padding=1)

		# self.deconv1 = Deconv(ngf * 16, int(ngf * 8))
		# self.deconv2 = Deconv(ngf * 16, int(ngf * 4))
		# self.deconv3 = Deconv(ngf * 8, int(ngf * 2))
		# self.deconv4 = Deconv(ngf * 4, int(ngf))
		self.deconv5 = nn.Conv2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1)

		self.tan = nn.Tanh()

		self.leaky_relu = nn.LeakyReLU(0.2)

		self.instance_normal2 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal3 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal4 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal5 = nn.InstanceNorm2d(512,affine=True)

		self.instance_normal_de_1 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_2 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_3 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal_de_4 = nn.InstanceNorm2d(32,affine=True)

	def forward(self, input):

		# [batch,32,h,w]
		encoder1 = self.leaky_relu(self.conv1(input)) #local network
		# [batch,64,h/2,w/2]        
		encoder2 = self.leaky_relu(self.instance_normal2(self.conv2(encoder1))) #local network
		# [batch,128,h/4,w/4]        
		encoder3 = self.leaky_relu(self.instance_normal3(self.conv3(encoder2))) #local network
		# [batch,256,h/8,w/8]        
		encoder4 = self.leaky_relu(self.instance_normal4(self.conv4(encoder3))) #local network
		# [batch,512,h/16,w/16]        
		encoder5 = self.leaky_relu(self.instance_normal5(self.conv5(encoder4))) #local network

		################################## decoder #############################################
		# [batch,256,h/8,w/8]
		decoder1 = self.leaky_relu(self.instance_normal_de_1(self.deconv1(self.Upsample(encoder5))))
		# decoder1 = self.leaky_relu(self.instance_normal_de_1(self.deconv1(encoder5)))
		# [batch,512,h/8,w/8]
		decoder1 = torch.cat((decoder1, encoder4), 1)

		# [batch,128,h/4,w/4]
		decoder2 = self.leaky_relu(self.instance_normal_de_2(self.deconv2(self.Upsample(decoder1))))
		# decoder2 = self.leaky_relu(self.instance_normal_de_2(self.deconv2(decoder1)))
		# [batch,256,h/4,w/4]
		decoder2 = torch.cat((decoder2, encoder3), 1)

		# [batch,64,h/2,w/2]
		decoder3 = self.leaky_relu(self.instance_normal_de_3(self.deconv3(self.Upsample(decoder2))))
		# decoder3 = self.leaky_relu(self.instance_normal_de_3(self.deconv3(decoder2)))
		# [batch,128,h/2,w/2]
		decoder3 = torch.cat((decoder3, encoder2), 1)

		# [batch,32,h,w]
		decoder4 = self.leaky_relu(self.instance_normal_de_4(self.deconv4(self.Upsample(decoder3))))
		# decoder4 = self.leaky_relu(self.instance_normal_de_4(self.deconv4(decoder3)))
		# [batch,64,h,w]
		decoder4 = torch.cat((decoder4, encoder1), 1)

		# [batch,10,h,w]
		output = self.tan(self.deconv5(decoder4))
		# print(output.shape)
		return output

## use conv2d transpose
class UNet_Skip_NoIN(nn.Module):
	def __init__(self,input_nc,output_nc,ngf=32):
		super(UNet_Skip_NoIN,self).__init__()

		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_nc, ngf, 3, 1, 1)
		self.conv2 = nn.Conv2d(ngf, ngf*2, 4, 2, 1)
		self.conv3 = nn.Conv2d(ngf*2, ngf*4, 4, 2, 1)
		self.conv4 = nn.Conv2d(ngf*4, ngf*8, 4, 2, 1)
		self.conv5 = nn.Conv2d(ngf*8, ngf*16,4,2,1)

		#decoder
		self.Upsample=nn.Upsample(scale_factor = 2, mode='bilinear')

		self.deconv1 = nn.ConvTranspose2d(ngf * 16, int(ngf * 8), kernel_size=4, stride=2, padding=1)
		self.deconv2 = nn.ConvTranspose2d(ngf * 16, int(ngf * 4), kernel_size=4, stride=2, padding=1)
		self.deconv3 = nn.ConvTranspose2d(ngf * 8, int(ngf * 2), kernel_size=4, stride=2, padding=1)
		self.deconv4 = nn.ConvTranspose2d(ngf * 4, int(ngf), kernel_size=4, stride=2, padding=1)
		# self.deconv5 = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1)
		self.deconv5 = nn.Conv2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1)

		self.tan = nn.Tanh()

		self.leaky_relu = nn.LeakyReLU(0.2)

		self.instance_normal2 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal3 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal4 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal5 = nn.InstanceNorm2d(512,affine=True)

		self.instance_normal_de_1 = nn.InstanceNorm2d(256,affine=True)
		self.instance_normal_de_2 = nn.InstanceNorm2d(128,affine=True)
		self.instance_normal_de_3 = nn.InstanceNorm2d(64,affine=True)
		self.instance_normal_de_4 = nn.InstanceNorm2d(32,affine=True)

	def forward(self, input):

		# [batch,32,h,w]
		encoder1 = self.leaky_relu(self.conv1(input)) #local network
		# [batch,64,h/2,w/2]        
		encoder2 = self.leaky_relu(self.conv2(encoder1)) #local network
		# [batch,128,h/4,w/4]        
		encoder3 = self.leaky_relu(self.conv3(encoder2)) #local network
		# [batch,256,h/8,w/8]        
		encoder4 = self.leaky_relu(self.conv4(encoder3)) #local network
		# [batch,512,h/16,w/16]        
		encoder5 = self.leaky_relu(self.conv5(encoder4)) #local network
		# print(encoder5.shape)

		################################## decoder #############################################
		# [batch,256,h/8,w/8]
		# decoder1 = self.leaky_relu(self.deconv1(self.Upsample(encoder5)))
		decoder1 = self.leaky_relu(self.deconv1(encoder5))
		# print(decoder1.shape)
		# print(encoder4.shape)
		# [batch,512,h/8,w/8]
		decoder1 = torch.cat((decoder1, encoder4), 1)

		# [batch,128,h/4,w/4]
		# decoder2 = self.leaky_relu(self.deconv2(self.Upsample(decoder1)))
		decoder2 = self.leaky_relu(self.deconv2(decoder1))
		# [batch,256,h/4,w/4]
		decoder2 = torch.cat((decoder2, encoder3), 1)

		# [batch,64,h/2,w/2]
		# decoder3 = self.leaky_relu(self.deconv3(self.Upsample(decoder2)))
		decoder3 = self.leaky_relu(self.deconv3(decoder2))
		# [batch,128,h/2,w/2]
		decoder3 = torch.cat((decoder3, encoder2), 1)

		# [batch,32,h,w]
		# decoder4 = self.leaky_relu(self.deconv4(self.Upsample(decoder3)))
		decoder4 = self.leaky_relu(self.deconv4(decoder3))
		# [batch,64,h,w]
		decoder4 = torch.cat((decoder4, encoder1), 1)

		# [batch,10,h,w]
		output = self.tan(self.deconv5(decoder4))
		# print(output.shape)
		return output

## use upsampleing
class UNet_Skip_NoIN2(nn.Module):
	def __init__(self,input_nc,output_nc,ngf=32):
		super(UNet_Skip_NoIN2,self).__init__()

		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_nc, ngf, 3, 1, 1)
		self.conv2 = nn.Conv2d(ngf, ngf*2, 4, 2, 1)
		self.conv3 = nn.Conv2d(ngf*2, ngf*4, 4, 2, 1)
		self.conv4 = nn.Conv2d(ngf*4, ngf*8, 4, 2, 1)
		self.conv5 = nn.Conv2d(ngf*8, ngf*16,4,2,1)

		#decoder
		self.Upsample=nn.Upsample(scale_factor = 2, mode='bilinear')

		self.deconv1 = nn.Conv2d(ngf * 16, int(ngf * 8), kernel_size=3, stride=1, padding=1)
		self.deconv2 = nn.Conv2d(ngf * 16, int(ngf * 4), kernel_size=3, stride=1, padding=1)
		self.deconv3 = nn.Conv2d(ngf * 8, int(ngf * 2), kernel_size=3, stride=1, padding=1)
		self.deconv4 = nn.Conv2d(ngf * 4, int(ngf), kernel_size=3, stride=1, padding=1)
		# self.deconv5 = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1)
		self.deconv5 = nn.Conv2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1)

		self.tan = nn.Tanh()

		self.leaky_relu = nn.LeakyReLU(0.2)


	def forward(self, input):

		# [batch,32,h,w]
		encoder1 = self.leaky_relu(self.conv1(input)) #local network
		# [batch,64,h/2,w/2]        
		encoder2 = self.leaky_relu(self.conv2(encoder1)) #local network
		# [batch,128,h/4,w/4]        
		encoder3 = self.leaky_relu(self.conv3(encoder2)) #local network
		# [batch,256,h/8,w/8]        
		encoder4 = self.leaky_relu(self.conv4(encoder3)) #local network
		# [batch,512,h/16,w/16]        
		encoder5 = self.leaky_relu(self.conv5(encoder4)) #local network
		# print(encoder5.shape)

		################################## decoder #############################################
		# [batch,256,h/8,w/8]
		decoder1 = self.leaky_relu(self.deconv1(self.Upsample(encoder5)))
		# decoder1 = self.leaky_relu(self.deconv1(encoder5))
		# print(decoder1.shape)
		# print(encoder4.shape)
		# [batch,512,h/8,w/8]
		decoder1 = torch.cat((decoder1, encoder4), 1)

		# [batch,128,h/4,w/4]
		decoder2 = self.leaky_relu(self.deconv2(self.Upsample(decoder1)))
		# decoder2 = self.leaky_relu(self.deconv2(decoder1))
		# [batch,256,h/4,w/4]
		decoder2 = torch.cat((decoder2, encoder3), 1)

		# [batch,64,h/2,w/2]
		decoder3 = self.leaky_relu(self.deconv3(self.Upsample(decoder2)))
		# decoder3 = self.leaky_relu(self.deconv3(decoder2))
		# [batch,128,h/2,w/2]
		decoder3 = torch.cat((decoder3, encoder2), 1)

		# [batch,32,h,w]
		decoder4 = self.leaky_relu(self.deconv4(self.Upsample(decoder3)))
		# decoder4 = self.leaky_relu(self.deconv4(decoder3))
		# [batch,64,h,w]
		decoder4 = torch.cat((decoder4, encoder1), 1)

		# [batch,10,h,w]
		output = self.tan(self.deconv5(decoder4))
		# print(output.shape)
		return output

## use Deconv
class UNet_Skip_NoIN3(nn.Module):
	def __init__(self,input_nc,output_nc,ngf=32):
		super(UNet_Skip_NoIN3,self).__init__()

		## define local networks
		#encoder and downsampling
		self.conv1 = nn.Conv2d(input_nc, ngf, 3, 1, 1)
		self.conv2 = nn.Conv2d(ngf, ngf*2, 4, 2, 1)
		self.conv3 = nn.Conv2d(ngf*2, ngf*4, 4, 2, 1)
		self.conv4 = nn.Conv2d(ngf*4, ngf*8, 4, 2, 1)
		self.conv5 = nn.Conv2d(ngf*8, ngf*16,4,2,1)

		#decoder
		self.deconv1 = Deconv(ngf * 16, int(ngf * 8))
		self.deconv2 = Deconv(ngf * 16, int(ngf * 4))
		self.deconv3 = Deconv(ngf * 8, int(ngf * 2))
		self.deconv4 = Deconv(ngf * 4, int(ngf))

		self.deconv5 = nn.Conv2d(ngf * 2, output_nc, kernel_size=3, stride=1, padding=1)

		self.tan = nn.Tanh()

		self.leaky_relu = nn.LeakyReLU(0.2)


	def forward(self, input):

		# [batch,32,h,w]
		encoder1 = self.leaky_relu(self.conv1(input)) #local network
		# [batch,64,h/2,w/2]        
		encoder2 = self.leaky_relu(self.conv2(encoder1)) #local network
		# [batch,128,h/4,w/4]        
		encoder3 = self.leaky_relu(self.conv3(encoder2)) #local network
		# [batch,256,h/8,w/8]        
		encoder4 = self.leaky_relu(self.conv4(encoder3)) #local network
		# [batch,512,h/16,w/16]        
		encoder5 = self.leaky_relu(self.conv5(encoder4)) #local network

		################################## decoder #############################################
		# [batch,256,h/8,w/8]
		decoder1 = self.leaky_relu(self.deconv1(encoder5))
		# decoder1 = self.leaky_relu(self.deconv1(encoder5))

		# [batch,512,h/8,w/8]
		decoder1 = torch.cat((decoder1, encoder4), 1)

		# [batch,128,h/4,w/4]
		decoder2 = self.leaky_relu(self.deconv2(decoder1))
		# decoder2 = self.leaky_relu(self.deconv2(decoder1))
		# [batch,256,h/4,w/4]
		decoder2 = torch.cat((decoder2, encoder3), 1)

		# [batch,64,h/2,w/2]
		decoder3 = self.leaky_relu(self.deconv3(decoder2))
		# decoder3 = self.leaky_relu(self.deconv3(decoder2))
		# [batch,128,h/2,w/2]
		decoder3 = torch.cat((decoder3, encoder2), 1)

		# [batch,32,h,w]
		decoder4 = self.leaky_relu(self.deconv4(decoder3))
		# decoder4 = self.leaky_relu(self.deconv4(decoder3))
		# [batch,64,h,w]
		decoder4 = torch.cat((decoder4, encoder1), 1)

		# [batch,10,h,w]
		output = self.tan(self.deconv5(decoder4))
		# print(output.shape)
		return output

############################### VGG network #################################
class VGGLoss(nn.Module):
	def __init__(self, opt):
		super(VGGLoss, self).__init__()        
		self.vgg = Vgg19().cuda()
		if opt.losstype == 'L2':
			self.criterion = nn.MSELoss()
		elif opt.losstype == 'L1':
			self.criterion = nn.L1Loss()
		self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

	def forward(self, x, y):              
		x_vgg, y_vgg = self.vgg(x), self.vgg(y)
		loss = 0
		for i in range(len(x_vgg)):
			loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())   
		return loss

from torchvision import models
class Vgg19(torch.nn.Module):
	def __init__(self, requires_grad=False):
		super(Vgg19, self).__init__()
		vgg_pretrained_features = models.vgg19(pretrained=True).features
		self.slice1 = torch.nn.Sequential()
		self.slice2 = torch.nn.Sequential()
		self.slice3 = torch.nn.Sequential()
		self.slice4 = torch.nn.Sequential()
		self.slice5 = torch.nn.Sequential()
		for x in range(2):
			self.slice1.add_module(str(x), vgg_pretrained_features[x])
		for x in range(2, 7):
			self.slice2.add_module(str(x), vgg_pretrained_features[x])
		for x in range(7, 12):
			self.slice3.add_module(str(x), vgg_pretrained_features[x])
		for x in range(12, 21):
			self.slice4.add_module(str(x), vgg_pretrained_features[x])
		for x in range(21, 30):
			self.slice5.add_module(str(x), vgg_pretrained_features[x])
		if not requires_grad:
			for param in self.parameters():
				param.requires_grad = False

	def forward(self, X):
		h_relu1 = self.slice1(X)
		h_relu2 = self.slice2(h_relu1)        
		h_relu3 = self.slice3(h_relu2)        
		h_relu4 = self.slice4(h_relu3)        
		h_relu5 = self.slice5(h_relu4)                
		out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
		return out


class FeatureLoss(torch.nn.Module):

	def __init__(self, dir, w):
		super(FeatureLoss, self).__init__()

		self.net = VGG()
		self.net.load_state_dict(torch.load(dir))
		self.net.eval().cuda()

		# self.layer = ['r11','r12','r33','r43']
		self.layer = ['r11','r12','r32','r42']
		self.weights = w

	def forward(self, x):
		outputs = self.net(x, self.layer)
		# th.save(outputs, 'tmp.pt')
		# exit()
		result = []
		for i, feature in enumerate(outputs):
			result.append(feature.flatten() * self.weights[i])

		return torch.cat(result)



class VGG(nn.Module):
	def __init__(self, pool='max'):
		super(VGG, self).__init__()
		#vgg modules
		self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
		self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
		if pool == 'max':
			self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
			self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
		elif pool == 'avg':
			self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
			self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

	def forward(self, x, out_keys):
		out = {}
		out['r11'] = F.relu(self.conv1_1(x))
		out['r12'] = F.relu(self.conv1_2(out['r11']))
		out['p1'] = self.pool1(out['r12'])
		out['r21'] = F.relu(self.conv2_1(out['p1']))
		out['r22'] = F.relu(self.conv2_2(out['r21']))
		out['p2'] = self.pool2(out['r22'])
		out['r31'] = F.relu(self.conv3_1(out['p2']))
		out['r32'] = F.relu(self.conv3_2(out['r31']))
		out['r33'] = F.relu(self.conv3_3(out['r32']))
		out['r34'] = F.relu(self.conv3_4(out['r33']))
		out['p3'] = self.pool3(out['r34'])
		out['r41'] = F.relu(self.conv4_1(out['p3']))
		out['r42'] = F.relu(self.conv4_2(out['r41']))
		out['r43'] = F.relu(self.conv4_3(out['r42']))
		# out['r44'] = F.relu(self.conv4_4(out['r43']))
		# out['p4'] = self.pool4(out['r44'])
		# out['r51'] = F.relu(self.conv5_1(out['p4']))
		# out['r52'] = F.relu(self.conv5_2(out['r51']))
		# out['r53'] = F.relu(self.conv5_3(out['r52']))
		# out['r54'] = F.relu(self.conv5_4(out['r53']))
		# out['p5'] = self.pool5(out['r54'])
		return [out[key] for key in out_keys]


############################## Des19 network #################################
def mymean(input):
	[b,c,w,h]=input.shape
	mean=input.view(b,c,-1).mean(2)
	mean=mean.unsqueeze(-1).unsqueeze(-1)
	return mean#.reshape(b,c,1,1)

## the input of FC function is 4D channel [b,c,1,1], output is 4D: [b,c,1,1]
class FC_des19(nn.Module):
	def __init__(self,input_channel,output_channel,BIAS):
		super(FC_des19,self).__init__()
		self.fc_layer=nn.Linear(input_channel,output_channel,bias=BIAS)

	def forward(self,input):
		# if input is 4D [b,c,1,1],output [b,c,1,1]
		if len(input.shape) == 4:
			[b,c,w,h]=input.shape
			out=self.fc_layer(input.view(b,c))
			out=out.unsqueeze(2).unsqueeze(2)
		# if input is 2D [b,c] otuput [b,c]
		elif len(input.shape) == 2:
			out=self.fc_layer(input)
		# otherwise, error
		else:
			print('incorrectly input to FC layer')
			sys.exit(1) 


		return out

# self-define deconv layer for des19
class Deconv_des19(nn.Module):

	def __init__(self,input_channel,output_channel):
		super(Deconv_des19,self).__init__()
		## upsampling method (non-deterministic in pytorch)
		# self.upsampling=nn.Upsample(scale_factor=2, mode='nearest')

		self.conv1 = nn.Conv2d(input_channel,output_channel,4,stride=1,bias=False)
		self.conv2 = nn.Conv2d(output_channel,output_channel,4,stride=1,bias=False)

		# realize same padding in tensorflow
		self.padding=nn.ConstantPad2d((1, 2, 1, 2), 0)

	def forward(self,input):

		# print('Deco input shape,',input.shape[1])
		# Upsamp=self.upsampling(input)

		## hack upsampling method to make is deterministic
		Upsamp = input[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(input.size(0), input.size(1), input.size(2)*2, input.size(3)*2)

		out=self.conv1(self.padding(Upsamp))
		out=self.conv2(self.padding(out))

		# print('output shape,',out.shape)


		return out

class Net_Des19(nn.Module):
	def __init__(self,No_in=1):
		super(Net_Des19,self).__init__()

		self.ngf = 64
		self.output_nc1 = 64
		self.output_nc2 = 10
		self.input_nc = 5
		
		# self.use_dropout = opt.des19_dropout if opt is not None else None
		self.use_dropout = None

		######## define local networks ##################
		### conv layers
		layer_specs_out = [
			self.ngf * 2, # encoder2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
			self.ngf * 4, # encoder3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
			self.ngf * 8, # encoder4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
			self.ngf * 8, # encoder5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
			self.ngf * 8, # encoder6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
			self.ngf * 8, # encoder7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
			self.ngf * 8, # encoder8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
		]       
		layer_specs_in = [
			self.ngf, # encoder2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
			self.ngf * 2, # encoder3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
			self.ngf * 4, # encoder4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
			self.ngf * 8, # encoder5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
			self.ngf * 8, # encoder6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
			self.ngf * 8, # encoder7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
			self.ngf * 8, # encoder8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
		] 
		for i in range(8):
			if i==0:
				# encoder_1
				setattr(self,'encoder'+str(i+1), nn.Conv2d(self.input_nc, self.ngf, 4, 2, 1, bias=False))
			else:
				# encoder_2 - encoder_8
				setattr(self,'encoder'+str(i+1), nn.Conv2d(layer_specs_in[i-1], layer_specs_out[i-1], 4, 2, 1, bias=False))

		### deconv layer
		layer_specs_de = [
			self.ngf * 8,   # decoder8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8]
			self.ngf * 8,   # decoder7: [batch, 2, 2, ngf * 16 ] => [batch, 4, 4, ngf * 8]
			self.ngf * 8,   # decoder6: [batch, 4, 4, ngf * 16 ] => [batch, 8, 8, ngf * 8] #Dropout was 0.5 until here
			self.ngf * 8,   # decoder5: [batch, 8, 8, ngf * 16 ] => [batch, 16, 16, ngf * 8]
			self.ngf * 4,   # decoder4: [batch, 16, 16, ngf * 16 ] => [batch, 32, 32, ngf * 4]
			self.ngf * 2,   # decoder3: [batch, 32, 32, ngf * 8 ] => [batch, 64, 64, ngf * 2]
			self.ngf,       # decoder2: [batch, 64, 64, ngf * 4 ] => [batch, 128, 128, ngf]
			self.ngf,       # decoder1: [batch, 128, 128, ngf * 2 ] => [batch, 256, 256, output_nc1]
		]
		for i in range(8):
			if i==0:
				# decoder_8
				setattr(self,'decoder'+str(8-i), Deconv_des19(layer_specs_de[i], layer_specs_de[i]))
			elif i==7:
				# decoder_1
				setattr(self,'decoder'+str(8-i), Deconv_des19(layer_specs_de[i]*2, self.output_nc1))
			else:
				# decoder_2 - decoder_8
				setattr(self,'decoder'+str(8-i), Deconv_des19(layer_specs_de[i-1]*2, layer_specs_de[i]))

		## instance norm encoder
		layer_specs_innorm_en = [
			self.ngf * 2, # innorm_en2
			self.ngf * 4, # innorm_en3
			self.ngf * 8, # innorm_en4
			self.ngf * 8, # innorm_en5
			self.ngf * 8, # innorm_en6
			self.ngf * 8, # innorm_en7
		]       
		for i in range(6):
			# instance_en 2-7
			setattr(self,'instance_en'+str(i+2), nn.InstanceNorm2d(layer_specs_innorm_en[i],affine=True))

		## instance norm decoder
		layer_specs_innorm_de = [
			self.ngf * 8,   # innorm_de_8
			self.ngf * 8,   # innorm_de_7
			self.ngf * 8,   # innorm_de_6
			self.ngf * 8,   # innorm_de_5 
			self.ngf * 4,   # innorm_de_4 
			self.ngf * 2,   # innorm_de_3
			self.ngf,       # innorm_de_2
		]
		for i in range(7):
			# instance_de 2-8
			setattr(self,'instance_de'+str(8-i), nn.InstanceNorm2d(layer_specs_innorm_de[i],affine=True))


		################ global branch ################
		layer_specs_global_en = [
			self.ngf * 2, # global_fc1 
			self.ngf * 4, # global_fc2 globaltolocal_fc2
			self.ngf * 8, # global_fc3 globaltolocal_fc3
			self.ngf * 8, # global_fc4 globaltolocal_fc4
			self.ngf * 8, # global_fc5 globaltolocal_fc5
			self.ngf * 8, # global_fc6 globaltolocal_fc6
			self.ngf * 8, # global_fc7 globaltolocal_fc7
			self.ngf * 8, # global_fc8 globaltolocal_fc8
		]    
		layer_specs_global_de = [
			self.ngf * 8,   # global_de_fc8
			self.ngf * 8,   # global_de_fc7
			self.ngf * 8,   # global_de_fc6
			self.ngf * 8,   # global_de_fc5
			self.ngf * 4,   # global_de_fc4 
			self.ngf * 2,   # global_de_fc3
			self.ngf,       # global_de_fc2
			self.ngf,       # global_de_fc1
		]    

		### encoder ###
		for i in range(8):
			if i==0:
				# global_fc1
				setattr(self,'global_fc'+str(i+1),FC_des19(self.input_nc,layer_specs_global_en[i],True))
			elif i >=3:
				# global_fc4 - 8
				setattr(self,'global_fc'+str(i+1),FC_des19(layer_specs_global_en[i]*2,layer_specs_global_en[i],True))
			else:
				# global_fc2 - 3
				setattr(self,'global_fc'+str(i+1),FC_des19(layer_specs_global_en[i],layer_specs_global_en[i],True))

			if i>0:
				# globaltolocal_fc2-8
				setattr(self,'globaltolocal_fc'+str(i+1),FC_des19(layer_specs_global_en[i-1],layer_specs_global_en[i-1],False))


		### decoder ###
		for i in range(8):
			if i<=3:
				# global_de_fc_8 - 5
				setattr(self,'global_de_fc'+str(8-i),FC_des19(layer_specs_global_de[i]*2,layer_specs_global_de[i],True))
			elif i==7:
				# global_de_fc_1
				setattr(self,'global_de_fc'+str(8-i),FC_des19(layer_specs_global_de[i]*2,layer_specs_global_de[i],True))
			else:
				# global_de_fc_4 - 2
				setattr(self,'global_de_fc'+str(8-i),\
					FC_des19(layer_specs_global_de[i-1]+layer_specs_global_de[i],layer_specs_global_de[i],True))
			
			if i<=3:
				# globaltolocal_de_fc8 - 5
				setattr(self,'globaltolocal_de_fc'+str(8-i),FC_des19(layer_specs_global_de[i],layer_specs_global_de[i],False))
			elif i==7:
				# globaltolocal_de_fc1
				setattr(self,'globaltolocal_de_fc1',FC_des19(layer_specs_global_de[i],self.output_nc1,False))
			else:
				# globaltolocal_de_fc4 - 2
				setattr(self,'globaltolocal_de_fc'+str(8-i),FC_des19(layer_specs_global_de[i-1],layer_specs_global_de[i],False))
		


		######### last layers after maxpool ###############
		## conv
		setattr(self,'lastconv1',nn.Conv2d(64,64, 3, 1, 1, bias=False))
		setattr(self,'lastconv2',nn.Conv2d(64,32, 3, 1, 1, bias=False))
		setattr(self,'lastconv3',nn.Conv2d(32,9, 3, 1, 1, bias=True))

		## instance norm
		setattr(self,'lastinstance1',nn.InstanceNorm2d(64,affine=True))
		setattr(self,'lastinstance2',nn.InstanceNorm2d(32,affine=True))

		## global
		setattr(self,'lastglobal_fc0',FC_des19(128,64,True))
		setattr(self,'lastglobaltolocal_fc0',FC_des19(64,64,False))

		setattr(self,'lastglobal_fc1',FC_des19(128,32,True))
		setattr(self,'lastglobaltolocal_fc1',FC_des19(64,64,False))

		setattr(self,'lastglobal_fc2',FC_des19(64,9,True))
		setattr(self,'lastglobaltolocal_fc2',FC_des19(32,32,False))


		self.dropout = nn.Dropout(0.5)
		self.leaky_relu = nn.LeakyReLU(0.2)

		self.sig = nn.Sigmoid()
		self.tan = nn.Tanh()

		self.selu=nn.SELU()
		self.NoInput = No_in

	def forward(self, input): #[n,c,w,h]

		###### encoder ###########
		encoderList=[]
		GlobalNetwork_FCList=[]
		for i in range(8):
			conv=getattr(self,'encoder'+str(i+1))
			global_fc=getattr(self,'global_fc'+str(i+1))
			if i>0:
				globaltolocal_fc=getattr(self,'globaltolocal_fc'+str(i+1))
			if i>0 and i<7:
				inNorm=getattr(self,'instance_en'+str(i+1))
			# encoder1
			if i==0:
				encoder = conv(input) #local network
				# print("encoder1: ", encoder[0,0,0,0])
				GlobalNetwork_FC = self.selu(global_fc(mymean(input))) #global
				encoderList.append(encoder)
				GlobalNetwork_FCList.append(GlobalNetwork_FC)
			# encoder2-8
			else:
				encoder = conv(self.leaky_relu(encoderList[-1])) #local network
				# GlobalInput=   #global network
				# print('i: ',i, "shape :", GlobalInput.shape)
				GlobalNetwork_FC=self.selu(global_fc(torch.cat((GlobalNetwork_FCList[-1],mymean(encoder)),1))) #global network
				if i==7: # no instance for encoder8
					encoder = encoder+globaltolocal_fc(GlobalNetwork_FCList[-1]) #local 						
				else:
					encoder =inNorm(encoder)+globaltolocal_fc(GlobalNetwork_FCList[-1]) #local 
				encoderList.append(encoder)
				GlobalNetwork_FCList.append(GlobalNetwork_FC)
		
		# print("encoder7: ", encoder[0,0,0,0])

		######## decoder #########
		decoderList=[]
		GlobalNetwork_de_FCList=[]
		for i in range(8):
			deconv=getattr(self,'decoder'+str(8-i))
			global_de_fc=getattr(self,'global_de_fc'+str(8-i))
			globaltolocal_de_fc=getattr(self,'globaltolocal_de_fc'+str(8-i))
			if i<7:
				inNorm=getattr(self,'instance_de'+str(8-i))
			# decoder8
			if i==0:
				decoder = deconv(self.leaky_relu(encoderList[-1]))
				# print("decoder1_1: ", decoder[0,0,0,0])
				# GlobalInput_de=  #global network

				GlobalNetwork_de_FC=self.selu(global_de_fc(torch.cat((GlobalNetwork_FCList[-1],mymean(decoder)),1) )) #global network
		
				decoder=inNorm(decoder)+globaltolocal_de_fc(GlobalNetwork_FCList[-1]) #local 
				# print(decoder.shape)
				# print(encoderList[6-i].shape)
				# print(encoderList[-1].shape)

				if self.use_dropout:
					decoder = torch.cat((self.dropout(decoder), encoderList[6-i]), 1)
				else:
					decoder = torch.cat((decoder, encoderList[6-i]), 1)

				# print("decoder1_2: ", decoder[0,0,0,0])
				decoderList.append(decoder)
				GlobalNetwork_de_FCList.append(GlobalNetwork_de_FC)
			#decoder 7-1
			else:
				decoder = deconv(self.leaky_relu(decoderList[-1]))
				# GlobalInput_de=   #global network
				GlobalNetwork_de_FC=self.selu(global_de_fc(torch.cat((GlobalNetwork_de_FCList[-1],mymean(decoder)),1))) #global network
				if i==7:
					decoder=decoder+globaltolocal_de_fc(GlobalNetwork_de_FCList[-1]) #local 
				else:
					decoder=inNorm(decoder)+globaltolocal_de_fc(GlobalNetwork_de_FCList[-1]) #local 
					# for dropout 
					if self.use_dropout and i<=2:   
						decoder = torch.cat((self.dropout(decoder), encoderList[6-i]), 1)
					else:
						decoder = torch.cat((decoder, encoderList[6-i]), 1)

					# decoder = torch.cat((decoder, encoderList[6-i]), 1)

				decoderList.append(decoder)
				GlobalNetwork_de_FCList.append(GlobalNetwork_de_FC)

		######### max pool #############
		if decoderList[-1].shape[0] != self.NoInput:
			## seperate batch and No_Input channel
			local_latent = decoderList[-1].view(-1,self.NoInput,decoderList[-1].shape[1],decoderList[-1].shape[2],decoderList[-1].shape[3])
			global_latent = GlobalNetwork_de_FCList[-1].view(-1,self.NoInput,GlobalNetwork_de_FCList[-1].shape[1],GlobalNetwork_de_FCList[-1].shape[2],GlobalNetwork_de_FCList[-1].shape[3])

			maxlatent = torch.max(local_latent,dim=1)[0]
			maxlatent_global = torch.max(global_latent,dim=1)[0]
		else:
			maxlatent = torch.max(decoderList[-1],dim=0)[0].unsqueeze(0)
			maxlatent_global = torch.max(GlobalNetwork_de_FCList[-1],dim=0)[0].unsqueeze(0)


		###### last conv ###########
		LastConvList=[]
		Last_GlobalNetwork_FCList=[]
		for i in range(4):
			if i>=1:
				conv = getattr(self,'lastconv'+str(i))
			if i<=2:
				global_fc = getattr(self,'lastglobal_fc'+str(i))
				globaltolocal_fc = getattr(self,'lastglobaltolocal_fc'+str(i))
			if i==1 or i==2:
				inNorm = getattr(self,'lastinstance'+str(i))

			if i==0:
				# last_GlobalInput =   #global network
				last_GlobalNetwork_FC = self.selu(global_fc(torch.cat((maxlatent_global,mymean(maxlatent)),1) )) #global network
				last_conv = maxlatent+globaltolocal_fc(maxlatent_global) #local 
				
				LastConvList.append(last_conv)
				Last_GlobalNetwork_FCList.append(last_GlobalNetwork_FC)

			elif i==1 or i==2:
				last_conv = conv(LastConvList[-1]) #local network

				# last_GlobalInput =   #global network
				# print('last_GlobalInput: ', last_GlobalInput.shape)
				last_GlobalNetwork_FC = self.selu(global_fc(torch.cat((Last_GlobalNetwork_FCList[-1],mymean(last_conv)),1) )) #global network
				# print('last_GlobalNetwork_FC: ', last_GlobalNetwork_FC.shape)
				last_conv = inNorm(last_conv)+globaltolocal_fc(Last_GlobalNetwork_FCList[-1]) #local 
				
				LastConvList.append(self.leaky_relu(last_conv))
				Last_GlobalNetwork_FCList.append(last_GlobalNetwork_FC)

			elif i==3:
				
				LastConvList.append(self.tan(conv(LastConvList[-1])))


		return LastConvList[-1]


##########################################################################################################################
##########################################################################################################################
######################################## This is for Meta Learning #######################################################
##########################################################################################################################
##########################################################################################################################

#####################################################################################
################################ des19 network ######################################
#####################################################################################


## https://github.com/tristandeleu/pytorch-meta/blob/389e35ef9aa812f07ce50a3f3bd253c4efb9765c/torchmeta/modules/linear.py#
import torch.nn.functional as F

from collections import OrderedDict

class MetaLinear(nn.Linear):
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.linear(input, params['weight'], bias)

class MetaConv2d(nn.Conv2d):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)

        return F.conv2d(input, params['weight'], bias, self.stride,self.padding)

from torch.nn.modules.instancenorm import _InstanceNorm

class _MetaInstanceNorm(_InstanceNorm):
    def forward(self, input, params=None):
        self._check_input_dim(input)
        if params is None:
            params = OrderedDict(self.named_parameters())

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # print(self.track_running_stats)
        # print(self.track_running_stats)
        # print(self.track_running_stats)
        # print(self.track_running_stats)
        # print(self.track_running_stats)

        weight = params.get('weight', None)
        bias = params.get('bias', None)

        return F.instance_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

class MetaInstanceNorm2d(_MetaInstanceNorm):
    __doc__ = nn.InstanceNorm2d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

## the input of FC function is 4D channel [b,c,1,1], output is 4D: [b,c,1,1]
class MetaFC_des19(nn.Module):
	def __init__(self,input_channel,output_channel,BIAS):
		super(MetaFC_des19,self).__init__()
		self.fc_layer=MetaLinear(input_channel,output_channel,bias=BIAS)

	def forward(self,input,params=None):
		# if input is 4D [b,c,1,1],output [b,c,1,1]
		if len(input.shape) == 4:
			[b,c,w,h]=input.shape
			out=self.fc_layer(input.view(b,c),params)
			out=out.unsqueeze(2).unsqueeze(2)
		# if input is 2D [b,c] otuput [b,c]
		elif len(input.shape) == 2:
			out=self.fc_layer(input,params)
		# otherwise, error
		else:
			print('incorrectly input to FC layer')
			sys.exit(1) 


		return out

# self-define deconv layer for des19
class MetaDeconv_des19(BaseModel):
	def __init__(self,input_channel,output_channel):
		super(MetaDeconv_des19,self).__init__()
		## upsampling method (non-deterministic in pytorch)
		# self.upsampling=nn.Upsample(scale_factor=2, mode='nearest')

		self.conv1 = MetaConv2d(input_channel, output_channel, kernel_size=4, stride=1, padding=0, bias=False)
		self.conv2 = MetaConv2d(output_channel, output_channel, kernel_size=4, stride=1, padding=0, bias=False)

		# realize same padding in tensorflow
		self.padding=nn.ConstantPad2d((1, 2, 1, 2), 0)
		self._children_modules_parameters_cache = dict()

	def forward(self,input,index,params=None):

		## hack upsampling method to make is deterministic
		Upsamp = input[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(input.size(0), input.size(1), input.size(2)*2, input.size(3)*2)

		out=self.conv1(self.padding(Upsamp), params = self.get_subdict(params, 'netG.decoder{}.conv1'.format(8-index)))
		out=self.conv2(self.padding(out), params = self.get_subdict(params, 'netG.decoder{}.conv2'.format(8-index)))

		return out

class MetaNet_Des19(BaseModel):
	def __init__(self,opt):
		super(MetaNet_Des19,self).__init__()

		self.ngf = 64
		self.output_nc1 = 64
		self.output_nc2 = 10
		self.input_nc = 5
		self.use_dropout = opt.des19_dropout

		self._children_modules_parameters_cache = dict()

		######## define local networks ##################
		### conv layers
		layer_specs_out = [
			self.ngf * 2, # encoder2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
			self.ngf * 4, # encoder3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
			self.ngf * 8, # encoder4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
			self.ngf * 8, # encoder5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
			self.ngf * 8, # encoder6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
			self.ngf * 8, # encoder7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
			self.ngf * 8, # encoder8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
		]       
		layer_specs_in = [
			self.ngf, # encoder2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
			self.ngf * 2, # encoder3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
			self.ngf * 4, # encoder4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
			self.ngf * 8, # encoder5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
			self.ngf * 8, # encoder6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
			self.ngf * 8, # encoder7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
			self.ngf * 8, # encoder8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
		] 
		for i in range(8):
			if i==0:
				# encoder_1
				setattr(self,'encoder'+str(i+1), MetaConv2d(self.input_nc, self.ngf, kernel_size=4, stride=2, padding=1, bias=False))
			else:
				# encoder_2 - encoder_8
				setattr(self,'encoder'+str(i+1), MetaConv2d(layer_specs_in[i-1], layer_specs_out[i-1], kernel_size=4, stride=2, padding=1, bias=False))

		### deconv layer
		layer_specs_de = [
			self.ngf * 8,   # decoder8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8]
			self.ngf * 8,   # decoder7: [batch, 2, 2, ngf * 16 ] => [batch, 4, 4, ngf * 8]
			self.ngf * 8,   # decoder6: [batch, 4, 4, ngf * 16 ] => [batch, 8, 8, ngf * 8] #Dropout was 0.5 until here
			self.ngf * 8,   # decoder5: [batch, 8, 8, ngf * 16 ] => [batch, 16, 16, ngf * 8]
			self.ngf * 4,   # decoder4: [batch, 16, 16, ngf * 16 ] => [batch, 32, 32, ngf * 4]
			self.ngf * 2,   # decoder3: [batch, 32, 32, ngf * 8 ] => [batch, 64, 64, ngf * 2]
			self.ngf,       # decoder2: [batch, 64, 64, ngf * 4 ] => [batch, 128, 128, ngf]
			self.ngf,       # decoder1: [batch, 128, 128, ngf * 2 ] => [batch, 256, 256, output_nc1]
		]
		for i in range(8):
			if i==0:
				# decoder_8
				setattr(self,'decoder'+str(8-i), MetaDeconv_des19(layer_specs_de[i], layer_specs_de[i]))
			elif i==7:
				# decoder_1
				setattr(self,'decoder'+str(8-i), MetaDeconv_des19(layer_specs_de[i]*2, self.output_nc1))
			else:
				# decoder_2 - decoder_8
				setattr(self,'decoder'+str(8-i), MetaDeconv_des19(layer_specs_de[i-1]*2, layer_specs_de[i]))

		## instance norm encoder
		layer_specs_innorm_en = [
			self.ngf * 2, # innorm_en2
			self.ngf * 4, # innorm_en3
			self.ngf * 8, # innorm_en4
			self.ngf * 8, # innorm_en5
			self.ngf * 8, # innorm_en6
			self.ngf * 8, # innorm_en7
		]       
		for i in range(6):
			# instance_en 2-7
			setattr(self,'instance_en'+str(i+2), MetaInstanceNorm2d(layer_specs_innorm_en[i],affine=True))

		## instance norm decoder
		layer_specs_innorm_de = [
			self.ngf * 8,   # innorm_de_8
			self.ngf * 8,   # innorm_de_7
			self.ngf * 8,   # innorm_de_6
			self.ngf * 8,   # innorm_de_5 
			self.ngf * 4,   # innorm_de_4 
			self.ngf * 2,   # innorm_de_3
			self.ngf,       # innorm_de_2
		]
		for i in range(7):
			# instance_de 2-8
			setattr(self,'instance_de'+str(8-i), MetaInstanceNorm2d(layer_specs_innorm_de[i],affine=True))


		################ global branch ################
		layer_specs_global_en = [
			self.ngf * 2, # global_fc1 
			self.ngf * 4, # global_fc2 globaltolocal_fc2
			self.ngf * 8, # global_fc3 globaltolocal_fc3
			self.ngf * 8, # global_fc4 globaltolocal_fc4
			self.ngf * 8, # global_fc5 globaltolocal_fc5
			self.ngf * 8, # global_fc6 globaltolocal_fc6
			self.ngf * 8, # global_fc7 globaltolocal_fc7
			self.ngf * 8, # global_fc8 globaltolocal_fc8
		]    
		layer_specs_global_de = [
			self.ngf * 8,   # global_de_fc8
			self.ngf * 8,   # global_de_fc7
			self.ngf * 8,   # global_de_fc6
			self.ngf * 8,   # global_de_fc5
			self.ngf * 4,   # global_de_fc4 
			self.ngf * 2,   # global_de_fc3
			self.ngf,       # global_de_fc2
			self.ngf,       # global_de_fc1
		]    

		### encoder ###
		for i in range(8):
			if i==0:
				# global_fc1
				setattr(self,'global_fc'+str(i+1),MetaFC_des19(self.input_nc,layer_specs_global_en[i],True))
			elif i >=3:
				# global_fc4 - 8
				setattr(self,'global_fc'+str(i+1),MetaFC_des19(layer_specs_global_en[i]*2,layer_specs_global_en[i],True))
			else:
				# global_fc2 - 3
				setattr(self,'global_fc'+str(i+1),MetaFC_des19(layer_specs_global_en[i],layer_specs_global_en[i],True))

			if i>0:
				# globaltolocal_fc2-8
				setattr(self,'globaltolocal_fc'+str(i+1),MetaFC_des19(layer_specs_global_en[i-1],layer_specs_global_en[i-1],False))


		### decoder ###
		for i in range(8):
			if i<=3:
				# global_de_fc_8 - 5
				setattr(self,'global_de_fc'+str(8-i),MetaFC_des19(layer_specs_global_de[i]*2,layer_specs_global_de[i],True))
			elif i==7:
				# global_de_fc_1
				setattr(self,'global_de_fc'+str(8-i),MetaFC_des19(layer_specs_global_de[i]*2,layer_specs_global_de[i],True))
			else:
				# global_de_fc_4 - 2
				setattr(self,'global_de_fc'+str(8-i),MetaFC_des19(layer_specs_global_de[i-1]+layer_specs_global_de[i],layer_specs_global_de[i],True))
			
			if i<=3:
				# globaltolocal_de_fc8 - 5
				setattr(self,'globaltolocal_de_fc'+str(8-i),MetaFC_des19(layer_specs_global_de[i],layer_specs_global_de[i],False))
			elif i==7:
				# globaltolocal_de_fc1
				setattr(self,'globaltolocal_de_fc1',MetaFC_des19(layer_specs_global_de[i],self.output_nc1,False))
			else:
				# globaltolocal_de_fc4 - 2
				setattr(self,'globaltolocal_de_fc'+str(8-i),MetaFC_des19(layer_specs_global_de[i-1],layer_specs_global_de[i],False))
		


		######### last layers after maxpool ###############
		## conv
		setattr(self,'lastconv1',MetaConv2d(64,64, 3, 1, 1, bias=False))
		setattr(self,'lastconv2',MetaConv2d(64,32, 3, 1, 1, bias=False))
		setattr(self,'lastconv3',MetaConv2d(32,9, 3, 1, 1, bias=True))

		## instance norm
		setattr(self,'lastinstance1',MetaInstanceNorm2d(64,affine=True))
		setattr(self,'lastinstance2',MetaInstanceNorm2d(32,affine=True))

		## global
		setattr(self,'lastglobal_fc0',MetaFC_des19(128,64,True))
		setattr(self,'lastglobaltolocal_fc0',MetaFC_des19(64,64,False))

		setattr(self,'lastglobal_fc1',MetaFC_des19(128,32,True))
		setattr(self,'lastglobaltolocal_fc1',MetaFC_des19(64,64,False))

		setattr(self,'lastglobal_fc2',MetaFC_des19(64,9,True))
		setattr(self,'lastglobaltolocal_fc2',MetaFC_des19(32,32,False))


		self.dropout = nn.Dropout(0.5)
		self.leaky_relu = nn.LeakyReLU(0.2)

		self.sig = nn.Sigmoid()
		self.tan = nn.Tanh()

		self.selu=nn.SELU()
		# self.NoInput = opt.No_Input

	def forward(self, input, params=None, Noinput=None): #[n,c,w,h]

		###### encoder ###########
		encoderList=[]
		GlobalNetwork_FCList=[]
		for i in range(8):
			conv=getattr(self,'encoder'+str(i+1))
			global_fc=getattr(self,'global_fc'+str(i+1))
			if i>0:
				globaltolocal_fc=getattr(self,'globaltolocal_fc'+str(i+1))
			if i>0 and i<7:
				inNorm=getattr(self,'instance_en'+str(i+1))
			# encoder1
			if i==0:
				encoder = conv(input, params = self.get_subdict(params, 'netG.encoder1')) #local network
				# print("encoder1: ", encoder[0,0,0,0])
				GlobalNetwork_FC = self.selu(global_fc(mymean(input), params = self.get_subdict(params, 'netG.global_fc1.fc_layer'))) #global
				encoderList.append(encoder)
				GlobalNetwork_FCList.append(GlobalNetwork_FC)
			# encoder2-8
			else:
				encoder = conv(self.leaky_relu(encoderList[-1]), params = self.get_subdict(params, 'netG.encoder{}'.format(i+1))) #local network
				# GlobalInput=   #global network
				# print('i: ',i, "shape :", GlobalInput.shape)
				GlobalNetwork_FC=self.selu(global_fc(torch.cat((GlobalNetwork_FCList[-1],mymean(encoder)),1), params = self.get_subdict(params, 'netG.global_fc{}.fc_layer'.format(i+1)))) #global network
				if i==7: # no instance for encoder8
					encoder = encoder+globaltolocal_fc(GlobalNetwork_FCList[-1],params = self.get_subdict(params, 'netG.globaltolocal_fc{}.fc_layer'.format(i+1))) #local 						
				else:
					encoder = inNorm(encoder,params = self.get_subdict(params, 'netG.instance_en{}'.format(i+1)))\
					+globaltolocal_fc(GlobalNetwork_FCList[-1], params = self.get_subdict(params, 'netG.globaltolocal_fc{}.fc_layer'.format(i+1))) #local 
				encoderList.append(encoder)
				GlobalNetwork_FCList.append(GlobalNetwork_FC)
		
		######## decoder #########
		decoderList=[]
		GlobalNetwork_de_FCList=[]
		for i in range(8):
			deconv=getattr(self,'decoder'+str(8-i))
			global_de_fc=getattr(self,'global_de_fc'+str(8-i))
			globaltolocal_de_fc=getattr(self,'globaltolocal_de_fc'+str(8-i))
			if i<7:
				inNorm=getattr(self,'instance_de'+str(8-i))
			# decoder8
			if i==0:
				decoder = deconv(self.leaky_relu(encoderList[-1]),i,params)
				# print("decoder1_1: ", decoder[0,0,0,0])
				# GlobalInput_de=  #global network

				GlobalNetwork_de_FC = self.selu(global_de_fc(torch.cat((GlobalNetwork_FCList[-1],mymean(decoder)),1), params = self.get_subdict(params, 'netG.global_de_fc{}.fc_layer'.format(8-i)))) #global network
		
				decoder = inNorm(decoder, params = self.get_subdict(params, 'netG.instance_de{}'.format(8-i))) \
				+ globaltolocal_de_fc(GlobalNetwork_FCList[-1], params = self.get_subdict(params, 'netG.globaltolocal_de_fc{}.fc_layer'.format(8-i))) #local 
				# print(decoder.shape)
				# print(encoderList[6-i].shape)
				# print(encoderList[-1].shape)

				if self.use_dropout:
					decoder = torch.cat((self.dropout(decoder), encoderList[6-i]), 1)
				else:
					decoder = torch.cat((decoder, encoderList[6-i]), 1)

				# print("decoder1_2: ", decoder[0,0,0,0])
				decoderList.append(decoder)
				GlobalNetwork_de_FCList.append(GlobalNetwork_de_FC)
			#decoder 7-1
			else:
				decoder = deconv(self.leaky_relu(decoderList[-1]),i,params)
				# GlobalInput_de=   #global network
				GlobalNetwork_de_FC = self.selu(global_de_fc(torch.cat((GlobalNetwork_de_FCList[-1],mymean(decoder)),1), params = self.get_subdict(params, 'netG.global_de_fc{}.fc_layer'.format(8-i)))) #global network
				if i==7:
					decoder = decoder + globaltolocal_de_fc(GlobalNetwork_de_FCList[-1], params = self.get_subdict(params, 'netG.globaltolocal_de_fc{}.fc_layer'.format(8-i))) #local 
				else:
					decoder = inNorm(decoder, params = self.get_subdict(params, 'netG.instance_de{}'.format(8-i))) \
					+ globaltolocal_de_fc(GlobalNetwork_de_FCList[-1], params = self.get_subdict(params, 'netG.globaltolocal_de_fc{}.fc_layer'.format(8-i))) #local 
					# for dropout 
					if self.use_dropout and i<=2:   
						decoder = torch.cat((self.dropout(decoder), encoderList[6-i]), 1)
					else:
						decoder = torch.cat((decoder, encoderList[6-i]), 1)

					# decoder = torch.cat((decoder, encoderList[6-i]), 1)

				decoderList.append(decoder)
				GlobalNetwork_de_FCList.append(GlobalNetwork_de_FC)

		######### max pool #############
		# print('shape: ',decoderList[-1].shape)
		if decoderList[-1].shape[0] != Noinput:
			## seperate batch and No_Input channel
			local_latent = decoderList[-1].view(-1,Noinput,decoderList[-1].shape[1],decoderList[-1].shape[2],decoderList[-1].shape[3])
			global_latent = GlobalNetwork_de_FCList[-1].view(-1,Noinput,GlobalNetwork_de_FCList[-1].shape[1],GlobalNetwork_de_FCList[-1].shape[2],GlobalNetwork_de_FCList[-1].shape[3])

			maxlatent = torch.max(local_latent,dim=1)[0]
			maxlatent_global = torch.max(global_latent,dim=1)[0]
		else:
			maxlatent = torch.max(decoderList[-1],dim=0)[0].unsqueeze(0)
			maxlatent_global = torch.max(GlobalNetwork_de_FCList[-1],dim=0)[0].unsqueeze(0)
		# print('shape: ',maxlatent.shape)
		# print('shape: ',maxlatent_global.shape)


		###### last conv ###########
		LastConvList=[]
		Last_GlobalNetwork_FCList=[]
		for i in range(4):
			if i>=1:
				conv = getattr(self,'lastconv'+str(i))
			if i<=2:
				global_fc = getattr(self,'lastglobal_fc'+str(i))
				globaltolocal_fc = getattr(self,'lastglobaltolocal_fc'+str(i))
			if i==1 or i==2:
				inNorm = getattr(self,'lastinstance'+str(i))

			if i==0:
				# last_GlobalInput =   #global network
				last_GlobalNetwork_FC = self.selu(global_fc(torch.cat((maxlatent_global,mymean(maxlatent)),1), params = self.get_subdict(params, 'netG.lastglobal_fc{}.fc_layer'.format(i)))) #global network
				last_conv = maxlatent+globaltolocal_fc(maxlatent_global, params = self.get_subdict(params, 'netG.lastglobaltolocal_fc{}.fc_layer'.format(i))) #local 
				
				LastConvList.append(last_conv)
				Last_GlobalNetwork_FCList.append(last_GlobalNetwork_FC)

			elif i==1 or i==2:
				last_conv = conv(LastConvList[-1], params = self.get_subdict(params, 'netG.lastconv{}'.format(i))) #local network

				# last_GlobalInput =   #global network
				# print('last_GlobalInput: ', last_GlobalInput.shape)
				if i==1:
					last_GlobalNetwork_FC = self.selu(global_fc(torch.cat((Last_GlobalNetwork_FCList[-1],mymean(last_conv)),1), params = self.get_subdict(params, 'netG.lastglobal_fc{}.fc_layer'.format(i)))) #global network
				else:
					last_GlobalNetwork_FC = self.selu(global_fc(torch.cat((Last_GlobalNetwork_FCList[-1],mymean(last_conv)),1))) #last global fc2 are not in the graph

				# print('last_GlobalNetwork_FC: ', last_GlobalNetwork_FC.shape)
				last_conv = inNorm(last_conv,params = self.get_subdict(params, 'netG.lastinstance{}'.format(i)))+globaltolocal_fc(Last_GlobalNetwork_FCList[-1], params = self.get_subdict(params, 'netG.lastglobaltolocal_fc{}.fc_layer'.format(i))) #local 
				
				LastConvList.append(self.leaky_relu(last_conv))
				Last_GlobalNetwork_FCList.append(last_GlobalNetwork_FC)

			elif i==3:
				
				LastConvList.append(self.tan(conv(LastConvList[-1], params = self.get_subdict(params, 'netG.lastconv{}'.format(i)))))


		return LastConvList[-1]

