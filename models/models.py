import numpy as np
import torch
import os
# from util.util import *
from .base_model import BaseModel
from . import networks
from .renderer import *
import matplotlib.pyplot as plt
from PIL import Image
import copy
from torchvision import transforms    


def create_model(opt):
	model = MyModel()
	model.initialize(opt)
	print("model [%s] was created" % (model.name()))
	return model


class MyModel(BaseModel):
	def name(self):
		return 'MyModel'
	
	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		# if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
		# 	torch.backends.cudnn.benchmark = True

		self.isTrain = opt.isTrain
		self.old_lr = opt.lr
		self.Init_Method = opt.Init_Method if not self.opt.Init_train else 'GT'

		################## network structure ############################
		if opt.Net_Option=='MLP':
			path = opt.init_path  
			diff_path = os.path.join(path,'diff.png')  
			normal_path = os.path.join(path,'normal.png')  
			rough_path = os.path.join(path,'rough.png')  
			spec_path = os.path.join(path,'spec.png')  

			Diff = Image.open(diff_path).convert('RGB')
			Normal = Image.open(normal_path).convert('RGB') 
			Rough = Image.open(rough_path).convert('RGB') 
			Spec = Image.open(spec_path).convert('RGB') 

			Diff_Tensor=transforms.ToTensor()(Diff).permute(1,2,0).cuda()**(2.2)
			Normal_Tensor=transforms.ToTensor()(Normal).permute(1,2,0).cuda() 
			Rough_Tensor=transforms.ToTensor()(Rough).permute(1,2,0)[...,0:1].cuda() 
			Spec_Tensor=transforms.ToTensor()(Spec).permute(1,2,0).cuda()**(2.2)

			self.cat_IniFe=torch.cat((Normal_Tensor,Diff_Tensor,Rough_Tensor,Spec_Tensor),dim=2)
			print(self.cat_IniFe.shape)

			self.Init=None
			# Generator network
			self.netG_input_nc = opt.No_Input * opt.input_nc  
 
			self.netG = networks.define_G(self.netG_input_nc, opt.output_nc,opt.Net_Option, gpu_ids=self.gpu_ids)   
			# print('aaa: ',list(self.Init).is_leaf) # True    
			if opt.isTrain:
				self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))                            
		elif opt.Net_Option=='Des19Net' or opt.Net_Option=='UNetS' or opt.Net_Option=='Siren':

			self.Init=None
			# Generator network
			if opt.maxpool:
				self.netG_input_nc = 3  
			else:
				self.netG_input_nc = opt.No_Input * 3  
 
			self.netG = networks.define_G(self.netG_input_nc, opt, gpu_ids=self.gpu_ids) 
			print(self.netG)  
			print('Skip connection: ', opt.SkipCon)
			print('Instance Norm: ', opt.InNorm)
			print('n_layers: ', opt.layers_n)

			if opt.isTrain:
				if opt.freeze:
					self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))   
				else:
					self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))  
		elif opt.Net_Option!='NA':
			if opt.Same_Opti:
				# self.Init= torch.empty((256, 256,10), requires_grad=True, device="cuda")
				# # print(self.Init.is_leaf) # True
				# if opt.Init_Method=='U':
				# 	torch.nn.init.uniform_(self.Init, a=0.0, b=1.0)
				# elif opt.Init_Method=='N':
				# 	torch.nn.init.normal_(self.Init)


				# if opt.Opt_Method=='ADAM':
				# 	self.optimizer_G = torch.optim.Adam([self.Init], lr=opt.lr, betas=(opt.beta1, 0.999))                            
				# else:
				# 	self.optimizer_G = torch.optim.LBFGS([self.Init])     
				print('using same optimizer and init for all pixels or lines')
			else:
				print('using different optimizer and init for all pixels or lines')

		# elif opt.Net_Option=='Siren':
		# 	self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))  

		################### Load Network #############################
		## if manually load
		# if not self.isTrain or opt.continue_train or opt.load_pretrain:
		# 	pretrained_path = '' if not self.isTrain else opt.load_pretrain
		# 	self.load_network(self.netG, 'G', opt.which_iter, pretrained_path)            
		
		## if des19 network, load from npy file
		if opt.Net_Option=='Des19Net' or opt.Net_Option=='UNetS':

			if opt.LoadType=='npy':
				self.des19_npy=np.load(opt.Des19Net_npy_path,allow_pickle=True).item()
				self.LoadDes19Net(self.netG,self.des19_npy)

			elif opt.LoadType=='pth':
				pretrained_path = opt.load_pretrain
				assert pretrained_path!='',"load pretrain should not be empty"
				self.load_network(self.netG, 'G', opt.which_iter, pretrained_path)            
			elif opt.LoadType=='no_init':
				print('no initialization for Des 19 network')
			else:
				raise TypeError('no load type here!!')

			if opt.Load_Optim and opt.inner_Optim=='Adam':
				## if load optimizer state for testing
				if opt.Meta_test:
					self.load_optimizer(self.optimizer_G, 'inner_Optim', opt.which_iter, pretrained_path)            
				## if load optimizer state for training
				else:
					self.load_optimizer(self.optimizer_G, 'Meta_Optim', opt.which_iter, pretrained_path)            

			if opt.Loss_des19:
				self.params=copy.deepcopy(self.netG.state_dict())
				# self.params=copy.copy(self.netG.state_dict())
			self.Positionmap_des19=PositionMap_Des19().cuda(self.gpu_ids[0])

			# [W,H,2] --> [N,2,W,H]
			if opt.optim=='real':
				self.extra_input = self.Positionmap_des19.permute(2,0,1).unsqueeze(0).repeat(opt.No_Input,1,1,1)
				self.extra_input_single = self.Positionmap_des19.permute(2,0,1).unsqueeze(0).repeat(opt.No_Input,1,1,1)
			elif opt.optim=='synreal' or opt.optim=='MGsyn':
				self.extra_input = self.Positionmap_des19.permute(2,0,1).unsqueeze(0).repeat(opt.No_Input*(opt.batchSize+1),1,1,1)
				self.extra_input_single = self.Positionmap_des19.permute(2,0,1).unsqueeze(0).repeat(opt.No_Input,1,1,1)
			elif opt.Meta_train or opt.Meta_test or opt.Finetune_des19:
				self.extra_input = self.Positionmap_des19.permute(2,0,1).unsqueeze(0).unsqueeze(0).repeat(opt.batchSize, opt.No_Input,1,1,1) ## [B,N,2,1,1] as extra input
			else:
				self.extra_input = self.Positionmap_des19.permute(2,0,1).unsqueeze(0).repeat(opt.No_Input,1,1,1)

		################### initialization #############################
		if self.isTrain and not opt.Meta_train and not opt.Meta_test and not opt.Finetune_des19:
			if self.Init_Method=='18Des':
				
				path = './init'  
				diff_path = join(path,'diff.png')  
				normal_path = join(path,'normal.png')  
				rough_path = join(path,'rough.png')  
				spec_path = join(path,'spec.png')  

				Diff = Image.open(diff_path).convert('RGB')
				Normal = Image.open(normal_path).convert('RGB') 
				Rough = Image.open(rough_path).convert('RGB') 
				Spec = Image.open(spec_path).convert('RGB') 

				Diff_Tensor=transforms.ToTensor()(Diff).permute(1,2,0).cuda()**(2.2)
				Normal_Tensor=transforms.ToTensor()(Normal).permute(1,2,0).cuda() 
				Rough_Tensor=transforms.ToTensor()(Rough).permute(1,2,0)[...,0:1].cuda() 
				Spec_Tensor=transforms.ToTensor()(Spec).permute(1,2,0).cuda()**(2.2)

				self.cat_IniFe=torch.cat((Normal_Tensor,Diff_Tensor,Rough_Tensor,Spec_Tensor),dim=2)
				print(self.cat_IniFe.shape)
			elif self.Init_Method=='C':
				mean_diff=torch.mean(TrainData[:,:,:,3:6],dim=[0,1,2])
				mean_spec=torch.mean(TrainData[:,:,:,9:12],dim=[0,1,2])
				mean_norm=torch.mean(TrainData[:,:,:,0:3],dim=[0,1,2])
				mean_rough=torch.mean(TrainData[:,:,:,6:9],dim=[0,1,2])
				
				InitNormal=mean_norm.clone().unsqueeze(0).unsqueeze(0).repeat(256,256,1)
				InitDiff=mean_diff.clone().unsqueeze(0).unsqueeze(0).repeat(256,256,1)
				InitRough=mean_rough.clone().unsqueeze(0).unsqueeze(0).repeat(256,256,1)
				InitSpec=mean_spec.clone().unsqueeze(0).unsqueeze(0).repeat(256,256,1)
				
				# save initilization
				ini_diff_path = os.path.join(input_dir, 'initi_diff.jpg')
				ini_norm_path = os.path.join(input_dir, 'initi_norm.jpg')
				ini_rough_path = os.path.join(input_dir, 'initi_rough.jpg')
				ini_spec_path = os.path.join(input_dir, 'initi_spec.jpg')

				In_save_norm=util.tensor2im(InitNormal, gamma=False)
				In_save_diff=util.tensor2im(InitDiff, gamma=True)
				In_save_rough=util.tensor2im(InitRough, gamma=False)
				In_save_spec=util.tensor2im(InitSpec, gamma=True)

				util.save_image(In_save_norm, ini_norm_path)		
				util.save_image(In_save_diff, ini_diff_path)		
				util.save_image(In_save_rough, ini_rough_path)		
				util.save_image(In_save_spec, ini_spec_path)		

				mean_ini=torch.cat((mean_norm,mean_diff,mean_rough[0:1],mean_spec))
			elif self.Init_Method=='19Des':
				init_path = os.path.join(opt.init_path,'{}.png'.format(opt.fileNo)) 
				fullimage = Image.open(init_path).convert('RGB')
				w, h = fullimage.size 

				w5 = int(w / 4)
				Normal = fullimage.crop((0, 0, w5, h))    
				Diff = fullimage.crop((w5, 0, 2*w5, h))    
				Rough = fullimage.crop((2*w5, 0, 3*w5, h))    
				Spec = fullimage.crop((3*w5, 0, 4*w5, h)) 
				Diff_Tensor=transforms.ToTensor()(Diff).permute(1,2,0).cuda()**(2.2)
				Normal_Tensor=transforms.ToTensor()(Normal).permute(1,2,0).cuda() 
				Rough_Tensor=transforms.ToTensor()(Rough).permute(1,2,0)[...,0:1].cuda() 
				Spec_Tensor=transforms.ToTensor()(Spec).permute(1,2,0).cuda()**(2.2)

				self.cat_IniFe=torch.cat((Normal_Tensor,Diff_Tensor,Rough_Tensor,Spec_Tensor),dim=2)
				print(self.cat_IniFe.shape)
			elif self.Init_Method=='GT':
				print('have GT feature maps')
		else:
			print('meta learning or finetuning')

		if self.opt.loss_type=='L1':
			print('use L1 loss')			
			self.criterion = torch.nn.L1Loss()
		elif self.opt.loss_type=='L2':
			print('use L2 loss')
			self.criterion = torch.nn.MSELoss()		

		if self.opt.VGGloss:
			print('use vgg loss')
			self.criterionVGG = networks.VGGLoss(self.gpu_ids[0])
		if self.opt.noL1Loss:
			print('no L1 loss')

		self.Position_map=PositionMap(256,256,3).cuda(self.gpu_ids[0])
		# self.Position_map_int=PositionMap_int(256,256,3)

		self.normal_MEAN = torch.tensor([0.5,0.5,1]).repeat(1,256,256,1).cuda()

		if opt.gaussian_L1:
			self.GauFilter, self.padding=Gaussian_filter(kernel_size=35,sigma=10)

	def forward(self ,in_image_linear, gt_fe, LightPos, steps, IO=None, params=None, input_coords=None):

		if in_image_linear.dim()==4:
			L_No,C,W,H=in_image_linear.shape
			Batch=1
			In_dim=4
			in_image_linear = in_image_linear.unsqueeze(0)
		elif in_image_linear.dim()==5:
			In_dim=5
			Batch,L_No,C,W,H=in_image_linear.shape

		if self.opt.Meta_train or self.opt.Meta_test or self.opt.Finetune_des19:
			assert IO is not None, "meta learning should specify in or out"

			### lineat to log domain for input; [0,1] -> [-1,1]
			in_image=logTensor(torch.clamp(in_image_linear,0,1))*2-1

			if self.opt.Net_Option=='Des19Net':
				extra_input=self.extra_input
				net_in = torch.cat((in_image,extra_input),dim=2) 
			else:
				net_in = in_image

			if Batch==1:
				net_in=net_in.squeeze(0) # [1,N,5,W,H] -> [N,5, W,H]
			else:
				net_in=net_in.reshape(Batch*L_No,-1,256,256) # [B,N,5,W,H] -> [B*N, 5, W,H]

			if self.opt.Net_Option =='Siren':
				# print(input_coords[0:4,...])
				Net_Out = self.netG.forward(input_coords, params)
				Net_Out = Net_Out.view(256,256,self.opt.output_nc).unsqueeze(0)
			else:
				Net_Out = self.netG.forward(net_in,params,L_No).permute(0,2,3,1)  # [B, 9, W,H] -> [B,W,H,9]
				Net_Out=process(Net_Out)# [-1,1] -> [0,1]

			loss_total = 0

			# 9 channels and des19 net
			if self.opt.Net_Option=='Des19Net':
				out_normal=Net_Out[...,0:2]
				out_diff=Net_Out[...,2:5]
				out_rough=Net_Out[...,5:6]
				out_spec=Net_Out[...,6:9]
				out_normal=Process_des19normal(out_normal, self.opt.gpu_ids[0]) #[0,1] --> [-1,1] normalize
			# 10 channels and other network
			else:
				if self.opt.debug_type=='albedo':
					out_diff=Net_Out
					cat_fe_out = out_diff
				else:
					if self.opt.use_height:
						out_normal=Net_Out[...,0:1]
						out_diff=Net_Out[...,1:4]
						out_rough=Net_Out[...,4:5]
						out_spec=Net_Out[...,5:8]					
					else:
						out_normal=Net_Out[...,0:3]
						out_diff=Net_Out[...,3:6]
						out_rough=Net_Out[...,6:7]
						out_spec=Net_Out[...,7:10]
					out_normal = ProcessNormal(out_normal)
					cat_fe_out = torch.cat((out_normal,out_diff,out_rough,out_spec),dim=-1) #[B,W,H,10]

			if not self.opt.debug:
				Render_Fake=Batchrender_NumberPointLight_FixedCamera(out_diff, out_spec, out_normal, out_rough, LightPos, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi) #[B,N,W,H,C]
				Render_Fake=logTensor(torch.clamp(Render_Fake,0,1))*2-1

			# print('Render_Fake: ',Render_Fake.shape)
			#### outer loss to train meta  
			if IO=='O':
				### feature maps loss
				gt_normal = gt_fe[...,0:3]
				gt_diff = gt_fe[...,3:6]
				gt_rough = gt_fe[...,6:7]
				gt_spec = gt_fe[...,7:10]	

				if self.opt.debug_type=='albedo':

					loss_Fe_O = self.criterion(gt_diff,out_diff)
				else:
					loss_Fe_O = self.criterion(gt_normal,out_normal) + self.criterion(gt_diff,out_diff) + self.criterion(gt_rough,out_rough) + self.criterion(gt_spec,out_spec)
				loss_Fe_O = loss_Fe_O * 0.25 * self.opt.lambda_syn
				# loss_Fe_O = self.criterion(cat_fe_out, gt_fe) * self.opt.lambda_syn 

				### rerender image loss
				loss_Ren_O = 0
				loss_RenVGG_O = torch.tensor([0.0])
				loss_RenL1_O = torch.tensor([0.0])

				if self.opt.renderloss_O:
					LightPos_O = Create_NumberPointLightPosition_VaryDist(1, self.opt.rand_light, self.opt.gpu_ids[0], self.opt)
					if In_dim == 5:
						gtRen_O = Batchrender_NumberPointLight_FixedCamera(gt_diff, gt_spec, gt_normal, gt_rough, LightPos_O, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi) #[B,N, W,H,C]
						OutRen_O = Batchrender_NumberPointLight_FixedCamera(out_diff, out_spec, out_normal, out_rough, LightPos_O, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi) #[B,N, W,H,C]
					elif In_dim == 4:
						gtRen_O = SingleRender_NumberPointLight_FixedCamera(gt_diff, gt_spec, gt_normal, gt_rough, LightPos_O, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi) #[B, W,H,C]
						OutRen_O = SingleRender_NumberPointLight_FixedCamera(out_diff, out_spec, out_normal, out_rough, LightPos_O, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi) #[B, W,H,C]
					
					# LDR: 2.2 correction;
					gtRen_O=logTensor(torch.clamp(gtRen_O,0,1))
					OutRen_O=logTensor(torch.clamp(OutRen_O,0,1))

					## L1 loss
					loss_RenL1_O = self.criterion(OutRen_O, gtRen_O)*self.opt.lambda_L1

					## VGG loss
					if self.opt.VGGloss:
						if In_dim==4:
							out_vg = VGGpreprocess(OutRen_O.permute(0,3,1,2))	
							gt_vg = VGGpreprocess(gtRen_O.permute(0,3,1,2))
						elif In_dim==5:
							out_vg = VGGpreprocess(OutRen_O[0,...].permute(0,3,1,2))	
							gt_vg = VGGpreprocess(gtRen_O[0,...].permute(0,3,1,2))
							# for j in range(Batch):
							# 	out_vg = VGGpreprocess(OutRen_O[j,...].permute(0,3,1,2))	
							# 	gt_vg = VGGpreprocess(gtRen_O[j,...].permute(0,3,1,2))
							# 	loss_RenVGG_O += self.criterionVGG(out_vg,gt_vg )*self.opt.lambda_vgg
							# loss_RenVGG_O=loss_RenVGG_O/Batch

						if self.opt.fp16:
							out_vg = out_vg.half()
							gt_vg = gt_vg.half()

						loss_RenVGG_O = self.criterionVGG(out_vg, gt_vg)*self.opt.lambda_vgg

					loss_Ren_O = (loss_RenL1_O+loss_RenVGG_O)*0.5*self.opt.lambda_renloss_O
				
				loss_total += loss_Fe_O+loss_Ren_O
				# print('outer loss: ', loss_total)

			#### inner loss for optimization and finetuning 
			if IO=='I':
				## L1 loss
				if not self.opt.debug:
					assert Render_Fake.shape[0]==1, "batch size error in the inner loop, should be 1"
					loss_Re_L1=0
					if not self.opt.noL1Loss:
						loss_Re_L1 =  self.criterion( Render_Fake[0,...].permute(0,3,1,2), in_image[0,...])*self.opt.lambda_L1
					loss_total += loss_Re_L1

					# VGG loss
					loss_Re_vgg=0
					if self.opt.VGGloss:
						# [-1,1]- > [0,1] VGG normalize
						fake_vg=VGGpreprocess(process(Render_Fake[0,...].permute(0,3,1,2)))	
						real_vg=VGGpreprocess(process(in_image[0,...]))

						loss_Re_vgg = self.criterionVGG(fake_vg,real_vg )*self.opt.lambda_vgg
					loss_total += loss_Re_vgg
				else:
					gt_normal = gt_fe[...,0:3]
					gt_diff = gt_fe[...,3:6]
					gt_rough = gt_fe[...,6:7]
					gt_spec = gt_fe[...,7:10]	
					# print(gt_diff.shape)
					# print(out_diff.shape)
					if self.opt.debug_type=='albedo':
						loss_Fe_I = self.criterion(gt_diff,out_diff)
					else:
						loss_Fe_I = self.criterion(gt_normal,out_normal) + self.criterion(gt_diff,out_diff) + self.criterion(gt_rough,out_rough) + self.criterion(gt_spec,out_spec)
					
					loss_total = loss_Fe_I * 0.25 * self.opt.lambda_syn
					
			Loss={'loss_total':loss_total}

			if not self.opt.debug:
				if IO=='I':
					Output = {'gt_fe':gt_fe,'out_fe':cat_fe_out,'out_re':Render_Fake[0,...],'gt_re':in_image[0,...].permute(0,2,3,1)}
				elif IO=='O':
					Output = {'gt_fe':gt_fe,'out_fe':cat_fe_out,'out_re':Render_Fake[0,...],'gt_re':in_image[0,...].permute(0,2,3,1)}
					Loss.update({'loss_ren_vgg':loss_RenVGG_O*0.5*self.opt.lambda_renloss_O, 'loss_ren_L1':loss_RenL1_O*0.5*self.opt.lambda_renloss_O, 'loss_fe':loss_Fe_O})
					if self.opt.renderloss_O:
						# print(OutRen_O.shape)
						Output.update({'outren_rand':OutRen_O[0,...],'gtren_rand':gtRen_O[0,...]})

				return Loss, Output

			else:
				Output = {'gt_fe':gt_fe,'out_fe':cat_fe_out}

				return Loss, Output



		else:
			if self.opt.Net_Option=='MLP':
				# [B]
				x = torch.randint(0,256,(batch,))
				y = torch.randint(0,256,(batch,))

				# x = torch.arange(128).to(dtype=torch.long).repeat(2)
				# y = torch.arange(2).to(dtype=torch.long).repeat_interleave(128)
				# batch=256

				Des_Fea=self.cat_IniFe[x,y,...]
				# print('Des_Fea: ',Des_Fea)
				#[0,1] -> [-1,1] [B,1,C]
				XY_in=2*torch.cat((y.float().unsqueeze(-1)/255,1.0-x.float().unsqueeze(-1)/255),dim=1).cuda()-1	
				# print('XY_in: ',XY_in.shape)

				zeros=torch.zeros(batch,1).cuda()

				XYZ_in = torch.cat((XY_in,zeros), dim=1) #[B,3]

				#[L,3,W,H] -> [B,L,3]
				Render_in=in_image[:,:,x,y].permute(2,0,1)

				if self.opt.input_nc==6:
					# Light dir [L,3] -> [B,L,3]
					L_vec = LightPos.unsqueeze(0).repeat(batch,1,1)-XYZ_in.unsqueeze(1)
					L_vec = normalize_vec(L_vec)
					if self.opt.gamma_in:
						Render_in=(Render_in+EPSILON)**(1/2.2)*2-1
					else:
						Render_in=Render_in*2-1

					Net_in=torch.cat((Render_in,L_vec), dim=2).view(batch,-1)

				elif self.opt.input_nc==9:
						# Light dir [L,3] -> [B,L,3]
						L_vec = LightPos.unsqueeze(0).repeat(batch,1,1)-XYZ_in.unsqueeze(1)
						L_vec = normalize_vec(L_vec)

						# camera dir [3] -> [B,L,3]
						CamPos=torch.tensor([0.,0.,2.14]).cuda() # [B,L,c]
						V_vec = CamPos.unsqueeze(0).unsqueeze(0).repeat(batch,L_No,1)-XYZ_in.unsqueeze(1)
						V_vec = normalize_vec(V_vec)			

						# [B,L,6] -> [B, L*6]
						Render_in=Render_in*2-1
						Net_in=torch.cat((Render_in,L_vec,V_vec), dim=2).view(batch,-1)
						# print('Net_in: ',Net_in.shape)


				# [B,10]
				Net_Out = self.netG.forward(Net_in)
				Net_Out = (Net_Out+1)*0.5

				# print('Net_Out: ',Net_Out)

				# if not self.opt.Normal_XYZ:
				# 	normal_temp = Net_Out[:,0:2]*2-1
				# 	normal = torch.cat((normal_temp,Net_Out[:,2:3]),dim=1)
				# 	# print('a,', self.opt.Normal_XYZ)
				# else:
				# 	normal=Net_Out[:,0:3]
					# print('b,', self.opt.Normal_XYZ)

				# loss_G_L1 =  self.criterionL1(Net_Out[:,0:3],Des_Fea[:,0:3])
				if self.opt.Init_train:
					loss_G_L1 =  self.criterionL1(Net_Out,Des_Fea) 
				else:
					# [B,L,C], [0,1]
					Render_Fake=SingleRender(Net_Out[:,3:6], Net_Out[:,7:10], Net_Out[:,0:3], Net_Out[:,6:7], LightPos, XYZ_in.unsqueeze(1), self.gpu_ids[0], not self.opt.no_CoCamLi)
					if not self.opt.HDR:
						Render_Fake=torch.clamp(Render_Fake,0,1)
					
					if self.opt.gamma_in:
						Render_Fake=(Render_Fake+EPSILON)**(1/2.2)*2-1
					else:
						Render_Fake=Render_Fake*2-1


					# Render_Fake=torch.clamp(Render_Fake,0,1)
					# print('Render_Fake: ',Render_Fake)
					loss_G_L1 =  self.criterionL1(Render_Fake, Render_in) 
				
				return loss_G_L1
			elif self.opt.Net_Option=='Des19Net' or self.opt.Net_Option=='UNetS':

				# LDR: log correction; [0,1] -> [-1,1]
				if not self.opt.no_gamma_in and not self.opt.HDR:
					in_image=logTensor(torch.clamp(in_image_linear,0,1))*2-1
				# HDR: log correction;
				elif not self.opt.no_gamma_in and self.opt.HDR:
					in_image=torch.log(in_image_linear+EPSILON+1)
				# other:[0,1] -> [-1,1]
				else:
					in_image=in_image_linear*2-1 
				
				if self.opt.randinput and not self.opt.Init_train and self.opt.Net_Option!='Des19Net':
					rand_index=torch.randperm(L_No)
					in_image=in_image[rand_index]
					LightPos=LightPos[rand_index]
					# print(rand_index)
				
				###### for debug ########
				if self.opt.Init_train:
					if self.opt.maxpool:
						net_in = in_image # [B,N,C,W,H]
					else:
						net_in=in_image.view(self.opt.batchSize,-1,256,256) # [B,N,C,W,H] -> [B, N*C, W,H]
					# print(net_in.shape)
					Net_Out = self.netG.forward(net_in).permute(0,2,3,1)  # [B, N*C, W,H] -> [B,W,H,N*C]
				else:
					if self.opt.maxpool:
						net_in = in_image.unsqueeze(0) # [1, N,C,W,H]
					elif self.opt.Net_Option=='Des19Net':
						# [N,3,W,H] + [N,2,W,H] = [N,5,W,H]	
						# print('in_image: ', in_image.shape)
						# print('self.extra_input: ', self.extra_input.shape)
						net_in = torch.cat((in_image,self.extra_input),dim=1) 
						# print('input: ', net_in.shape)
					else:
						net_in = in_image.view(-1,256,256).unsqueeze(0) # [N,C,W,H] -> [1, N*C, W,H]

					# print('input shape: ', net_in[0,0,0,0])
					# print('input shape: ', net_in[0,4,0,0])
					Net_Out = self.netG.forward(net_in).permute(0,2,3,1)  # [1, N*C, W,H] -> [1,W,H,N*C]
				
				Net_Out=process(Net_Out)# [-1,1] -> [0,1]

				loss_Init = 0

				loss_Re=0
				loss_dif = 0
				loss_normal = 0
				loss_rough = 0
				loss_spec = 0
				Gaussian = None
				loss_total = 0
				## init training
				if self.opt.Init_train:
					## feature loss
					loss_Fe =  self.criterion(Net_Out, gt_fe) 

					## render loss
					Newlight = Create_NumberPointLightPosition(self.opt.batchSize, self.opt.rand_light, self.opt.gpu_ids[0])
					# [B,W,H,C]
					Render_Fake=SingleRender_NumberPointLight_FixedCamera(Net_Out[...,3:6], Net_Out[...,7:10], Net_Out[...,0:3], Net_Out[...,6:7], Newlight, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
					Render_gt=SingleRender_NumberPointLight_FixedCamera(gt_fe[...,3:6], gt_fe[...,7:10], gt_fe[...,0:3], gt_fe[...,6:7], Newlight, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
					
					# LDR: 2.2 correction; [0,1] -> [-1,1]
					if not self.opt.HDR and not self.opt.no_gamma_in:
						Render_gt=torch.clamp(Render_gt,0,1)
						Render_Fake=torch.clamp(Render_Fake,0,1)
						Render_Fake=logTensor(Render_Fake)*2-1
						Render_gt=logTensor(Render_gt)*2-1
					# HDR: log correction;
					elif self.opt.HDR and not self.opt.no_gamma_in:
						Render_Fake=torch.log(Render_Fake+EPSILON+1)
					else:
						Render_Fake=Render_Fake*2-1

					loss_Re =  self.criterion(Render_Fake, Render_gt) 
					Render = {'gt_render': Render_gt, 'fake_render': Render_Fake}
				
					loss_total =  loss_Re + loss_Fe 

					Loss = {'loss_total': loss_total,'loss_Re': loss_Re, 'loss_Fe': loss_Fe}

					return Loss,Net_Out,net_in,Render
				else:
					if steps < self.opt.init_iter and self.opt.Net_Option !='Des19Net':
						loss_Init =  self.criterion(Net_Out.squeeze(0),self.cat_IniFe) 
						loss_total+=loss_Init
					else:
						##### seperate output to different feature maps #####
						# 9 channels
						if self.opt.Net_Option=='Des19Net':
							out_normal=Net_Out[...,0:2]
							out_diff=Net_Out[...,2:5]
							out_rough=Net_Out[...,5:6]
							out_spec=Net_Out[...,6:9]
							out_normal=Process_des19normal(out_normal, self.opt.gpu_ids[0]) #[0,1] --> [-1,1] normalize
						# 10 channels
						else:
							out_normal=Net_Out[...,0:3]
							out_diff=Net_Out[...,3:6]
							out_rough=Net_Out[...,6:7]
							out_spec=Net_Out[...,7:10]
							out_normal = ProcessNormal(out_normal)

						if self.opt.optim=='real':
							# [B,W,H,C]
							Render_Fake=SingleRender_NumberPointLight_FixedCamera(out_diff, out_spec, out_normal, out_rough, LightPos, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
						elif self.opt.optim=='synreal' or self.opt.optim=='MGsyn':
							Render_Fake=SingleRender_NumberPointLight_FixedCamera(out_diff[0:1,...], out_spec[0:1,...], out_normal[0:1,...], out_rough[0:1,...], LightPos[0:self.opt.No_Input,...], self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
							# Render_Fake = torch.cat((Render_Fake_real,in_image.permute(0,2,3,1)[1:2,...]),dim=0)
							in_image=in_image[0:self.opt.No_Input,...]

						# syn 
						loss_Fe_L1 = 0
						if self.opt.optim == 'synreal'or self.opt.optim=='MGsyn':
							cat_fe_out = torch.cat((out_normal[1:2,...],out_diff[1:2,...],out_rough[1:2,...],out_spec[1:2,...]),dim=-1) #[B,W,H,10]
							loss_Fe_L1 = self.criterion(cat_fe_out, gt_fe)*self.opt.lambda_syn 
						loss_total += loss_Fe_L1

						diff_loss = 0
						# if False:
						# 	diff_loss = torch.exp(-self.criterion(out_diff[0:1,...], in_image.permute(0,2,3,1)))
						# 	loss_total += diff_loss

						# LDR: 2.2 correction; [0,1] -> [-1,1]
						if not self.opt.HDR and not self.opt.no_gamma_in:
							Render_Fake=logTensor(torch.clamp(Render_Fake,0,1))*2-1
						# HDR: log correction;
						elif self.opt.HDR and not self.opt.no_gamma_in:
							Render_Fake=torch.log(Render_Fake+EPSILON+1)
						else:
							Render_Fake=Render_Fake*2-1

						## L1 loss
						loss_Re_L1=0
						if not self.opt.noL1Loss:
							if self.opt.gaussian_L1:
								# [-1,1] -> [0,1] Gaussian filter
								RenderFake_Gaussian = self.GauFilter(self.padding(process(Render_Fake.permute(0,3,1,2))))
								in_image_Gaussian = self.GauFilter(self.padding(process(in_image)))
								loss_Re_L1 =  self.criterion( RenderFake_Gaussian, in_image_Gaussian)*self.opt.lambda_L1	

								# print('before in: ',in_image)
								# print('after in: ',in_image_Gaussian)
								# print('before render: ',Render_Fake.shape)
								# print('after render: ',RenderFake_Gaussian.shape)

								# plt.figure(0)
								# imshow(tensorNormalize(in_image).permute(0,2,3,1).cpu())
								# plt.figure(1)
								# imshow(in_image_Gaussian.permute(0,2,3,1).cpu())
								# plt.figure(2)
								# imshow(Render_Fake.detach().cpu())
								# plt.figure(3)
								# imshow(RenderFake_Gaussian.permute(0,2,3,1).detach().cpu())
								# plt.show()						
								Gaussian = {'Fake':RenderFake_Gaussian.detach().cpu(),'Real':in_image_Gaussian.cpu()} #[B,C,W,H]
							else:
								loss_Re_L1 =  self.criterion( Render_Fake, in_image.permute(0,2,3,1))*self.opt.lambda_L1
						loss_total += loss_Re_L1

						## VGG loss
						loss_Re_vgg=0
						if self.opt.VGGloss and steps>=self.opt.iter1_VGG:
							## decaying VGG weight or not
							if self.opt.decay_lambda_VGG:
								lambda_vgg = self.opt.lambda_vgg * (steps - self.opt.iter1_VGG)/(self.opt.iter2_VGG - self.opt.iter1_VGG) if steps <= self.opt.iter2_VGG else self.opt.lambda_vgg
							else:
								lambda_vgg = self.opt.lambda_vgg
							# [-1,1]- > [0,1] VGG normalize
							fake_vg=VGGpreprocess(process(Render_Fake.permute(0,3,1,2)))	
							real_vg=VGGpreprocess(process(in_image))
							loss_Re_vgg = self.criterionVGG(fake_vg,real_vg )*lambda_vgg
						loss_total += loss_Re_vgg

						## Loss for Des19
						Loss_des19=0
						if self.opt.Loss_des19:
							Lambda_Loss_des19 = self.opt.lambda_des19loss*torch.exp(torch.tensor([-self.opt.No_Input*1.0])).cuda()
							Loss_des19 = Lambda_Loss_des19*self.LossDes19Net(self.netG,self.params)
						loss_total += Loss_des19

						## SA loss
						loss_SA=0
						Out_SA=None
						if self.opt.SA and steps >= self.opt.iter_beforeSA:
							# print('SA')
							NewLightPos = Create_NumberPointLightPosition_SA(self.opt.No_Input, self.opt.rand_light, self.opt.gpu_ids[0])
							# [N,W,H,C]
							if self.opt.optim=='synreal'or self.opt.optim=='MGsyn':
								Render_Fake2=SingleRender_NumberPointLight_FixedCamera(out_diff[0:1,...], out_spec[0:1,...], out_normal[0:1,...], out_rough[0:1,...], NewLightPos, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
							else:	
								Render_Fake2=SingleRender_NumberPointLight_FixedCamera(out_diff, out_spec, out_normal, out_rough, NewLightPos, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)

							# [N,C,W,H]
							Render_Fake2=Render_Fake2.permute(0,3,1,2)

							# LDR: 2.2 correction; [0,1] -> [-1,1]
							if not self.opt.HDR and not self.opt.no_gamma_in:
								Render_Fake2=torch.clamp(Render_Fake2,0,1)
								Render_Fake2=logTensor(Render_Fake2)*2-1
							# HDR: log correction;
							elif self.opt.HDR and not self.opt.no_gamma_in:
								Render_Fake2=torch.log(Render_Fake2+EPSILON+1)
							else:
								Render_Fake2=Render_Fake2*2-1

							# if self.opt.detach is True:
							# 	SA_in = Render_Fake2.view(-1,256,256).unsqueeze(0).detach()
							# else:
							# 	SA_in = Render_Fake2.view(-1,256,256).unsqueeze(0)

							if self.opt.maxpool:
								SA_in = Render_Fake2.unsqueeze(0) # [1, N,C,W,H]
							elif self.opt.Net_Option=='Des19Net':
								# [N,3,W,H] + [N,2,W,H] = [N,5,W,H]					
								SA_in = torch.cat((Render_Fake2,self.extra_input_single),dim=1) 
								# print('input: ', SA_in.shape)							
							else:
								SA_in = Render_Fake2.view(-1,256,256).unsqueeze(0) # [N,C,W,H] -> [1, N*C, W,H]

							Net_Out2 = self.netG.forward(SA_in).permute(0,2,3,1)  # [B,C,W,H] -> [B,W,H,C]
							Net_Out2=process(Net_Out2)# [-1,1] -> [0,1]

							net_in = SA_in
							if self.opt.optim=='synreal'or self.opt.optim=='MGsyn':							
								loss_SA = self.opt.lambda_SA * self.criterion(Net_Out2, Net_Out[0:1,...]) 						
							else:
								loss_SA = self.opt.lambda_SA * self.criterion(Net_Out2, Net_Out) 
							
							Render_Loss_SA = 0
							if self.opt.SA_render:
								# print('SA Render')
								##### seperate output to different feature maps #####
								# 9 channels
								if self.opt.Net_Option=='Des19Net':
									out_normal_SA=Net_Out2[...,0:2]
									out_diff_SA=Net_Out2[...,2:5]
									out_rough_SA=Net_Out2[...,5:6]
									out_spec_SA=Net_Out2[...,6:9]
									out_normal_SA=Process_des19normal(out_normal_SA, self.opt.gpu_ids[0])
								# 10 channels
								else:
									out_normal_SA=Net_Out2[...,0:3]
									out_diff_SA=Net_Out2[...,3:6]
									out_rough_SA=Net_Out2[...,6:7]
									out_spec_SA=Net_Out2[...,7:10]
									out_normal_SA = ProcessNormal(out_normal_SA)

								Render_Fake_SA=SingleRender_NumberPointLight_FixedCamera(out_diff_SA, out_spec_SA, out_normal_SA, out_rough_SA, LightPos, \
																						self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)

								# LDR: 2.2 correction; [0,1] -> [-1,1]
								if not self.opt.HDR and not self.opt.no_gamma_in:
									Render_Fake_SA=torch.clamp(Render_Fake_SA,0,1)
									Render_Fake_SA=logTensor(Render_Fake_SA)*2-1

								if not self.opt.noL1Loss:
							
									if self.opt.gaussian_L1:
										# [-1,1] -> [0,1] Gaussian filter
										RenderFake_Gaussian_SA = self.GauFilter(self.padding(process(Render_Fake_SA.permute(0,3,1,2))))
										in_image_Gaussian = self.GauFilter(self.padding(process(in_image)))
										loss_Re_L1_SA =  self.criterion( RenderFake_Gaussian_SA, in_image_Gaussian)*self.opt.lambda_L1	
							
										# Gaussian = {'Fake':RenderFake_Gaussian.detach().cpu(),'Real':in_image_Gaussian.cpu()} #[B,C,W,H]
									else:
										loss_Re_L1_SA =  self.criterion( Render_Fake_SA, in_image.permute(0,2,3,1))*self.opt.lambda_L1

								if self.opt.VGGloss and steps>=self.opt.iter1_VGG:
									fake_vg_SA=VGGpreprocess(process(Render_Fake_SA.permute(0,3,1,2)))	
									real_vg=VGGpreprocess(process(in_image))
									loss_Re_vgg_SA = self.criterionVGG(fake_vg_SA,real_vg )*self.opt.lambda_vgg

									Render_Loss_SA = loss_Re_L1_SA + loss_Re_vgg_SA

							loss_SA = loss_SA + Render_Loss_SA
							Out_SA = {'Out_SA':  Net_Out2, 'Render_fake_in':Render_Fake2}

						loss_total += loss_SA

						## for regularization term
						loss_reg = 0
						if self.opt.reg:
							## diff regular
							diff_MEAN = torch.mean(in_image_linear,dim=0).permute(1,2,0).unsqueeze(0)
							loss_dif = self.criterion(out_diff,diff_MEAN)

							# plt.figure(1)
							# imshow(diff_MEAN.cpu())
							# plt.show()

							## normal regular
							normal_MEAN = torch.tensor([0.5,0.5,1]).repeat(1,W,H,1).cuda()
							# print('normal_MEAN ',normal_MEAN.shape)
							loss_normal = self.criterion(out_normal,self.normal_MEAN)

							## rough regular
							# rough_MEAN = torch.mean(Net_Out[...,6:7],dim=(1,2)).repeat(1,W,H,1)
							rough_MEAN = torch.tensor([0.5]).repeat(1,W,H,1).cuda()
							
							# print('rough_MEAN ',rough_MEAN.shape)
							loss_rough = self.criterion(out_rough,rough_MEAN)
							
							## spec regular
							spec_MEAN = torch.mean(out_spec,dim=(1,2)).repeat(1,W,H,1)
							# print('spec_MEAN ',spec_MEAN.shape)
							loss_spec = self.criterion(out_spec,spec_MEAN)

							## decaying weight for regularization
							lambda_reg = self.opt.lambda_reg * (self.opt.total_iter - steps)/(self.opt.total_iter - self.opt.init_iter)
							
							# loss_reg = lambda_reg * 0.25*(loss_dif + 10*loss_normal + 10*loss_rough + loss_spec)
							loss_reg = lambda_reg * 0.5*(loss_dif + 10*loss_normal)
						loss_total += loss_reg

					# loss_total =  loss_Re_L1 + loss_Re_vgg + loss_SA + loss_reg + loss_Init + Loss_des19
			
					Loss={'loss_total':loss_total, 'loss_Re_vgg':loss_Re_vgg, 'loss_Fe_L1':loss_Fe_L1, 'loss_Re_L1':loss_Re_L1,'Loss_des19':Loss_des19, 'diff_loss':diff_loss,'loss_SA':loss_SA, 'loss_reg':loss_reg,'loss_Init':loss_Init,
							'loss_dif':loss_dif, 'loss_normal':loss_normal, 'loss_rough':loss_rough, 'loss_spec':loss_spec }



					return Loss, Gaussian, Out_SA

					# return loss_total,loss_SA, loss_Re_vgg, Net_Out,net_in

	def inference_real(self, in_image_linear,in_image_test, LightPos_train, LightPos_test):
		
		L_No,C,W,H=in_image_linear.shape

		with torch.no_grad():

			if self.opt.Net_Option=='Des19Net' or self.opt.Net_Option=='UNetS':

				# LDR: 2.2 correction; [0,1] -> [-1,1]
				if not self.opt.no_gamma_in and not self.opt.HDR:
					print('LDR')
					in_image=logTensor(in_image_linear)*2-1
				# HDR: log correction;
				elif not self.opt.no_gamma_in and self.opt.HDR:
					print('HDR')
					in_image=torch.log(in_image_linear+EPSILON+1)
				# other:[0,1] -> [-1,1]
				else:
					in_image=in_image*2-1 

				if self.opt.randinput:
					rand_index=torch.randperm(L_No)
					in_image=in_image[rand_index]
					# LightPos=LightPos[rand_index]
					print(rand_index)

				if self.opt.maxpool:
					net_in = in_image.unsqueeze(0) # [1, N,C,W,H]
				elif self.opt.Net_Option=='Des19Net':
					# [N,3,W,H] + [N,2,W,H] = [N,5,W,H]					
					net_in = torch.cat((in_image,self.extra_input_single),dim=1) 
					# print('input: ', net_in.shape)					
				else:
					net_in = in_image.view(-1,256,256).unsqueeze(0) # [N,C,W,H] -> [1, N*C, W,H]

				# Net_Out = self.netG.forward(in_image.view(L_No*3,256,256).unsqueeze(0)).squeeze(0).permute(1,2,0)
				Net_Out = self.netG.forward(net_in).squeeze(0).permute(1,2,0)
				Net_Out=process(Net_Out)# [-1,1] -> [0,1]
			else:

				XYZ_in=self.Position_map.view(-1,3)
				# print('XYZ_in: ', XYZ_in.shape)
				
				Pos=self.Position_map_int[:,:,0:2].view(-1,2).numpy().astype(int)
				batch,C=Pos.shape

				#[L,3,W,H] -> [B,L,3]
				Render_in=in_image[:,:,Pos[:,0],Pos[:,1]].permute(2,0,1)

				if self.opt.input_nc==6:
					# Light dir [L,3] -> [B,L,3]
					L_vec = LightPos.unsqueeze(0).repeat(batch,1,1)-XYZ_in.unsqueeze(1)
					L_vec = normalize_vec(L_vec)
					if self.opt.gamma_in:
						Render_in=Render_in**(1/2.2)*2-1
					else:
						Render_in=Render_in*2-1
					Net_in=torch.cat((Render_in,L_vec), dim=2).view(batch,-1)

				elif self.opt.input_nc==9:
					# Light dir [L,3] -> [B,L,3]
					L_vec = LightPos.unsqueeze(0).repeat(batch,1,1)-XYZ_in.unsqueeze(1)
					L_vec = normalize_vec(L_vec)

					# camera dir [3] -> [B,L,3]
					CamPos=torch.tensor([0.,0.,2.14]).cuda() # [B,L,c]
					V_vec = CamPos.unsqueeze(0).unsqueeze(0).repeat(batch,L_No,1)-XYZ_in.unsqueeze(1)
					V_vec = normalize_vec(V_vec)			

					# [B,L,6] -> [B, L*6]
					Render_in=Render_in*2-1
					Net_in=torch.cat((Render_in,L_vec,V_vec), dim=2).view(batch,-1)
					# print('Net_in: ',Net_in.shape)

				Net_Out = self.netG.forward(Net_in)
				Net_Out = (Net_Out+1)*0.5

				print('Net_Out:',Net_Out.shape)

				Net_Out = Net_Out.view(256,256,10)

			##### seperate output to different feature maps #####
			# 9 channels
			if self.opt.Net_Option=='Des19Net':
				normal=Net_Out[...,0:2]
				diff=Net_Out[...,2:5]
				rough=Net_Out[...,5:6]
				spec=Net_Out[...,6:9]
				normal=Process_des19normal(normal, self.opt.gpu_ids[0])
			# 10 channels
			else:
				normal=Net_Out[...,0:3]
				diff=Net_Out[...,3:6]
				rough=Net_Out[...,6:7]
				spec=Net_Out[...,7:10]
				normal = ProcessNormal(normal)

			# # [B,W,H,C]
			fake_re_in=SingleRender_NumberPointLight_FixedCamera(diff.unsqueeze(0), spec.unsqueeze(0), normal.unsqueeze(0), rough.unsqueeze(0), LightPos_train, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
			fake_re_test=SingleRender_NumberPointLight_FixedCamera(diff.unsqueeze(0), spec.unsqueeze(0), normal.unsqueeze(0), rough.unsqueeze(0), LightPos_test, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)

			Out={'diff':diff, 'normal':normal, 'rough':rough, 'spec':spec, 'Render_fake_in':fake_re_in,'Render_fake_test':fake_re_test}

			## compute the loss for test images
			# LDR: 2.2 correction; [0,1] -> [-1,1]
			if not self.opt.HDR and not self.opt.no_gamma_in:
				in_image_test=logTensor(torch.clamp(in_image_test,0,1))*2-1
				fake_re_test=logTensor(torch.clamp(fake_re_test,0,1))*2-1

			loss_Re_test =  self.criterion( in_image_test.permute(0,2,3,1), fake_re_test) 

			if self.opt.SA:
				NewLightPos = Create_NumberPointLightPosition_SA(self.opt.No_Input, self.opt.rand_light, self.opt.gpu_ids[0])
				# print(NewLightPos)

				Render_in_SA=SingleRender_NumberPointLight_FixedCamera(diff.unsqueeze(0), spec.unsqueeze(0), normal.unsqueeze(0), rough.unsqueeze(0), NewLightPos, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
				Render_in_SA=Render_in_SA.permute(0,3,1,2)

				# LDR: 2.2 correction; [0,1] -> [-1,1]
				if not self.opt.HDR and not self.opt.no_gamma_in:
					Render_in_SA=torch.clamp(Render_in_SA,0,1)
					Render_in_SA=logTensor(Render_in_SA)*2-1
				# HDR: log correction;
				elif self.opt.HDR and not self.opt.no_gamma_in:
					Render_in_SA=torch.log(Render_in_SA+EPSILON+1)
				else:
					Render_in_SA=Render_in_SA*2-1

				if self.opt.randinput:
					rand_index=torch.randperm(L_No)
					Render_in_SA=Render_in_SA[rand_index]
					# LightPos=LightPos[rand_index]
					print(rand_index)

				if self.opt.maxpool:
					Net_In_SA = Render_in_SA.unsqueeze(0) # [1, N,C,W,H]
				elif self.opt.Net_Option=='Des19Net':
					# [N,3,W,H] + [N,2,W,H] = [N,5,W,H]
					# print('render in shape :',Render_in_SA.shape)					
					# print('self.extra_input :',self.extra_input_single.shape)					
					Net_In_SA = torch.cat((Render_in_SA,self.extra_input_single),dim=1) 
					# print('input: ', Net_In_SA.shape)		
				else:
					Net_In_SA = Render_in_SA.view(-1,256,256).unsqueeze(0) # [N,C,W,H] -> [1, N*C, W,H]

				# Net_In_SA = Render_in_SA.view(L_No*3,256,256).unsqueeze(0)
				Net_Out_SA = self.netG.forward(Net_In_SA).squeeze(0).permute(1,2,0)
				Net_Out_SA=process(Net_Out_SA)# [-1,1] -> [0,1]
							
				# 9 channels
				if self.opt.Net_Option=='Des19Net':
					normal_SA=Net_Out_SA[...,0:2]
					diff_SA=Net_Out_SA[...,2:5]
					rough_SA=Net_Out_SA[...,5:6]
					spec_SA=Net_Out_SA[...,6:9]
					normal_SA=Process_des19normal(normal_SA, self.opt.gpu_ids[0])
				# 10 channels
				else:
					normal_SA=Net_Out_SA[...,0:3]
					diff_SA=Net_Out_SA[...,3:6]
					rough_SA=Net_Out_SA[...,6:7]
					spec_SA=Net_Out_SA[...,7:10]
					normal_SA = ProcessNormal(normal_SA)


				Out_SA = {'diff':diff_SA, 'normal':normal_SA, 'rough':rough_SA, 'spec':spec_SA, 'Render_fake_in':Render_in_SA,'Render_fake_debug':Net_In_SA}
			else:
				Out_SA = None
		# print('fake_re:',fake_re.shape)


			##### debug SA ########
			if self.opt.debugSA:
				# LDR: 2.2 correction; [0,1] -> [-1,1]
				if self.opt.maxpool:
					Net_In_debugSA = in_image_test.unsqueeze(0) # [1, N,C,W,H]
				elif self.opt.Net_Option=='Des19Net':
					### 2 test images map to 7
					# in_image_test_wh = in_image_test.permute(0,1,3,2)
					# in_image_test_fw = torch.flip(in_image_test,[2])
					# in_image_test_fh = torch.flip(in_image_test,[3])

					# in_image_test_wh = in_image_test
					# in_image_test_fw = in_image_test
					# in_image_test_fh = in_image_test
					# plt.figure(0)
					# imshow(in_image_test_wh.cpu())
					# plt.figure(1)
					# imshow(in_image_test_fw.cpu())
					# plt.figure(2)
					# imshow(in_image_test_fh.cpu())
					# plt.show()
					# in_image_test_temp = torch.cat((in_image_test,in_image_test_wh,in_image_test_fw,in_image_test_fh[0:1,...]),dim=0)
					# in_image_test_temp = torch.cat((in_image_test,in_image_test_wh,in_image_test_fw,in_image_test_fh[0:1,...]),dim=0)

					# print(in_image_test_temp.shape)
					# print(self.extra_input_single.shape)

					# [N,3,W,H] + [N,2,W,H] = [N,5,W,H]								 	
					Net_In_debugSA = torch.cat((in_image_test,self.extra_input_single),dim=1) 
					# print('input: ', Net_In_debugSA.shape)		
				else:
					Net_In_debugSA = in_image_test.view(-1,256,256).unsqueeze(0) # [N,C,W,H] -> [1, N*C, W,H]

				# Net_In_SA = Render_in_SA.view(L_No*3,256,256).unsqueeze(0)
				Net_Out_debugSA = self.netG.forward(Net_In_debugSA).squeeze(0).permute(1,2,0)
				Net_Out_debugSA=process(Net_Out_debugSA)# [-1,1] -> [0,1]
							
				# 9 channels
				if self.opt.Net_Option=='Des19Net':
					normal_debugSA=Net_Out_debugSA[...,0:2]
					diff_debugSA=Net_Out_debugSA[...,2:5]
					rough_debugSA=Net_Out_debugSA[...,5:6]
					spec_debugSA=Net_Out_debugSA[...,6:9]
					normal_debugSA=Process_des19normal(normal_debugSA, self.opt.gpu_ids[0])
				# 10 channels
				else:
					normal_debugSA=Net_Out_debugSA[...,0:3]
					diff_debugSA=Net_Out_debugSA[...,3:6]
					rough_debugSA=Net_Out_debugSA[...,6:7]
					spec_debugSA=Net_Out_debugSA[...,7:10]
					normal_debugSA = ProcessNormal(normal_debugSA)

				Out_debugSA = {'diff':diff_debugSA, 'normal':normal_debugSA, 'rough':rough_debugSA, 'spec':spec_debugSA,'debug_in1':in_image_test_wh,'debug_in2':in_image_test_fw,'debug_in3':in_image_test_fh}
			else:
				Out_debugSA = None

		return Out, Out_SA, Out_debugSA, loss_Re_test

	def inference_debug(self, in_image_linear,LightPos_test):
		
		L_No,C,W,H=in_image_linear.shape

		with torch.no_grad():

			in_image=logTensor(in_image_linear)*2-1

			net_in = torch.cat((in_image,self.extra_input_single),dim=1) 

			# Net_Out = self.netG.forward(in_image.view(L_No*3,256,256).unsqueeze(0)).squeeze(0).permute(1,2,0)
			Net_Out = self.netG.forward(net_in).squeeze(0).permute(1,2,0)
			Net_Out=process(Net_Out)# [-1,1] -> [0,1]
				
			normal=Net_Out[...,0:2]
			diff=Net_Out[...,2:5]
			rough=Net_Out[...,5:6]
			spec=Net_Out[...,6:9]
			normal=Process_des19normal(normal, self.opt.gpu_ids[0])

			fake_re_test=SingleRender_NumberPointLight_FixedCamera(diff.unsqueeze(0), spec.unsqueeze(0), normal.unsqueeze(0), rough.unsqueeze(0), LightPos_test, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)


			Out={'diff':diff, 'normal':normal, 'rough':rough, 'spec':spec,'Render_fake_test':fake_re_test}

		return Out

	def inference_syn(self, in_image_linear, LightPos):
		
		L_No,C,W,H=in_image_linear.shape

		with torch.no_grad():

			if self.opt.Net_Option=='Des19Net' or self.opt.Net_Option=='UNetS':
				# LDR: 2.2 correction; [0,1] -> [-1,1]
				if not self.opt.no_gamma_in and not self.opt.HDR:
					print('LDR')
					in_image=logTensor(in_image_linear)*2-1
				# HDR: log correction;
				elif not self.opt.no_gamma_in and self.opt.HDR:
					print('HDR')
					in_image=torch.log(in_image_linear+EPSILON+1)
				# other:[0,1] -> [-1,1]
				else:
					in_image=in_image*2-1 

				if self.opt.randinput:
					rand_index=torch.randperm(L_No)
					in_image=in_image[rand_index]
					# LightPos=LightPos[rand_index]
					print(rand_index)

				if self.opt.maxpool:
					net_in = in_image.unsqueeze(0) # [1, N,C,W,H]
				elif self.opt.Net_Option=='Des19Net':
					# [N,3,W,H] + [N,2,W,H] = [N,5,W,H]					
					net_in = torch.cat((in_image,self.extra_input_single),dim=1) 
				else:
					net_in = in_image.view(-1,256,256).unsqueeze(0) # [N,C,W,H] -> [1, N*C, W,H]
				# Net_Out = self.netG.forward(in_image.view(L_No*3,256,256).unsqueeze(0)).squeeze(0).permute(1,2,0)
				Net_Out = self.netG.forward(net_in).squeeze(0).permute(1,2,0)
				Net_Out=process(Net_Out)# [-1,1] -> [0,1]
			else:

				XYZ_in=self.Position_map.view(-1,3)
				# print('XYZ_in: ', XYZ_in.shape)
				
				Pos=self.Position_map_int[:,:,0:2].view(-1,2).numpy().astype(int)
				batch,C=Pos.shape

				#[L,3,W,H] -> [B,L,3]
				Render_in=in_image[:,:,Pos[:,0],Pos[:,1]].permute(2,0,1)

				if self.opt.input_nc==6:
					# Light dir [L,3] -> [B,L,3]
					L_vec = LightPos.unsqueeze(0).repeat(batch,1,1)-XYZ_in.unsqueeze(1)
					L_vec = normalize_vec(L_vec)
					if self.opt.gamma_in:
						Render_in=Render_in**(1/2.2)*2-1
					else:
						Render_in=Render_in*2-1
					Net_in=torch.cat((Render_in,L_vec), dim=2).view(batch,-1)

				elif self.opt.input_nc==9:
					# Light dir [L,3] -> [B,L,3]
					L_vec = LightPos.unsqueeze(0).repeat(batch,1,1)-XYZ_in.unsqueeze(1)
					L_vec = normalize_vec(L_vec)

					# camera dir [3] -> [B,L,3]
					CamPos=torch.tensor([0.,0.,2.14]).cuda() # [B,L,c]
					V_vec = CamPos.unsqueeze(0).unsqueeze(0).repeat(batch,L_No,1)-XYZ_in.unsqueeze(1)
					V_vec = normalize_vec(V_vec)			

					# [B,L,6] -> [B, L*6]
					Render_in=Render_in*2-1
					Net_in=torch.cat((Render_in,L_vec,V_vec), dim=2).view(batch,-1)
					# print('Net_in: ',Net_in.shape)

				Net_Out = self.netG.forward(Net_in)
				Net_Out = (Net_Out+1)*0.5

				print('Net_Out:',Net_Out.shape)

				Net_Out = Net_Out.view(256,256,10)

			##### seperate output to different feature maps #####
			# 9 channels
			if self.opt.Net_Option=='Des19Net':
				normal=Net_Out[...,0:2]
				diff=Net_Out[...,2:5]
				rough=Net_Out[...,5:6]
				spec=Net_Out[...,6:9]
				normal=Process_des19normal(normal, self.opt.gpu_ids[0])
				# temp_out = torch.cat((normal,diff,rough,spec),dim=1)
			# 10 channels
			else:
				normal=Net_Out[...,0:3]
				diff=Net_Out[...,3:6]
				rough=Net_Out[...,6:7]
				spec=Net_Out[...,7:10]
				normal = ProcessNormal(normal)

			# # [B,W,H,C]
			fake_re_in=SingleRender_NumberPointLight_FixedCamera(diff.unsqueeze(0), spec.unsqueeze(0), normal.unsqueeze(0), rough.unsqueeze(0), LightPos, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
				
			rand_LightPos = Create_NumberPointLightPosition_SA(self.opt.No_Input, self.opt.rand_light, self.opt.gpu_ids[0])
			Render_fake=SingleRender_NumberPointLight_FixedCamera(diff.unsqueeze(0), spec.unsqueeze(0), normal.unsqueeze(0), rough.unsqueeze(0), rand_LightPos, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
			# normal_gt = gt_fe[:,0:256,:]
			# diff_gt = gt_fe[:,256:2*256,:]**2.2
			# rough_gt = gt_fe[:,2*256:3*256,:]
			# spec_gt = gt_fe[:,3*256:4*256,:]**2.2
			# normal_gt = ProcessNormal(normal_gt)
			# temp_gt = torch.cat((normal_gt,diff_gt,rough_gt,spec_gt),dim=1)

			# Render_gt=SingleRender_NumberPointLight_FixedCamera(diff_gt.unsqueeze(0), spec_gt.unsqueeze(0), normal_gt.unsqueeze(0), rough_gt.unsqueeze(0), rand_LightPos, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)

			# Out={'diff':diff, 'normal':normal, 'rough':rough, 'spec':spec, 'Render_fake_in':fake_re_in,'Render_fake':Render_fake,'Render_gt':Render_gt}
			Out={'diff':diff, 'normal':normal, 'rough':rough, 'spec':spec,'Render_fake':Render_fake, 'Render_fake_in':fake_re_in}

			## compute the loss for feature maps
			# print(temp_out.shape)
			# print(gt_fe.shape)
			# loss_Fe =  self.criterion(temp_out, temp_gt) 


			if self.opt.SA:
				NewLightPos = Create_NumberPointLightPosition_SA(self.opt.No_Input, self.opt.rand_light, self.opt.gpu_ids[0])
				# print(NewLightPos)

				Render_in_SA=SingleRender_NumberPointLight_FixedCamera(diff.unsqueeze(0), spec.unsqueeze(0), normal.unsqueeze(0), rough.unsqueeze(0), NewLightPos, self.Position_map, self.opt.gpu_ids[0], not self.opt.no_CoCamLi)
				Render_in_SA=Render_in_SA.permute(0,3,1,2)

				# LDR: 2.2 correction; [0,1] -> [-1,1]
				if not self.opt.HDR and not self.opt.no_gamma_in:
					Render_in_SA=torch.clamp(Render_in_SA,0,1)
					Render_in_SA=logTensor(Render_in_SA)*2-1
				# HDR: log correction;
				elif self.opt.HDR and not self.opt.no_gamma_in:
					Render_in_SA=torch.log(Render_in_SA+EPSILON+1)
				else:
					Render_in_SA=Render_in_SA*2-1

				if self.opt.randinput:
					rand_index=torch.randperm(L_No)
					Render_in_SA=Render_in_SA[rand_index]
					# LightPos=LightPos[rand_index]
					print(rand_index)

				if self.opt.maxpool:
					Net_In_SA = Render_in_SA.unsqueeze(0) # [1, N,C,W,H]
				elif self.opt.Net_Option=='Des19Net':
					# [N,3,W,H] + [N,2,W,H] = [N,5,W,H]					
					Net_In_SA = torch.cat((Render_in_SA,self.extra_input_single),dim=1) 
					print('input: ', Net_In_SA.shape)		
				else:
					Net_In_SA = Render_in_SA.view(-1,256,256).unsqueeze(0) # [N,C,W,H] -> [1, N*C, W,H]

				# Net_In_SA = Render_in_SA.view(L_No*3,256,256).unsqueeze(0)
				Net_Out_SA = self.netG.forward(Net_In_SA).squeeze(0).permute(1,2,0)
				Net_Out_SA=process(Net_Out_SA)# [-1,1] -> [0,1]
							
				# 9 channels
				if self.opt.Net_Option=='Des19Net':
					normal_SA=Net_Out_SA[...,0:2]
					diff_SA=Net_Out_SA[...,2:5]
					rough_SA=Net_Out_SA[...,5:6].repeat(1,1,3)
					spec_SA=Net_Out_SA[...,6:9]
					normal_SA=Process_des19normal(normal_SA, self.opt.gpu_ids[0])
				# 10 channels
				else:
					normal_SA=Net_Out_SA[...,0:3]
					diff_SA=Net_Out_SA[...,3:6]
					rough_SA=Net_Out_SA[...,6:7].repeat(1,1,3)
					spec_SA=Net_Out_SA[...,7:10]
					normal_SA = ProcessNormal(normal_SA)


				Out_SA = {'diff':diff_SA, 'normal':normal_SA, 'rough':rough_SA, 'spec':spec_SA, 'Render_fake_in':Render_in_SA,'Render_fake_debug':Net_In_SA}
			else:
				Out_SA = None

			##### debug SA ########
			if self.opt.debugSA:
				# LDR: 2.2 correction; [0,1] -> [-1,1]
				if self.opt.maxpool:
					Net_In_debugSA = in_image_test.unsqueeze(0) # [1, N,C,W,H]
				elif self.opt.Net_Option=='Des19Net':
					### 2 test images map to 7
					# in_image_test_wh = in_image_test.permute(0,1,3,2)
					# in_image_test_fw = torch.flip(in_image_test,[2])
					# in_image_test_fh = torch.flip(in_image_test,[3])

					in_image_test_wh = in_image_test
					in_image_test_fw = in_image_test
					in_image_test_fh = in_image_test
					# plt.figure(0)
					# imshow(in_image_test_wh.cpu())
					# plt.figure(1)
					# imshow(in_image_test_fw.cpu())
					# plt.figure(2)
					# imshow(in_image_test_fh.cpu())
					# plt.show()
					in_image_test_temp = torch.cat((in_image_test,in_image_test_wh,in_image_test_fw,in_image_test_fh[0:1,...]),dim=0)
					in_image_test_temp = torch.cat((in_image_test,in_image_test_wh,in_image_test_fw,in_image_test_fh[0:1,...]),dim=0)

					# [N,3,W,H] + [N,2,W,H] = [N,5,W,H]								 	
					Net_In_debugSA = torch.cat((in_image_test_temp,self.extra_input),dim=1) 
					# print('input: ', Net_In_debugSA.shape)		
				else:
					Net_In_debugSA = in_image_test.view(-1,256,256).unsqueeze(0) # [N,C,W,H] -> [1, N*C, W,H]

				# Net_In_SA = Render_in_SA.view(L_No*3,256,256).unsqueeze(0)
				Net_Out_debugSA = self.netG.forward(Net_In_debugSA).squeeze(0).permute(1,2,0)
				Net_Out_debugSA=process(Net_Out_debugSA)# [-1,1] -> [0,1]
							
				# 9 channels
				if self.opt.Net_Option=='Des19Net':
					normal_debugSA=Net_Out_debugSA[...,0:2]
					diff_debugSA=Net_Out_debugSA[...,2:5]
					rough_debugSA=Net_Out_debugSA[...,5:6].repeat(1,1,3)
					spec_debugSA=Net_Out_debugSA[...,6:9]
					normal_debugSA=Process_des19normal(normal_debugSA, self.opt.gpu_ids[0])
				# 10 channels
				else:
					normal_debugSA=Net_Out_debugSA[...,0:3]
					diff_debugSA=Net_Out_debugSA[...,3:6]
					rough_debugSA=Net_Out_debugSA[...,6:7].repeat(1,1,3)
					spec_debugSA=Net_Out_debugSA[...,7:10]
					normal_debugSA = ProcessNormal(normal_debugSA)

				Out_debugSA = {'diff':diff_debugSA, 'normal':normal_debugSA, 'rough':rough_debugSA, 'spec':spec_debugSA,'debug_in1':in_image_test_wh,'debug_in2':in_image_test_fw,'debug_in3':in_image_test_fh}
			else:
				Out_debugSA = None

		# return Out, Out_SA, Out_debugSA,loss_Fe
		return Out
	
	def save(self, which_epoch):
		self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
		self.save_optimizer(self.optimizer_G, 'Meta_Optim', which_epoch, self.gpu_ids)

	def LoadDes19Net(self, netG, des19_npy):
		# print('start loading des19')
		# des19_npy=np.load(npy_path,allow_pickle=True).item()
		params=netG.state_dict()
		# print(params)
		for key in params:
			# print('before: ',params[key].requires_grad)
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

	def LoadDes19Net_params(self, netG, load_params):
		# print('start loading des19')
		# des19_npy=np.load(npy_path,allow_pickle=True).item()
		params=netG.state_dict()
		# print(params)
		for key in params:
			# print('before: ',params[key].requires_grad)
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


	def LossDes19Net(self, netG, params):
		# print('start loading des19')
		criterion = torch.nn.L1Loss()

		TotalLoss = 0
		index=0
		# print(params['encoder1.weight'][0,0,0,:])
		# print(params['encoder1.weight'][0,0,0,:])

		for key,param in netG.named_parameters():
			# print(key)
			index +=1
			if 'instance' in key:
				TotalLoss+=criterion(param,params[key])
			elif 'global' in key:
				if 'weight' in key:
					TotalLoss+=criterion(param,params[key])
				else:
					TotalLoss+=criterion(param,params[key])
			elif 'lastconv3.bias' in key:
				TotalLoss+=criterion(param,params[key])
			else:
				TotalLoss+=criterion(param,params[key])
		# print(TotalLoss)
		return TotalLoss

	def update_learning_rate(self):
		lr = self.old_lr/2.0  

		for param_group in self.optimizer_G.param_groups:
			param_group['lr'] = lr
		print('update learning rate: %f -> %f' % (self.old_lr, lr))
		self.old_lr = lr


		
