import numpy as np
import torch
import random
import sys

import torchvision.transforms.functional as TF

from util.util import *
import matplotlib.pyplot as plt
import torch.distributions as tdist
from numpy import nan


def PositionMap(width,height,channel):

	Position_map_cpu = torch.zeros((width,height,channel))
	for w in range(width):
		for h in range(height):
			Position_map_cpu[h][w][0] = 2*w/(width-1) - 1
			#Position_map[h][w][0] = 2*(width-w-1)/(width-1) - 1
			Position_map_cpu[h][w][1] = 2*(height-h-1)/(height-1) - 1
			#Position_map[h][w][1] = 2*h/(height-1) - 1
			Position_map_cpu[h][w][2] = 0

	return Position_map_cpu

def PositionMap_Des19(width=256,height=256,channel=2):
	
	Position_map_cpu = torch.zeros((width,height,channel))
	for w in range(width):
		for h in range(height):
			Position_map_cpu[h][w][0] = 2*w/(width-1) - 1
			#Position_map[h][w][0] = 2*(width-w-1)/(width-1) - 1
			Position_map_cpu[h][w][1] = 2*(height-h-1)/(height-1) - 1
			#Position_map[h][w][1] = 2*h/(height-1) - 1
	return Position_map_cpu

def PositionMap_int(width,height,channel):

	Position_map_cpu = torch.zeros((width,height,channel))
	for w in range(width):
		for h in range(height):
			Position_map_cpu[w][height-h-1][0] = w
			Position_map_cpu[w][height-h-1][1] = (height-h-1)
			Position_map_cpu[w][height-h-1][2] = 0

	return Position_map_cpu
	
############################################## Creating Light & Camera Positions ################################################

#### randomly sample Number direction (for the rendering in backpropgation during training )
def Cosine_Distribution_Number(Number, r_max, mydevice):

	u_1= torch.rand((Number,1),device=mydevice)*r_max 	# rmax: 0.95 (default)
	# print('u_1:', u_1.shape)

	u_2= torch.rand((Number,1),device=mydevice)
	# u_2= 1.0/Number*torch.arange(1,Number+1,dtype=torch.float32).unsqueeze(-1).to(mydevice)

	r = torch.sqrt(u_1)
	theta = 2*PI*u_2

	x = r*torch.cos(theta)
	y = r*torch.sin(theta)
	z = torch.sqrt(1-r*r)

	temp_out = torch.cat([x,y,z],1)

	return temp_out


def Cosine_Distribution_Number_Center(Number, r_max, mydevice):

	u_1= torch.rand((Number,1),device=mydevice)*0 	# rmax: 0.95 (default)
	# u_1= torch.rand((Number,1),device=mydevice)*0+ r_max	# rmax: 0.95 (default)

	u_2= torch.rand((Number,1),device=mydevice)
	# u_2= 1.0/Number*torch.arange(1,Number+1,dtype=torch.float32).unsqueeze(-1).to(mydevice)

	r = torch.sqrt(u_1)
	theta = 2*PI*u_2

	x = r*torch.cos(theta)
	y = r*torch.sin(theta)
	z = torch.sqrt(1-r*r)

	temp_out = torch.cat([x,y,z],1)

	return temp_out


def Des19_normalized_random_direction(Number, angle, lowEps = 0.001):
	r1 = torch.rand(Number, 1)*angle + lowEps
	r2 =  torch.rand(Number, 1)
	r = torch.sqrt(r1)
	phi = 2 * PI * r2
	#min alpha = atan(sqrt(1-r^2)/r)
	x = r * torch.cos(phi)
	y = r * torch.sin(phi)
	z = torch.sqrt(1.0 - torch.square(r))
	finalVec = torch.cat((x, y, z), dim=-1) #Dimension here should be [N, 3]
	return finalVec.cuda()



def Create_NumberPointLightPosition_test(Near_Number,r_max, mydevice):

	mydevice=torch.device('cuda')

	rand_light = Cosine_Distribution_Number(Near_Number, r_max, mydevice)

	# Origin ([-1,1],[-1,1],0)
	# Origin=torch.tensor([0.0,0.0],device=mydevice)
	# Origin_xy = torch.rand((Near_Number,2),device=mydevice)*2-1
	Origin_xy = torch.rand((Near_Number,2),device=mydevice)*0
	Origin = torch.cat([Origin_xy,torch.zeros((Near_Number,1),device=mydevice)],1)

	m=tdist.Normal(torch.tensor([1.0]),torch.tensor([0.5]))
	Distance=m.sample((Near_Number,2)).to(mydevice)
	Light_po=Origin+rand_light*5.0

	return Light_po

def Generate_height(opt):

	if opt.large_height:
		Height_max = Height_max_h
		Heigt_min = Heigt_min_h
	else:
		Height_max = Height_max_l
		Heigt_min = Heigt_min_l
	
	# lightDistance = torch.exp(torch.normal(Height_mean,Height_Va))
	lightDistance = torch.rand(1, dtype=torch.float32) * (Height_max - Heigt_min) + Heigt_min

	return lightDistance

## torch version light in hemisphere + around center light
## return [N,3]
def Create_Des19Light(Near_Number, r_max, mydevice, opt):
	# currentLightPos = torch.rand((1, 2), dtype=torch.float32)*1.5 - 0.75
	currentLightPos = torch.rand((1, 2), dtype=torch.float32)*1 - 0.5
	dist = Generate_height(opt)

	currentLightPos = torch.cat((currentLightPos, torch.ones(1,1)*dist),dim=-1).cuda()
	# print('dist: ', dist)
	if Near_Number>1:
		currentLightPos2 = Des19_normalized_random_direction(Near_Number - 1, angle=r_max, lowEps = minEps) * dist #getting the pos and adding the multi light dim.
		currentLightPos = torch.cat((currentLightPos, currentLightPos2), dim = 0)

	return currentLightPos


# new "ambient light" should have [batchSize, nbRenderings, nbLights, 1, 1, 3]
def Create_Des19AmbientLight(): #Maybe use the same cos distribution as in the other direction and pick a distance from a exp normal distribution
	ambiantDir = Des19_normalized_random_direction(1, angle = 0.8, lowEps = minEps) #Here remove angles below 25 ° from the surface

	distance = torch.exp(torch.normal(torch.tensor(np.log(30.0)), torch.tensor(0.15))).cuda().to(torch.float32)
	ambiantPos = ambiantDir.cuda()*distance

	return ambiantPos

def Create_MultiDes19AmbientLight(num):
	ambi_list = torch.empty(num,3)
	for i in range(num):
		ambi = Create_Des19AmbientLight()
		ambi_list[i]=ambi
	return ambi_list

def Des19_addNoise(renderings, noise=None):

	if noise is not None:
		return noise
	else:
		stddevNoise = torch.exp(torch.normal(torch.tensor(np.log(0.005)), torch.tensor(0.3)))
		noise = torch.normal(torch.tensor(0.0), stddevNoise, size = renderings.shape).cuda()

		return noise

# def NLight_Vary(Near_Number, r_max, device):

# 	rand_light = Cosine_Distribution_Number(Near_Number, r_max, device)
# 	# dist = Generate_height(opt)

# 	# dist = torch.exp(torch.normal(torch.tensor([1.0]), torch.tensor([0.2]))).cuda()
# 	dist = 2.14 ** torch.normal(torch.tensor([1.0]), torch.tensor([0.1]))
# 	Light_po = rand_light * dist.cuda()

# 	return Light_po


def NLight_VaryH(Near_Number, r_max, device):

	rand_light = Cosine_Distribution_Number(Near_Number, r_max, device)
	# dist = Generate_height(opt)

	# dist = torch.exp(torch.normal(torch.tensor([1.0]), torch.tensor([0.2]))).cuda()
	dist = 4.0 ** torch.normal(torch.tensor([1.0]), torch.tensor([0.15]))
	Light_po = rand_light * dist.cuda()

	return Light_po

def NLight_VaryH_Jitter(Near_Number, device):

	Jitter = torch.normal(torch.tensor([1.0]), torch.tensor([0.1]))*0.001

	rand_light = Cosine_Distribution_Number(Near_Number, Jitter.to(device), device)

	dist = 4.0 ** torch.normal(torch.tensor([1.0]), torch.tensor([0.15]))
	Light_po = rand_light * dist.cuda()


	return Light_po


# fixed light with origin 0
def NLight_Fix(Near_Number, r_max, device):

	return Cosine_Distribution_Number(Near_Number, r_max, device)*4.0


############################################# Rendering Function ###################################################################
# Single Render: each scene rendered under single light (camera) positin
# Batch Render: each scene rendered under multiple light (camera) position at the same time

## render algorithm
def Rendering_Algorithm(diff,spec,rough,NdotH,NdotL,VdotN,VdotH, no_spec=False):

	########################### rendering alogorithm ###########################################################################
	if no_spec:
		diff=diff/PI
	else:
		diff=diff*(1-spec)/PI

	## only one channel for roughness
	rough2 = rough*rough
	NdotH2 = NdotH*NdotH
	
	### psudocode: GGTX distribution with PI removed
	#### D #####
	deno_D = torch.max((rough2*rough2 - 1)*NdotH2+1,EPSILON_tensor)
	D = (rough2/deno_D)**2
	D = D/PI

	#### G #####
	G_1 = 1/torch.max(NdotL*(1-rough2/2)+rough2/2,EPSILON_tensor)
	G_2 = 1/torch.max(VdotN*(1-rough2/2)+rough2/2,EPSILON_tensor)
	G = G_1*G_2

	#### F #####
	if no_spec:
		F = spec + (1-spec)*(1-VdotH)**5
	else:
		F = spec+(1-spec)*2**((-5.55473*VdotH - 6.98316)*VdotH)

	specular = G*F*D/(4+EPSILON)

	# [B,C]
	FinalColor = PI*(diff+specular)*NdotL 
	# FinalColor = PI*(diff)#*NdotL

	return FinalColor


## render B svbrdf under B light positions
## input: diff,spec 3 channel; roughness 1 channel; normal 3 channel [-1,1]
## output: [B/N,W,H,C]. (B==N)
## !!! B (batch size) must equal N (Number of point light position) !!!
def SingleRender_NumberPointLight_FixedCamera(normal, diff, rough, spec, LightPos, Pos, mydevice, CamLi_co, fix_intensity=False, lightinten=16.0, no_spec=False):

	assert normal.shape[-1]==3, "normal channel wrong"
	assert rough.shape[-1]==1, "roughness channel wrong"

	if normal.dim()==3:
		normal = normal.unsqueeze(0)
		diff = diff.unsqueeze(0)
		rough = rough.unsqueeze(0)
		spec = spec.unsqueeze(0)

	if LightPos.dim()!=4:
		LightPos = LightPos.unsqueeze(1).unsqueeze(1)


	Near_Number=LightPos.shape[0]
	# print(LightPos.shape)

	if not CamLi_co:
		CameraPosition=torch.tensor([0.,0.,2.14],device=mydevice).unsqueeze(0).repeat(Near_Number,1)
	else:
		CameraPosition=LightPos.clone()

	Pos=Pos.repeat(Near_Number,1,1,1)

	L_vec = LightPos-Pos
	dist_l_sq = (L_vec**2).sum(3, keepdim=True)
	L_vec = normalize_vec(L_vec)
	normal = normalize_vec(normal)

	V_vec = CameraPosition-Pos
	V_vec = normalize_vec(V_vec)

	## Half vector of view and light direction [N,W,H,C]
	H_vec = normalize_vec((L_vec + V_vec)/2)

	# [B/N,W,H,C]
	NdotL = (normal*L_vec).sum(3, keepdim = True).clamp(0, 1)
	NdotH = (normal*H_vec).sum(3, keepdim = True).clamp(0, 1)
	VdotH = (V_vec*H_vec).sum(3, keepdim = True).clamp(0, 1)
	VdotN = (V_vec*normal).sum(3, keepdim = True).clamp(0, 1)

	if fix_intensity:
		print('fix intensity')
		intensity = 1.0
	else:
		intensity = lightinten /dist_l_sq
		# intensity = lightinten /dist_l_sq

	FinalColor = intensity* Rendering_Algorithm(diff, spec, rough, NdotH, NdotL, VdotN, VdotH, no_spec=no_spec)

	return FinalColor


