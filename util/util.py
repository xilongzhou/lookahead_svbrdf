from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms    
import torch.nn as nn
from collections import OrderedDict
import time
import sys


PI=np.pi
EPSILON=1e-6
diff_thred=0.0000001
val_thred=0.0001
iter_thred=20000

EPSILON_tensor = torch.tensor(EPSILON).cuda()

CROP_SIZE = 256

minEps = 0.001 #allows near 90degrees angles
maxEps = 0.02 #removes all angles below 8.13 degrees. see helpers.tf_generate_normalized_random_direction for the equation to calculate it.

des19_lightDistance = 2.197

Height_max_h = 10.0
Heigt_min_h = 1.0
Height_max_l = 5.0
Heigt_min_l = 1.0

def logTensor(x):
    return  (torch.log(x+0.01) - np.log(0.01)) / (np.log(1.01)-np.log(0.01))

def InverseLogTensor(x):
    temp=x*(np.log(1.01)-np.log(0.01))+np.log(0.01)
    return torch.exp(temp)-0.01

def iter_thred_factor(Num, Option):
    if Option=='PerPixel':
        factor=1
    elif Option=='PerLine':
        factor=Num
    return factor


def load_light_txt(name, verbose=True):
    with open(name,'r') as f:
        lines = f.readlines()
        wlvs = []
        for line in lines[0:]:
            line = line[:-1]
            camera_pos = [float(i) for i in line.split(',')]
            # camera_pos = [i for i in line.split(',')]
            # print(camera_pos)
            wlvs.append(camera_pos)
        wlvs=np.array(wlvs)
        if verbose:
            print(wlvs)
        return wlvs



def load_shift_txt(name, verbose=True):
    with open(name,'r') as f:
        lines = f.readlines()
        for line in lines[0:]:
            line = line[:-1]
            shift = [float(i) for i in line.split(',')]
            # camera_pos = [i for i in line.split(',')]
        shift=np.array(shift)
        if verbose:
            print(shift)
        return shift

def load_lightimg_txt(name, verbose=True):

    # load light
    path = os.path.join(name, 'camera_pos.txt')
    with open(path,'r') as f:
        lines = f.readlines()
        wlvs = []
        for line in lines[0:]:
            line = line[:-1]
            camera_pos = [float(i) for i in line.split(',')]
            # camera_pos = [i for i in line.split(',')]
            # print(camera_pos)
            wlvs.append(camera_pos)
        wlvs=np.array(wlvs)
        if verbose:
            print(wlvs)


    # load img size
    path = os.path.join(name, 'image_size.txt')
    with open(path,'r') as f:
        lines = f.readlines()
        for line in lines[0:]:
            line = line[:-1]
            img_size = [float(i) for i in line.split(',')]
            # camera_pos = [i for i in line.split(',')]
            # print(camera_pos)
        img_size=np.array(img_size)
        if verbose:
            print(img_size)

    wlvs = 2*wlvs/img_size
    if verbose:
        print(wlvs)

    return wlvs


def load_trainlight_txt(name):
    with open(name,'r') as f:
        lines = f.readlines()
        wlvs = []
        for line in lines[0:7]:
            line = line[:-1]
            camera_pos = [float(i) for i in line.split(',')]
            # camera_pos = [i for i in line.split(',')]
            # print(camera_pos)
            wlvs.append(camera_pos)
        wlvs=np.array(wlvs)
        print(wlvs)
        return wlvs

def load_testlight_txt(name):
    with open(name,'r') as f:
        lines = f.readlines()
        wlvs = []
        for line in lines[7:9]:
            line = line[:-1]
            camera_pos = [float(i) for i in line.split(',')]
            # camera_pos = [i for i in line.split(',')]
            # print(camera_pos)
            wlvs.append(camera_pos)
        wlvs=np.array(wlvs)
        print(wlvs)
        return wlvs



def lognp(inputimage):
    return  (np.log(inputimage+0.01) - np.log(0.01)) / (np.log(1.01)-np.log(0.01))

def savenpy(path,tensor):
    np.save(path,tensor.detach().cpu())

def imshow(img):
    if isinstance(img,(np.ndarray))==False:
        npimg = img.numpy()
    else:
        npimg=img

    # [B,C,W,H] --> [B,W,H,C]
    if npimg.shape[1] == 3:
        npimg = npimg.transpose(0,2,3,1)

    # [B,W,H,C]
    if npimg.ndim==4:
        npimg = npimg[0,:,:,:]
    # [W,H,C]
    elif npimg.ndim==3:
        npimg = npimg
    # [B,N,W,H,C]
    elif npimg.ndim==5:
        npimg = npimg[0,0,:,:,:]

    plt.axis("off")
    npimg = npimg*255
    npimg = npimg.clip(0,255)
    #npimg = np.transpose(npimg, (1, 2, 0))
    npimg = npimg.astype(np.uint8)
    # print('size in imshow: ',npimg.shape)
    plt.imshow(npimg)
    # plt.show()

# [-1,1] -> [0,1]
def process(image_tensor, imtype=np.uint8):
    return ( image_tensor + 1) / 2.0 

##### input: [W,H,3] or [B,W,H,3]
def normalize_vec(input_tensor):
    
    assert input_tensor.shape[-1]==3, "input channel should be 3 but got {}".format(input_tensor.shape[-1])

    ## speed test manually vs norm()
    if input_tensor.dim() == 3:  
        NormalizedNorm_len=input_tensor[:,:,0]*input_tensor[:,:,0]+input_tensor[:,:,1]*input_tensor[:,:,1]+input_tensor[:,:,2]*input_tensor[:,:,2]
        NormalizedNorm_len=torch.sqrt(NormalizedNorm_len)
        # NormalizedNorm_len=torch.norm(input_tensor,2,2)
        # print('shape:',NormalizedNorm_len.shape)
        NormalizedNorm = input_tensor/(NormalizedNorm_len[:,:,np.newaxis]+EPSILON)
        return NormalizedNorm
    elif input_tensor.dim() == 4:
        NormalizedNorm_len=input_tensor[:,:,:,0]*input_tensor[:,:,:,0]+input_tensor[:,:,:,1]*input_tensor[:,:,:,1]+input_tensor[:,:,:,2]*input_tensor[:,:,:,2]
        NormalizedNorm_len=torch.sqrt(NormalizedNorm_len)
        # NormalizedNorm_len=torch.norm(input_tensor,2,3)
        # print('shape:',NormalizedNorm_len.shape)
        NormalizedNorm = input_tensor/(NormalizedNorm_len[:,:,:,np.newaxis]+EPSILON)
        return NormalizedNorm
    elif input_tensor.dim() == 2:
        NormalizedNorm_len=input_tensor[:,0]*input_tensor[:,0]+input_tensor[:,1]*input_tensor[:,1]+input_tensor[:,2]*input_tensor[:,2]
        NormalizedNorm_len=torch.sqrt(NormalizedNorm_len)

        NormalizedNorm = input_tensor/(NormalizedNorm_len[:,np.newaxis]+EPSILON)

        return NormalizedNorm
    else:       
        print('incorrectly input')
        return

def normalize(input_tensor):
    return input_tensor / torch.sqrt((input_tensor ** 2).sum(-1, keepdim=True))

#[0,1] [H,W,C] or [B, H,W,C]
def ProcessNormal(opt, normal): #
    ### attention !!! remember normalization need to be done on [-1,1] domain 
    # [0, 1] => [-1, 1] && normalize
    if normal.shape[-1]==3:
        # normal = normal*2-1 # [0,1] --> [-1,1]
        # normal = torch.cat([normal[:,:,0:2], torch.clamp(normal[:,:,2:3], EPSILON)], dim=2) # ([-1,1], [-1,1], [0,1])

        normal = torch.cat([normal[...,0:2]*2-1, normal[...,2:3]], dim=2) # ([-1,1], [-1,1], [0,1])

        return 0.5*normalize_vec(normal)+0.5 # [-1,1] --> [0,1] (times 3 to make normal stronger)

    elif normal.shape[-1]==2:
        normal=normal*2-1
        reconstruct_n=torch.cat((normal*3,torch.ones([opt.res,opt.res,1]).cuda()),dim=-1) #[-1,1] (times 3 to make normal stronger)
        reconstruct_n=0.5*normalize_vec(reconstruct_n)+0.5 #[0,1]
        return reconstruct_n

    ## height map --> normal
    elif normal.shape[-1]==1:
        return height_to_normal(normal, intensity=opt.HN_factor) #[0,1]



###### this is for process des 19 normal
# input [B,W,H,2], [0, 1] --> normalized [-1,1]
def Process_des19normal(normal):
    if normal.dim()==4:
        b,w,h,c=normal.shape
        ### [0,1] --> [-1,1]
        normal=normal*2-1
        ### add 1 as 3rd channel && normalization
        reconstruct_n=torch.cat((normal*3, torch.ones([b,w,h,1]).cuda()),dim=-1) #[-1,1]
        reconstruct_n=0.5*normalize_vec(reconstruct_n)+0.5 #[0,1]
        ### [-1,1] --> [0,1]
        # reconstruct_n=Deprocess(reconstruct_n)
    elif normal.dim()==3:
        w,h,c=normal.shape
        ### [0,1] --> [-1,1]
        normal=normal*2-1
        ### add 1 as 3rd channel && normalization
        reconstruct_n=torch.cat((normal*3, torch.ones([w,h,1]).cuda()),dim=-1) #[-1,1]
        reconstruct_n=0.5*normalize_vec(reconstruct_n)+0.5 #[0,1]

        ### [-1,1] --> [0,1]
        # reconstruct_n=Deprocess(reconstruct_n)

    return reconstruct_n


def NormMeanStd(image_tensor, Mean, Std):
    return ( image_tensor - Mean) / Std 

def Inverese_NormMeanStd(image_tensor, Mean, Std):
    return image_tensor*Std + Mean 

def RenderProcess(opt, image):
    # LDR: 2.2 correction; [0,1] -> [-1,1]
    if opt.LDR and not opt.no_gamma_in:
        image=torch.clamp(image,0,1)
        image=logTensor(image)*2-1
    # HDR: log correction;
    elif not opt.LDR and not opt.no_gamma_in:
        image=torch.log(image+EPSILON+1)
    else:
        image=image*2-1
    return image

def VGGpreprocess(x):  
    mean = torch.tensor([0.485, 0.456, 0.406]).cuda().unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225]).cuda().unsqueeze(-1).unsqueeze(-1)
    return (x-mean)/std

def save_loss(loss_dict, save_dir, step, save_name=None):

    if save_name is None:
        plt.figure()
        for i in loss_dict:
            plt.plot(step, loss_dict[i], label='%s' % i)
        plt.legend()
        plt.savefig(save_dir+'/losses.png')
        plt.close()
    else:
        plt.figure()
        for i in loss_dict:
            plt.plot(step, loss_dict[i], label='%s' % i)
        plt.legend()
        plt.savefig(save_dir+'/losses%s.png'%save_name)
        plt.close()


def tensor2im(image_tensor, imtype=np.uint8, normalize=False, gamma=False, InverseLog=False):

    image_numpy = image_tensor.cpu().float().numpy()
    # print('1: ',image_numpy.shape)
    if normalize:
        image_numpy = (image_numpy + 1) / 2.0 
        
        if InverseLog:
            temp=image_numpy*(np.log(1.01)-np.log(0.01))+np.log(0.01)
            image_numpy=np.exp(temp)-0.01

        if gamma:
            image_numpy=(image_numpy+EPSILON)**(1/2.2)
    else:

        if InverseLog:
            temp=image_numpy*(np.log(1.01)-np.log(0.01))+np.log(0.01)
            image_numpy=np.exp(temp)-0.01

        if gamma:
            image_numpy=(image_numpy+EPSILON)**(1/2.2)

    image_numpy = np.clip(image_numpy*255, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    # print('2: ',image_numpy.shape)

    return image_numpy.astype(imtype)


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
def Gaussian_filter(kernel_size=7, sigma=2.5, channels=3):
    print('kernel size: ',kernel_size, "sigma: ", sigma)
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2*PI*variance))*torch.exp(-torch.sum((xy_grid - mean)**2, dim=-1)/(2*variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    padding = nn.ReflectionPad2d(int((kernel_size-1)*0.5))

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size,groups=channels, bias=False).cuda()

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter,padding


def save_image(image_numpy, image_path):

    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path, format='JPEG', subsampling=0, quality=100)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def matrix_evaluator( model, params, lam, syn_In, gt_Fe, LightPos_syn, regu_coef=1.0, lam_damping=10.0):
    """
    Constructor function that can be given to CG optimizer
    Works for both type(lam) == float and type(lam) == np.ndarray
    """
    if type(lam) == np.ndarray:
        lam = torch.from_numpy(lam).float().cuda()

    def evaluator(v):
        # print('evaluator')
        hvp = hessian_vector_product(model, v, params, syn_In, gt_Fe, LightPos_syn)
        Av = (1.0 + regu_coef) * v + hvp / (lam + lam_damping)
        return Av

    return evaluator

# @torch.enable_grad()
def hessian_vector_product(model, vector, params, syn_In, gt_Fe, LightPos_syn):
    """
    Performs hessian vector product on the train set in task with the provided vector
    """
    # print('perform hessian')

    loss_imaml,inner_Output = model(in_image_linear=syn_In[0:1,...].permute(0,1,4,2,3), gt_fe=gt_Fe[0:1,...], LightPos=LightPos_syn, steps=0, IO='I', params = params)
    grad_ft = torch.autograd.grad(loss_imaml['loss_total'], params.values(),create_graph=True)
    flat_in_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
    h = torch.sum(flat_in_grad * vector.cuda())
    hvp = torch.autograd.grad(h, params.values())
    hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])

    # print('hvp_flat: ', hvp_flat[0])
    # print()
    return hvp_flat



def cg_solve(f_Ax, b, cg_iters=2, callback=None, verbose=False, residual_tol=1e-20, x_init=None):
    """
    Goal: Solve Ax=b equivalent to minimizing f(x) = 1/2 x^T A x - x^T b
    Assumption: A is PSD, no damping term is used here (must be damped externally in f_Ax)
    Algorithm template from wikipedia
    Verbose mode works only with numpy
    """
       
    if type(b) == torch.Tensor:
        x = torch.zeros(b.shape[0]) if x_init is None else x_init
        x = x.to(b.device)
        if b.dtype == torch.float16:
            x = x.half()
        r = b - f_Ax(x)
        p = r.clone()
    elif type(b) == np.ndarray:
        x = np.zeros_like(b) if x_init is None else x_init
        r = b - f_Ax(x)
        p = r.copy()
    else:
        print("Type error in cg")

    fmtstr = "%10i %10.3g %10.3g %10.3g"
    titlestr = "%10s %10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm", "obj fn"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose:
            obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
            norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
            print(fmtstr % (i, r.dot(r), norm_x, obj_fn))

        rdotr = r.dot(r)
        Ap = f_Ax(p)
        alpha = rdotr/(p.dot(Ap))
        x = x + alpha * p
        r = r - alpha * Ap
        newrdotr = r.dot(r)
        beta = newrdotr/rdotr
        p = r + beta * p

        print(i, ' residual: ', newrdotr)

        if newrdotr < residual_tol:
            # print("Early CG termination because the residual was small")
            break

    if callback is not None:
        callback(x)
    if verbose: 
        obj_fn = 0.5*x.dot(f_Ax(x)) - 0.5*b.dot(x)
        norm_x = torch.norm(x) if type(x) == torch.Tensor else np.linalg.norm(x)
        print(fmtstr % (i, r.dot(r), norm_x, obj_fn))
    return x

def regularization_loss(model, w_1, lam=0.0):
    """
    Add a regularization loss onto the weights
    The proximal term regularizes around the point w_0
    Strength of regularization is lambda
    lambda can either be scalar (type float) or ndarray (numpy.ndarray)
    """
    index = 0
    delta = 0
    # for w0, w1 in zip(w_0, w_1):
    for i in model:
        # print(i,' shape1: ', w_0[i].shape, 'shape2: ', w_1[i].shape)
        index += 1
        delta += 0.5*torch.sum((model[i] - w_1[i])**2) * lam

    return delta


############################################## normal height conversion ######################################3

# img_in: [H,W,C] or [B,H,W,C]
def height_to_normal(img_in, mode='tangent_space', normal_format='gl', intensity=10.0):
    """Atomic function: Normal (https://docs.substance3d.com/sddoc/normal-172825289.html)

    Args:
        img_in (tensor): Input image.
        mode (str, optional): 'tangent space' or 'object space'. Defaults to 'tangent_space'.
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
        use_input_alpha (bool, optional): Use input alpha. Defaults to False.
        use_alpha (bool, optional): Enable the alpha channel. Defaults to False.
        intensity (float, optional): Normalized height map multiplier on dx, dy. Defaults to 1.0/3.0.
        max_intensity (float, optional): Maximum height map multiplier. Defaults to 3.0.

    Returns:
        Tensor: Normal image.
    """
    # grayscale_input_check(img_in, "input height field")
    assert img_in.shape[-1]==1, 'should be grayscale image'

    def roll_row(img_in, n):
        if img_in.dim()==3:
            return img_in.roll(-n, 0)
        elif img_in.dim()==4:
            return img_in.roll(-n, 1)

    def roll_col(img_in, n):
        if img_in.dim()==3:
            return img_in.roll(-n, 1)
        elif img_in.dim()==4:
            return img_in.roll(-n, 2)

    def norm(vec): 
        vec = vec.div(vec.norm(2.0, -1, keepdim=True))
        return vec

    # intensity = 10.0
    # intensity = (intensity * 2.0 - 1.0) * max_intensity * img_size / 256.0 # magic number to match sbs, check it later
    dx = roll_col(img_in, -1) - img_in
    dy = roll_row(img_in, -1) - img_in
    if normal_format == 'gl':
        img_out = torch.cat((intensity*dx, -intensity*dy, torch.ones_like(dx)), -1)
    elif normal_format == 'dx':
        img_out = torch.cat((intensity*dx, intensity*dy, torch.ones_like(dx)), -1)
    else:
        img_out = torch.cat((-intensity*dx, intensity*dy, torch.ones_like(dx)), -1)

    img_out = norm(img_out)
    if mode == 'tangent_space':
        img_out = img_out / 2.0 + 0.5

    return img_out




to_tensor = lambda a: torch.as_tensor(a, dtype=torch.float)

to_zero_one = lambda a: a / 2.0 + 0.5

def frequency_transform(img_in, normal_format='dx'):
    """Calculate convolution at multiple frequency levels.

    Args:
        img_in (tensor): input image
        normal_format (str, optional): switch for inverting the vertical 1-D convolution direction ('dx'|'gl'). Defaults to 'dx'.

    Returns:
        List[List[Tensor]]: list of convoluted images (in X and Y direction respectively)
    """
    def create_mipmaps(img_in, mipmaps_level, keep_size=False):
        """Create mipmap levels for an input image using box filtering.

        Args:
            img_in (tensor): input image
            mipmaps_level (int): number of mipmap levels
            keep_size (bool, optional): switch for restoring the original image size after downsampling. Defaults to False.

        Returns:
            List[Tensor]: mipmap stack
        """
        mipmaps = []
        img_mm = img_in
        last_shape = img_in.shape[2]
        for i in range(mipmaps_level):
            img_mm = manual_resize(img_mm, -1) if img_mm.shape[2] > 1 else img_mm
            mipmaps.append(img_mm if not keep_size else \
                           mipmaps[-1] if last_shape == 1 else \
                           img_mm.expand_as(img_in) if last_shape == 2 else \
                           automatic_resize(img_mm, i + 1))
            last_shape = img_mm.shape[2]
        return mipmaps

    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))
    print(in_size_log2)

    # Create mipmap levels for R and G channels
    img_in = img_in[:, :2, :, :]
    mm_list = [img_in]
    if in_size_log2 > 4:
        mm_list.extend(create_mipmaps(img_in, in_size_log2 - 4))

    # Define convolution operators
    def conv_x(img):
        img_bw = torch.clamp(img - roll_col(img, -1), -0.5, 0.5)
        img_fw = torch.clamp(roll_col(img, 1) - img, -0.5, 0.5)
        return (img_fw + img_bw) * 0.5 + 0.5

    def conv_y(img):
        dr = -1 if normal_format == 'dx' else 1
        img_bw = torch.clamp(img - roll_row(img, dr), -0.5, 0.5)
        img_fw = torch.clamp(roll_row(img, -dr) - img, -0.5, 0.5)
        return (img_fw + img_bw) * 0.5 + 0.5



    conv_ops = [conv_x, conv_y]

    # Init blended images
    img_freqs = [[], []]

    # Low frequencies (for 16x16 images only)
    img_4 = mm_list[-1]
    img_4_scale = [None, None, None, img_4]
    for i in range(3):
        img_4_scale[i] = transform_2d(img_4, x1=to_zero_one(2.0 ** (3 - i)), y2=to_zero_one(2.0 ** (3 - i)))
    for i, scale in enumerate([8.0, 4.0, 2.0, 1.0]):
        for c in (0, 1):
            img_4_c = conv_ops[c](img_4_scale[i][:, [c], :, :])
            if scale > 1.0:
                img_4_c = transform_2d(img_4_c, mipmap_mode='manual', x1=to_zero_one(1.0 / scale), y2=to_zero_one(1.0 / scale))
            img_freqs[c].append(img_4_c)

    # Other frequencies
    for i in range(len(mm_list) - 1):
        for c in (0, 1):
            img_i_c = conv_ops[c](mm_list[-2 - i][:, [c], :, :])
            img_freqs[c].append(img_i_c)

    return img_freqs

def normal_to_height(img_in, normal_format='dx', relief_balance=[0.5, 0.5, 0.5], opacity=0.36, max_opacity=1.0):
    """Non-atomic function: Normal to Height (https://docs.substance3d.com/sddoc/normal-to-height-159450591.html)

    Args:
        img_in (tensor): Input image.
        normal_format (str, optional): 'gl' or 'dx'. Defaults to 'dx'.
        relief_balance (list, optional): Adjust the extent to which the different frequencies influence the final result. 
        This is largely dependent on the input map and requires a fair bit of tweaking. Defaults to [0.5, 0.5, 0.5].
        opacity (float, optional): Normalized global opacity of the effect. Defaults to 0.36.
        max_opacity (float, optional): Maximum global opacity of the effect. Defaults to 1.0.

    Returns:
        Tensor: Height image.
    """

    assert img_in.shape[2] == img_in.shape[3], 'input image must be in square shape'
    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))
    assert in_size_log2 >= 7, 'input size must be at least 128'

    # Construct variables
    low_freq = to_tensor(relief_balance[0])
    mid_freq = to_tensor(relief_balance[1])
    high_freq = to_tensor(relief_balance[2])
    opacity = to_tensor(opacity) * max_opacity

    # Frequency transform for R and G channels
    img_freqs = frequency_transform(img_in[:,:2,:,:], normal_format)
    img_blend = [None, None]

    # Low frequencies (for 16x16 images only)
    for i in range(4):
        for c in (0, 1):
            img_i_c = img_freqs[c][i]
            blend_opacity = torch.clamp(0.0625 * 2 * (8 >> i) * low_freq * 100 * opacity, 0.0, 1.0)
            img_blend[c] = img_i_c if img_blend[c] is None else blend(img_i_c, img_blend[c], blending_mode='add_sub', opacity=blend_opacity)

    # Mid frequencies
    for i in range(min(2, len(img_freqs[0]) - 4)):
        for c in (0, 1):
            img_i_c = img_freqs[c][i + 4]
            # print(c)
            # print(img_i_c.shape)
            # print(img_blend[c].shape)            
            blend_opacity = torch.clamp(0.0156 * 2 * (2 >> i) * mid_freq * 100 * opacity, 0.0, 1.0)
            img_blend[c] = blend(img_i_c, manual_resize(img_blend[c], 1), blending_mode='add_sub', opacity=blend_opacity)

    # High frequencies
    for i in range(min(6, len(img_freqs[0]) - 6)):
        for c in (0, 1):
            img_i_c = img_freqs[c][i + 6]
            blend_opacity = torch.clamp(0.0078 * 0.0625 * (32 >> i) * high_freq * 100 * opacity, 0.0, 1.0) if i < 5 else \
                            torch.clamp(0.0078 * 0.0612 * high_freq * 100 * opacity)
            # print(c)
            # print(img_i_c.shape)
            # print(img_blend[c].shape)

            img_blend[c] = blend(img_i_c, manual_resize(img_blend[c], 1), blending_mode='add_sub', opacity=blend_opacity)

    # Combine both channels
    img_out = blend(img_blend[0], img_blend[1], blending_mode='add_sub', opacity=0.5)
    return img_out

def automatic_resize(img_in, scale_log2, filtering='bilinear'):
    """Progressively resize an input image.

    Args:
        img_in (tensor): input image
        scale_log2 (int): size change relative to the input resolution (after log2)
        filtering (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.

    Returns:
        Tensor: resized image
    """
    # Check input validity
    assert filtering in ('bilinear', 'nearest')
    in_size_log2 = int(np.log2(img_in.shape[2]))
    out_size_log2 = max(in_size_log2 + scale_log2, 0)

    # Equal size
    if out_size_log2 == in_size_log2:
        img_out = img_in
    # Down-sampling (regardless of filtering)
    elif out_size_log2 < in_size_log2:
        img_out = img_in
        for _ in range(in_size_log2 - out_size_log2):
            img_out = manual_resize(img_out, -1)
    # Up-sampling (progressive bilinear filtering)
    elif filtering == 'bilinear':
        img_out = img_in
        for _ in range(scale_log2):
            img_out = manual_resize(img_out, 1)
    # Up-sampling (nearest sampling)
    else:
        img_out = manual_resize(img_in, scale_log2, filtering)

    return img_out


def blend(img_fg=None, img_bg=None, blend_mask=None, blending_mode='copy', cropping=[0.0,1.0,0.0,1.0], opacity=1.0):
    """Atomic function: Blend (https://docs.substance3d.com/sddoc/blending-modes-description-132120605.html)

    Args:
        img_fg (tensor, optional): Foreground image (G or RGB(A)). Defaults to None.
        img_bg (tensor, optional): Background image (G or RGB(A)). Defaults to None.
        blend_mask (tensor, optional): Blending mask (G only). Defaults to None.
        blending_mode (str, optional): 
            copy|add|subtract|multiply|add_sub|max|min|divide|switch|overlay|screen|soft_light. 
            Defaults to 'copy'.
        cropping (list, optional): [left, right, top, bottom]. Defaults to [0.0,1.0,0.0,1.0].
        opacity (float, optional): Alpha mask. Defaults to 1.0.

    Returns:
        Tensor: Blended image.
    """
    if img_fg is not None:
        img_fg = to_tensor(img_fg)
    else:
        img_fg = to_tensor(0.0)
        img_fg_alpha = 0.0
    if img_bg is not None:
        img_bg = to_tensor(img_bg)
        if len(img_fg.shape):
            assert img_fg.shape[1] == img_bg.shape[1], 'foreground and background image type does not match' 
    else:
        img_bg = to_tensor(0.0)
        img_bg_alpha = 0.0
    if blend_mask is not None:
        blend_mask = to_tensor(blend_mask)
        # grayscale_input_check(blend_mask, 'blend mask')
        weight = blend_mask * opacity
    else:
        weight = opacity

    # compute output alpha channel
    use_alpha = False
    if len(img_fg.shape) and img_fg.shape[1] == 4:
        img_fg_alpha = img_fg[:,[3],:,:]
        img_fg = img_fg[:,:3,:,:]
        use_alpha = True
    if len(img_bg.shape) and img_bg.shape[1] == 4:
        img_bg_alpha = img_bg[:,[3],:,:]
        img_bg = img_bg[:,:3,:,:]
        use_alpha = True
    if use_alpha:
        weight = weight * img_fg_alpha
        img_out_alpha = weight + img_bg_alpha * (1.0 - weight)

    clamp_max = 1.0
    clamp_min = 0.0

    if blending_mode == 'add_sub': 
        img_fg = (img_fg - 0.5) * 2.0
        img_out = torch.clamp(img_fg * weight + img_bg, clamp_min, clamp_max)
    
    # apply cropping
    if cropping[0] == 0.0 and cropping[1] == 1.0 and cropping[2] == 0.0 and cropping[3] == 1.0:
        img_out_crop = img_out
    else:    
        start_row = math.floor(cropping[2] * img_out.shape[2])
        end_row = math.floor(cropping[3] * img_out.shape[2])
        start_col = math.floor(cropping[0] * img_out.shape[3])
        end_col = math.floor(cropping[1] * img_out.shape[3])
        img_out_crop = img_bg.clone()
        img_out_crop[:,:,start_row:end_row, start_col:end_col] = img_out[:,:,start_row:end_row, start_col:end_col]

    if use_alpha == True:
        img_out_crop = torch.cat([img_out_crop, img_out_alpha], dim=1)

    return img_out_crop

def manual_resize(img_in, scale_log2, filtering='bilinear'):
    """Manually resize an input image (all-in-one sampling).

    Args:
        img_in (tensor): input image
        scale_log2 (int): size change relative to input (after log2).
        filtering (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.

    Returns:
        Tensor: resized image
    """
    # Check input validity
    assert filtering in ('bilinear', 'nearest')
    in_size = img_in.shape[2]
    in_size_log2 = int(np.log2(in_size))
    out_size_log2 = max(in_size_log2 + scale_log2, 0)
    out_size = 1 << out_size_log2

    # Equal size
    if out_size_log2 == in_size_log2:
        img_out = img_in
    else:
        row_grid, col_grid = torch.meshgrid(torch.linspace(1, out_size * 2 - 1, out_size), torch.linspace(1, out_size * 2 - 1, out_size))
        sample_grid = torch.stack([col_grid, row_grid], 2).expand(img_in.shape[0], out_size, out_size, 2)
        sample_grid = sample_grid / (out_size * 2) * 2.0 - 1.0
        # Down-sampling
        if out_size_log2 < in_size_log2:
            img_out = torch.nn.functional.grid_sample(img_in, sample_grid, filtering, 'zeros', align_corners=False)
        # Up-sampling
        else:
            sample_grid = sample_grid * in_size / (in_size + 2)
            img_in_pad = torch.nn.functional.pad(img_in, (1, 1, 1, 1), mode='circular')
            img_out = torch.nn.functional.grid_sample(img_in_pad, sample_grid, filtering, 'zeros', align_corners=False)

    return img_out

def resize_color(color, num_channels):
    """Resize color to a specified number of channels.

    Args:
        color (tensor): input color
        num_channels (int): target number of channels

    Raises:
        ValueError: Resizing failed due to channel mismatch.

    Returns:
        Tensor: resized color
    """
    assert color.ndim == 1
    assert num_channels >= 1 and num_channels <= 4

    # Match background color with image channels
    if len(color) > num_channels:
        color = color[:num_channels]
    elif num_channels == 4 and len(color) == 3:
        color = th.cat((color, th.ones(1)))
    elif len(color) == 1:
        color = color.repeat(num_channels)
    elif len(color) != num_channels:
        raise ValueError('Channel mismatch between input image and background color')

    return color

def transform_2d(img_in, tile_mode=3, sample_mode='bilinear', mipmap_mode='auto', mipmap_level=0, x1=1.0, x1_max=1.0, x2=0.5, x2_max=1.0,
                 x_offset=0.5, x_offset_max=1.0, y1=0.5, y1_max=1.0, y2=1.0, y2_max=1.0, y_offset=0.5, y_offset_max=1.0,
                 matte_color=[0.0, 0.0, 0.0, 1.0]):
    """Atomic function: Transform 2D (https://docs.substance3d.com/sddoc/transformation-2d-172825332.html)

    Args:
        img_in (tensor): input image
        tile_mode (int, optional): 0=no tile, 
                                   1=horizontal tile, 
                                   2=vertical tile, 
                                   3=horizontal and vertical tile. Defaults to 3.
        sample_mode (str, optional): 'bilinear' or 'nearest'. Defaults to 'bilinear'.
        mipmap_mode (str, optional): 'auto' or 'manual'. Defaults to 'auto'.
        mipmap_level (int, optional): Mipmap level. Defaults to 0.
        x1 (float, optional): Entry in the affine transformation matrix, same for the below. Defaults to 1.0.
        x1_max (float, optional): . Defaults to 1.0.
        x2 (float, optional): . Defaults to 0.5.
        x2_max (float, optional): . Defaults to 1.0.
        x_offset (float, optional): . Defaults to 0.5.
        x_offset_max (float, optional): . Defaults to 1.0.
        y1 (float, optional): . Defaults to 0.5.
        y1_max (float, optional): . Defaults to 1.0.
        y2 (float, optional): . Defaults to 1.0.
        y2_max (float, optional): . Defaults to 1.0.
        y_offset (float, optional): . Defaults to 0.5.
        y_offset_max (float, optional): . Defaults to 1.0.
        matte_color (list, optional): background color. Defaults to [0.0, 0.0, 0.0, 1.0].

    Returns:
        Tensor: Transformed image.
    """
    assert sample_mode in ('bilinear', 'nearest')
    assert mipmap_mode in ('auto', 'manual')

    gs_padding_mode = 'zeros'
    gs_interp_mode = sample_mode

    x1 = to_tensor((x1 * 2.0 - 1.0) * x1_max).squeeze()
    x2 = to_tensor((x2 * 2.0 - 1.0) * x2_max).squeeze()
    x_offset = to_tensor((x_offset * 2.0 - 1.0) * x_offset_max).squeeze()
    y1 = to_tensor((y1 * 2.0 - 1.0) * y1_max).squeeze()
    y2 = to_tensor((y2 * 2.0 - 1.0) * y2_max).squeeze()
    y_offset = to_tensor((y_offset * 2.0 - 1.0) * y_offset_max).squeeze()
    matte_color = to_tensor(matte_color).view(-1)
    matte_color = resize_color(matte_color, img_in.shape[1])

    # compute mipmap level
    mm_level = mipmap_level
    det = torch.abs(x1 * y2 - x2 * y1)
    if det < 1e-6:
        print('Warning: singular transformation matrix may lead to unexpected results.')
        mm_level = 0
    elif mipmap_mode == 'auto':
        inv_h1 = torch.sqrt(x2 * x2 + y2 * y2)
        inv_h2 = torch.sqrt(x1 * x1 + y1 * y1)
        max_compress_ratio = torch.max(inv_h1, inv_h2)
        # !! this is a hack !!
        upper_limit = 2895.329
        thresholds = to_tensor([upper_limit / (1 << i) for i in reversed(range(12))])
        mm_level = torch.sum(max_compress_ratio > thresholds).item()
        # Special cases
        is_pow2 = lambda x: torch.remainder(torch.log2(x), 1.0) == 0
        if torch.abs(x1) == torch.abs(y2) and x2 == 0 and y1 == 0 and is_pow2(torch.abs(x1)) or \
           torch.abs(x2) == torch.abs(y1) and x1 == 0 and y2 == 0 and is_pow2(torch.abs(x2)):
            scale = torch.max(torch.abs(x1), torch.abs(x2))
            if torch.remainder(x_offset * scale, 1.0) == 0 and torch.remainder(y_offset * scale, 1.0) == 0:
                mm_level = max(0, mm_level - 1)

    # mipmapping (optional)
    if mm_level > 0:
        mm_level = min(mm_level, int(np.floor(np.log2(img_in.shape[2]))))
        img_mm = automatic_resize(img_in, -mm_level)
        img_mm = manual_resize(img_mm, mm_level)
        assert img_mm.shape == img_in.shape
    else:
        img_mm = img_in

    # compute sampling tensor
    res_x, res_y = img_in.shape[3], img_in.shape[2]
    theta_first_row = torch.stack([x1, y1, x_offset * 2.0])
    theta_second_row = torch.stack([x2, y2, y_offset * 2.0])
    theta = torch.stack([theta_first_row, theta_second_row]).unsqueeze(0).expand(img_in.shape[0],2,3)
    sample_grid = torch.nn.functional.affine_grid(theta, img_in.shape, align_corners=False)

    if tile_mode in (1, 3):
        sample_grid[:,:,:,0] = (torch.remainder(sample_grid[:,:,:,0] + 1.0, 2.0) - 1.0) * res_x / (res_x + 2)
    if tile_mode in (2, 3):
        sample_grid[:,:,:,1] = (torch.remainder(sample_grid[:,:,:,1] + 1.0, 2.0) - 1.0) * res_y / (res_y + 2)

    # Deduce background color from the image if tiling is not fully applied
    if tile_mode < 3:
        img_mm = img_mm - matte_color[:, None, None]

    # Pad input image
    if tile_mode == 0:
        img_pad = img_mm
    else:
        pad_arr = [[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]]
        img_pad = torch.nn.functional.pad(img_mm, pad_arr[tile_mode], mode='circular')

    # compute output
    img_out = torch.nn.functional.grid_sample(img_pad, sample_grid, mode=gs_interp_mode, padding_mode=gs_padding_mode, align_corners=False)

    # Add the background color back after sampling
    if tile_mode < 3:
        img_out = img_out + matte_color[:, None, None]

    return img_out


import ctypes
import fnmatch
import importlib
import inspect
import numpy as np
import os
import shutil
import sys
import types
import io
import pickle
import re
import requests
import html
import hashlib
import glob
import tempfile
import urllib
import urllib.request
import uuid

from typing import Any, List, Tuple, Union

_dnnlib_cache_dir = None


def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if 'DNNLIB_CACHE_DIR' in os.environ:
        return os.path.join(os.environ['DNNLIB_CACHE_DIR'], *paths)
    if 'HOME' in os.environ:
        return os.path.join(os.environ['HOME'], '.cache', 'dnnlib', *paths)
    if 'USERPROFILE' in os.environ:
        return os.path.join(os.environ['USERPROFILE'], '.cache', 'dnnlib', *paths)
    return os.path.join(tempfile.gettempdir(), '.cache', 'dnnlib', *paths)

def open_url(url: str, cache_dir: str = None, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False, cache: bool = True) -> Any:
    """Download the given URL and return a binary-mode file object to access the data."""
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    # assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path('downloads')

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file) # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


from torchvision.models.vgg import vgg19

class TextureDescriptor(nn.Module):

    def __init__(self, device):
        super(TextureDescriptor, self).__init__()
        self.device = device
        self.outputs = []

        # get VGG19 feature network in evaluation mode
        self.net = vgg19(True).features.to(device)
        self.net.eval()

        # change max pooling to average pooling
        for i, x in enumerate(self.net):
            if isinstance(x, nn.MaxPool2d):
                self.net[i] = nn.AvgPool2d(kernel_size=2)

        def hook(module, input, output):
            self.outputs.append(output)

        #for i in [6, 13, 26, 39]: # with BN
        for i in [4, 9, 18, 27]: # without BN
            self.net[i].register_forward_hook(hook)

        # weight proportional to num. of feature channels [Aittala 2016]
        self.weights = [1, 2, 4, 8, 8]

        # this appears to be standard for the ImageNet models in torchvision.models;
        # takes image input in [0,1] and transforms to roughly zero mean and unit stddev
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    def forward(self, x):
        self.outputs = []

        # run VGG features
        x = self.net(x)
        self.outputs.append(x)

        result = []
        batch = self.outputs[0].shape[0]

        for i in range(batch):
            temp_result = []
            for j, F in enumerate(self.outputs):

                # print(j, ' shape: ', F.shape)

                F_slice = F[i,:,:,:]
                f, s1, s2 = F_slice.shape
                s = s1 * s2
                F_slice = F_slice.view((f, s))

                # Gram matrix
                G = torch.mm(F_slice, F_slice.t()) / s
                temp_result.append(G.flatten())
            temp_result = torch.cat(temp_result)

            result.append(temp_result)
        return torch.stack(result)

    def eval_CHW_tensor(self, x):
        "only takes a pytorch tensor of size B * C * H * W"
        assert len(x.shape) == 4, "input Tensor cannot be reduced to a 3D tensor"
        x = (x - self.mean) / self.std
        return self.forward(x.to(self.device))


class TDLoss(nn.Module):
    def __init__(self, GT_img, device, num_pyramid):
        super(TDLoss, self).__init__()
        # create texture descriptor
        self.net_td = TextureDescriptor(device) 
        # fix parameters for evaluation 
        for param in self.net_td.parameters(): 
            param.requires_grad = False 

        self.num_pyramid = num_pyramid

        self.GT_td = self.compute_td_pyramid(GT_img.to(device))


    def forward(self, img):

        # td1 = self.compute_td_pyramid(img1)
        td = self.compute_td_pyramid(img)

        tdloss = (td - self.GT_td).abs().mean() 

        return tdloss


    def compute_td_pyramid(self, img): # img: [0,1]
        """compute texture descriptor pyramid

        Args:
            img (tensor): 4D tensor of image (NCHW)
            num_pyramid (int): pyramid level]

        Returns:
            Tensor: 2-d tensor of texture descriptor
        """    
        # print('img type',img[0,:,0,0])
        # print('img type',img.dtype)

        # if img.dtype=='torch.uint8':

        td = self.net_td.eval_CHW_tensor(img) 
        for scale in range(self.num_pyramid):
            td_ = self.net_td.eval_CHW_tensor(nn.functional.interpolate(img, scale_factor = 1.0/(2.0**(scale+1)), mode='bilinear', align_corners=True, recompute_scale_factor=True))
            td = torch.cat([td, td_], dim=1) 
        return td
