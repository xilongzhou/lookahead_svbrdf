import os
import torch
import sys

import re
import warnings

from collections import OrderedDict

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        if opt.fileNo is None:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        else:
            self.save_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.fileNo+opt.name2)



    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, iter_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (iter_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()


    # helper saving function that can be used by subclasses
    def save_optimizer(self, optim, optim_label, iter_label, gpu_ids):
        save_filename = '%s_%s.pth' % (iter_label, optim_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(optim.state_dict(), save_path)


    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, iter_label, save_dir=''):        
        save_filename = '%s_net_%s.pth' % (iter_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            print('................loading network{}..........'.format(iter_label))
            # net = torch.load(save_path)
            # for parameters in net:
            #     print()
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:   
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}                    
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
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


    def load_optimizer(self, optim, optim_label, iter_label, save_dir=''):        
        save_filename = '%s_%s.pth' % (iter_label, optim_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)        
        if not os.path.isfile(save_path):
            raise('%s not exists yet!' % save_path)
        else:
            print('.................. loading {} at {}..........'.format(optim_label, iter_label))
            # net = torch.load(save_path)
            # for parameters in net:
            #     print()
            #network.load_state_dict(torch.load(save_path))
            optim.load_state_dict(torch.load(save_path))


    def update_learning_rate():
        pass



    """
    Base class for PyTorch meta-learning modules. These modules accept an
    additional argument `params` in their `forward` method.
    Notes
    -----
    Objects inherited from `MetaModule` are fully compatible with PyTorch
    modules from `torch.nn.Module`. The argument `params` is a dictionary of
    tensors, with full support of the computation graph (for differentiation).
    """

    def meta_named_parameters(self, prefix='', recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)

        for elem in gen:
            ## this is only for des19
            if self.opt.Net_Option=='Des19Net':
                flag = 'netG' in elem[0] and 'lastglobal_fc2' not in elem[0]            
                if flag: 
                    # print('elem: ',elem[0])
                    yield elem
            elif self.opt.Net_Option=='UNetS':
                flag = 'netG' in elem[0]
                if flag:
                    # print('elem: ',elem[0], ' shape: ', elem[1].shape)
                    yield elem
            elif self.opt.Net_Option=='Siren':
                yield elem


    def meta_parameters(self, recurse=True):
        for name, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(self, params, key=None):
        if params is None:
            return None
        all_names = tuple(params.keys())
        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names
            else:
                key_escape = re.escape(key)
                key_re = re.compile(r'^{0}\.(.+)'.format(key_escape))
                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r'\1', k) for k in all_names if key_re.match(k) is not None]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn('Module `{0}` has no parameter corresponding to the '
                          'submodule named `{1}` in the dictionary `params` '
                          'provided as an argument to `forward()`. Using the '
                          'default parameters for this submodule. The list of '
                          'the parameters in `params`: [{2}].'.format(
                          self.__class__.__name__, key, ', '.join(all_names)),
                          stacklevel=2)
            return None


        return OrderedDict([(name, params[f'{key}.{name}']) for name in names])