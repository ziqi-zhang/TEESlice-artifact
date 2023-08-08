import torch
import knockoff.models.zoo as zoo
import timm
import os
import os.path as osp
import sys
import pickle
from functools import partial

import knockoff.models.cifar
import knockoff.models.mnist
import knockoff.models.imagenet

timm_model_list = ['gluon_resnet18_v1b', 'resnet18', 'ssl_resnet18', 'swsl_resnet18']

def get_pretrained_model(model_name, pretrained=None, pretrained_dir=None, \
    modelfamily='cifar', model_arch='resnet18', num_classes=10):

    # pretrained: imagenet_for_cifar
    if pretrained:
        print(f'=> Loading pretrained model {pretrained}')
        return zoo.get_net(model_arch, modelfamily, pretrained=pretrained, \
            num_classes=num_classes)
    
    # pretrained model: timm or downloaded
    pretrained_model = eval('knockoff.models.{}.{}'.format(modelfamily, model_arch))(pretrained=False, \
        num_classes=num_classes)
    ori_state_dict = pretrained_model.state_dict()

    if model_name in timm_model_list:
        print(f'=> loading timm model {model_name}')
        pretrained_ckpt = timm.create_model(model_name, pretrained=True, num_classes=num_classes).state_dict()
    else:
        print(f'loading downloaded model {model_name}')
        if ('FractalDB' in model_name) or ('kaggle' in model_name):
            pretrained_path = osp.join(pretrained_dir, f'{model_name}.pth')
            pretrained_ckpt = torch.load(pretrained_path)
        
        elif 'places365' in model_name:
            pretrained_path = osp.join(pretrained_dir, f'{model_name}.pth.tar')
            # Remove the original module, but keep it around
            main_pickle = sys.modules.pop('pickle')
            # Get a modified copy of the module
            import pickle as modified_pickle
            modified_pickle.load = partial(modified_pickle.load, encoding='latin1')
            modified_pickle.Unpickler = partial(modified_pickle.Unpickler, encoding='latin1')
            pretrained_ckpt = torch.load(pretrained_path, map_location=lambda storage, loc: storage, \
                pickle_module=modified_pickle)['state_dict']
            pretrained_ckpt = {str.replace(k,'module.',''): v for k,v in pretrained_ckpt.items()}
            # Recover original module
            sys.modules['pickle'] = main_pickle
        
        # Change the fc to 'num_classes'
        for name, param in pretrained_ckpt.items():
            if 'fc' in name:
                print(model_name, name, param.data.size())
                pretrained_ckpt[name] = ori_state_dict[name]
    
    if pretrained_ckpt['conv1.weight'].size(-1) != ori_state_dict['conv1.weight'].size(-1):
        pretrained_ckpt['conv1.weight'] = pretrained_ckpt['conv1.weight'][:,:,2:-2,2:-2]
    pretrained_model.load_state_dict(pretrained_ckpt)

    return pretrained_model



def get_stored_net(model_name, model_arch, model_family, pretrained=None, \
    num_classes=10, ckpt_path=None, pretrained_path=None, ret_acc=False):
    """
    Get stored models: 
    * trained ckpt: dict, including state_dict, epoch, acc...
    * timm models
    * pretrained models in 'downloaded_models'
    """

    if ckpt_path:
        model = eval('knockoff.models.{}.{}'.format(model_family, model_arch))(pretrained=False, \
            num_classes=num_classes)
        print("=> loading checkpoint '{}'".format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        epoch = ckpt['epoch']
        best_test_acc = ckpt['best_acc']
        model.load_state_dict(ckpt['state_dict'])
        print("=> loaded checkpoint (epoch {}, acc={:.2f})".format(epoch, best_test_acc))
        if ret_acc:
            return model, best_test_acc
        return model
    
    if pretrained or (model_name in timm_model_list):
        return get_pretrained_model(model_name=model_name, pretrained=pretrained, \
            pretrained_dir=None, modelfamily=model_family, model_arch=model_arch, \
            num_classes=num_classes)
    
    if pretrained_path:
        pretrained_dir = '/'.join(pretrained_path.split('/')[:-1])
        return get_pretrained_model(model_name=model_name, pretrained=None, \
            pretrained_dir=pretrained_dir, modelfamily=model_family, model_arch=model_arch, \
            num_classes=num_classes)