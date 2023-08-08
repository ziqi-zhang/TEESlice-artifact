import os, sys
import copy
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os.path as osp
import json

from doctor.meminf_whitebox import *
from doctor.meminf_blackbox import *
from doctor.meminf_shadow import *
from doctor.meminf_whitebox_feature import *
from doctor.modinv import *
from doctor.attrinf import *
from doctor.modsteal import *
from demoloader.train import *
from demoloader.DCGAN import *
from utils.define_models import *
from demoloader.dataloader import *
from log_utils import *
from distill import *

import gol
gol._init()
gol.set_value("debug", False)

def load_target_model(target_model, args):
    target_path = osp.join(args.victim_dir, f"target.pth")
    target_model = target_model.cuda()
    print(f"Load model from {target_path}")
    target_model.load_state_dict(torch.load(target_path))
    target_model.eval()
    return target_model

def generate_transferset(target_model, queryset, args):
    target_model = load_target_model(target_model, args)
    refdataset = copy.deepcopy(queryset)
    
    if isinstance(refdataset.dataset, torch.utils.data.dataset.ConcatDataset):
        num_datasets = len(refdataset.dataset.datasets)
        for dataset_idx in range(num_datasets):
            transforms = refdataset.dataset.datasets[dataset_idx].transform.transforms
            refdataset.dataset.datasets[dataset_idx].transform.transforms = transforms[:-1]
            print(f"Query transforms {queryset.dataset.datasets[dataset_idx].transform.transforms}")
            print(f"Ref transforms {refdataset.dataset.datasets[dataset_idx].transform.transforms}")
    elif isinstance(refdataset.dataset, UTKFaceDataset):
        transforms = refdataset.dataset.transform.transforms
        refdataset.dataset.transform.transforms = transforms[:-1]
        print(f"Query transforms {queryset.dataset.transform.transforms}")
        print(f"Ref transforms {refdataset.dataset.transform.transforms}")
    else:
        raise NotImplementedError
    
    idx_set = set(range(len(queryset)))
    transferset = []
    
    start_B = 0
    end_B = args.trasnferset_budgets
    batch_size = 64
    with tqdm(total=args.trasnferset_budgets) as pbar:
        for t, B in enumerate(range(start_B, end_B, batch_size)):
            idxs = np.random.choice(list(idx_set), replace=False,
                                    size=min(batch_size, args.trasnferset_budgets - len(transferset)))
            idx_set = idx_set - set(idxs)

            if len(idx_set) == 0:
                print('=> Query set exhausted. Now repeating input examples.')
                idx_set = set(range(len(queryset)))

            x_t = torch.stack([queryset[i][0] for i in idxs]).cuda()
            gt_t = torch.Tensor([queryset[i][1] for i in idxs]).cuda()
            y_t = target_model(x_t).cpu()
            ref_x_t = [refdataset[i][0] for i in idxs]

            # if hasattr(raw_dataset, 'samples'):
            #     # Any DatasetFolder (or subclass) has this attribute
            #     # Saving image paths are space-efficient
            #     img_t = [raw_dataset.samples[queryset.indices[i]][0] for i in idxs]  # Image paths
            # elif hasattr(raw_dataset, 'getitem_to_numpy'):
            #     # st()
            #     # img = self.queryset.getitem_to_numpy(idxs[0])
            #     img_t = [raw_dataset.getitem_to_numpy(queryset.indices[i]) for i in idxs]
            #     # st()
            #     # if isinstance(self.queryset.getitem_to_numpy(0), torch.Tensor):
            #     #     img_t = [(x.numpy()*255).astype(np.uint8).transpose(1,2,0) for x in img_t]
            # elif isinstance(raw_dataset, torch.utils.data.dataset.ConcatDataset):
            #     # STL10
            #     st()
            # else:
            #     # Otherwise, store the image itself
            #     # But, we need to store the non-transformed version
            #     st()
            #     img_t = [raw_dataset.data[queryset.indices[i]] for i in idxs]
            #     if isinstance(raw_dataset.data[0], torch.Tensor):
            #         img_t = [x.numpy() for x in img_t]
            #     # st()

            for i in range(x_t.size(0)):
                # img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
                img_t_i = np.asarray(ref_x_t[i])
                if isinstance(refdataset.dataset, UTKFaceDataset):
                    img_t_i = img_t_i.transpose(1,2,0)
                    img_t_i = (img_t_i*255).astype(np.uint8)
                transferset.append((img_t_i, y_t[i].cpu().squeeze(), gt_t[i].cpu()))

            pbar.update(x_t.size(0))
            
    transferset_path = osp.join(args.out_path, "transferset.pickle")
    with open(transferset_path, 'wb') as wf:
        pickle.dump(transferset, wf)
    print('=> transfer set ({} samples) written to: {}'.format(len(transferset), transferset_path))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model')
    parser.add_argument('--victim_dir', type=str)
    parser.add_argument('--budgets', metavar='B', type=str,
                        help='Comma separated values of budgets. Knockoffs will be trained for each budget.')
    parser.add_argument('--trasnferset_budgets', type=int, default=1000)
    parser.add_argument('--argmaxed', action='store_true', help='Only consider argmax labels', default=False)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--shadow-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--shadow-lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', action="store_true", default=False)
    args = parser.parse_args()
    params = vars(args)

    torch.manual_seed(17)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, pretrained_model = prepare_dataset(
        args.dataset.lower(), args.model_arch, pretrained=args.pretrained
    )
    
    transferset_path = osp.join(args.out_path, "transferset.pickle")
    # if not os.path.exists(transferset_path):
    generate_transferset(target_model, shadow_train, args)

