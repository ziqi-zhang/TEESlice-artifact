import os
import torch
import pandas
import torchvision
torch.manual_seed(0)
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms
from torchvision.datasets import STL10 as RawSTL10
import numpy as np
import os.path as osp

from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple
from pdb import set_trace as st
from sklearn.model_selection import train_test_split
import knockoff.config as cfg

class STL10(RawSTL10):
    def __init__(self, train=True, transform=None, target_transform=None, download=True):
        root = osp.join(cfg.DATASET_ROOT, 'stl10')
        if train:
            split = 'train'
        else:
            split = 'test'
        super().__init__(root, split, None, transform, target_transform, True)



class UTKFace(torch.utils.data.Dataset):
    # def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
    def __init__(self, train=True, transform=None, target_transform=None, attr = "age")-> None:

        # self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.files = os.listdir(self.root+'/UTKFace/processed/')
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]
        if attr == "gender":
            self.classes = list(range(2))
        elif attr == "race":
            self.classes = list(range(4))
        elif attr == "age":
            self.classes = list(range(117))
        else:
            raise RuntimeError
        

        self.raw_lines = []
        # self.samples = []
        for txt_file in self.files:
            with open(self.root+'/UTKFace/processed/' + txt_file, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4:
                        continue
                    if attrs[1] == "":
                        continue
                    self.raw_lines.append(image_name+'jpg')
                    # self.samples.append(

                    # )

        train_lines, test_lines = train_test_split(self.raw_lines, test_size=0.1, random_state=37)
        if train:
            self.lines = train_lines
        else:
            self.lines = test_lines
        print(self.raw_lines[:10])
        print("UTKFace raw samples: ", len(self.raw_lines))
        # exit()

    def getitem_to_numpy(self, index):
        image_path = os.path.join(self.root+'/UTKFace/UTKface_aligned_cropped/UTKFace', self.lines[index]+'.chip.jpg').rstrip()
        image = Image.open(image_path).convert('RGB')
        image = transforms.ToTensor()(image)
        image = (image.numpy()*255).astype(np.uint8).transpose(1,2,0)
        return image


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        
        attrs = self.lines[index].split('_')
        # print(self.lines[index], attrs, attrs[0], attrs[1], attrs[2])

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(self.root+'/UTKFace/UTKface_aligned_cropped/UTKFace', self.lines[index]+'.chip.jpg').rstrip()

        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target

class UTKFaceAge(UTKFace):
    # def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
    def __init__(self, train=True, transform=None, target_transform=None)-> None:
        super().__init__(train, transform, target_transform, "age")

class UTKFaceRace(UTKFace):
    # def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
    def __init__(self, train=True, transform=None, target_transform=None)-> None:
        super().__init__(train, transform, target_transform, "race")


class CelebA(torch.utils.data.Dataset):
    base_folder = "celeba"

    # def __init__(
    #         self,
    #         root: str,
    #         attr_list: str,
    #         target_type: Union[List[str], str] = "attr",
    #         transform: Optional[Callable] = None,
    #         target_transform: Optional[Callable] = None,
    # ) -> None:
    def __init__(
            self, train=True, transform=None, target_transform=None,
    ) -> None:
        attr_list = [[31, 39]]
        target_type = ["attr", "attr"]
        self.classes = list(range(4))

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.transform = transform
        self.target_transform =target_transform
        self.attr_list = attr_list

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

        # original size is 200k, select 50k
        torch.manual_seed(123)
        perm = torch.randperm(self.attr.size(0))
        train_idx = perm[:10000]
        test_idx = perm[100000:150000]
        if train:
            self.attr = self.attr[train_idx]
            self.filename = self.filename[train_idx]
        else:
            self.attr = self.attr[test_idx]
            self.filename = self.filename[test_idx]

    def getitem_to_numpy(self, index):
        # image_path = os.path.join(self.root+'/UTKFace/UTKface_aligned_cropped/UTKFace', self.lines[index]+'.chip.jpg').rstrip()
        # image = Image.open(image_path).convert('RGB')
        image = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
        image = transforms.ToTensor()(image)
        image = (image.numpy()*255).astype(np.uint8).transpose(1,2,0)
        return image


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t, nums in zip(self.target_type, self.attr_list):
            if t == "attr":
                final_attr = 0
                for i in range(len(nums)):
                    final_attr += 2 ** i * self.attr[index][nums[i]]
                target.append(final_attr)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
