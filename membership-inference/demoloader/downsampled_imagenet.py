from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
from typing import Any, Callable, Optional, Tuple
from pdb import set_trace as st

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict
def load_databatch(data_folder, idx, img_size=64, prefix='train_data_batch_'):
    data_file = os.path.join(data_folder, prefix)

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']
    mean_image = d['mean']

    # x = x/np.float32(255)
    # mean_image = mean_image/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    # x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    # X_train = x[0:data_size, :, :, :]
    # Y_train = y[0:data_size]
    # X_train_flip = X_train[:, :, :, ::-1]
    # Y_train_flip = Y_train
    # X_train = np.concatenate((X_train, X_train_flip), axis=0)
    # Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)
    
    return x, y, mean_image


class DownsampledImagenet(VisionDataset):


    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            img_size=64,
    ) -> None:

        super(DownsampledImagenet, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set
        
        self.data: Any = []
        self.targets = []
        for i in range(1,11):
            print(f"Loading batch {i}")
            data, target, _ = load_databatch(root, i, img_size)
            self.data.append(data)
            self.targets.extend(target)
            # if i > 2:
            #     break

        self.data = np.vstack(self.data).reshape(-1, 3, img_size, img_size)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        
        self._load_meta()


    def _load_meta(self) -> None:
        self.classes = list(range(1000))
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")



