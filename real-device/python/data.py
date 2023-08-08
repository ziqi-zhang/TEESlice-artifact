import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

num_classes_dict = {
    "CIFAR10":10,
    "CIFAR100":100,
    "IDC":2,
    "IDCR":2,
}

def get_data(dataset, data_path, batch_size, num_workers):
    # assert dataset in ["CIFAR10", "CIFAR100", "IDC"]
    assert dataset in num_classes_dict.keys()
    print('Loading dataset {} from {}'.format(dataset, data_path))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_sampler = None
    val_sampler = None
    if dataset in ["CIFAR10",  "CIFAR100"]:
        ds = getattr(datasets, dataset.upper())
        path = os.path.join(data_path, dataset.lower())
        train_set = ds(path, train=True, download=True, transform=transform_train)
        val_set = ds(path, train=True, download=True, transform=transform_test)
        test_set = ds(path, train=False, download=True, transform=transform_test)
    elif dataset == "IDC":
        train_set = datasets.ImageFolder(root=f'{data_path}/idc_dataset/train', transform=transform_train)
        val_set = datasets.ImageFolder(root=f'{data_path}/idc_dataset/val', transform=transform_test)
        test_set = datasets.ImageFolder(root=f'{data_path}/idc_dataset/test', transform=transform_test)
    elif dataset == "IDCR":
        train_set = datasets.ImageFolder(root=f'{data_path}/idc_dataset_resize/train', transform=transform_train)
        val_set = datasets.ImageFolder(root=f'{data_path}/idc_dataset_resize/val', transform=transform_test)
        test_set = datasets.ImageFolder(root=f'{data_path}/idc_dataset_resize/test', transform=transform_test)
    else:
        raise Exception("Invalid dataset %s"%dataset)

    loaders = {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': torch.utils.data.DataLoader(
            val_set,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
       }

    return loaders