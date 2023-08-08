

NORMALIZE_DICT = {
    'mnist':    dict( mean=(0.1307,),                std=(0.3081,) ),
    # 'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    # 'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'CIFAR10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'CIFAR100': dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'stl10': dict( mean= (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),
    'STL10': dict( mean= (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),
    'UTKFaceRace': dict( mean= (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),
    'utkfacerace': dict( mean= (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ),

}

IMGSIZE_DICT = {
    'mnist': 28,
    'cifar10': 32,
    'cifar100': 32,
    'CIFAR10': 32,
    'CIFAR100': 32,
    'stl10': 64,
    'STL10': 64,
    'UTKFaceRace': 64,
    'utkfacerace': 64,

}