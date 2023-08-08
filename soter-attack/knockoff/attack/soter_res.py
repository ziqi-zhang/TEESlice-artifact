# resnet18: 21 layers, vgg16_bn: 14 layers, alexnet: 6 layers
# theta_dict = {
#     'resnet18': [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0], 
#     'vgg16_bn': [1.0, 0.8, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0], 
#     'alexnet': [1.0, 0.9, 0.7, 0.5, 0.4, 0.3, 0.0], 
#     'resnet34': [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0], 
#     'vgg19_bn': [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0], 
# }

CIFAR10_resnet18_acc =  95.91
CIFAR10_resnet18_soter_stealing_50_protect_params =  [0, 3024640, 4980352, 7653760, 8613504, 10982272, 11131840, 11173962]
CIFAR10_resnet18_soter_stealing_50_protect_flops =  [0, 94486528.0, 226852864.0, 306675712.0, 439123968.0, 479051776.0, 518766592.0, 556651530.0]
CIFAR10_resnet18_soter_stealing_50_acc = [ 92.89, 92.61, 90.94, 86.42, 74.88, 64.41, 57.16, 21.57 ]
CIFAR10_resnet18_soter_stealing_50_fidelity = [ 94.61, 93.93, 92.36, 87.57, 75.60, 65.07, 57.85, 21.87 ]
CIFAR10_resnet18_soter_stealing_50_asr = [ 100.00, 99.73, 99.73, 95.08, 54.64, 42.08, 30.87, 8.74 ]

CIFAR10_vgg16_bn_acc =  92.95
CIFAR10_vgg16_bn_soter_stealing_50_protect_params =  [0, 1845888, 6567552, 9518976, 12323328, 12362304, 14723136, 14728266]
CIFAR10_vgg16_bn_soter_stealing_50_protect_flops =  [0, 75612160.0, 122818560.0, 170041344.0, 264527872.0, 304308224.0, 313749504.0, 313754634.0]
CIFAR10_vgg16_bn_soter_stealing_50_acc = [ 84.06, 81.52, 75.97, 66.46, 51.62, 50.68, 46.72, 12.30 ]
CIFAR10_vgg16_bn_soter_stealing_50_fidelity = [ 85.72, 83.16, 77.11, 67.45, 52.14, 51.07, 47.12, 12.46 ]
CIFAR10_vgg16_bn_soter_stealing_50_asr = [ 100.00, 100.00, 99.16, 84.96, 51.25, 50.97, 43.18, 25.35 ]

CIFAR10_alexnet_acc =  83.57
CIFAR10_alexnet_soter_stealing_50_protect_params =  [0, 884992, 1192384, 1856320, 2446400, 2455872, 2458442]
CIFAR10_alexnet_soter_stealing_50_protect_flops =  [0, 14155776.0, 33816576.0, 44433408.0, 53870592.0, 56279040.0, 56281610.0]
CIFAR10_alexnet_soter_stealing_50_acc = [ 83.70, 76.90, 67.66, 56.36, 43.90, 36.68, 24.13 ]
CIFAR10_alexnet_soter_stealing_50_fidelity = [ 97.80, 83.94, 72.02, 59.70, 46.23, 38.13, 24.49 ]
CIFAR10_alexnet_soter_stealing_50_asr = [ 100.00, 99.05, 93.02, 79.37, 52.38, 36.19, 16.51 ]

CIFAR10_resnet34_acc =  96.71
CIFAR10_resnet34_soter_stealing_50_protect_params =  [0, 783488, 8145280, 12314112, 17844928, 20243264, 21166016, 21282122]
CIFAR10_resnet34_soter_stealing_50_protect_flops =  [0, 115638272.0, 382222336.0, 627965952.0, 821002240.0, 953335808.0, 1104625664.0, 1161450506.0]
CIFAR10_resnet34_soter_stealing_50_acc = [ 92.77, 93.10, 91.56, 87.07, 81.06, 68.59, 58.70, 23.48 ]
CIFAR10_resnet34_soter_stealing_50_fidelity = [ 93.66, 93.72, 92.18, 87.42, 81.54, 68.72, 58.83, 23.53 ]
CIFAR10_resnet34_soter_stealing_50_asr = [ 100.00, 100.00, 97.82, 78.20, 55.59, 29.97, 20.44, 8.72 ]

CIFAR10_vgg19_bn_acc =  92.49
CIFAR10_vgg19_bn_soter_stealing_50_protect_params =  [0, 2952960, 8858880, 12440128, 17238208, 19896256, 20046400, 20051530]
CIFAR10_vgg19_bn_soter_stealing_50_protect_flops =  [0, 75546624.0, 198316032.0, 264527872.0, 302350336.0, 359022592.0, 398737408.0, 398742538.0]
CIFAR10_vgg19_bn_soter_stealing_50_acc = [ 90.93, 87.34, 76.47, 65.11, 56.32, 42.29, 46.05, 14.78 ]
CIFAR10_vgg19_bn_soter_stealing_50_fidelity = [ 94.39, 89.60, 77.91, 66.21, 56.94, 42.58, 46.60, 14.75 ]
CIFAR10_vgg19_bn_soter_stealing_50_asr = [ 99.72, 100.00, 92.68, 72.68, 60.85, 39.72, 39.72, 22.25 ]

CIFAR100_resnet18_acc =  81.66
CIFAR100_resnet18_soter_stealing_500_protect_params =  [0, 3024640, 4980352, 7653760, 8613504, 10982272, 11131840, 11220132]
CIFAR100_resnet18_soter_stealing_500_protect_flops =  [0, 94486528.0, 226852864.0, 306675712.0, 439123968.0, 479051776.0, 518766592.0, 556697700.0]
CIFAR100_resnet18_soter_stealing_500_acc = [ 80.48, 79.28, 75.82, 70.17, 61.26, 52.94, 47.15, 15.41 ]
CIFAR100_resnet18_soter_stealing_500_fidelity = [ 90.72, 86.92, 81.11, 73.57, 63.48, 54.32, 48.35, 15.54 ]
CIFAR100_resnet18_soter_stealing_500_asr = [ 100.00, 100.00, 99.69, 97.83, 73.99, 64.40, 56.97, 26.01 ]

CIFAR100_vgg16_bn_acc =  74.48
CIFAR100_vgg16_bn_soter_stealing_500_protect_params =  [0, 1845888, 6567552, 9518976, 12323328, 12362304, 14723136, 14774436]
CIFAR100_vgg16_bn_soter_stealing_500_protect_flops =  [0, 75612160.0, 122818560.0, 170041344.0, 264527872.0, 304308224.0, 313749504.0, 313800804.0]
CIFAR100_vgg16_bn_soter_stealing_500_acc = [ 71.10, 66.06, 57.41, 49.60, 36.12, 35.48, 27.51, 10.31 ]
CIFAR100_vgg16_bn_soter_stealing_500_fidelity = [ 83.87, 73.52, 62.23, 52.92, 37.94, 37.16, 28.47, 10.62 ]
CIFAR100_vgg16_bn_soter_stealing_500_asr = [ 99.66, 98.98, 96.95, 88.81, 67.80, 65.42, 60.34, 40.68 ]

CIFAR100_alexnet_acc = [ 58.42, 58.42, 58.42, 58.42, 58.42, 58.42, 58.42 ]
CIFAR100_alexnet_soter_stealing_500_protect_params =  [0, 884992, 1192384, 1856320, 2446400, 2455872, 2481572]
CIFAR100_alexnet_soter_stealing_500_protect_flops =  [0, 14155776.0, 33816576.0, 44433408.0, 53870592.0, 56279040.0, 56304740.0]
CIFAR100_alexnet_soter_stealing_500_acc = [ 58.43, 50.83, 45.65, 39.90, 27.01, 21.52, 12.09 ]
CIFAR100_alexnet_soter_stealing_500_fidelity = [ 96.86, 66.98, 56.57, 48.16, 31.47, 24.67, 13.44 ]
CIFAR100_alexnet_soter_stealing_500_asr = [ 100.00, 99.16, 97.06, 92.86, 80.67, 63.45, 39.92 ]

CIFAR100_resnet34_acc =  83.63
CIFAR100_resnet34_soter_stealing_500_protect_params =  [0, 783488, 8145280, 12314112, 17844928, 20243264, 21166016, 21328292]
CIFAR100_resnet34_soter_stealing_500_protect_flops =  [0, 115638272.0, 382222336.0, 627965952.0, 821002240.0, 953335808.0, 1104625664.0, 1161496676.0]
CIFAR100_resnet34_soter_stealing_500_acc = [ 82.08, 81.47, 77.37, 72.43, 62.70, 56.12, 51.60, 16.02 ]
CIFAR100_resnet34_soter_stealing_500_fidelity = [ 91.18, 89.59, 82.32, 75.91, 65.07, 57.85, 53.01, 16.35 ]
CIFAR100_resnet34_soter_stealing_500_asr = [ 100.00, 100.00, 99.69, 88.04, 68.71, 59.51, 49.69, 20.55 ]

CIFAR100_vgg19_bn_acc =  72.65
CIFAR100_vgg19_bn_soter_stealing_500_protect_params =  [0, 2951424, 8854272, 12433344, 17229120, 19885632, 20035392, 20086692]
CIFAR100_vgg19_bn_soter_stealing_500_protect_flops =  [0, 75546624.0, 198316032.0, 264527872.0, 302350336.0, 359022592.0, 398737408.0, 398788708.0]
CIFAR100_vgg19_bn_soter_stealing_500_acc = [ 70.71, 66.32, 56.67, 45.66, 31.63, 24.33, 24.47, 9.68 ]
CIFAR100_vgg19_bn_soter_stealing_500_fidelity = [ 84.72, 75.64, 61.35, 49.13, 33.55, 25.58, 25.59, 9.74 ]
CIFAR100_vgg19_bn_soter_stealing_500_asr = [ 98.97, 97.59, 92.10, 80.41, 70.45, 56.70, 55.67, 41.58 ]

STL10_resnet18_acc =  86.99
STL10_resnet18_soter_stealing_50_protect_params =  [0, 3024640, 4980352, 7653760, 8613504, 10982272, 11139520, 11181642]
STL10_resnet18_soter_stealing_50_protect_flops =  [0, 94486528.0, 226852864.0, 306675712.0, 439123968.0, 479051776.0, 524468224.0, 562353162.0]
STL10_resnet18_soter_stealing_50_acc = [ 82.01, 80.83, 81.79, 79.89, 51.88, 38.36, 64.26, 31.54 ]
STL10_resnet18_soter_stealing_50_fidelity = [ 86.97, 84.04, 83.34, 80.51, 51.88, 37.94, 63.67, 31.04 ]
STL10_resnet18_soter_stealing_50_asr = [ 100.00, 100.00, 99.69, 93.27, 36.70, 20.18, 25.38, 10.70 ]

STL10_vgg16_bn_acc =  90.03
STL10_vgg16_bn_soter_stealing_50_protect_params =  [0, 1845888, 6567552, 9518976, 12323328, 12362304, 14723136, 14728266]
STL10_vgg16_bn_soter_stealing_50_protect_flops =  [0, 302448640.0, 491274240.0, 680165376.0, 1058111488.0, 1217232896.0, 1254998016.0, 1255003146.0]
STL10_vgg16_bn_soter_stealing_50_acc = [ 84.53, 86.94, 85.99, 85.46, 78.35, 77.62, 71.45, 23.01 ]
STL10_vgg16_bn_soter_stealing_50_fidelity = [ 88.33, 90.44, 88.33, 87.35, 78.95, 78.05, 71.64, 23.65 ]
STL10_vgg16_bn_soter_stealing_50_asr = [ 100.00, 100.00, 100.00, 99.42, 88.05, 87.76, 73.18, 33.24 ]

STL10_alexnet_acc =  76.54
STL10_alexnet_soter_stealing_50_protect_params =  [0, 884992, 1192384, 1856320, 2446400, 2455872, 2458442]
STL10_alexnet_soter_stealing_50_protect_flops =  [0, 14155776.0, 92798976.0, 135266304.0, 144703488.0, 154337280.0, 154339850.0]
STL10_alexnet_soter_stealing_50_acc = [ 76.56, 37.60, 30.93, 32.10, 14.75, 18.43, 15.76 ]
STL10_alexnet_soter_stealing_50_fidelity = [ 98.59, 39.91, 32.74, 34.42, 14.93, 18.96, 15.91 ]
STL10_alexnet_soter_stealing_50_asr = [ 100.00, 58.45, 38.03, 30.28, 11.97, 12.32, 9.86 ]

STL10_resnet34_acc =  87.84
STL10_resnet34_soter_stealing_50_protect_params =  [0, 783488, 8145280, 12314112, 17852608, 20250944, 21173696, 21289802]
STL10_resnet34_soter_stealing_50_protect_flops =  [0, 115638272.0, 382222336.0, 627965952.0, 826703872.0, 959037440.0, 1110327296.0, 1167152138.0]
STL10_resnet34_soter_stealing_50_acc = [ 78.90, 81.81, 81.62, 79.76, 76.65, 71.60, 71.49, 36.42 ]
STL10_resnet34_soter_stealing_50_fidelity = [ 82.58, 85.49, 83.81, 80.89, 77.41, 71.71, 71.51, 36.51 ]
STL10_resnet34_soter_stealing_50_asr = [ 100.00, 100.00, 100.00, 82.46, 49.54, 24.62, 16.62, 4.31 ]

STL10_vgg19_bn_acc =  88.67
STL10_vgg19_bn_soter_stealing_50_protect_params =  [0, 2951424, 8854272, 12433344, 17229120, 19885632, 20035392, 20040522]
STL10_vgg19_bn_soter_stealing_50_protect_flops =  [0, 302186496.0, 793264128.0, 1058111488.0, 1209401344.0, 1436090368.0, 1594949632.0, 1594954762.0]
STL10_vgg19_bn_soter_stealing_50_acc = [ 82.90, 86.17, 85.75, 80.06, 69.55, 64.36, 68.81, 28.65 ]
STL10_vgg19_bn_soter_stealing_50_fidelity = [ 86.78, 89.15, 86.95, 80.24, 70.01, 64.30, 68.58, 28.49 ]
STL10_vgg19_bn_soter_stealing_50_asr = [ 100.00, 100.00, 99.12, 86.73, 80.83, 57.82, 49.26, 25.37 ]

UTKFaceRace_resnet18_acc =  90.78
UTKFaceRace_resnet18_soter_stealing_50_protect_params =  [0, 3024640, 4980352, 7653760, 8613504, 10982272, 11139520, 11178564]
UTKFaceRace_resnet18_soter_stealing_50_protect_flops =  [0, 94486528.0, 226852864.0, 306675712.0, 439123968.0, 479051776.0, 524468224.0, 562350084.0]
UTKFaceRace_resnet18_soter_stealing_50_acc = [ 85.65, 76.43, 66.98, 57.77, 55.18, 49.77, 49.68, 45.69 ]
UTKFaceRace_resnet18_soter_stealing_50_fidelity = [ 89.06, 77.79, 68.12, 58.31, 55.90, 50.00, 49.73, 45.87 ]
UTKFaceRace_resnet18_soter_stealing_50_asr = [ 100.00, 100.00, 100.00, 100.00, 87.93, 71.26, 42.53, 12.07 ]

UTKFaceRace_vgg16_bn_acc =  91.51
UTKFaceRace_vgg16_bn_soter_stealing_50_protect_params =  [0, 1845888, 6567552, 9518976, 12323328, 12362304, 14723136, 14725188]
UTKFaceRace_vgg16_bn_soter_stealing_50_protect_flops =  [0, 302448640.0, 491274240.0, 680165376.0, 1058111488.0, 1217232896.0, 1254998016.0, 1255000068.0]
UTKFaceRace_vgg16_bn_soter_stealing_50_acc = [ 81.97, 80.74, 66.89, 59.85, 49.82, 49.14, 52.91, 46.32 ]
UTKFaceRace_vgg16_bn_soter_stealing_50_fidelity = [ 83.83, 82.88, 67.21, 60.08, 49.86, 49.14, 53.00, 46.14 ]
UTKFaceRace_vgg16_bn_soter_stealing_50_asr = [ 100.00, 100.00, 100.00, 100.00, 100.00, 100.00, 99.14, 32.00 ]

UTKFaceRace_alexnet_acc =  90.01
UTKFaceRace_alexnet_soter_stealing_50_protect_params =  [0, 884992, 1192384, 1856320, 2446400, 2455872, 2456900]
UTKFaceRace_alexnet_soter_stealing_50_protect_flops =  [0, 14155776.0, 92798976.0, 135266304.0, 144703488.0, 154337280.0, 154338308.0]
UTKFaceRace_alexnet_soter_stealing_50_acc = [ 89.78, 58.86, 55.13, 51.00, 46.41, 48.00, 47.73 ]
UTKFaceRace_alexnet_soter_stealing_50_fidelity = [ 98.27, 60.17, 55.68, 51.68, 47.37, 49.18, 48.86 ]
UTKFaceRace_alexnet_soter_stealing_50_asr = [ 100.00, 92.71, 68.22, 38.48, 21.87, 16.62, 15.16 ]

UTKFaceRace_resnet34_acc =  91.51
UTKFaceRace_resnet34_soter_stealing_50_protect_params =  [0, 783488, 8145280, 12314112, 17852608, 20250944, 21173696, 21286724]
UTKFaceRace_resnet34_soter_stealing_50_protect_flops =  [0, 115638272.0, 382222336.0, 627965952.0, 826703872.0, 959037440.0, 1110327296.0, 1167149060.0]
UTKFaceRace_resnet34_soter_stealing_50_acc = [ 88.83, 87.83, 84.06, 79.06, 68.26, 62.67, 55.22, 45.59 ]
UTKFaceRace_resnet34_soter_stealing_50_fidelity = [ 92.96, 92.19, 87.69, 81.93, 69.62, 63.85, 56.45, 46.73 ]
UTKFaceRace_resnet34_soter_stealing_50_asr = [ 100.00, 100.00, 100.00, 100.00, 84.38, 65.62, 42.90, 18.47 ]

UTKFaceRace_vgg19_bn_acc =  90.69
UTKFaceRace_vgg19_bn_soter_stealing_50_protect_params =  [0, 2951424, 8854272, 12433344, 17229120, 19885632, 20035392, 20037444]
UTKFaceRace_vgg19_bn_soter_stealing_50_protect_flops =  [0, 302186496.0, 793264128.0, 1058111488.0, 1209401344.0, 1436090368.0, 1594949632.0, 1594951684.0]
UTKFaceRace_vgg19_bn_soter_stealing_50_acc = [ 89.60, 87.87, 81.06, 70.48, 57.95, 52.86, 52.68, 46.46 ]
UTKFaceRace_vgg19_bn_soter_stealing_50_fidelity = [ 94.19, 91.46, 83.51, 72.25, 58.45, 53.36, 53.13, 46.82 ]
UTKFaceRace_vgg19_bn_soter_stealing_50_asr = [ 100.00, 100.00, 100.00, 100.00, 99.71, 95.69, 93.39, 38.22 ]