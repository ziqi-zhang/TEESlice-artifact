# python scripts/stealing_result_layer.py
CIFAR100_resnet18_acc =  79.06
CIFAR100_resnet18_block_deep_params = [0, 51300, 4771940, 8445028, 9625700, 10544740, 10840164, 11070308, 11144292, 11218276, 11220004]
CIFAR100_resnet18_block_deep_flops = [0, 51300, 75581540.0, 134350948.0, 209913956.0, 268732516.0, 344361060.0, 403277924.0, 479037540.0, 554797156.0, 556697700.0]
CIFAR100_resnet18_block_deep_50_acc = [79.86, 27.08, 21.27, 15.35, 12.15, 10.98, 9.82, 8.32, 6.97, 4.11, 4.18]
CIFAR100_resnet18_block_deep_50_fidelity = [97.73, 28.84, 22.43, 15.95, 12.43, 11.01, 9.98, 8.42, 6.94, 4.16, 4.28]
CIFAR100_resnet18_block_deep_50_asr = [ 100.00, 80.19, 75.97, 60.06, 53.25, 45.45, 37.66, 30.19, 26.30, 24.03, 23.05 ]
CIFAR100_resnet18_block_shallow_params = [0, 1728, 75712, 149696, 379840, 675264, 1594304, 2774976, 6448064, 11168704, 11220004]
CIFAR100_resnet18_block_shallow_flops = [0, 1900544, 77660160.0, 153419776.0, 212336640.0, 287965184.0, 346783744.0, 422346752.0, 481116160.0, 556646400.0, 556697700.0]
CIFAR100_resnet18_block_shallow_50_acc = [79.86, 75.69, 66.29, 62.12, 51.38, 48.45, 37.96, 29.0, 23.89, 19.55, 2.61]
CIFAR100_resnet18_block_shallow_50_fidelity = [97.73, 85.05, 71.79, 66.31, 54.3, 50.2, 39.3, 29.72, 24.26, 20.16, 2.61]
CIFAR100_resnet18_block_shallow_50_asr = [ 100.00, 100.00, 96.75, 89.29, 72.73, 61.04, 50.00, 40.58, 33.77, 32.14, 26.62 ]
CIFAR100_resnet18_block_large_mag_params = [1.0, 112153.0, 1121524.0, 3364570.0, 5607617.0, 7850663.0, 8972186.0, 10093709.0, 10654471.0, 11215232.0]
CIFAR100_resnet18_block_large_mag_flops = [16.0, 29379688.0, 123448904.0, 240496128.0, 335173152.0, 421851552.0, 463734912.0, 505259392.0, 525921312.0, 551364608.0]
CIFAR100_resnet18_block_large_mag_50_acc = [79.8, 66.0, 56.36, 40.11, 23.02, 10.31, 6.62, 4.58, 3.59, 2.66]
CIFAR100_resnet18_block_large_mag_50_fidelity = [97.57, 70.49, 58.79, 41.23, 23.28, 10.4, 6.76, 4.75, 3.84, 2.77]
CIFAR100_resnet18_block_large_mag_50_asr = [ 100.00, 99.35, 91.56, 72.73, 54.22, 34.74, 31.82, 28.57, 25.65, 23.38 ]

CIFAR100_resnet18_shadownet_stealing_50_acc = 80.51
CIFAR100_resnet18_shadownet_stealing_50_fidelity = 92.42
CIFAR100_resnet18_shadownet_stealing_50_asr = 100.0

# python scripts/stealing_result_nettailor.py
CIFAR100_resnet18_nettailor_acc = 76.79
CIFAR100_resnet18_nettailor_task_param = 711524.0
CIFAR100_resnet18_nettailor_task_flops = 29868032.0
CIFAR100_resnet18_nettailor_stealing_50_acc = 2.91
CIFAR100_resnet18_nettailor_stealing_50_fidelity = 2.99
CIFAR100_resnet18_nettailor_stealing_50_asr = 26.62

CIFAR100_resnet18_nettailor_stealing_victim_50_acc = 2.91
CIFAR100_resnet18_nettailor_stealing_victim_50_fidelity = 2.99
CIFAR100_resnet18_nettailor_stealing_victim_50_asr = 26.62
CIFAR100_resnet18_nettailor_stealing_hybrid_50_acc = 2.66
CIFAR100_resnet18_nettailor_stealing_hybrid_50_fidelity = 2.59
CIFAR100_resnet18_nettailor_stealing_hybrid_50_asr = 20.45
CIFAR100_resnet18_nettailor_stealing_backbone_50_acc = 4.56
CIFAR100_resnet18_nettailor_stealing_backbone_50_fidelity = 4.55
CIFAR100_resnet18_nettailor_stealing_backbone_50_asr = 28.25

CIFAR100_resnet18_acc =  79.06
CIFAR100_resnet18_block_deep_params =  [0, 5130, 4725770, 8398858, 9579530, 10498570, 10793994, 11024138, 11098122, 11172106, 11173834]
CIFAR100_resnet18_block_deep_flops =  [0, 5130, 75535370.0, 134304778.0, 209867786.0, 268686346.0, 344314890.0, 403231754.0, 478991370.0, 554750986.0, 556651530.0]
CIFAR100_resnet18_block_deep_500_acc =  [79.76, 70.11, 61.18, 50.4, 45.36, 42.54, 40.08, 36.36, 32.03, 20.99, 19.16]
CIFAR100_resnet18_block_deep_500_fidelity =  [95.39, 75.72, 64.04, 52.54, 46.62, 43.43, 40.8, 37.32, 32.46, 21.1, 19.41]
CIFAR100_resnet18_block_deep_500_asr = [ 100.00, 99.68, 99.03, 96.43, 90.91, 83.12, 72.40, 55.52, 50.32, 29.55, 28.90 ]
CIFAR100_resnet18_block_shallow_params =  [0, 1728, 75712, 149696, 379840, 675264, 1594304, 2774976, 6448064, 11168704, 11173834]
CIFAR100_resnet18_block_shallow_flops =  [0, 1900544, 77660160.0, 153419776.0, 212336640.0, 287965184.0, 346783744.0, 422346752.0, 481116160.0, 556646400.0, 556651530.0]
CIFAR100_resnet18_block_shallow_500_acc =  [79.77, 78.01, 74.25, 72.54, 67.79, 66.69, 62.88, 57.94, 49.89, 40.25, 11.66]
CIFAR100_resnet18_block_shallow_500_fidelity =  [95.4, 88.49, 81.95, 78.95, 72.46, 70.39, 66.06, 59.9, 51.5, 41.19, 11.57]
CIFAR100_resnet18_block_shallow_500_asr = [ 100.00, 100.00, 100.00, 97.40, 89.94, 86.69, 77.27, 63.64, 56.49, 45.45, 23.05 ]
CIFAR100_resnet18_block_large_mag_params =  [1.0, 112153.0, 1121524.0, 3364570.0, 5607617.0, 7850663.0, 8972186.0, 10093709.0, 10654471.0, 11215232.0]
CIFAR100_resnet18_block_large_mag_flops =  [16.0, 29379688.0, 123448904.0, 240496128.0, 335173152.0, 421851552.0, 463734912.0, 505259392.0, 525921312.0, 551364608.0]
CIFAR100_resnet18_block_large_mag_500_acc =  [79.65, 74.84, 69.81, 58.66, 45.21, 30.93, 23.1, 16.98, 14.7, 12.61]
CIFAR100_resnet18_block_large_mag_500_fidelity =  [92.02, 81.29, 73.28, 60.98, 46.22, 30.84, 23.43, 17.31, 14.86, 12.79]
CIFAR100_resnet18_block_large_mag_500_asr = [ 100.00, 100.00, 96.75, 77.92, 56.17, 39.29, 34.42, 28.57, 25.65, 26.30 ]

CIFAR100_resnet18_shadownet_stealing_500_acc = 80.51
CIFAR100_resnet18_shadownet_stealing_500_fidelity = 92.42
CIFAR100_resnet18_shadownet_stealing_500_asr = 100.0

CIFAR100_resnet18_nettailor_acc  = 76.79
CIFAR100_resnet18_nettailor_task_param = 711524.0
CIFAR100_resnet18_nettailor_task_flops = 29868032.0
CIFAR100_resnet18_nettailor_stealing_500_acc = 7.78
CIFAR100_resnet18_nettailor_stealing_500_fidelity = 7.77
CIFAR100_resnet18_nettailor_stealing_500_asr = 25.32

CIFAR100_resnet18_nettailor_stealing_victim_500_acc = 7.78
CIFAR100_resnet18_nettailor_stealing_victim_500_fidelity = 7.77
CIFAR100_resnet18_nettailor_stealing_victim_500_asr = 25.32
CIFAR100_resnet18_nettailor_stealing_hybrid_500_acc = 10.9
CIFAR100_resnet18_nettailor_stealing_hybrid_500_fidelity = 11.21
CIFAR100_resnet18_nettailor_stealing_hybrid_500_asr = 25.97
CIFAR100_resnet18_nettailor_stealing_backbone_500_acc = 18.33
CIFAR100_resnet18_nettailor_stealing_backbone_500_fidelity = 19.26
CIFAR100_resnet18_nettailor_stealing_backbone_500_asr = 29.87


CIFAR100_resnet18_block_deep_50_gen_gap =  [0.3914, 0.2422, 0.1247, 0.0389, 0.0131, 0.0038, 0.0005, 0.0004, 0.0009, 0.002, 0.0012]
CIFAR100_resnet18_block_deep_50_conf_gap =  [0.4584, 0.2287, 0.1038, 0.0262, 0.0096, 0.0037, 0.0013, 0.0006, 0.0013, 0.0021, 0.0013]
CIFAR100_resnet18_block_deep_50_top3_acc=  [0.8616, 0.6345, 0.5575, 0.5108, 0.5015, 0.5043, 0.501, 0.5019, 0.504, 0.5045, 0.5054]
CIFAR100_resnet18_block_deep_50_mode0_acc=  [0.8608, 0.6183, 0.545, 0.5078, 0.5001, 0.502, 0.5001, 0.5023, 0.5026, 0.5013, 0.5022]
CIFAR100_resnet18_block_deep_50_mode3_acc=  [0.9078, 0.5595, 0.5068, 0.5032, 0.5038, 0.5014, 0.501, 0.5003, 0.5001, 0.5009, 0.5006]
CIFAR100_resnet18_block_shallow_50_gen_gap =  [0.3914, 0.4399, 0.4081, 0.3953, 0.3055, 0.2191, 0.0556, 0.0325, 0.0057, 0.0043, 0.0025]
CIFAR100_resnet18_block_shallow_50_conf_gap =  [0.4584, 0.4899, 0.3862, 0.3536, 0.2427, 0.1626, 0.0359, 0.0174, 0.0057, 0.0028, 0.0015]
CIFAR100_resnet18_block_shallow_50_top3_acc=  [0.8616, 0.7971, 0.6631, 0.6414, 0.5837, 0.5526, 0.5075, 0.5042, 0.5027, 0.5025, 0.5067]
CIFAR100_resnet18_block_shallow_50_mode0_acc=  [0.8608, 0.7894, 0.6624, 0.6415, 0.5816, 0.55, 0.5066, 0.5029, 0.5029, 0.5014, 0.5034]
CIFAR100_resnet18_block_shallow_50_mode3_acc=  [0.9078, 0.8137, 0.6581, 0.6684, 0.5786, 0.5453, 0.5047, 0.5012, 0.5007, 0.5011, 0.5005]
CIFAR100_resnet18_block_large_mag_50_gen_gap =  [0.3965, 0.3172, 0.2859, 0.1746, 0.0957, 0.0313, 0.015, 0.0029, 0.0001, 0.0006]
CIFAR100_resnet18_block_large_mag_50_conf_gap =  [0.4705, 0.2634, 0.2433, 0.1162, 0.038, 0.0089, 0.0034, 0.0011, 0.0004, 0.0]
CIFAR100_resnet18_block_large_mag_50_top3_acc=  [0.8372, 0.5883, 0.586, 0.5427, 0.5089, 0.5032, 0.5007, 0.5003, 0.5001, 0.5]
CIFAR100_resnet18_block_large_mag_50_mode0_acc=  [0.8293, 0.5898, 0.5893, 0.541, 0.5112, 0.5013, 0.5005, 0.5001, 0.5, 0.5]
CIFAR100_resnet18_block_large_mag_50_mode3_acc=  [0.8923, 0.5735, 0.565, 0.5156, 0.5039, 0.5006, 0.5002, 0.5, 0.5, 0.5]

CIFAR100_resnet18_block_deep_500_gen_gap =  [0.3887, 0.4129, 0.4077, 0.206, 0.1113, 0.0425, 0.0231, 0.0065, 0.0073, 0.0037, 0.0007]
CIFAR100_resnet18_block_deep_500_conf_gap =  [0.4468, 0.4933, 0.3985, 0.1484, 0.0734, 0.029, 0.0154, 0.0056, 0.0037, 0.0035, 0.0025]
CIFAR100_resnet18_block_deep_500_top3_acc=  [0.6573, 0.7041, 0.6787, 0.5524, 0.5262, 0.5072, 0.5053, 0.5, 0.502, 0.4978, 0.5044]
CIFAR100_resnet18_block_deep_500_mode0_acc=  [0.8263, 0.811, 0.5824, 0.5113, 0.5025, 0.5006, 0.5007, 0.5015, 0.5009, 0.5007, 0.5016]
CIFAR100_resnet18_block_deep_500_mode3_acc=  [0.8907, 0.7278, 0.558, 0.5089, 0.5025, 0.501, 0.5006, 0.5008, 0.5007, 0.5004, 0.5008]
CIFAR100_resnet18_block_shallow_500_gen_gap =  [0.3887, 0.3926, 0.3935, 0.3919, 0.3917, 0.3775, 0.2571, 0.1979, 0.0679, 0.0213, -0.0034]
CIFAR100_resnet18_block_shallow_500_conf_gap =  [0.4468, 0.4516, 0.4512, 0.4509, 0.4224, 0.3906, 0.2299, 0.1629, 0.0528, 0.0171, 0.0016]
CIFAR100_resnet18_block_shallow_500_top3_acc=  [0.6572, 0.657, 0.6574, 0.657, 0.6525, 0.6408, 0.5724, 0.5489, 0.5179, 0.5028, 0.5014]
CIFAR100_resnet18_block_shallow_500_mode0_acc=  [0.8264, 0.8292, 0.827, 0.8197, 0.7234, 0.672, 0.5612, 0.5337, 0.5063, 0.5015, 0.5003]
CIFAR100_resnet18_block_shallow_500_mode3_acc=  [0.8907, 0.8863, 0.863, 0.826, 0.679, 0.6255, 0.5335, 0.519, 0.5026, 0.5012, 0.5]
CIFAR100_resnet18_block_large_mag_500_gen_gap =  [0.395, 0.3961, 0.3213, 0.2071, 0.1169, 0.0509, 0.0266, 0.0093, 0.0059, 0.0038]
CIFAR100_resnet18_block_large_mag_500_conf_gap =  [0.4635, 0.3982, 0.3035, 0.155, 0.0686, 0.0233, 0.0103, 0.0042, 0.002, 0.0007]
CIFAR100_resnet18_block_large_mag_500_top3_acc=  [0.8254, 0.6752, 0.6186, 0.5553, 0.5234, 0.5082, 0.5015, 0.5013, 0.5009, 0.5005]
CIFAR100_resnet18_block_large_mag_500_mode0_acc=  [0.8197, 0.6755, 0.6154, 0.5539, 0.5237, 0.5073, 0.5025, 0.5008, 0.5005, 0.5]
CIFAR100_resnet18_block_large_mag_500_mode3_acc=  [0.8796, 0.7215, 0.6215, 0.5416, 0.51, 0.5018, 0.5004, 0.5011, 0.5004, 0.5001]

CIFAR100_resnet18_block_deep_100_gen_gap =  [0.3925, 0.2799, 0.2066, 0.0743, 0.0411, 0.0121, 0.0075, 0.0031, 0.0036, 0.0002, 0.0003]
CIFAR100_resnet18_block_deep_100_conf_gap =  [0.4536, 0.3358, 0.1597, 0.0457, 0.0198, 0.0076, 0.0041, 0.0016, 0.0022, 0.0008, 0.0005]
CIFAR100_resnet18_block_deep_100_top3_acc=  [0.6609, 0.6622, 0.5812, 0.5166, 0.507, 0.504, 0.4994, 0.4991, 0.5019, 0.5027, 0.5003]
CIFAR100_resnet18_block_deep_100_mode0_acc=  [0.8326, 0.6337, 0.5126, 0.5021, 0.5001, 0.4999, 0.5, 0.4999, 0.4999, 0.5, 0.5]
CIFAR100_resnet18_block_deep_100_mode3_acc=  [0.8952, 0.5201, 0.5069, 0.5009, 0.5001, 0.5001, 0.5, 0.4999, 0.5, 0.4999, 0.5001]
CIFAR100_resnet18_block_shallow_100_gen_gap =  [0.3925, 0.4071, 0.4175, 0.4186, 0.4004, 0.3751, 0.2353, 0.1684, 0.0554, 0.017, 0.0025]
CIFAR100_resnet18_block_shallow_100_conf_gap =  [0.4536, 0.4687, 0.4737, 0.4698, 0.4135, 0.3675, 0.1918, 0.126, 0.0362, 0.0122, 0.0007]
CIFAR100_resnet18_block_shallow_100_top3_acc=  [0.6609, 0.6665, 0.668, 0.6745, 0.6536, 0.6338, 0.5619, 0.5384, 0.5082, 0.5053, 0.5]
CIFAR100_resnet18_block_shallow_100_mode0_acc=  [0.8324, 0.8347, 0.805, 0.7793, 0.6837, 0.6377, 0.5442, 0.5223, 0.5051, 0.5002, 0.5]
CIFAR100_resnet18_block_shallow_100_mode3_acc=  [0.8952, 0.866, 0.7776, 0.7257, 0.629, 0.5874, 0.5247, 0.5111, 0.5023, 0.5004, 0.5]
CIFAR100_resnet18_block_large_mag_100_gen_gap =  [0.3965, 0.3902, 0.3133, 0.1933, 0.1029, 0.0293, 0.0167, 0.0052, 0.003, 0.0006]
CIFAR100_resnet18_block_large_mag_100_conf_gap =  [0.4681, 0.384, 0.284, 0.1307, 0.0447, 0.0101, 0.0041, 0.0017, 0.0007, -0.0002]
CIFAR100_resnet18_block_large_mag_100_top3_acc=  [0.8324, 0.6705, 0.6046, 0.5491, 0.5147, 0.5025, 0.5011, 0.5012, 0.5002, 0.5003]
CIFAR100_resnet18_block_large_mag_100_mode0_acc=  [0.8256, 0.6681, 0.6056, 0.5486, 0.5149, 0.5023, 0.5016, 0.5014, 0.5005, 0.5003]
CIFAR100_resnet18_block_large_mag_100_mode3_acc=  [0.8859, 0.7024, 0.5996, 0.5346, 0.5039, 0.5008, 0.5007, 0.5009, 0.5, 0.5]

CIFAR100_resnet18_block_deep_300_gen_gap =  [0.3914, 0.4101, 0.3647, 0.1537, 0.0811, 0.0267, 0.0187, 0.0042, 0.0059, 0.0005, 0.0039]
CIFAR100_resnet18_block_deep_300_conf_gap =  [0.4495, 0.4989, 0.3106, 0.1036, 0.0453, 0.0173, 0.0075, 0.0039, 0.002, 0.0027, 0.001]
CIFAR100_resnet18_block_deep_300_top3_acc=  [0.6602, 0.7347, 0.6428, 0.5354, 0.515, 0.505, 0.5017, 0.5021, 0.5011, 0.5005, 0.4982]
CIFAR100_resnet18_block_deep_300_mode0_acc=  [0.8282, 0.7472, 0.5544, 0.5081, 0.5023, 0.4992, 0.4999, 0.5003, 0.4992, 0.4997, 0.4999]
CIFAR100_resnet18_block_deep_300_mode3_acc=  [0.8921, 0.6125, 0.5336, 0.5064, 0.501, 0.5002, 0.5, 0.5001, 0.4999, 0.4999, 0.5001]
CIFAR100_resnet18_block_shallow_300_gen_gap =  [0.3913, 0.3984, 0.3963, 0.396, 0.3899, 0.3753, 0.2545, 0.1921, 0.067, 0.0264, 0.0017]
CIFAR100_resnet18_block_shallow_300_conf_gap =  [0.4495, 0.4552, 0.4547, 0.4546, 0.4202, 0.3857, 0.2255, 0.1543, 0.0458, 0.0187, 0.0007]
CIFAR100_resnet18_block_shallow_300_top3_acc=  [0.6601, 0.6584, 0.6612, 0.6605, 0.651, 0.6378, 0.573, 0.5408, 0.5123, 0.5023, 0.5001]
CIFAR100_resnet18_block_shallow_300_mode0_acc=  [0.8282, 0.8308, 0.8276, 0.8168, 0.7156, 0.6635, 0.557, 0.5312, 0.5063, 0.5007, 0.5]
CIFAR100_resnet18_block_shallow_300_mode3_acc=  [0.8921, 0.886, 0.8564, 0.8109, 0.6706, 0.617, 0.5321, 0.5179, 0.5015, 0.5006, 0.5]
CIFAR100_resnet18_block_large_mag_300_gen_gap =  [0.4023, 0.3949, 0.3175, 0.2008, 0.1117, 0.042, 0.0225, 0.0148, 0.0069, 0.0073]
CIFAR100_resnet18_block_large_mag_300_conf_gap =  [0.4701, 0.3934, 0.2994, 0.1451, 0.058, 0.0166, 0.0085, 0.0029, 0.0017, 0.0008]
CIFAR100_resnet18_block_large_mag_300_top3_acc=  [0.8301, 0.6658, 0.612, 0.5494, 0.5184, 0.5034, 0.5023, 0.5005, 0.5002, 0.5001]
CIFAR100_resnet18_block_large_mag_300_mode0_acc=  [0.8226, 0.6729, 0.6118, 0.5491, 0.5209, 0.5059, 0.503, 0.5023, 0.5007, 0.5001]
CIFAR100_resnet18_block_large_mag_300_mode3_acc=  [0.8748, 0.7136, 0.6228, 0.5348, 0.5033, 0.5003, 0.5003, 0.5001, 0.5002, 0.5001]

CIFAR100_resnet18_shadownet_50_gen_gap = 0.39
CIFAR100_resnet18_shadownet_50_conf_gap = 0.46
CIFAR100_resnet18_shadownet_50_top3_acc= 0.66
CIFAR100_resnet18_shadownet_50_mode0_acc= 0.83
CIFAR100_resnet18_shadownet_50_mode3_acc= 0.89

CIFAR100_resnet18_shadownet_100_gen_gap = 0.39
CIFAR100_resnet18_shadownet_100_conf_gap = 0.45
CIFAR100_resnet18_shadownet_100_top3_acc= 0.66
CIFAR100_resnet18_shadownet_100_mode0_acc= 0.8323
CIFAR100_resnet18_shadownet_100_mode3_acc= 0.89

CIFAR100_resnet18_shadownet_300_gen_gap = 0.39
CIFAR100_resnet18_shadownet_300_conf_gap = 0.44
CIFAR100_resnet18_shadownet_300_top3_acc= 0.65
CIFAR100_resnet18_shadownet_300_mode0_acc= 0.82
CIFAR100_resnet18_shadownet_300_mode3_acc= 0.89

CIFAR100_resnet18_shadownet_500_gen_gap = 0.39
CIFAR100_resnet18_shadownet_500_conf_gap = 0.44
CIFAR100_resnet18_shadownet_500_top3_acc= 0.65
CIFAR100_resnet18_shadownet_500_mode0_acc= 0.8373
CIFAR100_resnet18_shadownet_500_mode3_acc= 0.89

CIFAR100_resnet18_soter_stealing_500_protect_params =  [0, 3024640, 4980352, 7653760, 8613504, 10982272, 11131840, 11220132]
CIFAR100_resnet18_soter_stealing_500_protect_flops =  [0, 94486528.0, 226852864.0, 306675712.0, 439123968.0, 479051776.0, 518766592.0, 556697700.0]
CIFAR100_resnet18_soter_stealing_500_acc = [ 80.48, 79.28, 75.82, 70.17, 61.26, 52.94, 47.15, 15.41 ]
CIFAR100_resnet18_soter_stealing_500_fidelity = [ 90.72, 86.92, 81.11, 73.57, 63.48, 54.32, 48.35, 15.54 ]
CIFAR100_resnet18_soter_stealing_500_asr = [ 100.00, 100.00, 99.69, 97.83, 73.99, 64.40, 56.97, 26.01 ]

CIFAR100_resnet18_soter_500_gen_gap =  [0.4083, 0.4052, 0.3499, 0.2965, 0.2175, 0.1307, 0.0774, 0.001]
CIFAR100_resnet18_soter_500_conf_gap =  [0.4387, 0.4373, 0.3445, 0.2785, 0.1417, 0.077, 0.0439, 0.0023]
CIFAR100_resnet18_soter_500_top3_acc=  [0.622, 0.6339, 0.6058, 0.5772, 0.5576, 0.5267, 0.5144, 0.502]
CIFAR100_resnet18_soter_500_mode0_acc=  [0.8112, 0.7631, 0.624, 0.5797, 0.5137, 0.506, 0.5014, 0.5008]
CIFAR100_resnet18_soter_500_mode3_acc=  [0.8519, 0.7678, 0.614, 0.5715, 0.5028, 0.5008, 0.5002, 0.5001]

CIFAR100_resnet18_soter_100_gen_gap =  [0.4369, 0.4137, 0.3403, 0.2827, 0.2082, 0.1259, 0.0735, -0.0005]
CIFAR100_resnet18_soter_100_conf_gap =  [0.4682, 0.4336, 0.3339, 0.2666, 0.1199, 0.0638, 0.0344, -0.0009]
CIFAR100_resnet18_soter_100_top3_acc=  [0.6358, 0.6346, 0.5989, 0.5768, 0.5524, 0.524, 0.5118, 0.499]
CIFAR100_resnet18_soter_100_mode0_acc=  [0.8136, 0.7064, 0.6138, 0.5745, 0.5088, 0.5034, 0.5013, 0.4996]
CIFAR100_resnet18_soter_100_mode3_acc=  [0.8337, 0.6966, 0.604, 0.5663, 0.501, 0.5003, 0.5, 0.4999]