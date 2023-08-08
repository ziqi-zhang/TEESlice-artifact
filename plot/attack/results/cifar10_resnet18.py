# python scripts/stealing_result_layer.py
CIFAR10_resnet18_acc = 95.47
CIFAR10_resnet18_block_deep_params = [0, 5130, 4725770, 8398858, 9579530, 10498570, 10793994, 11024138, 11098122, 11172106, 11173834]
CIFAR10_resnet18_block_deep_flops = [0, 5130, 75535370.0, 134304778.0, 209867786.0, 268686346.0, 344314890.0, 403231754.0, 478991370.0, 554750986.0, 556651530.0]
CIFAR10_resnet18_block_deep_50_acc = [95.39, 87.55, 78.58, 64.21, 64.9, 35.12, 35.4, 41.51, 24.55, 21.71, 24.78]
CIFAR10_resnet18_block_deep_50_fidelity = [99.72, 88.97, 79.66, 64.74, 65.18, 35.37, 35.36, 41.78, 24.84, 21.92, 25.06]
CIFAR10_resnet18_block_deep_50_asr = [ 100.00, 100.00, 99.18, 97.00, 75.48, 48.50, 31.34, 26.98, 15.80, 9.81, 10.35 ]
CIFAR10_resnet18_block_shallow_params = [0, 1728, 75712, 149696, 379840, 675264, 1594304, 2774976, 6448064, 11168704, 11173834]
CIFAR10_resnet18_block_shallow_flops = [0, 1900544, 77660160.0, 153419776.0, 212336640.0, 287965184.0, 346783744.0, 422346752.0, 481116160.0, 556646400.0, 556651530.0]
CIFAR10_resnet18_block_shallow_50_acc = [95.39, 93.94, 90.26, 87.43, 79.06, 74.11, 56.71, 49.97, 40.42, 34.05, 14.97]
CIFAR10_resnet18_block_shallow_50_fidelity = [99.72, 96.14, 91.54, 88.48, 80.0, 74.51, 57.03, 50.23, 40.77, 34.37, 15.27]
CIFAR10_resnet18_block_shallow_50_asr = [ 100.00, 100.00, 97.00, 85.56, 60.76, 42.23, 26.16, 22.62, 14.17, 13.62, 7.90 ]
CIFAR10_resnet18_block_large_mag_params = [1.0, 111692.0, 1116916.0, 3350747.0, 5584577.0, 7818407.0, 8935323.0, 10052237.0, 10610695.0, 11169152.0]
CIFAR10_resnet18_block_large_mag_flops = [16.0, 29303228.0, 124018880.0, 241073472.0, 335580480.0, 422073920.0, 463879712.0, 505306848.0, 525919872.0, 551318528.0]
CIFAR10_resnet18_block_large_mag_50_acc = [95.4, 89.92, 85.17, 75.32, 60.59, 40.04, 30.96, 24.99, 22.97, 19.4]
CIFAR10_resnet18_block_large_mag_50_fidelity = [99.47, 91.14, 86.03, 76.02, 61.11, 40.59, 31.45, 25.53, 23.34, 19.73]
CIFAR10_resnet18_block_large_mag_50_asr = [ 100.00, 99.73, 93.73, 63.49, 40.33, 20.44, 14.71, 11.44, 11.44, 9.54 ]

CIFAR10_resnet18_shadownet_stealing_50_acc = 95.58
CIFAR10_resnet18_shadownet_stealing_50_fidelity = 98.24
CIFAR10_resnet18_shadownet_stealing_50_asr = 100.0

# python scripts/stealing_result_nettailor.py
CIFAR10_resnet18_nettailor_acc = 93.65
CIFAR10_resnet18_nettailor_task_param = 520586
CIFAR10_resnet18_nettailor_task_flops = 21266432
CIFAR10_resnet18_nettailor_stealing_50_acc = 17.33
CIFAR10_resnet18_nettailor_stealing_50_fidelity = 17.64
CIFAR10_resnet18_nettailor_stealing_50_asr = 10.35

CIFAR10_resnet18_nettailor_stealing_victim_50_acc = 17.33
CIFAR10_resnet18_nettailor_stealing_victim_50_fidelity = 17.64
CIFAR10_resnet18_nettailor_stealing_victim_50_asr = 10.35
CIFAR10_resnet18_nettailor_stealing_hybrid_50_acc = 31.4
CIFAR10_resnet18_nettailor_stealing_hybrid_50_fidelity = 31.72
CIFAR10_resnet18_nettailor_stealing_hybrid_50_asr = 14.99
CIFAR10_resnet18_nettailor_stealing_backbone_50_acc = 25.63
CIFAR10_resnet18_nettailor_stealing_backbone_50_fidelity = 25.6
CIFAR10_resnet18_nettailor_stealing_backbone_50_asr = 11.72

CIFAR10_resnet18_block_deep_50_gen_gap =  [0.1543, 0.1593, 0.1821, 0.0621, 0.0094, 0.0016, -0.0014, -0.0033, -0.0009, -0.0043, -0.006]
CIFAR10_resnet18_block_deep_50_conf_gap =  [0.1635, 0.1943, 0.1776, 0.0555, 0.0099, 0.0011, -0.004, -0.0027, -0.0012, -0.0043, -0.0067]
CIFAR10_resnet18_block_deep_50_top3_acc=  [0.6409, 0.6573, 0.5927, 0.5164, 0.5046, 0.5031, 0.5029, 0.5006, 0.5, 0.5001, 0.5031]
CIFAR10_resnet18_block_deep_50_mode0_acc=  [0.6898, 0.6501, 0.5489, 0.5059, 0.4987, 0.4993, 0.501, 0.4991, 0.4976, 0.4987, 0.4982]
CIFAR10_resnet18_block_deep_50_mode3_acc=  [0.7002, 0.6368, 0.5316, 0.5076, 0.5007, 0.5006, 0.5001, 0.5003, 0.5003, 0.5002, 0.5002]
CIFAR10_resnet18_block_shallow_50_gen_gap =  [0.1543, 0.1651, 0.1805, 0.1739, 0.1409, 0.1217, 0.0539, 0.0385, 0.0077, 0.0007, -0.0072]
CIFAR10_resnet18_block_shallow_50_conf_gap =  [0.1635, 0.1721, 0.1837, 0.1756, 0.1405, 0.12, 0.0524, 0.0339, 0.0054, -0.0014, -0.0074]
CIFAR10_resnet18_block_shallow_50_top3_acc=  [0.641, 0.6277, 0.6057, 0.5921, 0.5662, 0.5485, 0.5157, 0.5092, 0.5082, 0.5046, 0.5025]
CIFAR10_resnet18_block_shallow_50_mode0_acc=  [0.6898, 0.6659, 0.6146, 0.5986, 0.5655, 0.5487, 0.5123, 0.5053, 0.5025, 0.5009, 0.5006]
CIFAR10_resnet18_block_shallow_50_mode3_acc=  [0.7, 0.6751, 0.6219, 0.6069, 0.5716, 0.5526, 0.5131, 0.5043, 0.5025, 0.5008, 0.5001]
CIFAR10_resnet18_block_large_mag_50_gen_gap =  [0.1522, 0.1598, 0.1372, 0.106, 0.0703, 0.0357, 0.0227, 0.003, 0.0031, -0.0023]
CIFAR10_resnet18_block_large_mag_50_conf_gap =  [0.1611, 0.159, 0.1402, 0.1015, 0.0606, 0.0234, 0.0116, 0.0047, 0.0022, 0.0011]
CIFAR10_resnet18_block_large_mag_50_top3_acc=  [0.6141, 0.5611, 0.5622, 0.5367, 0.5186, 0.5055, 0.5026, 0.5012, 0.5013, 0.5007]
CIFAR10_resnet18_block_large_mag_50_mode0_acc=  [0.6918, 0.5912, 0.575, 0.5343, 0.5129, 0.5025, 0.5017, 0.5002, 0.5003, 0.5002]
CIFAR10_resnet18_block_large_mag_50_mode3_acc=  [0.7018, 0.6013, 0.5786, 0.5293, 0.5092, 0.5034, 0.5024, 0.5025, 0.5001, 0.5001]

CIFAR10_resnet18_shadownet_50_gen_gap = 0.15
CIFAR10_resnet18_shadownet_50_conf_gap = 0.16
CIFAR10_resnet18_shadownet_50_top3_acc= 0.63
CIFAR10_resnet18_shadownet_50_mode0_acc= 0.6953
CIFAR10_resnet18_shadownet_50_mode3_acc= 0.70

CIFAR10_resnet18_soter_stealing_50_protect_params =  [0, 3024640, 4980352, 7653760, 8613504, 10982272, 11131840, 11173962]
CIFAR10_resnet18_soter_stealing_50_protect_flops =  [0, 94486528.0, 226852864.0, 306675712.0, 439123968.0, 479051776.0, 518766592.0, 556651530.0]
CIFAR10_resnet18_soter_stealing_50_acc = [ 92.89, 92.61, 90.94, 86.42, 74.88, 64.41, 57.16, 21.57 ]
CIFAR10_resnet18_soter_stealing_50_fidelity = [ 94.61, 93.93, 92.36, 87.57, 75.60, 65.07, 57.85, 21.87 ]
CIFAR10_resnet18_soter_stealing_50_asr = [ 100.00, 99.73, 99.73, 95.08, 54.64, 42.08, 30.87, 8.74 ]

CIFAR10_resnet18_soter_50_gen_gap =  [0.1537, 0.1458, 0.1191, 0.1029, 0.0851, 0.0555, 0.023, -0.0015]
CIFAR10_resnet18_soter_50_conf_gap =  [0.151, 0.1389, 0.1184, 0.1016, 0.0725, 0.0363, 0.0162, -0.0008]
CIFAR10_resnet18_soter_50_top3_acc=  [0.5479, 0.5484, 0.5438, 0.534, 0.5103, 0.504, 0.5025, 0.5002]
CIFAR10_resnet18_soter_50_mode0_acc=  [0.5309, 0.5267, 0.5368, 0.5302, 0.5042, 0.5017, 0.5007, 0.5002]
CIFAR10_resnet18_soter_50_mode3_acc=  [0.5387, 0.5329, 0.5464, 0.5413, 0.5006, 0.5003, 0.5002, 0.5001]