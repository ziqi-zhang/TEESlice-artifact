# python scripts/stealing_result_layer.py
STL10_resnet18_acc =  87.36
STL10_resnet18_block_deep_params = [0, 5130, 4725770, 8398858, 9579530, 10498570, 10793994, 11024138, 11098122, 11172106, 11173834]
STL10_resnet18_block_deep_flops =  [0, 5130, 75535370.0, 134304778.0, 209867786.0, 268686346.0, 344314890.0, 403231754.0, 478991370.0, 554750986.0, 556651530.0]
STL10_resnet18_block_deep_50_acc = [ 87.45, 86.03, 80.64, 69.38, 61.59, 49.75, 41.76, 40.20, 39.71, 23.23, 24.31 ]
STL10_resnet18_block_deep_50_fidelity = [ 99.51, 93.10, 84.67, 71.28, 62.11, 50.20, 42.35, 40.21, 39.70, 23.56, 23.98 ]
STL10_resnet18_block_deep_50_asr = [ 100.00, 100.00, 100.00, 96.72, 82.99, 54.33, 41.79, 26.27, 20.00, 11.94, 10.75 ]
STL10_resnet18_block_shallow_params = [0, 1728, 75712, 149696, 379840, 675264, 1594304, 2774976, 6448064, 11168704, 11173834]
STL10_resnet18_block_shallow_flops = [0, 1900544, 77660160.0, 153419776.0, 212336640.0, 287965184.0, 346783744.0, 422346752.0, 481116160.0, 556646400.0, 556651530.0]
STL10_resnet18_block_shallow_50_acc = [ 87.45, 85.05, 83.54, 81.16, 78.47, 77.55, 69.81, 59.70, 44.30, 33.73, 19.02 ]
STL10_resnet18_block_shallow_50_fidelity = [ 99.51, 91.60, 88.72, 85.40, 81.05, 80.25, 71.64, 61.25, 44.70, 33.34, 18.94 ]
STL10_resnet18_block_shallow_50_asr = [ 100.00, 100.00, 100.00, 99.70, 95.22, 88.06, 62.69, 44.48, 26.87, 18.21, 10.15 ]
STL10_resnet18_block_large_mag_params =  [1.0, 111769.0, 1117684.0, 3353050.0, 5588417.0, 7823783.0, 8941466.0, 10059149.0, 10617991.0, 11176832.0]
STL10_resnet18_block_large_mag_flops = [16.0, 31241308.0, 127817568.0, 246026080.0, 341195616.0, 428261856.0, 470327456.0, 511998912.0, 532721184.0, 559182848.0]
STL10_resnet18_block_large_mag_50_acc = [ 87.40, 77.08, 79.26, 80.21, 75.92, 62.26, 48.61, 36.02, 28.30, 23.44 ]
STL10_resnet18_block_large_mag_50_fidelity = [ 99.40, 80.06, 81.45, 81.78, 76.72, 62.30, 48.50, 36.11, 28.41, 23.56 ]
STL10_resnet18_block_large_mag_50_asr = [ 100.00, 99.70, 96.72, 81.49, 54.03, 31.34, 18.21, 11.94, 8.06, 9.55 ]

STL10_resnet18_shadownet_stealing_50_acc = 88.39
STL10_resnet18_shadownet_stealing_50_fidelity = 97.2
STL10_resnet18_shadownet_stealing_50_asr = 100.0

# python scripts/stealing_result_nettailor.py
STL10_resnet18_nettailor_acc = 86.225
STL10_resnet18_nettailor_task_param = 520586.0
STL10_resnet18_nettailor_task_flops = 21266432.0
STL10_resnet18_nettailor_stealing_50_acc = 32.6625
STL10_resnet18_nettailor_stealing_50_fidelity = 32.6625
STL10_resnet18_nettailor_stealing_50_asr = 11.64

STL10_resnet18_nettailor_stealing_victim_50_acc = 32.77
STL10_resnet18_nettailor_stealing_victim_50_fidelity = 32.67
STL10_resnet18_nettailor_stealing_victim_50_asr = 11.34
STL10_resnet18_nettailor_stealing_hybrid_50_acc = 29.19
STL10_resnet18_nettailor_stealing_hybrid_50_fidelity = 29.27
STL10_resnet18_nettailor_stealing_hybrid_50_asr = 10.75
STL10_resnet18_nettailor_stealing_backbone_50_acc = 32.77
STL10_resnet18_nettailor_stealing_backbone_50_fidelity = 32.56
STL10_resnet18_nettailor_stealing_backbone_50_asr = 10.45

STL10_resnet18_block_deep_50_gen_gap =  [0.2348, 0.2446, 0.2465, 0.1671, 0.1135, 0.0415, 0.0237, -0.0126, -0.0194, -0.0022, -0.0018]
STL10_resnet18_block_deep_50_conf_gap =  [0.2445, 0.2761, 0.2558, 0.1446, 0.0824, 0.0236, 0.0128, -0.0013, -0.006, -0.0022, -0.0047]
STL10_resnet18_block_deep_50_top3_acc=  [0.7226, 0.6866, 0.6452, 0.5583, 0.5226, 0.5092, 0.5014, 0.5046, 0.5011, 0.5029, 0.5023]
STL10_resnet18_block_deep_50_mode0_acc=  [0.7608, 0.6598, 0.5937, 0.5169, 0.5106, 0.5025, 0.5017, 0.5025, 0.5008, 0.5006, 0.5003]
STL10_resnet18_block_deep_50_mode3_acc=  [0.7728, 0.6515, 0.5672, 0.5109, 0.5028, 0.5, 0.5008, 0.5, 0.5, 0.5002, 0.5002]
STL10_resnet18_block_shallow_50_gen_gap =  [0.2345, 0.2369, 0.2449, 0.2492, 0.2286, 0.2197, 0.1612, 0.1314, 0.0563, 0.0258, -0.0077]
STL10_resnet18_block_shallow_50_conf_gap =  [0.2445, 0.2451, 0.2541, 0.2582, 0.2362, 0.2185, 0.1532, 0.1266, 0.055, 0.0163, -0.0009]
STL10_resnet18_block_shallow_50_top3_acc=  [0.7228, 0.6851, 0.6845, 0.6749, 0.6471, 0.6129, 0.5677, 0.5374, 0.5128, 0.5008, 0.5034]
STL10_resnet18_block_shallow_50_mode0_acc=  [0.7609, 0.7422, 0.7129, 0.6906, 0.6462, 0.616, 0.5588, 0.5234, 0.5028, 0.4997, 0.5]
STL10_resnet18_block_shallow_50_mode3_acc=  [0.7726, 0.7591, 0.7174, 0.6915, 0.6426, 0.6071, 0.5388, 0.5158, 0.5038, 0.5, 0.5]
STL10_resnet18_block_large_mag_50_gen_gap =  [0.2425, 0.2569, 0.232, 0.2163, 0.1837, 0.112, 0.0582, 0.0068, -0.0006, -0.0083]
STL10_resnet18_block_large_mag_50_conf_gap =  [0.2471, 0.2598, 0.2363, 0.2133, 0.1703, 0.0905, 0.0454, 0.0122, 0.0022, -0.0048]
STL10_resnet18_block_large_mag_50_top3_acc=  [0.6709, 0.6077, 0.6077, 0.6218, 0.5849, 0.5326, 0.5135, 0.5009, 0.4995, 0.4991]
STL10_resnet18_block_large_mag_50_mode0_acc=  [0.75, 0.6429, 0.6335, 0.5911, 0.5431, 0.5095, 0.5026, 0.5017, 0.5009, 0.5005]
STL10_resnet18_block_large_mag_50_mode3_acc=  [0.7546, 0.6438, 0.6183, 0.5742, 0.5358, 0.5072, 0.5066, 0.5014, 0.5042, 0.5006]

STL10_resnet18_shadownet_50_gen_gap = 0.24
STL10_resnet18_shadownet_50_conf_gap = 0.24
STL10_resnet18_shadownet_50_top3_acc= 0.60
STL10_resnet18_shadownet_50_mode0_acc= 0.7483
STL10_resnet18_shadownet_50_mode3_acc= 0.76


STL10_resnet18_soter_stealing_50_protect_params =  [0, 3024640, 4980352, 7653760, 8613504, 10982272, 11139520, 11181642]
STL10_resnet18_soter_stealing_50_protect_flops =  [0, 94486528.0, 226852864.0, 306675712.0, 439123968.0, 479051776.0, 524468224.0, 562353162.0]
STL10_resnet18_soter_stealing_50_acc = [ 82.01, 80.83, 81.79, 79.89, 51.88, 38.36, 64.26, 31.54 ]
STL10_resnet18_soter_stealing_50_fidelity = [ 86.97, 84.04, 83.34, 80.51, 51.88, 37.94, 63.67, 31.04 ]
STL10_resnet18_soter_stealing_50_asr = [ 100.00, 100.00, 99.69, 93.27, 36.70, 20.18, 25.38, 10.70 ]

STL10_resnet18_soter_50_gen_gap =  [0.2538, 0.2372, 0.2129, 0.1748, 0.1569, 0.1246, 0.0852, -0.0098]
STL10_resnet18_soter_50_conf_gap =  [0.2691, 0.2475, 0.2134, 0.178, 0.1508, 0.1105, 0.0673, 0.0013]
STL10_resnet18_soter_50_top3_acc=  [0.6337, 0.6309, 0.5935, 0.5746, 0.5765, 0.5448, 0.5166, 0.4991]
STL10_resnet18_soter_50_mode0_acc=  [0.6338, 0.5985, 0.56, 0.5528, 0.5103, 0.504, 0.5029, 0.5]
STL10_resnet18_soter_50_mode3_acc=  [0.6403, 0.6151, 0.5626, 0.5626, 0.5094, 0.5031, 0.5008, 0.5]