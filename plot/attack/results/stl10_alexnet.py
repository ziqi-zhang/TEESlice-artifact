# python scripts/stealing_result_layer.py
STL10_alexnet_acc =  76.5375
STL10_alexnet_block_deep_params =  [0, 2570, 592650, 1477642, 2141578, 2448970, 2458442]
STL10_alexnet_block_deep_flops =  [0, 2570, 9439754.0, 23595530.0, 66062858.0, 144706058.0, 154339850.0]
STL10_alexnet_block_deep_50_acc =  [76.55, 75.17, 52.33, 13.55, 13.5, 12.39, 15.82]
STL10_alexnet_block_deep_50_fidelity =  [99.16, 89.84, 55.48, 13.54, 13.69, 12.64, 16.57]
STL10_alexnet_block_deep_50_asr = [ 100.00, 100.00, 97.90, 26.92, 21.68, 15.38, 13.64 ]
STL10_alexnet_block_shallow_params =  [0, 9472, 316864, 980800, 1865792, 2455872, 2458442]
STL10_alexnet_block_shallow_flops =  [0, 9633792.0, 88276992.0, 130744320.0, 144900096.0, 154337280.0, 154339850.0]
STL10_alexnet_block_shallow_50_acc =  [76.55, 69.06, 65.55, 62.95, 29.61, 26.04, 14.7]
STL10_alexnet_block_shallow_50_fidelity =  [99.16, 81.06, 74.31, 69.81, 31.54, 27.48, 15.41]
STL10_alexnet_block_shallow_50_asr = [ 100.00, 100.00, 99.30, 97.90, 22.03, 20.28, 12.59 ]
STL10_alexnet_block_large_mag_params =  [1.0, 24573.0, 245729.0, 737185.0, 1228641.0, 1720097.0, 1965825.0, 2211553.0, 2334417.0, 2457280.0]
STL10_alexnet_block_large_mag_flops =  [256.0, 7280064.0, 37535704.0, 81161672.0, 101125008.0, 121985584.0, 132673368.0, 143450176.0, 148898048.0, 154339840.0]
STL10_alexnet_block_large_mag_50_acc =  [76.4, 73.67, 70.25, 62.42, 53.06, 31.66, 24.16, 19.49, 17.46, 18.02]
STL10_alexnet_block_large_mag_50_fidelity =  [96.53, 88.61, 81.38, 69.85, 58.11, 33.89, 24.94, 20.56, 18.32, 19.11]
STL10_alexnet_block_large_mag_50_asr = [ 100.00, 100.00, 100.00, 100.00, 94.76, 54.20, 22.03, 14.69, 12.94, 14.34 ]

STL10_alexnet_shadownet_stealing_50_acc = 76.53
STL10_alexnet_shadownet_stealing_50_fidelity = 64.75
STL10_alexnet_shadownet_stealing_50_asr = 100.0

# python scripts/stealing_result_nettailor.py
STL10_alexnet_nettailor_acc = 84.175
STL10_alexnet_nettailor_task_param = 479114.0
STL10_alexnet_nettailor_task_flops = 17039360.0
STL10_alexnet_nettailor_stealing_50_acc = 17.125
STL10_alexnet_nettailor_stealing_50_fidelity = 17.7
STL10_alexnet_nettailor_stealing_50_asr = 16.43

STL10_alexnet_nettailor_stealing_victim_50_acc = 17.14
STL10_alexnet_nettailor_stealing_victim_50_fidelity = 17.69
STL10_alexnet_nettailor_stealing_victim_50_asr = 16.43
STL10_alexnet_nettailor_stealing_hybrid_50_acc = 24.15
STL10_alexnet_nettailor_stealing_hybrid_50_fidelity = 23.5
STL10_alexnet_nettailor_stealing_hybrid_50_asr = 6.29
STL10_alexnet_nettailor_stealing_backbone_50_acc = 32.75
STL10_alexnet_nettailor_stealing_backbone_50_fidelity = 30.68
STL10_alexnet_nettailor_stealing_backbone_50_asr = 5.59


STL10_alexnet_block_deep_50_gen_gap =  [0.2865, 0.2886, 0.2323, -0.008, -0.0068, -0.004, -0.0092]
STL10_alexnet_block_deep_50_conf_gap =  [0.292, 0.291, 0.1422, 0.0034, 0.0013, 0.0004, -0.0004]
STL10_alexnet_block_deep_50_top3_acc=  [0.6234, 0.6666, 0.5323, 0.5, 0.5, 0.5, 0.5]
STL10_alexnet_block_deep_50_mode0_acc=  [0.6477, 0.5849, 0.516, 0.5, 0.5, 0.5, 0.5]
STL10_alexnet_block_deep_50_mode3_acc=  [0.712, 0.5871, 0.4998, 0.5, 0.5, 0.5, 0.5]
STL10_alexnet_block_shallow_50_gen_gap =  [0.2865, 0.2935, 0.2628, 0.1702, 0.0212, -0.0138, -0.004]
STL10_alexnet_block_shallow_50_conf_gap =  [0.292, 0.297, 0.2539, 0.1581, 0.0167, -0.0015, -0.0002]
STL10_alexnet_block_shallow_50_top3_acc=  [0.6234, 0.604, 0.5715, 0.5325, 0.5046, 0.4998, 0.5]
STL10_alexnet_block_shallow_50_mode0_acc=  [0.6477, 0.6151, 0.5517, 0.5225, 0.4995, 0.4998, 0.5]
STL10_alexnet_block_shallow_50_mode3_acc=  [0.712, 0.6389, 0.5418, 0.504, 0.5, 0.5, 0.5]
STL10_alexnet_block_large_mag_50_gen_gap =  [0.2889, 0.308, 0.2985, 0.2597, 0.2157, 0.1422, 0.1092, 0.0443, 0.0194, 0.0065]
STL10_alexnet_block_large_mag_50_conf_gap =  [0.2989, 0.3069, 0.288, 0.2154, 0.1404, 0.0658, 0.0355, 0.0108, 0.0038, 0.0005]
STL10_alexnet_block_large_mag_50_top3_acc=  [0.6491, 0.6254, 0.6328, 0.582, 0.5472, 0.5146, 0.5054, 0.5031, 0.5057, 0.5092]
STL10_alexnet_block_large_mag_50_mode0_acc=  [0.6968, 0.6626, 0.6286, 0.584, 0.5483, 0.5132, 0.5066, 0.5018, 0.5043, 0.5002]
STL10_alexnet_block_large_mag_50_mode3_acc=  [0.7114, 0.6914, 0.6628, 0.6265, 0.5914, 0.5495, 0.5237, 0.5069, 0.5, 0.5]

STL10_alexnet_shadownet_50_gen_gap = 0.28
STL10_alexnet_shadownet_50_conf_gap = 0.29
STL10_alexnet_shadownet_50_top3_acc= 0.62
STL10_alexnet_shadownet_50_mode0_acc= 0.6557
STL10_alexnet_shadownet_50_mode3_acc= 0.72

STL10_alexnet_soter_stealing_50_protect_params =  [0, 884992, 1192384, 1856320, 2446400, 2455872, 2458442]
STL10_alexnet_soter_stealing_50_protect_flops =  [0, 14155776.0, 92798976.0, 135266304.0, 144703488.0, 154337280.0, 154339850.0]
STL10_alexnet_soter_stealing_50_acc = [ 76.56, 37.60, 30.93, 32.10, 14.75, 18.43, 15.76 ]
STL10_alexnet_soter_stealing_50_fidelity = [ 98.59, 39.91, 32.74, 34.42, 14.93, 18.96, 15.91 ]
STL10_alexnet_soter_stealing_50_asr = [ 100.00, 58.45, 38.03, 30.28, 11.97, 12.32, 9.86 ]

STL10_alexnet_soter_50_gen_gap =  [0.2846, 0.2908, 0.2702, 0.1742, 0.0375, 0.0105, 0.0074]
STL10_alexnet_soter_50_conf_gap =  [0.2994, 0.2968, 0.2399, 0.1452, 0.019, 0.0005, 0.0002]
STL10_alexnet_soter_50_top3_acc=  [0.6545, 0.6454, 0.5868, 0.5351, 0.4998, 0.5, 0.5]
STL10_alexnet_soter_50_mode0_acc=  [0.6292, 0.5972, 0.5332, 0.5091, 0.4986, 0.5, 0.5]
STL10_alexnet_soter_50_mode3_acc=  [0.6651, 0.6334, 0.5429, 0.5074, 0.5002, 0.5, 0.5]