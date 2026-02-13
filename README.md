# Artifact for the paper "No Privacy Left Outside: On the (In-)Security of TEE-Shielded DNN Partition for On-Device ML"

## On-device Evaluation 
If you want to use the on-device evaluation (SGX-based evaluation) part of this repo, please refer to our new repo [TAOISM: A TEE-based Confidential Heterogeneous Framework for DNN Models](https://github.com/ziqi-zhang/TAOISM). We cleaned some issues and present a better illustration of the code structure.

## Requirements
To reproduce the results in the paper, you should have Python 3.7 with ``scipy`` and ``matplotlib`` installed

## Scripts to Reproduce the results

Directory ``plot`` includes all the code and scripts to reproduce the results in the paper. We will introduce each directory and their correspondance in the paper. 


- ``plot/attack`` plots the main results of model stealing and membership inference attacks
  - To run the scripts, you should first enter the directory by ``cd plot/attack``
  - ``plot/attack/summarize_solution_result.py`` displays the results of prior TSDP results w.r.t model stealing accuracy and confidence-based membership inference attack accuracy (Section 3.5, Table 2). The command is ``python summarize_solution_result.py``
  - ``plot/attack/summarize_solution_result_other_metrics.py`` displays the results of prior TSDP results of other security metrics (fidelity, ASR, gradient-based membership inference attack, generalization gap, and confidence gap) (Append F.2, Table 10 to Table 14)
    - The results are saved in ``plot/attack/other_metrics_summarize_solution_csv``. By setting ``attr`` (line 80) you can get the results of different metrics. Viable ``attr`` includes ``['acc', 'fidelity', 'asr', 'gen_gap', 'conf_gap', 'mode0', 'mode3']``, where 'mode0' means confidence-based MIA attack and 'mode3' means gradient-based MIA attack. 
    - The command is ``python summarize_solution_result_other_metrics.py``
  - ``plot/attack/print_acc_mia_flops_optimal_point.py`` computes the ''sweet spot'' configuration (ie, Utility(C*) and %FLOPs(C*) ) w.r.t model stealing accuracy and confidence-based membership inference attack accuracy (Section 4.3, Table 3)
  - ``plot/attack/print_acc_mia_flops_optimal_point_other_metrics.py`` computes the ''sweet spot'' configuration (ie, Utility(C*) and %FLOPs(C*) ) w.r.t other metrics (Append F.3, Table 15 to Table 19)
    - The results are saved in ``plot/attack/other_metrics_optimal_config_csv``. By setting ``attr`` (line 88) you can get the results of different metrics. Viable ``attr`` includes ``['acc', 'fidelity', 'asr', 'gen_gap', 'conf_gap', 'mode0', 'mode3']``, where 'mode0' means confidence-based MIA attack and 'mode3' means gradient-based MIA attack. 
    - The command is ``python print_acc_mia_flops_optimal_point_other_metrics.py``
  - ``plot/attack/other_assumption.py`` prints the results of other assumptions (Section 6.1, Table 4, Append F.5), the results are saved in ``plot/attack/other_assumption_csv``
    - By setting ``attr = "acc"`` of line 76 in ``plot/attack/other_assumption.py``, you get the results of Table 4 (Section 6.1) and Table 9 (Append E)
    - By setting ``attr = "asr"`` or ``attr = "fidelity"`` of line 76 in ``plot/attack/other_assumption.py``, you get the results of Table 20 and Table 21 (Append F.5). 
    - The command is ``python other_assumption.py``
  - ``plot/attack/plot_acc_mia_flops_standard_one_fig.py`` plots the qualitative results of mode stealing and membership inference on AlexNet, ResNet18, and VGG16_BN (Section 4.3, Figure 3), the results are saved in ``acc_mia_flops_standard.pdf``. The command is ``python plot_acc_mia_flops_standard_one_fig.py``
  - ``plot/attack/plot_acc_mia_flops_standard_one_fig_append.py`` plots the qualitative results of mode stealing and membership inference on ResNet34 and VGG19_BN (Append E, Figure 8), the results are saved in ``acc_mia_flops_standard_append.pdf``. The command is ``python plot_acc_mia_flops_standard_one_fig_append.py``
  - ``plot/attack/plot_acc_flops_standard_five_models_one_fig.py`` plots the qualitative results of model stealing over all models and metrics (Append F.3 and Figure 9),  the results are saved in ``acc_flops_standard_five_models_one_fig.pdf``. The command is ``python plot_acc_flops_standard_five_models_one_fig.py``
  - ``plot/attack/plot_mia_flops_standard_five_models_one_fig.py`` plots the qualitative results of membership inference over all models and metrics (Append F.3 and Figure 10),  the results are saved in ``mia_flops_standard_five_models_one_fig.pdf``. The command is ``python plot_mia_flops_standard_five_models_one_fig.py``
  - ``plot/attack/results`` the flops and attack performance result raw data
- ``plot/accuracy_drop`` computes the wilcoxon p-value of accuracy drop in Section 6.2
  - The command is ``cd plot/accuracy_drop && python wilcoxon.py``
- ``plot/data_assumption`` plots the figures to evaluate the assumption of larger data (Section 6.1, Figure 5, Append H, Figure 11, Figure 12)
  - By setting ``attr`` (line 69 of ``plot_cifar100_accuracy.py``) you can get the results of different metrics. Viable ``attr`` includes ``['accuracy', 'fidelity', 'asr']``. 
  - The command is ``cd plot/data_assumption && python plot_cifar100_accuracy.py``. The results are saved in ``multi_arch_cifar100_compare_accuracy.pdf``, ``multi_arch_cifar100_compare_fidelity.pdf``, and ``multi_arch_cifar100_compare_asr.pdf``
- ``plot/flops_validation`` plots the relationship between %FLOPs and inference latency (Section 4.2, Append D, and Figure 7). The command is ``python flops_validation.py`` and the result is saved in ``plot/flops_validation/time_flops.pdf``
- ``plot/realdevice`` prints the results on the real devices (Section 5.C, Table VII, Table VI). The command ``python whole_time.py`` prints Table VI, and command ``python breakdown_time.py`` prints Table VII.

## Attack Implementation against SOTER
Directory ``soter-attack`` includes all the code and scripts to attack SOTER. The code structure is as follows
- ``soter-attack/scripts`` includes the scripts to run the code
- ``soter-attack/images`` include the results in the paper
  - ``soter_res.csv`` is the raw results of attacking SOTER
- ``soter-attack/knockoff`` is the source code to attack SOTER, the code is based on the original repository of KnockoffNet. All the directories except ``attack`` are the same as the original KnockoffNet code
  - ``soter-attack/knockoff/attack`` contains the code to attack SOTER
  - ``soter-attack/knockoff/attack/soter_recover_scalar.py`` is the attack code against SOTER

## Code for Model Stealing 
Directory ``model-stealing`` contains the code base for the part of model stealing 

- ``model-stealing/scripts`` contains the scripts of CIFAR datasets
  - ``model-stealing/scripts/adaptive`` perform adaptive attack (Section 8 and Append L)
  - ``model-stealing/scripts/nettailor`` contains the results of TEESlice
  - ``model-stealing/scripts/nettailor_multiarch_attack`` contains the script to evaluate the introduction attack surface of backbone (Section 6.2)
  - ``model-stealing/scripts/knockoff_layers.sh`` is the script for shielding deep layers (1), shielding shallow layers (2), shielding intermediate layers (4)
  - - ``model-stealing/scripts/knockoff_mag.sh`` is the script for shielding large magnitude weights (3),
- ``model-stealing/scripts_face`` contains the scripts for STL10 and UTKFace dataset, the structure is similar
- ``model-stealing/knockoff`` is the code to perform model stealing
  - ``model-stealing/knockoff/adversary`` implements the attack on all the model partition solutions
    - ``model-stealing/knockoff/adversary/train_layers.py``: attack against shielding deep layers (1), shielding shallow layers (2), shielding intermediate layers (4)
    - ``model-stealing/knockoff/adversary/train_mag.py``: attack against shielding large magnitude weights (3)
    - ``model-stealing/knockoff/adversary/train_nettailor.py``: attack against our approach
  - ``model-stealing/knockoff/adversary_adaptive`` the code to evaluate the introduction attack surface of backbone (Section 6.2) 
  - ``model-stealing/knockoff/nettailor`` the code of our approach on CIFAR10
  - ``model-stealing/knockoff/nettailor`` the code of our approach on STL10 and UTKFace

## Code for Membership Inference
Directory ``membership-inference`` contains the code base for the part of membership inference
- ``membership-inference/demoloader`` and ``membership-inference/doctor`` are the code to perform membership inference attack and are adopted from ML-Doctor
- ``membership-inference/scripts`` contains the script to run the code
- ``membership-inference/train_layer.py``: train and attack the models of shielding deep layers (1), shielding shallow layers (2), 
- ``membership-inference/train_mag.py``: train an attack the models against shielding large magnitude weights (3)
- ``membership-inference/train_soter.py``: train an attack the models against shielding intermediate layers (4)

## Code for Prototype on Real Devices
The system implementation on real device is in ``real-device`` directory. The directory includes both C++ code inside SGX and PyTorch code on GPU. 
- ``python`` includes the python interface to call the C code inside SGX
- ``nettailor`` and ``nettailor_224`` includes the scripts to run the run the code on real devices
- ``Enclave/sgxdnn.cpp`` is the implementation of DNN layers in the SGX
