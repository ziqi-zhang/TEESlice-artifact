import torch
import json
import os, sys
import os.path as osp
import time
import numpy as np

from nettailor_224.hetero_student_resnet import load_checkpoint, create_model

save_dir = "nettailor_224/models_nettailor/jetson"
if not osp.exists(save_dir):
    os.makedirs(save_dir)

for dataset_name in ["CIFAR10", "CIFAR100", "STL10", "UTKFaceRace"]:
    for model_name in ["alexnet", "resnet18", "vgg16_bn"]:

        if dataset_name in ["CIFAR10", "STL10"]:
            num_classes = 10
        elif dataset_name == "CIFAR100":
            num_classes = 100
        elif dataset_name == "UTKFaceRace":
            num_classes = 4
        else:
            raise NotImplementedError

        device = torch.device("cpu")
        model = create_model('resnet18', num_classes, max_skip=3)
        full_model_dir = f"nettailor_224/models_nettailor/models/CIFAR10-resnet18/resnet18-iterative-nettailor-3Skip-T10.0-C0.3-Pruned0"
        # import proj_utils
        load_checkpoint(model, model_dir=full_model_dir)


        # print('\n'+'='*30+'  Model  '+'='*30)
        # print(model)

        # print('\n'+'='*30+'  Parameters  '+'='*30)
        # for n, p in model.named_parameters():
        #     print("{:50} | {:10} | {:30} | {:20} | {}".format(
        #         n, 'Trainable' if p.requires_grad else 'Frozen' , 
        #         str(p.size()), str(np.prod(p.size())), str(p.type()))
        #     )

        # print(model.stats())
        # print(model.expected_complexity())

        iteration = 5
        batch_size = 16

        input = torch.rand(batch_size, 3, 224, 224)

        model.hetero_deploy()
        input = input.cuda()
        model.hetero_forward(input)

        inference_times = []
        throughputs = []

        for i in range(iteration):
            start = time.time()
            model.hetero_forward(input)
            end = time.time()
            elapse = (end-start) * 1000 / batch_size
            inference_times.append(elapse)
            throughputs.append(1000 / inference_times[-1])

        inf_mean = np.mean(inference_times)
        inf_std = np.std(inference_times)
        th_mean = np.mean(throughputs)
        th_std = np.std(throughputs)

        result = {
            "name": f"{model_name}_{dataset_name}"
        }
        mean, std = np.mean(inference_times), np.std(inference_times)
        th_mean, th_std = np.mean(throughputs), np.std(throughputs)
        result["mean"] = mean
        result["std"] = std
        result["inf_times"] = inference_times
        result["th_mean"] = th_mean
        result["th_std"] = th_std
        result["throughputs"] = throughputs

        path = osp.join(
            save_dir, f"inference_{dataset_name}_{model_name}.json"
        )

        with open(path, "w") as f:
            json.dump(result, f, indent=True)

