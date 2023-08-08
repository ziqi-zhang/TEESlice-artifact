import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.utils.data

from torch.autograd import Variable
import gol


class ccs_inversion(object):
    '''
    Model inversion is a kind of data reconstruct attack.
    This class we implement the attack on neural network,
    the attack goal is to generate data that is close to original data distribution.
    This attack was first described in Fredrikson's paper (Algorithm 1):
    "Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures" (CCS2015)
    -----------------------------NOTICE---------------------------
    If the model's output layer doesn't contain Softmax layer, please add it manually.
    And parameters will influence the quality of the reconstructed data significantly.
    --------------------------------------------------------------
    Args:
        ------------------------
        :param target_model: the target model which we are trying to reconstruct its training dataset
        :param input_size: the size of the model's input
        :param output_size: the size of the model's output
        :param target_label: the reconstructed output is belong to this class
        :param param_alpha: the number of iteration round
        :param param_beta, gamma, lambda: the hyperparameters in paper
    '''

    def __init__(self, target_model, input_size, output_size, target_label, param_alpha, param_beta, param_gamma, param_lambda, device):
        self.target_model = target_model
        self.input_size = input_size
        self.output_size = output_size
        self.target_label = target_label
        self.param_alpha = param_alpha
        self.param_beta = param_beta
        self.param_gamma = param_gamma
        self.param_lambda = param_lambda
        self.device = device

        self.target_model.to(self.device).eval()

    def model_invert(self):
        current_x = []
        cost_x = []
        current_x.append(Variable(torch.from_numpy(np.zeros(self.input_size, dtype=np.uint8))).float().to(self.device))
        for i in range(self.param_alpha):
            cost_x.append(self.invert_cost(current_x[i]).to(self.device))
            cost_x[i].backward()
            current_x.append((current_x[i] - self.param_lambda * current_x[i].grad).data)
            if self.invert_cost(current_x[i + 1]) <= self.param_gamma:
                print('Target cost value achieved')
                break
            elif i >= self.param_beta and self.invert_cost(current_x[i + 1]) >= max(cost_x[self.param_beta:i + 1]):
                print('Exceed beta')
                break
            if gol.get_value("debug"):
                break

        i = cost_x.index(min(cost_x))
        return current_x[i]

    def invert_cost(self, input_x):
        return 1 - self.target_model(input_x.requires_grad_(True))[0][self.target_label]

    def reverse_mse(self, ori_dataset):
        '''
        output the average MSE value of different classes
        :param ori_dataset: the data used to train the target model, please make sure setting the batch size as 1.
        :return: MSE value
        '''
        print("Processing Output size")
        reverse_data = []
        for i in range(self.output_size):
            self.target_label = i
            a = self.model_invert()
            reverse_data.append(a)
            print(f"Output size {i}/{self.output_size}")
        class_avg = [Variable(torch.from_numpy(np.zeros(self.input_size, dtype=np.uint8))).float().to(self.device) for _ in range(self.output_size)]
        class_mse = [0 for _ in range(self.output_size)]
        class_count = [0 for _ in range(self.output_size)]

        print("Processing ori_dataset")
        for i, (x, y) in enumerate(ori_dataset):
            x, y = x.to(self.device), y.to(self.device)
            class_avg[y] = class_avg[y] + x
            class_count[y] = class_count[y] + 1
            if i % 500 == 0:
                print(f"Ori_dataset {i}/{len(ori_dataset)}")
            if gol.get_value("debug"):
                break

        print("Figuring MSE")
        for i in range(self.output_size):
            class_mse[i] = self.figure_mse(class_avg[i] / class_count[i], (reverse_data[i]))
            print(f"Output size {i}/{self.output_size}")


        all_class_avg_mse = 0
        for i in range(self.output_size):
            all_class_avg_mse = all_class_avg_mse + class_mse[i]
        return all_class_avg_mse / self.output_size

    def figure_mse(self, recover_fig, ori_fig):
        '''
        :param recover_fig: figure recovered by model inversion attack, type:
        :param ori_fig: figure in the training dataset
        :return: MSE value of these two figures
        '''
        diff = nn.MSELoss()
        return diff(recover_fig, ori_fig)

        
        

def revealer_inversion(G, D, T, E, iden, device, noise = 100,lr=1e-3, momentum=0.9, lamda=100, iter_times=1500, clip_range=1):
    '''
    
    This model inversion attack was proposed by Zhang et al. in CVPR20
    "The Secret Revealer: Generative Model-Inversion Attacks Against Deep Neural Networks"
    '''
    iden = iden.view(-1).long().to(device)
    G, D, T, E = G.to(device), D.to(device), T.to(device), E.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    bs = iden.shape[0]
    
    G.eval()
    D.eval()
    T.eval()

    max_score = torch.zeros(bs)
    max_iden = torch.zeros(bs)
    z_hat = torch.zeros(bs, noise,1,1)

    cnt = 0
    for random_seed_sudo in range(10):
        tf = time.time()
        random_seed = random.randint(0,200)
        torch.manual_seed(random_seed) 
        torch.cuda.manual_seed(random_seed) 
        np.random.seed(random_seed) 
        random.seed(random_seed)

        z = torch.randn(bs, noise, 1, 1).to(device).float()
        z.requires_grad = True
        v = torch.zeros(bs, noise, 1, 1).to(device).float()
            
        for i in range(iter_times):
            fake = G(z)
            label = D(fake)
            out = T(fake)

            if z.grad is not None:
                z.grad.data.zero_()

            Prior_Loss = - label.mean()
            Iden_Loss = criterion(out, iden)
            Total_Loss = Prior_Loss + lamda * Iden_Loss

            Total_Loss.backward()
            
            v_prev = v.clone()
            gradient = z.grad.data

            v = momentum * v - lr * gradient
            z = z + ( - momentum * v_prev + (1 + momentum) * v)
            z = torch.clamp(z.detach(), -clip_range, clip_range).float()
            z.requires_grad = True

            Prior_Loss_val = Prior_Loss.item()
            Iden_Loss_val = Iden_Loss.item()

            if (i + 1) % 300 == 0:
                fake_img = G(z.detach())
                eval_prob = E(fake_img)
                eval_iden = torch.argmax(eval_prob, dim=1).view(-1)
                acc = iden.eq(eval_iden.long()).sum().item() * 1.0 / bs
            if gol.get_value("debug"):
                break


        fake = G(z)
        score = T(fake)
        eval_prob = E(fake)
        _, eval_iden = torch.max(eval_prob, dim=1)
        for i in range(bs):
            _, gtl = torch.max(score, 1)
            gt = gtl[i].item()
            if score[i, gt].item() > max_score[i].item():
                max_score[i] = score[i, gt]
                max_iden[i] = eval_iden[i]
                z_hat[i, :] = z[i, :]
            if eval_iden[i].item() == gt:
                cnt += 1

    print("Acc:{:.2f}\t".format(cnt * 1.0 / (bs*10)))
    return cnt * 1.0 / (bs*10)

def load_data(PATH_target, PATH_evaluation, target_model, evaluate_model):
    '''	
    Evaluate model is used to predict the identity based on the input reconstructed image.	
    If the evaluation classifier achieves high accuracy, the reconstructed image is considered to expose 
    private information about the target label.
    
    The evaluate model should be different from the target network because the reconstructed images may
    incorporate features that overfit the target network while being semantically meaningless.
    Moreover, the evaluation classifier should be highly performant.
    '''
    target_model.load_state_dict(torch.load(PATH_target))
    evaluate_model.load_state_dict(torch.load(PATH_evaluation))
    print("Finished Loading")

    return target_model, evaluate_model

def load_data_eval_model(PATH_evaluation, evaluate_model):
    evaluate_model.load_state_dict(torch.load(PATH_evaluation))
    print("Finished Loading")

    return evaluate_model

def prepare_GAN(data_type, discriminator, generator, PATH_1, PATH_2):
    discriminator.load_state_dict(torch.load(PATH_1))
    generator.load_state_dict(torch.load(PATH_2))

    iden = torch.zeros(10)

    if data_type.lower() == 'stl10' or  data_type.lower() == 'fmnist':
        for i in range(10):
            iden[i] = i

    elif data_type.lower() == 'utkface':
        for i in range(10):
            iden[i] = i % 4

    elif data_type.lower() == 'celeba':
        for i in range(10):
            iden[i] = i % 8
            
    return discriminator, generator, iden