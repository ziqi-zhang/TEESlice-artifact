import os
import glob
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
np.set_printoptions(threshold=np.inf)
from pdb import set_trace as st
from log_utils import *
import os.path as osp

# from opacus import PrivacyEngine
from torch.optim import lr_scheduler
# from opacus.utils import module_modification
from sklearn.metrics import f1_score, roc_auc_score
# from opacus.dp_model_inspector import DPModelInspector

import gol

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class attack_for_blackbox():
    def __init__(
        self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, 
        target_model, shadow_model, attack_model, device, logger
    ):
        self.device = device
        self.logger = logger

        self.TARGET_PATH = TARGET_PATH
        self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS

        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)

        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        print(f"Load target from {self.TARGET_PATH}")
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
        print(f"Load shadow from {self.SHADOW_PATH}")

        self.target_model.eval()
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)
        
        self.best_test_gndtrth = []
        self.best_test_predict = []
        self.best_test_probabe = []
        self.best_acc = -1
        self.best_state_dict = None

    def _get_data(self, model, inputs):
        result = model(inputs)
        
        output, _ = torch.sort(result, descending=True)
        # results = F.softmax(results[:,:5], dim=1)
        _, predicts = result.max(1)
        
        prediction = []
        for predict in predicts:
            prediction.append([1,] if predict else [0,])

        prediction = torch.Tensor(prediction)

        # final_inputs = torch.cat((results, prediction), 1)
        # print(final_inputs.shape)

        return output, prediction

    def prepare_dataset(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.shadow_model, inputs)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        self.logger.add_line(f"Finished Saving Train Dataset to {self.ATTACK_SETS + 'train.p'}")

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.target_model, inputs)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        self.logger.add_line(f"Finished Saving Test Dataset to {self.ATTACK_SETS + 'test.p'}")
        
    def prepare_test_dataset(self):
        print(f"Preparing test dataset for {self.TARGET_PATH}")
        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.target_model, inputs)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        self.logger.add_line(f"Finished Saving Test Dataset to {self.ATTACK_SETS + 'test.p'}")

    def train(self, epoch, result_path):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    output, prediction, members = pickle.load(f)
                    output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                    results = self.attack_model(output, prediction)
                    results = F.softmax(results, dim=1)

                    losses = self.criterion(results, members)
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
            self.logger.add_line("Saved Attack Train Ground Truth and Predict Sets")
            self.logger.add_line("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        final_result.append(1.*correct/total)
        self.logger.add_line( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result

    def test(self, epoch, result_path, best_result_path):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        output, prediction, members = pickle.load(f)
                        output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                        results = self.attack_model(output, prediction)
                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()
                        results = F.softmax(results, dim=1)

                        final_test_gndtrth.append(members.detach())
                        final_test_predict.append(predicted.detach())
                        final_test_probabe.append(results[:, 1].detach())

                        batch_idx += 1
                    except EOFError:
                        break
                    
        acc = correct/(1.0*total)

        if epoch or acc > self.best_acc:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            self.logger.add_line("Saved Attack Test Ground Truth and Predict Sets")
            self.logger.add_line("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))
            
            if acc > self.best_acc:
                self.best_acc = acc
                self.best_test_gndtrth = final_test_gndtrth
                self.best_test_predict = final_test_predict
                self.best_test_probabe = final_test_probabe
                self.best_state_dict = self.attack_model.state_dict()
                
                with open(best_result_path, "wb") as f:
                    pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

        final_result.append(1.*correct/total)
        self.logger.add_line( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result
    
    def eval_best_result(self):
        best_f1_score = f1_score(self.best_test_gndtrth, self.best_test_predict)
        best_roc_auc_score = roc_auc_score(self.best_test_gndtrth, self.best_test_probabe)
        self.logger.add_line("Best Acc: %f\n F1: %f\nAUC: %f" % (self.best_acc, best_f1_score, best_roc_auc_score))
        return best_f1_score, best_roc_auc_score, self.best_acc

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.best_state_dict, path)
        
    def loadModel(self, path):
        ckpt = torch.load(path)
        self.attack_model.load_state_dict(ckpt)

class attack_for_top3(attack_for_blackbox):
    def __init__(
        self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, 
        target_model, shadow_model, attack_model, device, logger
    ):
        super(attack_for_top3, self).__init__(SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, 
        target_model, shadow_model, attack_model, device, logger)
        
    def _get_data(self, model, inputs):
        result = model(inputs)
        
        output, _ = torch.sort(result, descending=True)
        output_top3 = F.softmax(output, dim=1)[:,:3]
        _, predicts = result.max(1)
        
        prediction = []
        for predict in predicts:
            prediction.append([1,] if predict else [0,])

        prediction = torch.Tensor(prediction)

        # final_inputs = torch.cat((results, prediction), 1)
        # print(final_inputs.shape)

        return output_top3, prediction

def attack_top3(
    TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, 
    target_model, shadow_model, attack_model, get_attack_set, num_classes
):
    MODELS_PATH = osp.join(ATTACK_PATH, "meminf_top3.pth")
    RESULT_PATH = osp.join(ATTACK_PATH, "meminf_top3.p")
    BEST_RESULT_PATH = osp.join(ATTACK_PATH, "meminf_best_top3.p")
    ATTACK_SETS = osp.join(ATTACK_PATH, "meminf_attack_top3_")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=ATTACK_PATH)

    epochs = 10
    attack = attack_for_top3(
        SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, device, logger
    )

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(epochs):
        flag = 1 if i == epochs-1 else 0
        logger.add_line("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH, BEST_RESULT_PATH)
        if gol.get_value("debug"):
            break
        
    res_best = attack.eval_best_result()

    attack.saveModel(MODELS_PATH)
    logger.add_line(f"Saved Attack Model to {MODELS_PATH}")
    print(f"{sys._getframe().f_code.co_name} finished")

    return res_train, res_test, res_best

def attack_top3_no_train(
    TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, 
    target_model, shadow_model, attack_model, get_attack_set, num_classes, shadow_model_dir
):
    SHADOW_MODELS_PATH = osp.join(shadow_model_dir, "meminf_top3.pth")
    RESULT_PATH = osp.join(ATTACK_PATH, "meminf_top3.p")
    BEST_RESULT_PATH = osp.join(ATTACK_PATH, "meminf_best_top3.p")
    ATTACK_SETS = osp.join(ATTACK_PATH, "meminf_attack_top3_")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=ATTACK_PATH)

    attack = attack_for_top3(
        SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, device, logger
    )
    
    attack.prepare_test_dataset()
    attack.loadModel(SHADOW_MODELS_PATH)
    res_test = attack.test(1, RESULT_PATH, BEST_RESULT_PATH)

    print(f"{sys._getframe().f_code.co_name} finished")

    return res_test

def attack_mode0_no_train(
    TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, 
    target_model, shadow_model, attack_model, get_attack_set, num_classes, shadow_model_dir
):
    SHADOW_MODELS_PATH = osp.join(shadow_model_dir, "meminf_attack0.pth")
    RESULT_PATH = osp.join(ATTACK_PATH, "meminf_attack0.p")
    BEST_RESULT_PATH = osp.join(ATTACK_PATH, "meminf_best_attack0.p")
    ATTACK_SETS = osp.join(ATTACK_PATH, "meminf_attack_mode0_")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=ATTACK_PATH)

    epochs = 10
    attack = attack_for_blackbox(
        SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, device, logger
    )
    
    attack.prepare_test_dataset()
    attack.loadModel(SHADOW_MODELS_PATH)
    res_test = attack.test(1, RESULT_PATH, BEST_RESULT_PATH)

    return res_test


def attack_mode0(
    TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, 
    target_model, shadow_model, attack_model, get_attack_set, num_classes
):
    MODELS_PATH = osp.join(ATTACK_PATH, "meminf_attack0.pth")
    RESULT_PATH = osp.join(ATTACK_PATH, "meminf_attack0.p")
    BEST_RESULT_PATH = osp.join(ATTACK_PATH, "meminf_best_attack0.p")
    ATTACK_SETS = osp.join(ATTACK_PATH, "meminf_attack_mode0_")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=ATTACK_PATH)

    epochs = 10
    attack = attack_for_blackbox(
        SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, device, logger
    )

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(epochs):
        flag = 1 if i == epochs-1 else 0
        logger.add_line("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH, BEST_RESULT_PATH)
        if gol.get_value("debug"):
            break
        
    res_best = attack.eval_best_result()

    attack.saveModel(MODELS_PATH)
    logger.add_line(f"Saved Attack Model to {MODELS_PATH}")
    print(f"{sys._getframe().f_code.co_name} finished")

    return res_train, res_test, res_best

