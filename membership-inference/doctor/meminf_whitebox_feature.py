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
        


class WhiteBoxFeatureAttackModel(nn.Module):
    def __init__(self, class_num, feature_size):
        super(WhiteBoxFeatureAttackModel, self).__init__()

        self.Output_Component = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(class_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        

        # self.Loss_Component = nn.Sequential(
        # 	nn.Dropout(p=0.2),
        # 	nn.Linear(1, 128),
        # 	nn.ReLU(),
        # 	nn.Linear(128, 64),
        # )

        # self.Gradient_Component = nn.Sequential(
        # 	nn.Dropout(p=0.2),
        # 	nn.Conv2d(1, 1, kernel_size=5, padding=2),
        # 	nn.BatchNorm2d(1),
        # 	nn.ReLU(),
        # 	nn.MaxPool2d(kernel_size=2),
        # 	nn.Flatten(),
        # 	nn.Dropout(p=0.2),
        # 	nn.Linear(total, 256),
        # 	nn.ReLU(),
        # 	nn.Dropout(p=0.2),
        # 	nn.Linear(256, 128),
        # 	nn.ReLU(),
        # 	nn.Linear(128, 64),
        # )

        self.Feature_Component = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.Label_Component = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(class_num, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.Encoder_Component = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, output, feature, label):
        Output_Component_result = self.Output_Component(output)
        # Loss_Component_result = self.Loss_Component(loss)
        # Gradient_Component_result = self.Gradient_Component(gradient)
        Feature_Component_result = self.Feature_Component(feature)
        Label_Component_result = self.Label_Component(label)

        # Loss_Component_result = F.softmax(Loss_Component_result, dim=1)
        # Gradient_Component_result = F.softmax(Gradient_Component_result, dim=1)

        # final_inputs = Output_Component_result
        # final_inputs = Loss_Component_result
        # final_inputs = Gradient_Component_result
        # final_inputs = Label_Component_result

        final_inputs = torch.cat((Output_Component_result, Feature_Component_result, Label_Component_result), 1)
        # final_inputs = torch.cat((Output_Component_result, Loss_Component_result, Gradient_Component_result, Label_Component_result), 1)
        final_result = self.Encoder_Component(final_inputs)

        return final_result

class attack_for_whitebox_feature():
    def __init__(
        self, TARGET_PATH, SHADOW_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, 
        target_model, shadow_model, attack_model, device, class_num, logger
    ):
        self.device = device
        self.class_num = class_num
        self.logger = logger

        self.ATTACK_SETS = ATTACK_SETS

        self.TARGET_PATH = TARGET_PATH
        self.target_model = target_model.to(self.device)
        print(f"Load target from {self.TARGET_PATH}")
        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        self.target_model.eval()


        self.SHADOW_PATH = SHADOW_PATH
        self.shadow_model = shadow_model.to(self.device)
        print(f"Load target from {self.SHADOW_PATH}")
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.target_criterion = nn.CrossEntropyLoss(reduction='none')
        self.attack_criterion = nn.CrossEntropyLoss()
        #self.optimizer = optim.SGD(self.attack_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

        self.attack_train_data = None
        self.attack_test_data = None
        
        self.best_test_gndtrth = []
        self.best_test_predict = []
        self.best_test_probabe = []
        self.best_acc = -1
        self.best_state_dict = None


    def _get_data(self, model, inputs, targets):
        results = model.forward_with_feature(inputs)
        outputs = results[0]
        ends= results[1]
        fc_input = ends[-1]

        # outputs = F.softmax(outputs, dim=1)
        # losses = self.target_criterion(results, targets)

        # gradients = []
        
        # for loss in losses:
        #     loss.backward(retain_graph=True)

        #     gradient_list = reversed(list(model.named_parameters()))

        #     for name, parameter in gradient_list:
        #         if 'weight' in name:
        #             gradient = parameter.grad.clone() # [column[:, None], row].resize_(100,100)
        #             gradient = gradient.unsqueeze_(0)
        #             gradients.append(gradient.unsqueeze_(0))
        #             break
        
        features = fc_input.detach().clone()
        # features = []
        # for feature in fc_input:
        #     features.append(feature)

        labels = []
        for num in targets:
            label = [0 for i in range(self.class_num)]
            label[num.item()] = 1
            labels.append(label)

        # gradients = torch.cat(gradients, dim=0)
        # losses = losses.unsqueeze_(1).detach()
        # features = torch.cat(features, dim=0)
        outputs, _ = torch.sort(outputs, descending=True)
        labels = torch.Tensor(labels)
        return outputs, features, labels

    def prepare_dataset(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # output, loss, gradient, label = self._get_data(self.shadow_model, inputs, targets)
                output, features, label = self._get_data(self.shadow_model, inputs, targets)

                pickle.dump((output, features, label, members), f)

        self.logger.add_line(f"Finished Saving Train Dataset to {self.ATTACK_SETS + 'train.p'}")

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # output, loss, gradient, label = self._get_data(self.target_model, inputs, targets)
                output, features, label = self._get_data(self.target_model, inputs, targets)
            
                pickle.dump((output, features, label, members), f)

            # pickle.dump((output, loss, gradient, label, members), open(self.ATTACK_PATH + "test.p", "wb"))

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
                    output, features, label, members = pickle.load(f)
                    output, features, label, members = output.to(self.device), features.to(self.device), label.to(self.device), members.to(self.device)

                    results = self.attack_model(output, features, label)
                    # results = F.softmax(results, dim=1)
                    losses = self.attack_criterion(results, members)
                    
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
                        output, features, label, members = pickle.load(f)
                        output, features, label, members = output.to(self.device), features.to(self.device), label.to(self.device), members.to(self.device)

                        results = self.attack_model(output, features, label)

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



def attack_mode_whitebox_feature(
    TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, 
    target_model, shadow_model, attack_model, get_attack_set, num_classes
):
    MODELS_PATH = osp.join(ATTACK_PATH, "meminf_attack_whitebox_feature.pth")
    RESULT_PATH = osp.join(ATTACK_PATH, "meminf_attack_whitebox_feature.p")
    BEST_RESULT_PATH = osp.join(ATTACK_PATH, "meminf_best_attack_whitebox_feature.p")
    ATTACK_SETS = osp.join(ATTACK_PATH, "meminf_attack_whitebox_feature_")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=ATTACK_PATH)

    attack = attack_for_whitebox_feature(
        TARGET_PATH, SHADOW_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, device, num_classes, logger
    )
    
    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(10):
        flag = 1 if i == 9 else 0
        logger.add_line("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH, BEST_RESULT_PATH)
        
    res_best = attack.eval_best_result()

    attack.saveModel(MODELS_PATH)
    logger.add_line("Saved Attack Model")

    return res_train, res_test, res_best

def get_feature_size(model):
    gradient_size = []
    gradient_list = reversed(list(model.named_parameters()))
    for name, parameter in gradient_list:
        if 'weight' in name:
            feature_size = parameter.shape[1]
            break

    return feature_size