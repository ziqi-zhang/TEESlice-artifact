import torch
import pickle
import torch.nn as nn
import torch.optim as optim

from utils.define_models import *
from sklearn.metrics import f1_score
import gol
from pdb import set_trace as st
import copy

class attack_training():
    def __init__(
        self, device, attack_trainloader, attack_testloader, 
        target_model, TARGET_PATH, ATTACK_PATH, logger, target_name
    ):
        self.device = device
        self.TARGET_PATH = TARGET_PATH
        self.ATTACK_PATH = ATTACK_PATH
        self.logger = logger

        self.target_model = target_model.to(self.device)
        if target_name == "pretrained":
            print(f"Pretrained model dont need to load from target path")
            self.training_target_model = copy.deepcopy(self.target_model)
            print(f"Load eval target model weights from {self.TARGET_PATH}")
            self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        else:
            print(f"Load target model weights from {self.TARGET_PATH}")
            self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
            self.training_target_model = self.target_model
        if target_name == "output":
            self.get_middle_output_training_target_model = self.get_final_output
            self.get_middle_output_target_model = self.get_final_output
            
        self.target_model.eval()

        self.attack_model = None

        self.attack_trainloader = attack_trainloader
        self.attack_testloader = attack_testloader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)
        self.dataset_type = None

    def _get_activation(self, name, activation):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    def init_attack_model(self, size, output_classes):
        x = torch.rand(size).to(self.device)
        input_classes = self.get_middle_output_training_target_model(x).flatten().shape[0]
        self.attack_model = attrinf_attack_model(inputs=input_classes, outputs=output_classes)
        self.attack_model.to(self.device)
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-3)
        if output_classes == 2:
            self.dataset_type = "binary"
        else:
            self.dataset_type = "macro"
            
    def get_final_output(self, x):
        temp = []
        for name, _ in self.training_target_model.named_parameters():
            if "weight" in name:
                temp.append(name)

        if 1 > len(temp):
            raise IndexError('layer is out of range')

        name = temp[-1].split('.')
        var = eval('self.training_target_model.' + name[0])
        out = {}
        var.register_forward_hook(self._get_activation(name[1], out))
        _ = self.training_target_model(x)
        
        return out[name[1]]

    def get_middle_output_training_target_model(self, x):
        temp = []
        for name, _ in self.training_target_model.named_parameters():
            if "weight" in name:
                temp.append(name)

        if 1 > len(temp):
            raise IndexError('layer is out of range')

        name = temp[-2].split('.')
        var = eval('self.training_target_model.' + name[0])
        out = {}
        var[int(name[1])].register_forward_hook(self._get_activation(name[1], out))
        _ = self.training_target_model(x)
        
        return out[name[1]]
    
    def get_middle_output_target_model(self, x):
        temp = []
        for name, _ in self.target_model.named_parameters():
            if "weight" in name:
                temp.append(name)

        if 1 > len(temp):
            raise IndexError('layer is out of range')

        name = temp[-2].split('.')
        var = eval('self.target_model.' + name[0])
        out = {}
        var[int(name[1])].register_forward_hook(self._get_activation(name[1], out))
        _ = self.target_model(x)
        
        return out[name[1]]

    # Training
    def train(self, epoch):
        self.attack_model.train()
        
        train_loss = 0
        correct = 0
        total = 0

        final_result = []
        final_gndtrth = []
        final_predict = []
        final_probabe = []

        for batch_idx, (inputs, [_, targets]) in enumerate(self.attack_trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            oracles = self.get_middle_output_training_target_model(inputs)
            outputs = self.attack_model(oracles)
            outputs = F.softmax(outputs, dim=1)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if gol.get_value("debug"):
                break
            
            if epoch:
                    final_gndtrth.append(targets)
                    final_predict.append(predicted)
                    final_probabe.append(outputs[:, 1])

        if epoch:
            final_gndtrth = torch.cat(final_gndtrth, dim=0).cpu().detach().numpy()
            final_predict = torch.cat(final_predict, dim=0).cpu().detach().numpy()
            final_probabe = torch.cat(final_probabe, dim=0).cpu().detach().numpy()

            test_f1_score = f1_score(final_gndtrth, final_predict, average=self.dataset_type)

            final_result.append(test_f1_score)

            with open(self.ATTACK_PATH + "attrinf_train.p", "wb") as f:
                pickle.dump((final_gndtrth, final_predict, final_probabe), f)

            self.logger.add_line("Saved Attack Test Ground Truth and Predict Sets")
            self.logger.add_line("Test F1: %f" % (test_f1_score))

        final_result.append(1.*correct/total)
        self.logger.add_line( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def test(self, epoch):
        self.attack_model.eval()

        correct = 0
        total = 0
        final_result = []
        final_gndtrth = []
        final_predict = []
        final_probabe = []

        with torch.no_grad():
            for inputs, [_, targets] in self.attack_testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                oracles = self.get_middle_output_target_model(inputs)
                outputs = self.attack_model(oracles)
                outputs = F.softmax(outputs, dim=1)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if epoch:
                    final_gndtrth.append(targets)
                    final_predict.append(predicted)
                    final_probabe.append(outputs[:, 1])

        if epoch:
            final_gndtrth = torch.cat(final_gndtrth, dim=0).cpu().numpy()
            final_predict = torch.cat(final_predict, dim=0).cpu().numpy()
            final_probabe = torch.cat(final_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_gndtrth, final_predict, average=self.dataset_type)

            final_result.append(test_f1_score)

            with open(self.ATTACK_PATH + "attrinf_test.p", "wb") as f:
                pickle.dump((final_gndtrth, final_predict, final_probabe), f)

            self.logger.add_line("Saved Attack Test Ground Truth and Predict Sets")
            self.logger.add_line("Test F1: %f" % (test_f1_score))

        final_result.append(1.*correct/total)
        self.logger.add_line( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result

    def saveModel(self):
        torch.save(self.attack_model.state_dict(), self.ATTACK_PATH + "attrinf_attack_model.pth")

def train_attack_model(
    TARGET_PATH, ATTACK_PATH, output_classes, device, target_model, 
    train_loader, test_loader, size, logger, target_name
):
    attack = attack_training(
        device, train_loader, test_loader, target_model, 
        TARGET_PATH, ATTACK_PATH, logger, target_name
    )
    attack.init_attack_model(size, output_classes)

    for epoch in range(100):
        flag = 1 if epoch==99 else 0
        logger.add_line("<======================= Epoch " + str(epoch+1) + " =======================>")
        logger.add_line("attack training")
        acc_train = attack.train(flag)
        logger.add_line("attack testing")
        acc_test = attack.test(flag)
        if gol.get_value("debug"):
                break

    attack.saveModel()
    logger.add_line("Saved Attack Model")
    logger.add_line("Finished!!!")


    return acc_train, acc_test