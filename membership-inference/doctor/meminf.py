# import os
# import glob
# import torch
# import pickle
# import numpy as np
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# np.set_printoptions(threshold=np.inf)
# from pdb import set_trace as st
# from log_utils import *
# import os.path as osp

# # from opacus import PrivacyEngine
# from torch.optim import lr_scheduler
# # from opacus.utils import module_modification
# from sklearn.metrics import f1_score, roc_auc_score
# # from opacus.dp_model_inspector import DPModelInspector

# import gol

# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.normal_(m.weight.data)
#         m.bias.data.fill_(0)
#     elif isinstance(m,nn.Linear):
#         nn.init.xavier_normal_(m.weight)
#         nn.init.constant_(m.bias, 0)

# class shadow():
#     def __init__(self, trainloader, testloader, model, device, batch_size, loss, optimizer, logger, epochs):
#         self.device = device
#         self.model = model.to(self.device)
#         self.trainloader = trainloader
#         self.testloader = testloader
#         self.epochs = epochs

#         self.criterion = loss
#         self.optimizer = optimizer
#         self.logger = logger

#         # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [50, 100], 0.1)
#         # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [int(epochs*0.6), int(epochs*0.8)], 0.5)
#         # self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [int(epochs*0.5), int(epochs*1)], 0.1)
#         self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, [int(epochs*0.5), int(epochs*0.75)], 0.1)


#     # Training
#     def train(self):
#         self.model.train()
        
#         train_loss = 0
#         correct = 0
#         total = 0
        
#         for batch_idx, (inputs, targets) in enumerate(self.trainloader):

#             inputs, targets = inputs.to(self.device), targets.to(self.device)

#             self.optimizer.zero_grad()
#             outputs = self.model(inputs)

#             loss = self.criterion(outputs, targets)
#             loss.backward()
#             self.optimizer.step()

#             train_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
            
#             if gol.get_value("debug"):
#                 break

#         self.scheduler.step()

#         self.logger.add_line( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/(batch_idx+1)))

#         return 1.*correct/total


#     def saveModel(self, path):
#         torch.save(self.model.state_dict(), path)

#     def get_noise_norm(self):
#         return self.noise_multiplier, self.max_grad_norm

#     def test(self):
#         self.model.eval()
#         test_loss = 0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, targets in self.testloader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs = self.model(inputs)

#                 loss = self.criterion(outputs, targets)

#                 test_loss += loss.item()
#                 _, predicted = outputs.max(1)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets).sum().item()
                
#                 if gol.get_value("debug"):
#                     break

#             self.logger.add_line( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))

#         return 1.*correct/total


        
# class attack_for_blackbox():
#     def __init__(
#         self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, 
#         target_model, shadow_model, attack_model, device, logger
#     ):
#         self.device = device
#         self.logger = logger

#         self.TARGET_PATH = TARGET_PATH
#         self.SHADOW_PATH = SHADOW_PATH
#         self.ATTACK_SETS = ATTACK_SETS

#         self.target_model = target_model.to(self.device)
#         self.shadow_model = shadow_model.to(self.device)

#         self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
#         print(f"Load target from {self.TARGET_PATH}")
#         self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
#         print(f"Load shadow from {self.SHADOW_PATH}")

#         self.target_model.eval()
#         self.shadow_model.eval()

#         self.attack_train_loader = attack_train_loader
#         self.attack_test_loader = attack_test_loader

#         self.attack_model = attack_model.to(self.device)
#         torch.manual_seed(0)
#         self.attack_model.apply(weights_init)

#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

#     def _get_data(self, model, inputs):
#         result = model(inputs)
        
#         output, _ = torch.sort(result, descending=True)
#         # results = F.softmax(results[:,:5], dim=1)
#         _, predicts = result.max(1)
        
#         prediction = []
#         for predict in predicts:
#             prediction.append([1,] if predict else [0,])

#         prediction = torch.Tensor(prediction)

#         # final_inputs = torch.cat((results, prediction), 1)
#         # print(final_inputs.shape)

#         return output, prediction

#     def prepare_dataset(self):
#         with open(self.ATTACK_SETS + "train.p", "wb") as f:
#             for inputs, targets, members in self.attack_train_loader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 output, prediction = self._get_data(self.shadow_model, inputs)
#                 # output = output.cpu().detach().numpy()
            
#                 pickle.dump((output, prediction, members), f)

#         self.logger.add_line(f"Finished Saving Train Dataset to {self.ATTACK_SETS + 'train.p'}")

#         with open(self.ATTACK_SETS + "test.p", "wb") as f:
#             for inputs, targets, members in self.attack_test_loader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 output, prediction = self._get_data(self.target_model, inputs)
#                 # output = output.cpu().detach().numpy()
            
#                 pickle.dump((output, prediction, members), f)

#         self.logger.add_line(f"Finished Saving Test Dataset to {self.ATTACK_SETS + 'test.p'}")

#     def train(self, epoch, result_path):
#         self.attack_model.train()
#         batch_idx = 1
#         train_loss = 0
#         correct = 0
#         total = 0

#         final_train_gndtrth = []
#         final_train_predict = []
#         final_train_probabe = []

#         final_result = []

#         with open(self.ATTACK_SETS + "train.p", "rb") as f:
#             while(True):
#                 try:
#                     output, prediction, members = pickle.load(f)
#                     output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

#                     results = self.attack_model(output, prediction)
#                     results = F.softmax(results, dim=1)

#                     losses = self.criterion(results, members)
#                     losses.backward()
#                     self.optimizer.step()

#                     train_loss += losses.item()
#                     _, predicted = results.max(1)
#                     total += members.size(0)
#                     correct += predicted.eq(members).sum().item()

#                     if epoch:
#                         final_train_gndtrth.append(members)
#                         final_train_predict.append(predicted)
#                         final_train_probabe.append(results[:, 1])

#                     batch_idx += 1
#                 except EOFError:
#                     break

#         if epoch:
#             final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
#             final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
#             final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

#             train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
#             train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

#             final_result.append(train_f1_score)
#             final_result.append(train_roc_auc_score)

#             with open(result_path, "wb") as f:
#                 pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
#             self.logger.add_line("Saved Attack Train Ground Truth and Predict Sets")
#             self.logger.add_line("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

#         final_result.append(1.*correct/total)
#         self.logger.add_line( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

#         return final_result

#     def test(self, epoch, result_path):
#         self.attack_model.eval()
#         batch_idx = 1
#         correct = 0
#         total = 0

#         final_test_gndtrth = []
#         final_test_predict = []
#         final_test_probabe = []

#         final_result = []

#         with torch.no_grad():
#             with open(self.ATTACK_SETS + "test.p", "rb") as f:
#                 while(True):
#                     try:
#                         output, prediction, members = pickle.load(f)
#                         output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

#                         results = self.attack_model(output, prediction)
#                         _, predicted = results.max(1)
#                         total += members.size(0)
#                         correct += predicted.eq(members).sum().item()
#                         results = F.softmax(results, dim=1)

#                         if epoch:
#                             final_test_gndtrth.append(members)
#                             final_test_predict.append(predicted)
#                             final_test_probabe.append(results[:, 1])

#                         batch_idx += 1
#                     except EOFError:
#                         break

#         if epoch:
#             final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
#             final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
#             final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

#             test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
#             test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

#             final_result.append(test_f1_score)
#             final_result.append(test_roc_auc_score)

#             with open(result_path, "wb") as f:
#                 pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

#             self.logger.add_line("Saved Attack Test Ground Truth and Predict Sets")
#             self.logger.add_line("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

#         final_result.append(1.*correct/total)
#         self.logger.add_line( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

#         return final_result

#     def delete_pickle(self):
#         train_file = glob.glob(self.ATTACK_SETS +"train.p")
#         for trf in train_file:
#             os.remove(trf)

#         test_file = glob.glob(self.ATTACK_SETS +"test.p")
#         for tef in test_file:
#             os.remove(tef)

#     def saveModel(self, path):
#         torch.save(self.attack_model.state_dict(), path)

# class attack_for_top3(attack_for_blackbox):
#     def __init__(
#         self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, 
#         target_model, shadow_model, attack_model, device, logger
#     ):
#         super(attack_for_top3, self).__init__(SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, 
#         target_model, shadow_model, attack_model, device, logger)
        
#     def _get_data(self, model, inputs):
#         result = model(inputs)
        
#         output, _ = torch.sort(result, descending=True)
#         output_top3 = F.softmax(output, dim=1)[:,:3]
#         _, predicts = result.max(1)
        
#         prediction = []
#         for predict in predicts:
#             prediction.append([1,] if predict else [0,])

#         prediction = torch.Tensor(prediction)

#         # final_inputs = torch.cat((results, prediction), 1)
#         # print(final_inputs.shape)

#         return output_top3, prediction
        
# class attack_for_whitebox():
#     def __init__(
#         self, TARGET_PATH, SHADOW_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, 
#         target_model, shadow_model, attack_model, device, class_num, logger
#     ):
#         self.device = device
#         self.class_num = class_num
#         self.logger = logger

#         self.ATTACK_SETS = ATTACK_SETS

#         self.TARGET_PATH = TARGET_PATH
#         self.target_model = target_model.to(self.device)
#         print(f"Load target from {self.TARGET_PATH}")
#         self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
#         self.target_model.eval()


#         self.SHADOW_PATH = SHADOW_PATH
#         self.shadow_model = shadow_model.to(self.device)
#         print(f"Load target from {self.SHADOW_PATH}")
#         self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
#         self.shadow_model.eval()

#         self.attack_train_loader = attack_train_loader
#         self.attack_test_loader = attack_test_loader

#         self.attack_model = attack_model.to(self.device)
#         torch.manual_seed(0)
#         self.attack_model.apply(weights_init)

#         self.target_criterion = nn.CrossEntropyLoss(reduction='none')
#         self.attack_criterion = nn.CrossEntropyLoss()
#         #self.optimizer = optim.SGD(self.attack_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
#         self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)

#         self.attack_train_data = None
#         self.attack_test_data = None
        

#     def _get_data(self, model, inputs, targets):
#         results = model(inputs)
#         # outputs = F.softmax(outputs, dim=1)
#         losses = self.target_criterion(results, targets)

#         gradients = []
        
#         for loss in losses:
#             loss.backward(retain_graph=True)

#             gradient_list = reversed(list(model.named_parameters()))

#             for name, parameter in gradient_list:
#                 if 'weight' in name:
#                     gradient = parameter.grad.clone() # [column[:, None], row].resize_(100,100)
#                     gradient = gradient.unsqueeze_(0)
#                     gradients.append(gradient.unsqueeze_(0))
#                     break

#         labels = []
#         for num in targets:
#             label = [0 for i in range(self.class_num)]
#             label[num.item()] = 1
#             labels.append(label)

#         gradients = torch.cat(gradients, dim=0)
#         losses = losses.unsqueeze_(1).detach()
#         outputs, _ = torch.sort(results, descending=True)
#         labels = torch.Tensor(labels)

#         return outputs, losses, gradients, labels

#     def prepare_dataset(self):
#         with open(self.ATTACK_SETS + "train.p", "wb") as f:
#             for inputs, targets, members in self.attack_train_loader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 output, loss, gradient, label = self._get_data(self.shadow_model, inputs, targets)

#                 pickle.dump((output, loss, gradient, label, members), f)

#         self.logger.add_line(f"Finished Saving Train Dataset to {self.ATTACK_SETS + 'train.p'}")

#         with open(self.ATTACK_SETS + "test.p", "wb") as f:
#             for inputs, targets, members in self.attack_test_loader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 output, loss, gradient, label = self._get_data(self.target_model, inputs, targets)
            
#                 pickle.dump((output, loss, gradient, label, members), f)

#             # pickle.dump((output, loss, gradient, label, members), open(self.ATTACK_PATH + "test.p", "wb"))

#         self.logger.add_line(f"Finished Saving Test Dataset to {self.ATTACK_SETS + 'test.p'}")

    
#     def train(self, epoch, result_path):
#         self.attack_model.train()
#         batch_idx = 1
#         train_loss = 0
#         correct = 0
#         total = 0

#         final_train_gndtrth = []
#         final_train_predict = []
#         final_train_probabe = []

#         final_result = []

#         with open(self.ATTACK_SETS + "train.p", "rb") as f:
#             while(True):
#                 try:
#                     output, loss, gradient, label, members = pickle.load(f)
#                     output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

#                     results = self.attack_model(output, loss, gradient, label)
#                     # results = F.softmax(results, dim=1)
#                     losses = self.attack_criterion(results, members)
                    
#                     losses.backward()
#                     self.optimizer.step()

#                     train_loss += losses.item()
#                     _, predicted = results.max(1)
#                     total += members.size(0)
#                     correct += predicted.eq(members).sum().item()

#                     if epoch:
#                         final_train_gndtrth.append(members)
#                         final_train_predict.append(predicted)
#                         final_train_probabe.append(results[:, 1])

#                     batch_idx += 1
#                 except EOFError:
#                     break	

#         if epoch:
#             final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
#             final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
#             final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

#             train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
#             train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

#             final_result.append(train_f1_score)
#             final_result.append(train_roc_auc_score)

#             with open(result_path, "wb") as f:
#                 pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
#             self.logger.add_line("Saved Attack Train Ground Truth and Predict Sets")
#             self.logger.add_line("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

#         final_result.append(1.*correct/total)
#         self.logger.add_line( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

#         return final_result


#     def test(self, epoch, result_path):
#         self.attack_model.eval()
#         batch_idx = 1
#         correct = 0
#         total = 0

#         final_test_gndtrth = []
#         final_test_predict = []
#         final_test_probabe = []

#         final_result = []

#         with torch.no_grad():
#             with open(self.ATTACK_SETS + "test.p", "rb") as f:
#                 while(True):
#                     try:
#                         output, loss, gradient, label, members = pickle.load(f)
#                         output, loss, gradient, label, members = output.to(self.device), loss.to(self.device), gradient.to(self.device), label.to(self.device), members.to(self.device)

#                         results = self.attack_model(output, loss, gradient, label)

#                         _, predicted = results.max(1)
#                         total += members.size(0)
#                         correct += predicted.eq(members).sum().item()

#                         results = F.softmax(results, dim=1)

#                         if epoch:
#                             final_test_gndtrth.append(members)
#                             final_test_predict.append(predicted)
#                             final_test_probabe.append(results[:, 1])

#                         batch_idx += 1
#                     except EOFError:
#                         break

#         if epoch:
#             final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
#             final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
#             final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

#             test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
#             test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

#             final_result.append(test_f1_score)
#             final_result.append(test_roc_auc_score)


#             with open(result_path, "wb") as f:
#                 pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

#             self.logger.add_line("Saved Attack Test Ground Truth and Predict Sets")
#             self.logger.add_line("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))

#         final_result.append(1.*correct/total)
#         self.logger.add_line( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

#         return final_result

#     def delete_pickle(self):
#         train_file = glob.glob(self.ATTACK_SETS +"train.p")
#         for trf in train_file:
#             os.remove(trf)

#         test_file = glob.glob(self.ATTACK_SETS +"test.p")
#         for tef in test_file:
#             os.remove(tef)

#     def saveModel(self, path):
#         torch.save(self.attack_model.state_dict(), path)


# def train_shadow_model(
#     PATH, device, shadow_model, train_loader, test_loader, 
#     batch_size, loss, optimizer, logger, args
# ):
#     epochs = args.shadow_epochs
#     model = shadow(
#         train_loader, test_loader, shadow_model, device, 
#         batch_size, loss, optimizer, logger, epochs
#     )
#     acc_train = 0
#     acc_test = 0

#     for i in range(epochs):
#         logger.add_line("<======================= Epoch " + str(i+1) + " =======================>")
#         logger.add_line("shadow training")

#         acc_train = model.train()
#         logger.add_line("shadow testing")
#         acc_test = model.test()


#         overfitting = round(acc_train - acc_test, 6)

#         logger.add_line('The overfitting rate is %s' % overfitting)
#         if gol.get_value("debug"):
#             break

#     # FILE_PATH = PATH + "shadow.pth"
#     FILE_PATH = os.path.join(PATH, "shadow.pth")
#     model.saveModel(FILE_PATH)
#     logger.add_line(f"saved shadow model to {FILE_PATH}!!!")
#     logger.add_line("Finished training!!!")

#     return acc_train, acc_test, overfitting


# def get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, batch_size):
#     mem_train, nonmem_train, mem_test, nonmem_test = list(shadow_train), list(shadow_test), list(target_train), list(target_test)

#     for i in range(len(mem_train)):
#         mem_train[i] = mem_train[i] + (1,)
#     for i in range(len(nonmem_train)):
#         nonmem_train[i] = nonmem_train[i] + (0,)
#     for i in range(len(nonmem_test)):
#         nonmem_test[i] = nonmem_test[i] + (0,)
#     for i in range(len(mem_test)):
#         mem_test[i] = mem_test[i] + (1,)


#     train_length = min(len(mem_train), len(nonmem_train))
#     test_length = min(len(mem_test), len(nonmem_test))

#     mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
#     non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
#     mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
#     non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    
#     attack_train = mem_train + non_mem_train
#     attack_test = mem_test + non_mem_test

#     attack_trainloader = torch.utils.data.DataLoader(
#         attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
#     attack_testloader = torch.utils.data.DataLoader(
#         attack_test, batch_size=batch_size, shuffle=True, num_workers=2)

#     return attack_trainloader, attack_testloader

# def attack_top3(
#     TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, 
#     target_model, shadow_model, attack_model, get_attack_set, num_classes
# ):
#     MODELS_PATH = ATTACK_PATH + "meminf_top3.pth"
#     RESULT_PATH = ATTACK_PATH + "meminf_top3.p"
#     ATTACK_SETS = ATTACK_PATH + "meminf_attack_top3_"
#     logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=ATTACK_PATH)

#     epochs = 50
#     attack = attack_for_top3(
#         SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, 
#         target_model, shadow_model, attack_model, device, logger
#     )

#     if get_attack_set:
#         attack.delete_pickle()
#         attack.prepare_dataset()

#     for i in range(epochs):
#         flag = 1 if i == epochs-1 else 0
#         logger.add_line("Epoch %d :" % (i+1))
#         res_train = attack.train(flag, RESULT_PATH)
#         res_test = attack.test(flag, RESULT_PATH)
#         if gol.get_value("debug"):
#             break

#     attack.saveModel(MODELS_PATH)
#     logger.add_line(f"Saved Attack Model to {MODELS_PATH}")
#     print(f"{sys._getframe().f_code.co_name} finished")

#     return res_train, res_test

# def attack_mode0(
#     TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, 
#     target_model, shadow_model, attack_model, get_attack_set, num_classes
# ):
#     MODELS_PATH = osp.join(ATTACK_PATH, "meminf_attack0.pth")
#     RESULT_PATH = osp.join(ATTACK_PATH, "meminf_attack0.p")
#     ATTACK_SETS = osp.join(ATTACK_PATH, "meminf_attack_mode0_")
#     logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=ATTACK_PATH)

#     epochs = 50
#     attack = attack_for_blackbox(
#         SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, 
#         target_model, shadow_model, attack_model, device, logger
#     )

#     if get_attack_set:
#         attack.delete_pickle()
#         attack.prepare_dataset()

#     for i in range(epochs):
#         flag = 1 if i == epochs-1 else 0
#         logger.add_line("Epoch %d :" % (i+1))
#         res_train = attack.train(flag, RESULT_PATH)
#         res_test = attack.test(flag, RESULT_PATH)
#         if gol.get_value("debug"):
#             break

#     attack.saveModel(MODELS_PATH)
#     logger.add_line(f"Saved Attack Model to {MODELS_PATH}")
#     print(f"{sys._getframe().f_code.co_name} finished")

#     return res_train, res_test


# def attack_mode3(
#     TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, 
#     target_model, shadow_model, attack_model, get_attack_set, num_classes
# ):
#     MODELS_PATH = ATTACK_PATH + "meminf_attack3.pth"
#     RESULT_PATH = ATTACK_PATH + "meminf_attack3.p"
#     ATTACK_SETS = ATTACK_PATH + "meminf_attack_mode3_"
#     logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=ATTACK_PATH)

#     attack = attack_for_whitebox(
#         TARGET_PATH, SHADOW_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, 
#         target_model, shadow_model, attack_model, device, num_classes, logger
#     )
    
#     if get_attack_set:
#         attack.delete_pickle()
#         attack.prepare_dataset()

#     for i in range(50):
#         flag = 1 if i == 49 else 0
#         logger.add_line("Epoch %d :" % (i+1))
#         res_train = attack.train(flag, RESULT_PATH)
#         res_test = attack.test(flag, RESULT_PATH)

#     attack.saveModel(MODELS_PATH)
#     logger.add_line("Saved Attack Model")

#     return res_train, res_test

# def get_gradient_size(model):
#     gradient_size = []
#     gradient_list = reversed(list(model.named_parameters()))
#     for name, parameter in gradient_list:
#         if 'weight' in name:
#             gradient_size.append(parameter.shape)

#     return gradient_size