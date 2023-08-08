import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from advertorch.attacks import LinfPGDAttack

def eval_model(model, testset, device):
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            nclasses = outputs.size(1)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    return acc


def eval_surrogate_fidelity(surrogate, victim, testset, device):
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    surrogate.eval()
    victim.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            surrogate_outputs = surrogate(inputs)
            _, surrogate_predicted = surrogate_outputs.max(1)
            victim_outputs = victim(inputs)
            _, victim_predicted = victim_outputs.max(1)
            total += targets.size(0)
            correct += surrogate_predicted.eq(victim_predicted).sum().item()

    fidelity = 100. * correct / total
    return fidelity


def advtest_output(model, loader, adversary):
    model.eval()

    total_ce = 0
    total = 0
    top1 = 0

    total = 0
    top1_clean = 0
    top1_adv = 0
    adv_success = 0
    adv_trial = 0
    for i, (batch, label) in enumerate(loader):
        batch, label = batch.to('cuda'), label.to('cuda')
        total += batch.size(0)
        out_clean = model(batch)
        _, pred_clean = out_clean.max(dim=1)
        
        advbatch = adversary.perturb(batch, pred_clean)

        out_adv = model(advbatch)
        _, pred_adv = out_adv.max(dim=1)
        

        clean_correct = pred_clean.eq(label)
        adv_trial += int(clean_correct.sum().item())
        adv_success += int(pred_adv[clean_correct].eq(label[clean_correct]).sum().detach().item())
        top1_clean += int(pred_clean.eq(label).sum().detach().item())
        top1_adv += int(pred_adv.eq(label).sum().detach().item())

        print('{}/{}...'.format(i+1, len(loader)))
        if i > 10:
            break
    return float(top1_clean)/total*100, float(top1_adv)/total*100, float(adv_trial-adv_success) / adv_trial *100


def eval_adversarial_transfer(surrogate, victim, testset, device):
    test_loader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    surrogate.eval()
    victim.eval()

    adversary = LinfPGDAttack(
        surrogate, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.2,
        nb_iter=20, eps_iter=0.02, 
        rand_init=True, clip_min=-2.42, clip_max=2.75,
        targeted=False)
    clean_top1, adv_top1, adv_sr = advtest_output(victim, test_loader, adversary)
    print(f"clean top1 {clean_top1}, adv top1 {adv_top1}, adv sr {adv_sr}")
    return clean_top1, adv_top1, adv_sr

def eval_surrogate_model(surrogate, victim, testset, device):
    surrogate_acc = eval_model(surrogate, testset, device)
    surrogate_fidelity = eval_surrogate_fidelity(surrogate, victim, testset, device)
    clean_top1, adv_top1, adv_sr = eval_adversarial_transfer(surrogate, victim, testset, device)

    return surrogate_acc, surrogate_fidelity, (clean_top1, adv_top1, adv_sr)