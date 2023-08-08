import sys, os
import os.path as osp
from pdb import set_trace as st
sys.path.append(os.path.abspath('.')) 
import time
import datetime
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
import teacher_resnet 
import student_resnet
import dataloader as dataloaders
import dataset_info
import proj_utils
from argparse import Namespace
from pdb import set_trace as st
import numpy as np
import copy

from knockoff.nettailor import teacher_alexnet
from knockoff.nettailor import student_alexnet

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', metavar='MODEL_DIR',
                    help='model directory')
parser.add_argument('--task', metavar='TASK',
                    help='task to train')
parser.add_argument('--teacher-fn', metavar='TEACHER_CHECKPOINT',
                    help='Teacher checkpoint filename')
parser.add_argument('--backbone', metavar='BACKBONE', default='resnet34',
                    help='backbone model architecture: ' + ' (default: resnet34)')
parser.add_argument('--max-skip', default=3, type=int)
parser.add_argument('--complexity-coeff', default=1.0, type=float)
parser.add_argument('--teacher-coeff', default=10.0, type=float)

parser.add_argument('--epochs', default=50, type=int, 
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--batch-size', default=32, type=int,
                    metavar='BS', help='batch size')
parser.add_argument('--lr', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay-epochs', default=20, type=int, 
                    metavar='LR_EPOCHS', help='number of epochs for each lr decay')
parser.add_argument('--momentum', default=0.9, type=float, 
                    metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--full-model-dir', metavar='MODEL_DIR', default='',)
parser.add_argument('--n-pruning-universal', metavar='THR', default=0, type=float)
parser.add_argument('--thr-pruning-proxy', metavar='THR', default=0.05, type=float)


parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', default=5, type=int,
                    metavar='N', help='eval frequency (default: 5)')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--log2file', action='store_true',
                    help='log output to file (under model_dir/train.log)')

parser.add_argument('--backbone-pretrain-dir', default=None)
parser.add_argument('--tolerance', default=0.03, type=float)
parser.add_argument('--start-prune-thr', default=0.5, type=float)
parser.add_argument('--start-prune-percent', default=0.3, type=float)
parser.add_argument('--iterative-prune-ratio', default=0.1, type=float)
parser.add_argument('--prune-interval', default=3, type=int)

parser.add_argument('--selected_data_dir', type=str)
parser.add_argument('--proxy_data_ratio', default=5, type=int)

args = parser.parse_args()
DEVICE = torch.device("cuda:0")
proj_utils.prep_output_folder(args.model_dir, args.evaluate)

def init_prune_proxies_by_thred(model, num_proxies=None, thr_proxies=None, percent_proxies=0.3,):
    alphas_proxies, alphas_layer, keep = [], [], []
    for layer in model.layers:
        aa = layer.alphas().data.cpu().numpy()
        alphas_layer.append(aa[0])
        alphas_proxies.append(aa[1:])
        keep.append([1.]*aa.size)

    # Remove task-specific layers
    meta_proxies = [(i, j) for i in range(len(alphas_proxies)) for j in range(alphas_proxies[i].size)]
    alphas_proxies = np.concatenate(alphas_proxies)
    to_rm = []

    # if num_proxies is not None:
    #     to_rm.extend(np.argsort(alphas_proxies)[:num_proxies].tolist())
    # elif percent_proxies is not None:
    #     to_rm.extend(np.argsort(alphas_proxies)[:int(len(alphas_proxies)*percent_proxies)].tolist())
    
    if thr_proxies is not None:
        if thr_proxies > max(alphas_proxies):
            to_rm.extend(np.argsort(alphas_proxies)[:int(len(alphas_proxies)*0.8)].tolist())
        else:
            to_rm.extend([i for i, aa in enumerate(alphas_proxies) if aa <= thr_proxies])

    if len(to_rm) < len(alphas_proxies) * percent_proxies:
        to_rm.extend(np.argsort(alphas_proxies)[:int(len(alphas_proxies)*percent_proxies)].tolist())

    # prune by perc
    # for i in range(len(model.layers)):
    #     adp_a = [aa for ii, (mm, aa) in enumerate(zip(meta_proxies, alphas_proxies)) if mm[0]==i]
    #     adp_i = [ii for ii, (mm, aa) in enumerate(zip(meta_proxies, alphas_proxies)) if mm[0]==i]
    #     if max(adp_a) == 0:
    #         to_rm.extend(adp_i)
    #     else:
    #         to_rm.extend([ii for ii, aa in zip(adp_i, adp_a) if aa/max(adp_a)<init_prune_ratio])

    for rm_idx in to_rm:
        layer_idx = meta_proxies[rm_idx][0]
        proxies_idx = meta_proxies[rm_idx][1]+1
        keep[layer_idx][proxies_idx] = 0.

    # Update keep variables
    for layer, k in zip(model.layers, keep):
        layer.keep_flag = k[:]
    return model

def iterative_prune_proxies(model, iterative_ratio):
    alphas_proxies, alphas_layer, keep = [], [], []
    for layer in model.layers:
        aa = layer.alphas().data.cpu().numpy()
        alphas_layer.append(aa[0])
        alphas_proxies.append(aa[1:])
        keep.append(layer.keep_flag)
    
    # Remove task-specific layers
    meta_proxies = [(i, j) for i in range(len(alphas_proxies)) for j in range(alphas_proxies[i].size)]
    to_rm = []

    num_remove = int(len(np.concatenate(alphas_proxies)) * iterative_ratio)
    if num_remove == 0:
        num_remove = 1
    valid_alphas = []
    for i in range(len(alphas_proxies)):
        for j in range(alphas_proxies[i].size):
            if keep[i][j+1]:
                valid_alphas.append(alphas_proxies[i][j])
    if len(valid_alphas) == 1:
        return model
    thred = np.sort(valid_alphas)[num_remove]
    for i in range(len(alphas_proxies)):
        for j in range(alphas_proxies[i].size):
            if keep[i][j+1] and alphas_proxies[i][j] < thred:
                keep[i][j+1] = 0

    print(f"+++++ num_total {len(np.concatenate(alphas_proxies))}, ratio {iterative_ratio}, num_remove {num_remove}, thred {thred}, valid alphas {valid_alphas}")
    # Update keep variables
    for layer, k in zip(model.layers, keep):
        layer.keep_flag = k[:]
    return model


def main():
    mode = 'train' if not args.evaluate else 'eval'
    logger = proj_utils.Logger(args.log2file, mode=mode, model_dir=args.model_dir)

    # Args
    logger.add_line(str(datetime.datetime.now()))
    logger.add_line("="*30+"   Arguments   "+"="*30)
    for k in args.__dict__:
        logger.add_line(' {:30}: {}'.format(k, str(args.__dict__[k])))

    # Data
    if mode == 'train':

        train_dataloader = dataloaders.get_dataloader(
            dataset=args.task, 
            batch_size=args.batch_size, 
            shuffle=True, 
            mode=mode, 
            num_workers=args.workers)
        logger.add_line("\n"+"="*30+"   Original data   "+"="*30)
        logger.add_line(str(train_dataloader.dataset))
        val_loader = dataloaders.get_dataloader(
            dataset=args.task,
            batch_size=args.batch_size, 
            shuffle=True, 
            mode='eval', 
            num_workers=args.workers)
        num_classes = val_loader.dataset.num_classes
        logger.add_line("\n"+"="*30+"   Validation data   "+"="*30)
        logger.add_line(str(val_loader.dataset))

    elif mode == 'eval':
        test_loader = dataloaders.get_dataloader(
            dataset=args.task, 
            batch_size=args.batch_size, 
            shuffle=False, 
            mode=mode, 
            num_workers=args.workers)
        num_classes = test_loader.dataset.num_classes
        logger.add_line("\n"+"="*30+"   Test data   "+"="*30)
        logger.add_line(str(test_loader.dataset))

    img_size = dataset_info.IMGSIZE_DICT[args.task]
    # Student model
    if args.backbone.startswith('resnet'):
        model = student_resnet.create_model(
            num_classes=num_classes, 
            max_skip=args.max_skip,
            backbone=args.backbone,
            pretrain_path=args.backbone_pretrain_dir,
        )
    elif args.backbone.startswith('alexnet'):
        model = student_alexnet.create_model(
            num_classes=num_classes, 
            max_skip=args.max_skip,
            backbone=args.backbone,
            pretrain_path=args.backbone_pretrain_dir,
        )
    elif args.backbone.startswith('wide_resnet'):
        model = student_resnet.create_model(
            num_classes=num_classes, 
            max_skip=args.max_skip,
            backbone=args.backbone
        )
    
    proj_utils.load_checkpoint
    universal_params = get_backbone_tensors(model)

    logger.add_line("="*30+"   Model   "+"="*30)
    logger.add_line(str(model))
    logger.add_line("="*30+"   Parameters   "+"="*30)
    logger.add_line(proj_utils.parameter_description(model))

    # Teacher model
    if args.teacher_fn is not None:
        if args.backbone.startswith('resnet'):
            teacher = teacher_resnet.create_teacher(args.backbone, pretrained=False, num_classes=num_classes, img_size=img_size)
        elif args.backbone.startswith('alexnet'):
            teacher = teacher_alexnet.create_teacher(args.backbone, pretrained=False, num_classes=num_classes, img_size=img_size)
        elif args.backbone.startswith('wide_resnet'):
            teacher = teacher_resnet_wide.create_teacher(args.backbone, pretrained=True, num_classes=num_classes)
        teacher.freeze()
        logger.add_line("\n"+"="*30+"   Teacher   "+"="*30)
        logger.add_line("Loading pretrained teacher from: " + args.teacher_fn)
        proj_utils.load_checkpoint(teacher, model_fn=args.teacher_fn)
        # teacher.load_pretrained(args.teacher_fn)
        teacher = teacher.to(DEVICE)
        teacher.eval()
    else:
        teacher = None
        

    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Resume from a checkpoint
    if mode == 'eval':
        logger.add_line("\n"+"="*30+"   Checkpoint   "+"="*30)
        logger.add_line("Loading checkpoint from: " + args.model_dir)
        proj_utils.load_checkpoint(model, model_dir=args.model_dir)
    if mode == 'train':
        print(f"Load student model from args.full_model_dir")
        proj_utils.load_checkpoint(model, model_dir=args.full_model_dir)

    model.to(DEVICE)

    model.get_task_param_flop()

    ############################ TRAIN #########################################
    if mode == 'train':
        
        ############################ Init Prune Model #########################################
        baseline_err, baseline_acc, _ = validate(val_loader, model, teacher, nn.CrossEntropyLoss(), logger)
        tolerance_acc = baseline_acc * (1 - args.tolerance)
        logger.add_line("\n" + "="*30+"   Accuracy Tolerance   "+"="*30)
        logger.add_line("\n" + f"baseline acc {baseline_acc}, tolerant acc {tolerance_acc}\n")

        logger.add_line("\n" + "="*30+"   Before Init Prune Model Stats   "+"="*30)
        logger.add_line(model.stats())
        # Layer pruning
        assert len(args.full_model_dir) > 0
        # if len(args.full_model_dir) > 0:
        # model.threshold_alphas(num_global=int(args.n_pruning_universal), thr_proxies=args.thr_pruning_proxy)
        model = init_prune_proxies_by_thred(
            model, thr_proxies=args.thr_pruning_proxy, percent_proxies=args.start_prune_percent
        )
        logger.add_line("\n" + "="*30+"   Init Prune Model Stats   "+"="*30)
        logger.add_line(model.stats())
        last_tolerable_state_dict, last_tolerable_acc, last_tolerable_keep_flag = None, -1, None

        # Optimizer
        parameters = [
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'proxies' in n], 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'alphas_params' in n], 'lr': args.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'ends_bn' in n], 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        ]
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs)
        best_acc1 = 0
        del parameters

        for ii, epoch in enumerate(range(args.epochs)):
            # Train for one epoch
            logger.add_line("\n"+"="*30+"   Train (Epoch {})   ".format(epoch)+"="*30)
            
            train(train_dataloader, model, teacher, criterion, optimizer, epoch, logger, args)
                    
            # optimizer = proj_utils.adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay_epochs, logger)
            # train(train_loader, model, teacher, criterion, optimizer, epoch, logger)

            if (epoch+1) % args.prune_interval == 0:
                err, acc, _ = validate(val_loader, model, teacher, nn.CrossEntropyLoss(), logger)
                logger.add_line("\n" + "="*30+ f"   Prune Interval Eval Before Prune, acc {acc:.4f}, tolerance {tolerance_acc:.4f}, last tolerable acc {last_tolerable_acc:.4f} "+"="*30)
                logger.add_line(model.stats())

                if acc > tolerance_acc:
                    last_tolerable_state_dict = copy.deepcopy(model.state_dict())
                    last_tolerable_acc = acc
                    last_tolerable_keep_flag = copy.deepcopy(model.get_keep_flags())
                    logger.add_line("\n Iterative Prune")
                    model = iterative_prune_proxies(model, args.iterative_prune_ratio)
                    logger.add_line("\n" + "="*30+ f"   Prune Interval After Prune State "+"="*30)
                    logger.add_line(model.stats())
                # else:
                #     if tolerant_model_state_dict is not None:
                #         logger.add_line("\n restore previous state dict")
                #         model.load_state_dict(tolerant_model_state_dict)
                # model = iterative_prune_proxies(model, args.iterative_prune_ratio)

                # logger.add_line("\n" + "="*30+ f"   Model State After Prune "+"="*30)
                # logger.add_line(model.stats())



            # if (epoch+1) % args.eval_freq == args.eval_freq-1 or epoch == args.epochs-1:
            if epoch == args.epochs-1:
                # Evaluate
                model.eval()
                logger.add_line(model.stats())
                logger.add_line("\n"+"="*30+"   Valid (Epoch {})   ".format(epoch)+"="*30)
                err, acc, run_time = validate(val_loader, model, teacher, nn.CrossEntropyLoss(), logger, epoch)
                logger.add_line(f"\nAcc {acc:.4f}, original {baseline_acc:.4f}, tolerance {tolerance_acc:.4f}, last tolerable {last_tolerable_acc:.4f}")
                
                scheduler.step()
                
                if last_tolerable_acc > acc:
                    acc = last_tolerable_acc
                    save_state_dict = last_tolerable_state_dict
                    save_keep_flag = last_tolerable_keep_flag
                else:
                    save_state_dict = model.state_dict()
                    save_keep_flag = model.get_keep_flags()
                # Save checkpoint
                proj_utils.save_checkpoint(args.model_dir, {
                        'epoch': epoch + 1,
                        'state_dict': save_state_dict,
                        'keep_flags': save_keep_flag,
                        'acc': acc,
                        'xent': -1
                    }, ignore_tensors=universal_params)

                logger.add_line(model.alphas_and_complexities())


    ############################ EVAL #########################################
    elif mode == 'eval':
        logger.add_line("="*30+"   Evaluation   "+"="*30)
        err, acc, run_time = validate(test_loader, model, teacher, nn.CrossEntropyLoss(), logger)

    logger.add_line('='*30+'  COMPLETED  '+'='*30)
    logger.add_line(model.stats())
    logger.add_line('[RUN TIME] {time.avg:.3f} sec/sample'.format(time=run_time))
    logger.add_line('[FINAL] {name:<30} {loss:.7f}'.format(name=args.task+'/crossentropy', loss=err))
    logger.add_line('[FINAL] {name:<30} {acc:.7f}'.format(name=args.task+'/accuracy', acc=acc))

def kdloss(y, teacher_scores, T=4):
    # weights = weights.unsqueeze(1)
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, reduce=False)
    loss = torch.sum(l_kl) / y.shape[0]
    return loss * (T**2)

def train(dataloader, model, teacher, criterion, optimizer, epoch, logger, synthesizer_args):
    batch_time = proj_utils.AverageMeter()
    data_time = proj_utils.AverageMeter()
    loss_avg = proj_utils.AverageMeter()
    complexity_avg = proj_utils.AverageMeter()
    teacher_avg = proj_utils.AverageMeter()
    acc_avg = proj_utils.AverageMeter()

    # switch to train mode
    model.train()
    teacher.eval()

    logger.add_line('Complexity coefficient:   {}'.format(args.complexity_coeff))
    logger.add_line('Teacher coefficient:      {}'.format(args.teacher_coeff))

    l2dist = nn.MSELoss()
    end = time.time()
    # for i, (images, labels, _) in enumerate(data_loader):
    #     if images.size(0) != args.batch_size:
    #         break
    #     images, labels = images.to(DEVICE), labels.to(DEVICE)
    # for i in range(synthesizer_args.kd_steps):
    #     images, _ = synthesizer.sample()
    for i, (images, _) in enumerate(dataloader):
        # if args.gpu is not None:
        #     images = images.cuda(args.gpu, non_blocking=True)
        images = images.to(DEVICE)

        # measure data loading time
        data_time.update(time.time() - end)
        
        # Teacher supervision
        with torch.no_grad():
            teacher_logit, ends_teacher = teacher.forward_with_block_ends(images)
            teacher_label = teacher_logit.max(1)[1]

        # Forward data through student
        logit, ends_model = model(images, return_internal=True)
        # loss = criterion(logit, labels)
        # loss = criterion(logit, teacher_logit.detach())
        loss = kdloss(logit, teacher_logit.detach())
        loss_avg.update(loss.item(), images.size(0))
        acc = proj_utils.accuracy(logit, teacher_label)
        acc_avg.update(acc.item(), images.size(0))

        
        # teacher_loss = l2dist(logit, teacher_logit.detach())
        teacher_loss = 0
        for e1, e2 in zip(ends_model, ends_teacher):
            if e1 is not None:
                teacher_loss += l2dist(e1, e2.detach())
        teacher_loss /= float(len(ends_model))
        teacher_avg.update(teacher_loss.item(), images.size(0))

        # Model complexity
        complexity = model.expected_complexity()
        complexity_avg.update(complexity.item(), 1)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss = loss + args.complexity_coeff * complexity + args.teacher_coeff * teacher_loss
        total_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i+1 % 100 == 0:
            logger.add_line(
                "TRAIN [{:5}][{:5}/{:5}] | Time {:6} Data {:6} Acc {:22} Loss {:16} Complexity {:7} Teacher Sup {:7}".format(
                    str(epoch), str(i), str(synthesizer_args.kd_steps), 
                    "{t.avg:.3f}".format(t=batch_time),
                    "{t.avg:.3f}".format(t=data_time),
                    "{t.val:.3f} (Avg: {t.avg:.3f})".format(t=acc_avg),
                    "{t.val:.3f} (Avg: {t.avg:.3f})".format(t=loss_avg),
                    "{t.val:.3f}".format(t=complexity_avg),
                    "{t.val:.3f}".format(t=teacher_avg)
                ))


def validate(data_loader, model, teacher, criterion, logger, epoch=None):
    batch_time = proj_utils.AverageMeter()
    loss_avg = proj_utils.AverageMeter()
    acc_avg = proj_utils.AverageMeter()
    loss_teacher_avg = proj_utils.AverageMeter()
    acc_teacher_avg = proj_utils.AverageMeter()
    complexity_avg = proj_utils.AverageMeter()

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward data through student
            logits = model(images)
            loss = criterion(logits, labels)
            loss_avg.update(loss.item(), images.size(0))
            acc = proj_utils.accuracy(logits, labels)
            acc_avg.update(acc.item(), images.size(0))
            complexity = model.expected_complexity()
            complexity_avg.update(complexity.item(), 1)
            
            if teacher is not None:
                # Forward data through teacher
                logits= teacher(images)
                loss = criterion(logits, labels)
                loss_teacher_avg.update(loss.item(), images.size(0))
                acc = proj_utils.accuracy(logits, labels)
                acc_teacher_avg.update(acc.item(), images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end, images.size(0))
            end = time.time()
            
            if i % args.print_freq == 0 or i == len(data_loader)-1:
                logger.add_line(
                    "Test [{:5}][{:5}/{:5}] | Time {:5} | Acc {:8} XEnt {:6} Complexity {:6} | Teacher: Acc {:8} XEnt {:6} ".format(
                        str(epoch), str(i), str(len(data_loader)), 
                        "{t.avg:.3f}".format(t=batch_time),
                        "{t.avg:.3f}".format(t=acc_avg),
                        "{t.avg:.3f}".format(t=loss_avg),
                        "{t.avg:.3f}".format(t=complexity_avg),
                        "{t.avg:.3f}".format(t=acc_teacher_avg),
                        "{t.avg:.3f}".format(t=loss_teacher_avg),
                    ))

    return loss_avg.avg, acc_avg.avg, batch_time


def get_backbone_tensors(model):
    tensors = {}
    for k in model.state_dict():
        if not ('proxies' in k or 'classifier' in k or 'alphas' in k or 'running' in k or 'tracked' in k or 'ends_bn' in k):
            if k.startswith('layer'):
                k_ckp = '.'.join(k.split('.')[:2] + k.split('.')[3:])
            elif k.startswith('classifier'):
                k_ckp = 'linear.{}'.format(k.split('.')[-1])
            else:
                k_ckp = k
            tensors[k_ckp] = k
    return tensors


if __name__ == '__main__':
    main()
