import time
import sys, os
import os.path as osp

from torch.functional import _return_inverse
sys.path.append(os.path.abspath('.')) 
import datetime
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn.functional as F
from pdb import set_trace as st
from knockoff.nettailor import teacher_resnet 
from knockoff.nettailor import student_resnet
from knockoff.nettailor import teacher_alexnet
from knockoff.nettailor import student_alexnet
from knockoff.nettailor import dataloader as dataloaders
from knockoff.nettailor import dataset_info
from knockoff.nettailor import proj_utils
from argparse import Namespace

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
parser.add_argument('--select_data_path', type=str)
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
parser.add_argument('--generator-dir', default=None)
parser.add_argument('--proxy_data_ratio', default=5, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--imagenet_pretrain', default=False, action="store_true")

args = parser.parse_args()
DEVICE = torch.device("cuda:0")
proj_utils.prep_output_folder(args.model_dir, args.evaluate)


    
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
            mode='train', 
            num_workers=args.workers,
            val_ratio=args.val_ratio)
        logger.add_line("\n"+"="*30+"   Original data   "+"="*30)
        logger.add_line(str(train_dataloader.dataset))
        logger.add_line("Length: " + f"{len(train_dataloader.dataset)}")

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
        teacher.freeze()
        logger.add_line("\n"+"="*30+"   Teacher   "+"="*30)
        logger.add_line("Loading pretrained teacher from: " + args.teacher_fn)
        proj_utils.load_checkpoint(teacher, model_fn=args.teacher_fn)
        # teacher.load_pretrained(args.teacher_fn)
        teacher = teacher.to(DEVICE)
        teacher.eval()
    else:
        teacher = None

    if args.backbone.startswith('alexnet'):
        model.classifier.load_state_dict(teacher.fc.state_dict())


    # Resume from a checkpoint
    if mode == 'eval':
        logger.add_line("\n"+"="*30+"   Checkpoint   "+"="*30)
        logger.add_line("Loading checkpoint from: " + args.model_dir)
        proj_utils.load_checkpoint(model, model_dir=args.model_dir)

    if mode == 'train' and len(args.full_model_dir) > 0:
        proj_utils.load_checkpoint(model, model_dir=args.full_model_dir)

    model = torch.nn.DataParallel(model)
    model.to(DEVICE)
    model.stats = model.module.stats
    model.expected_complexity = model.module.expected_complexity
    model.threshold_alphas = model.module.threshold_alphas

    ############################ TRAIN #########################################
    if mode == 'train':

        # Optimizer
        parameters = [
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'proxies' in n], 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'alphas_params' in n], 'lr': args.lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'ends_bn' in n], 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n], 'lr': args.lr, 'weight_decay': args.weight_decay}
        ]
        optimizer = torch.optim.SGD(parameters, args.lr, momentum=args.momentum, weight_decay=0.0004)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs)
        best_acc = 0
        del parameters
        
        # Layer pruning
        if len(args.full_model_dir) > 0:
            model.threshold_alphas(num_global=int(args.n_pruning_universal), thr_proxies=args.thr_pruning_proxy)
        logger.add_line("\n" + "="*30+"   Model Stats   "+"="*30)
        logger.add_line(model.stats())

        for ii, epoch in enumerate(range(args.epochs)):
            # Train for one epoch
            logger.add_line("="*30+"   Train (Epoch {})   ".format(epoch)+"="*30)
            optimizer = proj_utils.adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay_epochs, logger)
            train(train_dataloader, model, teacher, None, optimizer, epoch, logger, args)

            model.eval()
            logger.add_line(model.stats())
            logger.add_line("="*30+"   Valid (Epoch {})   ".format(epoch)+"="*30)
            err, acc, run_time = validate(val_loader, model, teacher, nn.CrossEntropyLoss(), logger, epoch)
            # (acc1, ), val_loss = eval_results['Acc'], eval_results['Loss']
            logger.add_line('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Lr={lr:.4f}'
                    .format(current_epoch=epoch, acc1=acc, lr=optimizer.param_groups[0]['lr']))
            scheduler.step()
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            
            if is_best:
                # remember best err and save checkpoint
                proj_utils.save_checkpoint(
                    args.model_dir, 
                    {'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'err': err,
                     'acc': acc})
                
        

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
                logits = teacher(images)
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
