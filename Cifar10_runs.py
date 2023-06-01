import argparse
import os
import shutil
import time
import random

import warnings

import torch


import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import vgg


import wandb
from utils import My_BatchSampler

import numpy as np
  



model_names_vgg = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

print(model_names_vgg)


parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices= model_names_vgg , 
                    help='model architecture: ' +  ' | '.join(model_names_vgg) +
                    ' (default: vgg_manual)')

parser.add_argument('--vgg_cfg', default='B', type=str, help='Configuration of VGG (A/AV/B/BV/D/DV/E/EV) (useful only for arch=vgg_manual)')
parser.add_argument('--bn', default=1, type=int, help='Whether to use Batch norm (useful only for arch=vgg_manual)')
parser.add_argument('--act', default='gelu', type=str, help='Activation (useful only for arch=vgg_manual)')


parser.add_argument('--wandb_project', default='ImplicitBias', type=str, help='Wandb project to add to')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--rho', default=0.9, type=float,
                    metavar='rho', help='RMSprop rho')
parser.add_argument('--epsilon', default=1e-8, type=float,
                    metavar='epsilon', help='RMSprop epsilon')
parser.add_argument('--augment', default=1, type=int,
                    metavar='augment', help='Whether to use data augmentation')

parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='opt', help='sgd/normalizedgd')


parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
#parser.add_argument('--width', default=1, type=int,
#                    metavar='W', help='multi-neurons (default: 1)')
parser.add_argument('--dropout', default=0, type=int,
                    metavar='dropout', help='Use dropout (0/1).')

parser.add_argument('--looper', default=1, type=int,
                    metavar='looper', help='loop over to perform large batch training.')

parser.add_argument('--data_subset', default=-1, type=int,
                    metavar='subset', help='Number of training data points to use.')

#parser.add_argument('--train_type', default=1, type=int,
#                    metavar='batchGD', help='Whether to do full batch descent (0) or stochastic batch descent (1).')


parser.add_argument('--num_variance_measurements', default=1000, type=int,
                    metavar='numvarmeasure', help='Num of samples to compute variance.')

parser.add_argument('--loss_type', default='xent', type=str,
                    metavar='losstype', help='mse/xent')

parser.add_argument('--drop_last', default='1', type=int,
                    metavar='drop_last', help='0/1 (drop last batch)')

parser.add_argument('--sample_mode', default='with_replacement', type=str,
                    metavar='sample_mode', help='random_shuffling/without_replacement/with_replacement/fixed_sequence/two_without_replacement')

parser.add_argument('--compute_top_eigenval', default=0, type=int,
                    metavar='topeigval', help='0/1')




parser.add_argument('--final_layer_init_scale', default=1., type=float,
                    metavar='final_layer_init_scale', help='Scaling factor for final layer (rel for vgg)')


parser.add_argument('--train_final_layer', default=1, type=int,
                    metavar='train_final_layer', help='Whether to freeze/train the final layer (rel for vgg)')


parser.add_argument('--compute_alignment', default=0, type=int,
                    metavar='compute_alignment', help='Whether to compute alignment with top eigenvector')

parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--batch_grad_loader', default=1, type=int,
                    help='Batch size of variance gradient and mean gradient computation during training. ')
parser.add_argument('--looper_grad_loader', default=1, type=int,
                    help='Batch size multiplier of variance gradient and mean gradient computation during training. ')


best_prec1 = 0


def topeigenalign(model, train_loader, criterion, device='cuda', gd_lr=0.001, num_steps=1000):
    v = [p.clone() for p in model.parameters()]
    
    for step in range(num_steps):
        model.zero_grad()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
        print (step, loss.float())    
        with torch.no_grad():
            for p in model.parameters(): 
                p -= gd_lr * p.grad
     
    curr_v = [p.clone() for p in model.parameters()]
    diff = [x - y for (x, y) in zip(v, curr_v)]
    eigenvector, eigenvalue, eigengap = topeigen_compute(model, train_loader, criterion)
    alignment_score = 0.
    norm_p = 0.
    for (x,y) in zip(diff, eigenvector):
        alignment_score += torch.sum(x * y).float()
        norm_p += torch.sum(x * x).float()
    alignment_score = alignment_score / (norm_p ** 0.5) 
    
    
    
    
    
    
    #return back to the original
    with torch.no_grad():
        lister = 0
        for p in model.parameters(): 
            p = v[lister]
            lister += 1
    
    return alignment_score, eigenvalue, eigengap
    
 
def eigengap_compute(model, train_loader, criterion, top_eigenvector, top_eigenvalue, device='cuda', num_measurements=1):
    
    
    loops = 0
    model.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        num_data = len(data)
        data, target = data.to(device), target.to(device)
        output = model(data)
        if loops == 0:
            loss = criterion(output, target)
        else:
            loss += criterion(output, target)
        loops += 1
        
        if loops >= num_measurements:
            break
        
    loss = loss / loops
    gradients = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=True)
    
    num_iterations = 100
    v = [torch.randn_like(p, device=device) for p in model.parameters()]
    norm_v = 0.
    for g in v:
        norm_v += torch.linalg.norm(g).item() ** 2
    for g in v:
        g /= (norm_v ** 0.5)
    
    for iter in range(num_iterations):
        gv = sum([torch.sum(x * y) for (x, y) in zip(gradients, v)])
        Hv = torch.autograd.grad(gv, [p for p in model.parameters() if p.requires_grad], retain_graph=True)
        
        dot_p = top_eigenvalue * sum([torch.sum(x * y) for (x, y) in zip(top_eigenvector, v)])
        for (p, q) in zip(Hv, top_eigenvector):
            p -= dot_p * q
        
        second_eigenvalue = 0.
        for p in Hv:
            second_eigenvalue += torch.linalg.norm(p).item() ** 2
            
        second_eigenvalue = second_eigenvalue ** 0.5     
        v = [p/second_eigenvalue for p in Hv]

    model.zero_grad() 
    return top_eigenvalue - second_eigenvalue
    
#current implementation only computes the parameters w.r.t. the first batch!
def topeigen_compute(model, train_loader, criterion, device='cuda', num_measurements=1):
    loops = 0
    model.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        num_data = len(data)
        data, target = data.to(device), target.to(device)
        output = model(data)
        if loops == 0:
            loss = criterion(output, target)
        else:
            loss += criterion(output, target)
        loops += 1
        
        if loops >= num_measurements:
            break
        
    loss = loss / loops
    gradients = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], create_graph=True)
    alignment = 0.
    
    
    num_iterations = 100
    v = [g.clone().detach() for g in gradients]
    
    
    norm_v = 0.
    for g in v:
        norm_v += torch.linalg.norm(g).item() ** 2
    for g in v:
        g /= (norm_v ** 0.5)
    
    for iter in range(num_iterations):   
        
        gv = sum([torch.sum(x * y) for (x, y) in zip(gradients, v)])
        Hv = torch.autograd.grad(gv, [p for p in model.parameters() if p.requires_grad], retain_graph=True)
        
        if iter == 0:
            alignment = sum([torch.sum(x * y) for (x, y) in zip(v, Hv)])
        
        top_eigenvalue = 0.
        for p in Hv:
            top_eigenvalue += torch.linalg.norm(p).item() ** 2
        top_eigenvalue = top_eigenvalue ** 0.5     
        v = [p/top_eigenvalue for p in Hv]
        
        
        

    model.zero_grad() 
    
    eigengap = eigengap_compute(model, train_loader, criterion, v, top_eigenvalue, num_measurements=num_measurements)
    alignment = alignment.item() /  top_eigenvalue
    
    return v, top_eigenvalue, eigengap, alignment
    
    

#current implementation only computes the parameters w.r.t. the first batch!    
def trace_compute(model, train_loader, criterion, device='cuda', num_measurements=1):
    loops = 0
    model.zero_grad()
    for batch_idx, (data, target) in enumerate(train_loader):
        num_data = len(data)
        data, target = data.to(device), target.to(device)
        output = model(data)
        if loops == 0:
            loss = criterion(output, target)
        else:
            loss += criterion(output, target)
        loops += 1
        if loops >= num_measurements:
            break
        
    loss = loss / loops
    gradients = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        
        
    trace = 0.
    num_vectors = 100
    #random 100 vectors
    for iter in range(num_vectors):
        v = [torch.randn_like(p, device=device) for p in model.parameters()]
        gv = sum([torch.sum(x * y) for (x, y) in zip(gradients, v)])
        Hv = torch.autograd.grad(gv, model.parameters(), retain_graph=True)
        trace += sum([torch.sum(x * y) for (x, y) in zip(Hv, v)]).item()
    trace /= num_vectors

    model.zero_grad()
    return trace
    
 

def MSEloss(output, target):
    rel_output = output[range(len(output)), target]
    return torch.mean( (rel_output - 1.) ** 2 + torch.sum(output ** 2, axis=-1) - rel_output ** 2 ).cuda()

def SqrtMSEloss(output, target):
    rel_output = output[range(len(output)), target]
    return torch.mean( (rel_output - 1.) ** 2 + torch.sum(output ** 2, axis=-1) - rel_output ** 2 ).cuda() ** 0.5


def main():
    global args, best_prec1
    args = parser.parse_args()

    # 1. Start a new run
    

    config = {}
    config['batch_size'] = args.batch_size * args.looper
    
    if args.arch != 'vgg_manual':
        config['arch'] = args.arch
    else:
        config['arch'] = args.arch + '_' + args.vgg_cfg + '_' + str(args.bn) + '_' + str(args.act)
        
   
    config['effective lr'] = args.lr   
    config['optimizer'] = args.optimizer

    
    config['weight_decay'] = args.weight_decay
    config['num_data_points'] = args.data_subset
    config['sample_mode'] = args.sample_mode
    config['augment'] = args.augment
    config['loss_type'] = args.loss_type
    config['batch_grad_loader'] = args.batch_grad_loader
    config['looper_grad_loader'] = args.looper_grad_loader
    config['manual seed'] = args.seed
    
    
    config['epochs'] = args.epochs 

    
    config['final_layer_init_scale'] = args.final_layer_init_scale
    config['train_final_layer'] = args.train_final_layer
  
    config['dropout'] = args.dropout
    config['data_augment'] = 0
    

   
    wandb.init(project=args.wandb_project, entity='inductive-bias', config=config, name='Model ' + ' '.join([str(key)+ ' ' + str(config[key]) for key in config.keys() ]), settings=wandb.Settings(start_method='fork') )

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if args.arch.startswith("resnet"):    
        model = torch.nn.DataParallel(resnet.__dict__[args.arch]())#args.width,dropout=args.dropout))
    elif args.arch.startswith("vgg"):  
        if args.arch != 'vgg_manual':
            model = torch.nn.DataParallel(vgg.__dict__[args.arch](num_classes=10, init_scale=args.final_layer_init_scale, train_final_layer=args.train_final_layer ))#args.width))  
        else:
            model = torch.nn.DataParallel(vgg.__dict__[args.arch](configuration=args.vgg_cfg, batch_norm=bool(args.bn), activation='relu', num_classes=10, init_scale=args.final_layer_init_scale, train_final_layer=args.train_final_layer ))
   
    
    model.cuda()
    wandb.watch(model)
    total_updates = 0
    
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(0)

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    if args.augment == 1:
        trainset=datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    else:
        trainset=datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]), download=True)
    
    if args.data_subset != -1:
        if os.path.exists('data/subset_indices_'+str(args.data_subset)+'.pt'):
            subset_indices = torch.load('data/subset_indices_'+str(args.data_subset)+'.pt')
        else:    
            subset_indices = torch.randperm(len(trainset))[:args.data_subset]
            torch.save(subset_indices, 'data/subset_indices_'+str(args.data_subset)+'.pt')
        trainset_sub = torch.utils.data.Subset(trainset, subset_indices)
    else:
        trainset_sub = trainset


    dataset_size=args.data_subset
    if dataset_size == -1:
        dataset_size=50000
    train_loader = torch.utils.data.DataLoader(
        trainset_sub,
        batch_sampler=My_BatchSampler(dataset_size=dataset_size, batch_size=args.batch_size, drop_last=args.drop_last, sample_mode = args.sample_mode), 
        num_workers=args.workers, 
        pin_memory=True
    )
    
    
  
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'xent':
        criterion = nn.CrossEntropyLoss().cuda()
    elif args.loss_type == 'mse':
        criterion = MSEloss
    elif args.loss_type == 'sqrtmse':
        criterion = SqrtMSEloss    
        
    if args.half:
        model.half()
        criterion.half()

    
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
       
        
        # evaluate on validation set, every 100 steps of update
        if epoch % 1 == 0 or epoch == args.start_epoch:
            prec1 = validate(val_loader, model, criterion, epoch)

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        total_updates = train(train_loader, model, criterion, optimizer, epoch, args.optimizer, args.looper, total_updates)
            
        
        
        #compute top eigenvalue of hessian every 1000 steps of update
        if epoch % min(500, args.epochs // 20) == 0 and args.compute_top_eigenval == 1:
            if args.loss_type == 'sqrtmse':
                hess_criterion = MSEloss
            else:
                hess_criterion = criterion
            num_meas = dataset_size // args.batch_size
            _, topeigenval, eigengap, alignment  = topeigen_compute(model, train_loader, hess_criterion, device='cuda', num_measurements=num_meas)
            _, topeigenval_crit, _, _ = topeigen_compute(model, train_loader, criterion, device='cuda', num_measurements=num_meas)
            wandb.log({'Topeigenvalue': topeigenval, 'Topeigenvalue_criterion': topeigenval_crit,  'eigengap': eigengap, "epoch": epoch, "alignment": alignment})

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        
        wandb.log({'Gradient_steps': (epoch + 1) * ( dataset_size//args.batch_size ) , 'epoch': epoch})
        
        
def train(train_loader, model, criterion, optimizer, epoch, opt, looper=1, total_updates=0):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    

    optimizer.zero_grad()        
    count_looper = 1

    
    for i, (input, target) in enumerate(train_loader):
            
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var) 

        train_loss = loss / looper
        # compute gradient and do SGD step
        if count_looper == 1:
            optimizer.zero_grad()
        train_loss.backward()
        
        if count_looper == looper:
            grad_norm = 0.
            for p in model.parameters():
                if p.requires_grad:
                    grad_norm += torch.linalg.norm(p.grad).float() ** 2
            
            if opt == 'normalizedgd':
                with torch.no_grad():
                    paramgrad_norm = 0.
                    for p in model.parameters():
                        if p.requires_grad:
                            paramgrad_norm += torch.linalg.norm(p.grad).float() ** 2
                    for p in model.parameters():
                        if p.requires_grad:
                            p.grad /= (paramgrad_norm ** 0.5)
            
            optimizer.step()
        output = output.float()
        if count_looper == 1:
            loss_ = loss.float()
        else:
            loss_ += loss.float()
            
        if count_looper == 1:    
            prec1 = accuracy(output.data, target)[0]
        else:
            prec1 += accuracy(output.data, target)[0]
        
        if count_looper == looper:
            losses.update(loss_.item() / looper, looper * input.size(0))
            top1.update(prec1.item() / looper, looper * input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

        if count_looper == looper:
            count_looper = 1
        else:
            count_looper += 1
        

    
    param_norm = 0.
    for p in model.parameters():
        param_norm += torch.linalg.norm(p).float() ** 2
    
    if np.isnan(losses.avg):
        #Game over! Stop.
        print ('Reached Nan!')
        exit(0)
    
    wandb.log({'Param norm': param_norm ** 0.5, "epoch": epoch})
    wandb.log({'Batch Grad norm': grad_norm ** 0.5, "epoch": epoch})
    wandb.log({'train loss': losses.avg, "epoch": epoch})
    wandb.log({'train prec': top1.avg, "epoch": epoch})

    return  total_updates
    

      
            
def validate(val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    
    wandb.log({'test loss': losses.avg, "epoch": epoch})
    wandb.log({'test prec': top1.avg, "epoch": epoch})

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
        

        
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
