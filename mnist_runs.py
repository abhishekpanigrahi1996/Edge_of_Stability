from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
import numpy as np
import sys
sys.path.insert(0, 'autograd-hacks')
import autograd_hacks
import random
import os 
from tqdm import tqdm

seed = 0 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.autograd.set_detect_anomaly(True)


#current implementation only computes the parameters w.r.t. the first batch!
def topeigen_compute(model, criterion, train_loader, device):
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
        
    alignment = alignment.item() /  top_eigenvalue    
    model.zero_grad()
    return v, top_eigenvalue, alignment    



#current implementation only computes the parameters w.r.t. the first batch!    
def trace_compute(model, criterion, train_loader, device):
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
    
    
    
def hessian_compute(model, criterion, train_loader, device, epoch, continuous_time, wandb_bool, wandb, additional_string=''):
    
        
    trace = trace_compute(model, criterion, train_loader, device)    
    v, top_eigenvalue, alignment = topeigen_compute(model, criterion, train_loader, device)

    if wandb_bool:   
        wandb.log({'hessian trace'+additional_string: trace, 'hessian top eigen'+additional_string: top_eigenvalue, 'alignment': alignment, 'continuous time': continuous_time, 'epoch': epoch})
    else:     
        print("The top Hessian eigenvalue" +additional_string + " of this model is %.4f"%top_eigenvalue)
        print("Trace of hessian" +additional_string + "  is %.4f"%trace)
 
        


def jacobian(model, criterion, train_loader, device, epoch, continuous_time, wandb_bool, wandb, additional_string=''):
    jacobian_frob_norm = 0.
    
    for batch_idx, (data, target) in enumerate(train_loader):
        num_data = len(data)
        model.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
 
        for i in range(num_data):    
            gradients = torch.autograd.grad(output[i], model.parameters(), retain_graph=True)
            
            
            for g in gradients:
                jacobian_frob_norm += torch.linalg.norm(g).item() ** 2
    
    jacobian_frob_norm = jacobian_frob_norm ** 0.5
    if wandb_bool:   
        wandb.log({'Frobenius norm Jacobian'+additional_string: jacobian_frob_norm, 'continuous time': continuous_time, 'epoch': epoch})
    else:     
        print("Frobenius norm of jacobian" +additional_string + " of this model is %.4f"%jacobian_frob_norm)

   
def regularizer_update(model, criterion, train_loader, device, epoch, continuous_time, wandb_bool, wandb, projection_epochs, projection_lr, lr, log_sharpness=1):
    v, top_eigenvalue, _ = topeigen_compute(model, criterion, train_loader, device)
    top_eigenvector = [p.detach() for p in v]
    

    autograd_hacks.clear_backprops(model)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        num_data = len(data)
        model.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)#F.mse_loss(torch.squeeze(output), 2. * target.to(torch.float32) - 1.)
        loss.backward(create_graph=True)
        
        
        
        autograd_hacks.compute_grad1(model)
        individual_grads = [[param.grad1[i] for param in model.parameters()] for i in range(len(data))]
        

        
        
        
        params = [p for p in model.parameters()]
        gradsH = [p.grad for p in model.parameters()]
 
        first_product = sum([torch.sum(x * y) for (x, y) in zip(gradsH, top_eigenvector)])
        
        second_grad = torch.autograd.grad(first_product, params, create_graph=True)
        
        second_product = sum([torch.sum(x * y) for (x, y) in zip(second_grad, top_eigenvector)])
        
        third_grad = torch.autograd.grad(second_product, params, create_graph=True)
        
        break
    
    
    with torch.no_grad():
        for (param, grad_vector) in zip(model.parameters(), third_grad):
            if log_sharpness == 1:
                param -= lr * grad_vector / top_eigenvalue
            else:
                param -= lr * grad_vector 

            
    model.zero_grad()       
    
    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        train_loss += loss.item() * len(data)
    train_loss /= len(train_loader.dataset) 
    if wandb_bool: 
        wandb.log({'train loss before GD projection': train_loss, 'continuous time': continuous_time, 'epoch': epoch})
    else:    
        print("Train loss before GD projection is %.4f"%train_loss)
    
  
    #Now again do GD to project back to the manifold
    for _ in tqdm(range(projection_epochs)):

        model.zero_grad()    
        train_loss = 0.
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            train_loss += loss.item() * len(data)
            
        with torch.no_grad():
            for param in model.parameters():
                param -= projection_lr * param.grad
       
  
    train_loss /= len(train_loader.dataset) 
    if wandb_bool: 
        wandb.log({'train loss': train_loss, 'Hess train loss': train_loss, 'continuous time': continuous_time, 'epoch': epoch})
    else:    
        print("Train loss is %.4f"%train_loss)
    
    
    
class Net(nn.Module):
    def __init__(self, activation):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation = activation
        
    def forward(self, x):
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'relusquared':
            x = F.relu(x) ** 2
        elif self.activation == 'quadratic':
            x = x ** 2    
        elif self.activation == 'tanh':
            x = F.tanh(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        
        x = self.fc2(x)
        output = x
        return output


def train(args, model, device, criterion, hess_criterion, train_loader, lr, epoch, continuous_time, optimizer, algo, wandb_bool, wandb):
    model.train()
    model.zero_grad()
    
    correct = 0.
    
    train_loss = 0.
    hess_train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        prec1 = accuracy(output.data, target)[0]
        correct += prec1 * len(data)
        loss.backward()
        train_loss += loss.item() * len(data)
        hess_train_loss += hess_criterion(output, target).item() * len(data)
        

    if algo == 'normalizedgd':    
        total_norm = 0.
        for param in model.parameters():
            total_norm += torch.linalg.norm(param.grad).item() ** 2
        total_norm = total_norm ** 0.5

        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad/total_norm 

            for param in model.parameters():
                param.grad.zero_()
    elif criterion == 'sqrtmse' and algo == 'GD':
        denom = 2 * ( train_loss / len(train_loader.dataset) ) ** 0.5
         
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad / denom

            for param in model.parameters():
                param.grad.zero_()   
    
    elif algo == 'rmsprop' or algo == 'GD':
        optimizer.step()
        optimizer.zero_grad()
    
    train_loss /= len(train_loader.dataset) 
    hess_train_loss /= len(train_loader.dataset) 
    

    
    if wandb_bool: 
        wandb.log({'train loss': train_loss, 'Hess train loss': hess_train_loss, 'Train Accuracy':  correct / len(train_loader.dataset), 'continuous time': continuous_time, 'epoch': epoch})
    else:    
        print("Train loss is %.4f"%train_loss)

         
def test(model, device, criterion, hess_criterion, test_loader, epoch, continuous_time, wandb_bool, wandb):
    model.eval()
    test_loss = 0
    hess_test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * len(data)
            hess_test_loss += hess_criterion(output, target).item() * len(data)
            
            prec1 = accuracy(output.data, target)[0]
            correct += prec1 * len(data)
            
    test_loss /= len(test_loader.dataset)
    hess_test_loss /= len(test_loader.dataset)

    if wandb_bool: 
        wandb.log( {'test loss': test_loss, 'Hess test loss': hess_test_loss, 'Accuracy':  correct / len(test_loader.dataset), 'continuous time': continuous_time, 'epoch': epoch } )
    else:    
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def MSEloss(output, target):
    rel_output = output[range(len(output)), target]
    return torch.mean( (rel_output - 1.) ** 2 + torch.sum(output ** 2, axis=-1) - rel_output ** 2 ).cuda()

def SqrtMSEloss(output, target):
    rel_output = output[range(len(output)), target]
    return torch.mean( (rel_output - 1.) ** 2 + torch.sum(output ** 2, axis=-1) - rel_output ** 2 ).cuda() ** 0.5




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--wandb', type=int, default=1,
                        help='For Saving the current Model')
    parser.add_argument('--activation', type=str, default='quadratic',
                        help='Choices: relu/quadratic/relusquared/tanh')
    parser.add_argument('--algo', type=str, default='normalizedgd',
                        help='Choices: normalizedgd/regularizer_minimize')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='Alpha for rmsprop')
    parser.add_argument('--max_training_examples', type=int, default=-1,
                        help='Number of training examples to consider')
    parser.add_argument('--projection_epochs', type=int, default=1000,
                        help='Number of training examples to consider')
    parser.add_argument('--projection_lr', type=float, default=0.01,
                        help='Number of training examples to consider')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume checkpoint')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='Frequency of saving checkpoints')
    parser.add_argument('--criterion', type=str, default='mse',
                        help='mse/sqrtmse')
    parser.add_argument('--hess_freq', type=int, default=100,
                        help='freq for hessian computation')
    parser.add_argument('--log_sharpness', type=int, default=1,
                        help='Regularizer update w.r.t. log sharpness?')
    parser.add_argument('--start', type=int, default=0,
                        help='Whether to start right from the start')
    parser.add_argument('--wandb_project', type=str, default=0,
                        help='Where to store the results on weights and bias')
    parser.add_argument('--save_dir', type=str, default=0,
                        help='Where to store the checkpoints')
    args = parser.parse_args()
    
    
    config = {}     
    config['lr'] = args.lr
    config['activation'] = args.activation
    config['epochs'] = args.epochs
    config['algo'] = args.algo
    config['max_training_examples'] = args.max_training_examples
    config['criterion'] = args.criterion
    config['log_sharpness'] = args.log_sharpness

    if args.wandb:
        wandb.init(project=args.wandb_project, config=config, name='MNIST ' + ' '.join([str(key)+ ' ' + str(config[key]) for key in config.keys() ]), settings=wandb.Settings(start_method='thread') )

    if args.criterion == 'mse':
        criterion = hess_criterion = MSEloss
    elif args.criterion == 'sqrtmse':
        criterion = MSEloss
        hess_criterion = MSEloss
    
    activation = args.activation
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    #hessian_kwargs = {'batch_size': 10000}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        #hessian_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    train_dataset = datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transform)
    test_dataset  = datasets.MNIST('mnist_data', train=False,
                       transform=transform)
    
    
    #idx = (dataset1.targets==0) | (dataset1.targets==1)
    #dataset1.targets = dataset1.targets[idx]
    #dataset1.data = dataset1.data[idx]
    
    if args.max_training_examples != -1:
        if os.path.exists('data/MNISTsubset_indices_'+str(args.max_training_examples)+'.pt'):
            subset_indices = torch.load('data/MNISTsubset_indices_'+str(args.max_training_examples)+'.pt')
        else:    
            subset_indices = torch.randperm(len(train_dataset))[:args.max_training_examples]
            torch.save(subset_indices, 'data/MNISTsubset_indices_'+str(args.max_training_examples)+'.pt')
        trainset_sub = torch.utils.data.Subset(train_dataset, subset_indices)
    else:
        trainset_sub = train_dataset
        

    train_loader = torch.utils.data.DataLoader(trainset_sub,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    
  
    model = Net(activation).to(device)
    if args.algo == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=args.alpha)
    elif args.algo == 'GD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)    
    else:
        optimizer = None
        
        
    if args.resume != '':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            
            model.load_state_dict(checkpoint['state_dict'])
            if args.algo  != 'normalizedgd' and args.algo  != 'regularizer_minimize':
                optimizer.load_state_dict(checkpoint['optim_state_dict'])
            
            print ('Loaded model from ', str(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            exit(0)    
        
    autograd_hacks.add_hooks(model)    

    
    if (args.start == 1 and (args.algo == 'normalizedgd' or args.criterion == 'sqrtmse' or args.algo  == 'regularizer_minimize')) or (args.algo == 'GD' and args.criterion == 'mse'):
        args.start_epoch = 0
    
    
    
    
    for epoch in tqdm(range(args.start_epoch, 1+args.epochs)):
        if args.algo  == 'regularizer_minimize':
            continuous_time = epoch * args.lr
        elif args.algo  == 'normalizedgd' :
            continuous_time = epoch * args.lr **2 / 4.
        elif args.criterion == 'sqrtmse':
            continuous_time = epoch * args.lr **2 / 8.
        else:
            continuous_time = epoch
     
            
        if args.algo == 'regularizer_minimize':   
            regularizer_update(model, criterion, train_loader, device, epoch, continuous_time, args.wandb, wandb, args.projection_epochs, args.projection_lr, args.lr, args.log_sharpness)
        else:    
            train(args, model, device, criterion, hess_criterion, train_loader, args.lr, epoch, continuous_time, optimizer, args.algo, args.wandb, wandb)
        test(model, device, criterion, hess_criterion, test_loader, epoch, continuous_time, args.wandb, wandb)

        if epoch % args.hess_freq == 0:
            hessian_compute(model, hess_criterion, train_loader, device, epoch, continuous_time, args.wandb, wandb)

        if (epoch - 1) % args.save_freq == 0 and args.save_model:
            
            if not os.path.exists(args.save_dir):
                os.mkdir(args.save_dir)
            
            if args.algo  == 'normalizedgd' or args.algo  == 'regularizer_minimize':
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optim_state_dict': None,
                }, filename=os.path.join(args.save_dir, 'checkpoint-'+str(epoch)+'.th'))
            else:    
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optim_state_dict': optimizer.state_dict(),
                }, filename=os.path.join(args.save_dir, 'checkpoint-'+str(epoch)+'.th'))
    
       

            
            
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



def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
    
    


if __name__ == '__main__':
    main()