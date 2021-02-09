from __future__ import print_function
import os
from time import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
from models.dataset import Dataset
from models.interaction_network import InteractionNetwork


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    for batch_idx, (data, target) in enumerate(train_loader):
        X, Ra = data['X'].to(device), data['Ra'].to(device)
        Ri, Ro = data['Ri'].to(device), data['Ro'].to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(X, Ra.float(), Ri.float(), Ro.float())
        loss = F.binary_cross_entropy(output.squeeze(2), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(X), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    print("epoch time: {0}s".format(time()-epoch_t0))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            X, Ra = data['X'].to(device), data['Ra'].to(device)
            Ri, Ro = data['Ri'].to(device), data['Ro'].to(device)
            target = target.to(device)
            output = model(X, Ra.float(), Ri.float(), Ro.float())
            N_correct = torch.sum((target==1).squeeze() & (output>0.5).squeeze())
            N_correct += torch.sum((target==0).squeeze() & (output<0.5).squeeze())
            N_total = target.shape[1]
            print(N_correct.item(), '/', N_total)
            accuracy += torch.sum(((target==1).squeeze() & 
                                   (output>0.5).squeeze()) |
                                  ((target==0).squeeze() & 
                                   (output<0.5).squeeze())).float()/target.shape[1]
            test_loss += F.binary_cross_entropy(output.squeeze(2), target, 
                                                reduction='mean').item() 

    test_loss /= len(test_loader.dataset)
    accuracy /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n, Accuracy: {:.4f}\n'
          .format(test_loss, accuracy))


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Interaction Network Implementation')
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
    parser.add_argument('--pt', type=str, default='2',
                        help='Cutoff pt value in GeV (default: 2)')
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

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda={0}".format(use_cuda))

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    graph_indir = "/scratch/gpfs/jdezoort/hitgraphs/IN_LP_ExaTrkX_{}/".format(args.pt)
    #graph_indir = "/tigress/jdezoort/IN_samples_endcaps/IN_LP_{}/".format(args.pt)
    graph_files = np.array(os.listdir(graph_indir))
    n_graphs = len(graph_files)

    IDs = np.arange(n_graphs)
    np.random.shuffle(IDs)
    partition = {'train': graph_files[IDs[:800]],  
                 'test':  graph_files[IDs[800:]]}
    
    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 6}
    train_set = Dataset(graph_indir, partition['train']) 
    train_loader = torch.utils.data.DataLoader(train_set, **params)
    test_set = Dataset(graph_indir, partition['test'])
    test_loader = torch.utils.data.DataLoader(test_set, **params)

    model = InteractionNetwork(3, 4, 4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        #scheduler.step()
    
        if args.save_model:
            torch.save(model.state_dict(), "IN_150_100_100_noSched_{}.pt".format(args.pt))


if __name__ == '__main__':
    main()
