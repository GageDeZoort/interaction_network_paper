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
        #target = target['y'].to(device)
        target = target.to(device)
        #print('train graph: X.shape={}, Ra.shape={}, Ri.shape={}, Ro.shape={}, y.shape={}'
        #      .format(X.shape, Ra.shape, Ri.shape, Ro.shape, target.shape))
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

def validate(model, device, val_loader):
    model.eval()
    best_discs = []
    for data, target in val_loader:
        X, Ra = data['X'].to(device), data['Ra'].to(device)
        Ri, Ro = data['Ri'].to(device), data['Ro'].to(device)
        #target = target['y'].to(device)
        target = target.to(device)
        output = model(X, Ra.float(), Ri.float(), Ro.float())
        N_correct = torch.sum((target==1).squeeze() & (output>0.5).squeeze())
        N_correct += torch.sum((target==0).squeeze() & (output<0.5).squeeze())
        N_total = target.shape[1]
        
        diff, best_disc = 100, 0
        best_tpr, best_tnr = 0, 0
        for disc in np.arange(0.2, 0.8, 0.01):
            true_pos = ((target==1).squeeze() & (output>disc).squeeze())
            true_neg = ((target==0).squeeze() & (output<disc).squeeze())
            false_pos = ((target==0).squeeze() & (output>disc).squeeze())
            false_neg = ((target==1).squeeze() & (output<disc).squeeze())
            N_tp, N_tn = torch.sum(true_pos).item(), torch.sum(true_neg).item()
            N_fp, N_fn = torch.sum(false_pos).item(), torch.sum(false_neg).item()
            true_pos_rate = N_tp/(N_tp + N_fn)
            true_neg_rate = N_tn/(N_tn + N_fp)
            delta = abs(true_pos_rate - true_neg_rate)
            if (delta < diff):
                diff, best_disc = delta, disc 
        best_discs.append(best_disc)
    
    #print("best_tpr", best_tpr, "\nbest_tnr", best_tnr)
    #print("diff=", diff, "\nbest_disc=", best_disc)
    print("val accuracy=", (N_tp+N_tn)/(N_tp+N_tn+N_fp+N_fn))
    return np.mean(best_discs)
   
def test(acc, model, device, test_loader, disc=0.5):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            X, Ra = data['X'].to(device), data['Ra'].to(device)
            Ri, Ro = data['Ri'].to(device), data['Ro'].to(device)
            #target = target['y'].to(device)
            target = target.to(device)
            t0 = time()
            output = model(X, Ra.float(), Ri.float(), Ro.float())
            acc = time() - t0
            N_correct = torch.sum((target==1).squeeze() & (output>0.5).squeeze())
            N_correct += torch.sum((target==0).squeeze() & (output<0.5).squeeze())
            N_total = target.shape[1]
            #print(N_correct.item(), '/', N_total)
            accuracy += torch.sum(((target==1).squeeze() & 
                                   (output>disc).squeeze()) |
                                  ((target==0).squeeze() & 
                                   (output<disc).squeeze())).float()/target.shape[1]
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
    parser.add_argument('--construction', type=str, default='heptrkx_classic',
                        help='graph construction method')

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

    graph_indir = "/interactionnetworkvol/interaction_network_paper/hitgraphs/{}_{}/".format(args.construction, args.pt)
    #graph_indir = "/tigress/jdezoort/IN_samples_endcaps/IN_LP_{}/".format(args.pt)
    graph_files = np.array(os.listdir(graph_indir))
    n_graphs = len(graph_files)

    IDs = np.arange(n_graphs)
    np.random.shuffle(IDs)
    partition = {'train': graph_files[IDs[:1]],  
                 'test':  graph_files[IDs[:100]],
                 'val': graph_files[IDs[:1]]}

    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 0}#6}
    train_set = Dataset(graph_indir, partition['train']) 
    train_loader = torch.utils.data.DataLoader(train_set, **params)
    test_set = Dataset(graph_indir, partition['test'])
    test_loader = torch.utils.data.DataLoader(test_set, **params)
    val_set = Dataset(graph_indir, partition['val'])
    val_loader = torch.utils.data.DataLoader(val_set, **params)

    model = InteractionNetwork(3, 4, 4).to(device)
    #model = InteractionNetwork(3, 1, 1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("total params", total_params)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)

    time_acc = 0
    time_avg = 0
    times = []

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        disc = validate(model, device, val_loader)
        print('optimal discriminant', disc)
        test(time_acc, model, device, test_loader, disc=disc)
        times.append(time_acc)
        scheduler.step()
    
        if args.save_model:
            torch.save(model.state_dict(), "{}_epoch{}_{}GeV.pt".format(args.construction,
                                                                        epoch, args.pt))

    for i in times:
        time_avg = time_avg + i

    time_avg = time_avg / len(times)
    timings = open("cpu_timing.txt", "a")
    timings.write("{0}s \n".format(time_avg))
    timings.close()

if __name__ == '__main__':
    main()
