import os 
import argparse
from time import time

import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch import optim
from torch_geometric.data import Data, DataLoader
from torch.optim.lr_scheduler import StepLR

from models.interaction_network import InteractionNetwork
from models.dataset import GraphDataset

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy(output.squeeze(1), data.y)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        losses.append(loss.item())
    print("...epoch time: {0}s".format(time()-epoch_t0))
    print("...epoch {}: train loss={}".format(epoch, np.mean(losses)))

def validate(model, device, val_loader):
    model.eval()
    best_discs = []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        output = model(data)
        N_correct = torch.sum((data.y==1).squeeze() & (output>0.5).squeeze())
        N_correct += torch.sum((data.y==0).squeeze() & (output<0.5).squeeze())
        N_total = data.y.shape[0]
        loss = F.binary_cross_entropy(output.squeeze(1), data.y)
        
        diff, best_disc = 100, 0
        best_tpr, best_tnr = 0, 0
        for disc in np.arange(0.2, 0.8, 0.01):
            true_pos = ((data.y==1).squeeze() & (output>disc).squeeze())
            true_neg = ((data.y==0).squeeze() & (output<disc).squeeze())
            false_pos = ((data.y==0).squeeze() & (output>disc).squeeze())
            false_neg = ((data.y==1).squeeze() & (output<disc).squeeze())
            N_tp, N_tn = torch.sum(true_pos).item(), torch.sum(true_neg).item()
            N_fp, N_fn = torch.sum(false_pos).item(), torch.sum(false_neg).item()
            true_pos_rate = N_tp/(N_tp + N_fn)
            true_neg_rate = N_tn/(N_tn + N_fp)
            delta = abs(true_pos_rate - true_neg_rate)
            if (delta < diff):
                diff, best_disc = delta, disc
        best_discs.append(best_disc)

    print("...val accuracy=", (N_tp+N_tn)/(N_tp+N_tn+N_fp+N_fn))
    return np.mean(best_discs)

def test(model, device, test_loader, disc=0.5):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            N_correct = torch.sum((data.y==1).squeeze() & (output>0.5).squeeze())
            N_correct += torch.sum((data.y==0).squeeze() & (output<0.5).squeeze())
            N_total = data.y.shape[0]
            accuracy += torch.sum(((data.y==1).squeeze() &
                                   (output>disc).squeeze()) |
                                  ((data.y==0).squeeze() &
                                   (output<disc).squeeze())).float()/data.y.shape[0]
            test_loss += F.binary_cross_entropy(output.squeeze(1), data.y,
                                                reduction='mean').item()

    test_loss /= len(test_loader)
    accuracy /= len(test_loader)
    print('...test loss: {:.4f}\n...test accuracy: {:.4f}'
          .format(test_loss, accuracy))


def main():

    # Training settings
    parser = argparse.ArgumentParser(description='PyG Interaction Network Implementation')
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
    parser.add_argument('--step-size', type=int, default=5,
                        help='Learning rate step size')
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
        

    home_dir = "/scratch/gpfs/jdezoort"
    indir = "{}/hitgraphs/{}_{}/".format(home_dir, args.construction, args.pt)
    
    graph_files = np.array(os.listdir(indir))
    graph_files = np.array([os.path.join(indir, graph_file)
                            for graph_file in graph_files])
    graph_paths = [os.path.join(indir, filename)
                   for filename in graph_files]
    n_graphs = len(graph_files)
    
    IDs = np.arange(n_graphs)
    np.random.shuffle(IDs)
    partition = {'train': graph_files[IDs[:1000]],
                 'test':  graph_files[IDs[1000:1400]],
                 'val': graph_files[IDs[1400:1500]]}
    
    params = {'batch_size': 1, 'shuffle': True, 'num_workers': 6}
    
    train_set = GraphDataset(graph_files=partition['train'])
    train_loader = DataLoader(train_set, **params)
    test_set = GraphDataset(graph_files=partition['test'])
    test_loader = DataLoader(test_set, **params)
    val_set = GraphDataset(graph_files=partition['val'])
    val_loader = DataLoader(val_set, **params)
    
    model = InteractionNetwork().to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print('total trainable params:', total_trainable_params)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size,
                       gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        print("---- Epoch {} ----".format(epoch))
        train(args, model, device, train_loader, optimizer, epoch)
        disc = validate(model, device, val_loader)
        print('...optimal discriminant', disc)
        test(model, device, test_loader, disc=disc)
        scheduler.step()
        
        if args.save_model:
            torch.save(model.state_dict(),
                       "trained_models/train1v3_40hu_{}_epoch{}_{}GeV.pt"
                       .format(args.construction, epoch, args.pt))
                

if __name__ == '__main__':
    main()



