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
from sklearn.metrics import roc_auc_score

def tensor_bound(inp):
    return torch.clamp(inp, min=0, max=1)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_t0 = time()
    losses = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze(1)
        loss = F.binary_cross_entropy(tensor_bound(output), y, reduction='mean')
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
    return np.mean(losses)

def validate(model, device, val_loader):
    model.eval()
    opt_thlds, accs = [], []
    for batch_idx, data in enumerate(val_loader):
        data = data.to(device)
        output = model(data.x, data.edge_index, data.edge_attr)
        y, output = data.y, output.squeeze()
        loss = F.binary_cross_entropy(tensor_bound(output), y, reduction='mean').item()
        
        # define optimal threshold (thld) where TPR = TNR 
        diff, opt_thld, opt_acc = 100, 0, 0
        best_tpr, best_tnr = 0, 0
        for thld in np.arange(0.15, 0.5, 0.001):
            TP = torch.sum((y==1) & (output>thld)).item()
            TN = torch.sum((y==0) & (output<thld)).item()
            FP = torch.sum((y==0) & (output>thld)).item()
            FN = torch.sum((y==1) & (output<thld)).item()
            #print(thld, TP, TN, FP, FN)
            acc = (TP+TN)/(TP+TN+FP+FN)
            #TPR, TNR = TP/(TP+FN), TN/(TN+FP)
            #TPR=TP/(TP+FN)
            TPR=0
            if (TP+FN) > 0:
                TPR = TP/(TP+FN)
            #TNR=TN/(TN+FP)
            TNR=0
            if (TN+FP) > 0:
                TNR=TN/(TN+FP)
            delta = abs(TPR-TNR)
            if (delta < diff): 
                diff, opt_thld, opt_acc = delta, thld, acc

        opt_thlds.append(opt_thld)
        accs.append(opt_acc)

    print("...val accuracy=", np.mean(accs))
    return np.mean(opt_thlds) 

def test(model, device, test_loader, thld=0.5):
    model.eval()
    losses, accs = [], []
    true = torch.Tensor()
    out_thresh = torch.Tensor()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr)
            out_thld = output > thld
            out_thresh = torch.cat((out_thresh, out_thld))
            true = torch.cat((true, data.y))
            #print(data.y, out_thld)
            #auc = roc_auc_score(data.y, out_thld)
            TP = torch.sum((data.y==1).squeeze() & 
                           (output>thld).squeeze()).item()
            TN = torch.sum((data.y==0).squeeze() & 
                           (output<thld).squeeze()).item()
            FP = torch.sum((data.y==0).squeeze() & 
                           (output>thld).squeeze()).item()
            FN = torch.sum((data.y==1).squeeze() & 
                           (output<thld).squeeze()).item()            
            acc = (TP+TN)/(TP+TN+FP+FN)
            loss = F.binary_cross_entropy(tensor_bound(output.squeeze(1)), data.y, 
                                          reduction='mean').item()
            accs.append(acc)
            losses.append(loss)
            #print(f"acc={TP+TN}/{TP+TN+FP+FN}={acc}")

    #print(true)
    #print(out_thresh)
    auc = roc_auc_score(true, out_thresh)
    print('...test loss: {:.4f}\n...test accuracy: {:.4f}'
          .format(np.mean(losses), np.mean(accs)))
    print("test auc: ", auc)
    return np.mean(losses), np.mean(accs)

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
    parser.add_argument('--sample', type=int, default=1, 
                        help='TrackML train_{} sample to train on')
    parser.add_argument('--hidden-size', type=int, default=40,
                        help='Number of hidden units per layer')

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
        
    home_dir = "../"
    indir = "{}/hitgraphs_{}/{}_{}/".format(home_dir, args.sample, args.construction, args.pt)
    
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
    
    model = InteractionNetwork(args.hidden_size).to(device)
    total_trainable_params = sum(p.numel() for p in model.parameters())
    print('total trainable params:', total_trainable_params)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size,
                       gamma=args.gamma)


    output = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    for epoch in range(1, args.epochs + 1):
        print("---- Epoch {} ----".format(epoch))
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        thld = validate(model, device, val_loader)
        print('...optimal threshold', thld)
        test_loss, test_acc = test(model, device, test_loader, thld=thld)
        scheduler.step()
        
        if args.save_model:
            torch.save(model.state_dict(),
                       "trained_models/train{}_PyG_{}_epoch{}_{}GeV_redo.pt"
                       .format(args.sample, args.construction, epoch, args.pt))

        output['train_loss'].append(train_loss)
        output['test_loss'].append(test_loss)
        output['test_acc'].append(test_acc)
    
        np.save('train_output/train{}_PyG_{}_{}GeV_redo'
                .format(args.sample, args.construction, args.pt),
                output)


if __name__ == '__main__':
    main()



