import os
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from .gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np
import random

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.utils.logger import logger, configure, dump_params

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def set_logger(args):
    time_tag = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    file_name = f"obg_{args.gnn}_{args.dataset}_{time_tag}"
    dump_dir = os.path.expanduser(os.path.join(args.filename, file_name))
    format_strings = ["stdout", "log", "csv"]
    configure(logger, dump_dir, format_strings)
    dump_params(logger.get_dir(), vars(args))

def set_global_seed(args):
    seed = args.seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available and args.device >= 0:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed(seed)

def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            pred = model(batch)
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type: 
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()

def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)

def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument("--runs", type=int, default=5, help="number of runs")
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="./results",
                        help='filename to output result (default: )')

    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--taiji", action="store_true", help="debug")
    args = parser.parse_args()
    return args

def main(args, run_id=0):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    if args.taiji:
        args.dataset_path = "/apdcephfs/share_1150325/ztjiaweixu-graph/dataset/ogb"
        args.filename = f"/apdcephfs/share_1150325/ztjiaweixu-graph/dumps/ogb/{args.dataset}"
    else:
        args.dataset_path =  "~/.ogb/dataset"
        args.filename = f"{args.filename}/{args.dataset}"
    args.seed += run_id
    set_global_seed(args)
    if run_id == 0:
        set_logger(args)

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name=args.dataset, root=args.dataset_path)

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-vanilla':
        model = GNN(gnn_type = 'gcn-vanilla', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        logger.record("Run", run_id)
        logger.record("Epoch", epoch)
        logger.record("Train", train_perf[dataset.eval_metric])
        logger.record("Validation", valid_perf[dataset.eval_metric])
        logger.record("Test", test_perf[dataset.eval_metric])
        logger.dump(epoch)

        train_curve.append(train_perf[dataset.eval_metric])
        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    logger.info(f'Finished training run {run_id}!')
    logger.info(
        f'Best val epoch: {best_val_epoch}, \
        Val Score: {valid_curve[best_val_epoch]}, \
        Test Score: {test_curve[best_val_epoch]}, \
        Train Score: {train_curve[best_val_epoch]}'
    )


    # if not args.filename == '':
    #     torch.save({'Val': valid_curve[best_val_epoch], 'Test': test_curve[best_val_epoch], 'Train': train_curve[best_val_epoch], 'BestTrain': best_train}, args.filename)

    return test_curve[best_val_epoch]

if __name__ == "__main__":
    args = get_args()
    all_runs = []
    for i in range(args.runs):
        all_runs.append(main(args, i))
        mean_acc = np.mean(all_runs)
        std_acc = np.std(all_runs)
        acc_str = ", ".join([f"{acc :.4f}" for acc in all_runs])
        logger.info(f"All runs: [{acc_str}]")
        logger.info(f"Average Test Accuracy: {mean_acc :.4f}(mean) {std_acc: .4f}(std)")
