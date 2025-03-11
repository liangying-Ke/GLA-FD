import warnings
warnings.filterwarnings("ignore")

import os
import torch
import models 
import datasets
import torch.nn as nn
import sklearn.metrics as skm 
import torch.nn.functional as F

from utils import *
from tqdm import tqdm
from pytorch_model_summary import summary
import configs


def _get_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if 'cosine' in args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=args.T_max, T_mult=2, eta_min=args.eta_min)
        if 'warmup' in args.scheduler:
            scheduler = models.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler)
    elif 'ReduceLROnPlateau' in args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.factor, patience=args.patience, verbose=args.verbose)
        if 'warmup' in args.scheduler:
            scheduler = models.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

    else:
        scheduler = None
    return optimizer, scheduler

def Learning(**kwargs):
    all_preds = []
    all_targets = []
    losses = AverageMeter('Loss', ':.4e')

    args = kwargs['args']
    model = kwargs['model']
    dataLoader = kwargs['dataLoader']
    criterions = kwargs['criterion']
    optimizer = kwargs['optimizer']
    train_infos = kwargs['infos'] 

    model.train() if args.phase == "train" else model.eval()
    for _ in tqdm(range(args.max_iteration_train if args.phase == 'train' else args.max_iteration_val)):
        inputs, targets = dataLoader.next()
        if inputs is None or targets is None:
            break
        with torch.set_grad_enabled(args.phase=="train"):
            outputs, embeds, _, _ = model(inputs)
            loss = criterions['CE'](outputs, targets)  

            if args.phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        outputs = F.softmax(outputs, dim=1)
        _, pred = torch.max(outputs, dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        losses.update(loss.item(), inputs.size(0))

    acc = skm.accuracy_score(all_targets, all_preds).astype(float)*100
    f1 = skm.f1_score(all_targets, all_preds, average='macro').astype(float)*100

    train_infos += f"\nTrain Loss: {losses.avg:.3f}"
    train_infos += f"\nTrain Accuracy: {acc:.2f}"
    train_infos += f"\nTrain F1-Score: {f1:.2f}"
    print('-'*100)
    print(train_infos)
    print('-'*100)
    return args, model, losses.avg, acc, f1


def main(args):
    configs.setup_seed(args.seed)
    for data_type in args.data_type:
        train_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='train')
        Train_DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=True, pin_memory=True, drop_last=True)
        
        args.max_iteration_train = len(Train_DataLoader)

        best_loss = float('inf')
        best_acc = float('-inf')
        best_f1 = float('-inf')

        model = models.LightWeightedModel(num_classes=args.classes).to(args.device)
        criterions = {'CE':nn.CrossEntropyLoss(label_smoothing=0.2),}
        optimizer, scheduler = _get_optimizer(args, model)

        input1 = torch.zeros([1, 3, args.img_size, args.img_size], device=args.device)
        print(summary(model, input1, show_input=False))
        print(args)
        
        for epoch in range(args.epochs):
            args.epoch = epoch + 1
            TrainDataLoaderPrefetcher = datasets.data_prefetcher(Train_DataLoader)

            train_infos = f"Database: {args.datasets}, Epoch: [{args.epoch:03d}/{args.epochs:03d}], Lr: {optimizer.param_groups[-1]['lr']}"
            args.phase = 'train'
            args, model, loss, acc, f1 = Learning(
                args=args, 
                model=model, 
                dataLoader=TrainDataLoaderPrefetcher, 
                criterion=criterions, 
                optimizer=optimizer, 
                infos=train_infos
            )

            if args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(loss)
            elif args.scheduler != '':
                scheduler.step()

            is_bestLoss = loss < best_loss
            is_bestF1 = f1 > best_f1
            is_bestAcc = acc > best_acc
            best_loss = min(loss, best_loss)
            best_f1 = max(f1, best_f1)
            best_acc = max(acc, best_acc)

            save_path = os.path.join(args.root_model, str(data_type)) 
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_checkpoint(os.path.join(save_path, f"backbone_ckpt.pth.tar"), {
                'args': args,
                'Accuracy': acc,
                'F1-Score': f1,
                'model_state_dict': model.state_dict(),
            }, is_bestF1, is_bestAcc, is_bestLoss)
            
            print(f'Best Loss: {best_loss:.6f}')
            print(f'Best F1-Score: {best_f1:.2f}')
            print(f'Best Accuracy: {best_acc:.2f}')
            print('='*100)


if __name__ == '__main__':
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in ['FV-USM', 'PLUSVein-FV3', 'MMCBNU_6000', 'UTFVP', 'NUPT-FPV']:
        args.datasets = dataset
        args = configs.get_dataset_params(args)
        main(args)