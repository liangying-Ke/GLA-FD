import warnings
warnings.filterwarnings("ignore")

import os
import torch
import models 
import datasets
import random
from utils import *
from tqdm import tqdm
from pytorch_model_summary import summary
from ptflops import get_model_complexity_info
import configs
import torch.nn.functional as F
import numpy as np
import sklearn.metrics as skm 
from sklearn.model_selection import KFold
from tqdm.contrib import tzip


def _get_model(args):
    model = models.Model(num_classes=args.classes).to(args.device)
    return model

dist_type='cosine'

def calculate_metrics(distances, labels, threshold):
    if dist_type == 'cosine':
        preds = np.greater(distances, threshold)
    elif dist_type == 'euclidean':
        preds = np.less(distances, threshold)

    tn, fp, fn, tp = skm.confusion_matrix(labels, preds).ravel()
    fpr = float(fp) / (tn + fp) * 100
    fnr = float(fn) / (tp + fn) * 100
    acc = float(tp + tn) / distances.size * 100
    return acc, fpr, fnr


def calculate_average_metrics(dists, labels, num_folds=5):
    dist_min, dist_max = np.min(dists), np.max(dists)
    thresholds = np.arange(0, np.ceil(dist_max), 0.01)
    print(f'Distance Min: {dist_min} Max: {dist_max}')
    eer_list = []
    acc_list = []
    folds = KFold(n_splits=num_folds, shuffle=True)
    for train_set, test_set in folds.split(labels):
        _acc_fold = []
        _fpr_fold = []
        _fnr_fold = []
        for threshold in thresholds:
            acc, fpr, fnr = calculate_metrics(dists[train_set], labels[train_set], threshold)
            _acc_fold.append(acc)
            _fpr_fold.append(fpr)
            _fnr_fold.append(fnr)
        eer_idx = np.nanargmin(np.absolute((np.array(_fnr_fold) - np.array(_fpr_fold))))
        eer = (_fpr_fold[eer_idx] + _fnr_fold[eer_idx]) / 2

        best_threshold = thresholds[np.argmax(_acc_fold)]
        acc, fpr, fnr = calculate_metrics(dists[test_set], labels[test_set], best_threshold)

        eer_list.append(eer)
        acc_list.append(acc)
    return np.mean(acc_list), np.mean(eer_list)


def evaluate(*kwargs):
    args = kwargs[0]
    model = kwargs[1]
    test_DataLoader = kwargs[2]
    
    dists = []
    labels = []
    embeds_list = []
    targets_list = []
    outputs_list = []
    model.eval()
    for inputs, targets in tqdm(test_DataLoader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        with torch.set_grad_enabled(False):
            outputs, embeds, _, _ = model(inputs)
            embeds = F.normalize(embeds, dim=1)
        _, outputs = torch.max(outputs, 1)
        outputs_list.extend(outputs.cpu().detach().numpy())
        embeds_list.extend(embeds.cpu().detach().numpy())
        targets_list.extend(targets.cpu().detach().numpy())
    
    outputs_list = np.array(outputs_list)
    embeds_list = np.array(embeds_list)
    targets_list = np.array(targets_list)

    for embed_A, target_A in tzip(embeds_list, targets_list):
        for embed_B, target_B in zip(embeds_list, targets_list):
            if dist_type == 'cosine':
                dist = np.dot(embed_A, embed_B) / (np.linalg.norm(embed_A) * np.linalg.norm(embed_B))
                dist = (dist + 1) / 2
            elif dist_type == 'euclidean':
                dist = np.sum((embed_A - embed_B) ** 2) ** 0.5

            label = int(target_A == target_B)
            dists.append(dist)
            labels.append(label)
    dists = np.array(dists, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    dists_1 = dists[labels==1]
    dists_2 = dists[labels==0]
    random.shuffle(dists_2)
    dists_2 = dists_2[:len(dists_1)]
    dists = np.hstack([dists_1, dists_2])

    labels_1 = labels[labels==1]
    labels_2 = labels[labels==0]
    random.shuffle(labels_2)
    labels_2 = labels_2[:len(labels_1)]
    labels = np.hstack([labels_1, labels_2])

    acc, eer = calculate_average_metrics(dists, labels)
    return eer, acc


def main(args, database_results={}):
    configs.setup_seed(args.seed)
    for data_type in args.data_type:
        test_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='test')
        test_DataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True)
        
        model = _get_model(args)
        input1 = torch.zeros([1, 3, args.img_size, args.img_size], device=args.device)
        print(summary(model, input1, show_input=False))
        macs, _ = get_model_complexity_info(model, (3, args.img_size, args.img_size), as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('='*100)

        best_eer = float('inf')
        for metrics in ['Loss', 'Acc', 'F1']:
            weights = torch.load(os.path.join(args.root_model, str(data_type), f"Backbone_ckpt.best{metrics}.pth.tar"))

            model.load_state_dict(weights['model_state_dict']) 
            eer, acc = evaluate(args, model, test_DataLoader)
            
            is_best = best_eer > eer
            best_eer = min(best_eer, eer)
            if is_best:
                database_results[f'{args.datasets}_{data_type}'] = {
                    'acc':f'{acc:.2f}',
                    'eer':f'{eer:.2f}'
                }
            print(f'Database: {args.datasets}, data_type: {data_type}, Metrics: {metrics}')
            print(f'Accuracy: {acc:.2f}')
            print(f'EER: {eer:04f}')
            print('-'*100)
        print('='*100)
    return database_results

if __name__ == '__main__':
    database_results = {}
    args = configs.get_all_params()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in ['FV-USM', 'PLUSVein-FV3', 'MMCBNU_6000', 'UTFVP', 'NUPT-FPV']:
        args.datasets = dataset
        args = configs.get_dataset_params(args)
        database_results = main(args, database_results)
        print(database_results)
