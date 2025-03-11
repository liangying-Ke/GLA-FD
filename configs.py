import os
import torch
import random
import argparse
import numpy as np


def get_basic_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default="FVUSM")
    parser.add_argument('--optim', type=str, default="adamw")
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args([])


def get_dataset_params(args):
    if args.datasets == 'FV-USM':
        args.split = 0.3
        args.classes = 492
        args.pad_height_width = 300
        args.data_type = [None]
        args.data_root = 'D:/LiangYing/Research/Databases/vein/FV-USM'
        args.root_model = './checkpoint/FV-USM'
        args.annot_file = './datasets/annotations_fvusm.pkl'

    elif args.datasets == 'PLUSVein-FV3':
        args.split = 0.3
        args.classes = 360
        args.pad_height_width = 736
        args.data_type = ['LED', 'LASER']
        args.data_root = 'D:/LiangYing/Research/Databases/vein/PLUSVein-FV3/PLUSVein-FV3-ROI_combined/ROI'
        args.root_model = './checkpoint/PLUSVein-FV3'
        args.annot_file = './datasets/annotations_plusvein.pkl'

    elif args.datasets == 'MMCBNU_6000':
        args.split = 0.2
        args.classes = 600
        args.pad_height_width = 128
        args.data_type = [None]
        args.data_root = 'D:/LiangYing/Research/Databases/vein/MMCBNU_6000/ROIs'
        args.root_model = './checkpoint/MMCBNU'
        args.annot_file = './datasets/annotations_mmcbnu.pkl'
            
    elif args.datasets == 'UTFVP':
        args.split = 0.5
        args.classes = 360
        args.pad_height_width = 672
        args.data_type = [None]
        args.data_root = 'D:/LiangYing/Research/Databases/vein/UTFVP/data'
        args.root_model = './checkpoint/UTFVP'
        args.annot_file = './datasets/annotations_utfvp.pkl'
            
    elif args.datasets == 'NUPT-FPV':
        args.split = 0.5
        args.classes = 840
        args.pad_height_width = 450
        args.data_type = [None]
        args.data_root = 'D:/LiangYing/Research/Databases/vein/NUPT-FPV'
        args.root_model = './checkpoint/NUPT'
        args.annot_file = './datasets/annotations_nupt.pkl'
    return args


def get_optim_params(args):
    if args.optim == 'adamw':
        args.lr = 2e-3
        args.weight_decay = 1e-2
    if args.optim == 'sgd':
        args.lr = 1e-1
        args.momentum = 0.9
        args.weight_decay = 2e-4
        
    if 'cosine' in args.scheduler:
        args.T_max = 16
        args.eta_min = 1e-6
        
    if 'ReduceLROnPlateau' in args.scheduler:
        args.factor = 0.9
        args.patience = 10
        args.verbose = True
    return args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def setup_seed(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def get_all_params():
    args = get_basic_params()
    args = get_optim_params(args)
    args = get_dataset_params(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args