import cv2
import torch
import pickle
import numpy as np
from torch.utils.data import Dataset

from torchvision.transforms import v2
import torch.nn.functional as F


def get_padding(image):
    imsize = image.shape
    max_length = max(imsize[1], imsize[2])
    h_padding = (max_length - imsize[1]) / 2
    v_padding = (max_length - imsize[2]) / 2

    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    return (int(l_pad), int(t_pad), int(r_pad), int(b_pad))


class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        return F.pad(img, get_padding(img), mode=self.padding_mode, value=self.fill)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.format(self.fill, self.padding_mode)
    

class ImagesDataset(Dataset):
    def __init__(self, args, data_type, phase='train'):
        self.args = args
        self.phase = phase
        self.data_type = data_type
        self._read_path_label()
        self._setup_transformation(self.phase)
        self._get_label_list()

    def _read_path_label(self):
        pkl = pickle.load(open(self.args.annot_file, 'rb'))

        if self.data_type is not None:
            pkl = pkl[self.data_type]
        if self.phase == 'train':
            self.data = pkl['Training_Set']
        elif self.phase == 'val':
            self.data = pkl['Validating_Set']
        elif self.phase == 'test':
            self.data = pkl['Testing_Set']
        else:
            raise ValueError("train mode must be in : Train or Validation")
        self.dataset_size = len(self.data)
        self._get_mean_std(pkl['Training_Set'])

    def _get_mean_std(self, data):
        dataset_size = len(data)
        self.mean = np.zeros(1)
        self.std = np.zeros(1)
        for idx in range(dataset_size):
            img = cv2.imread(data[idx]['path'], flags=0)
            self.mean += np.mean(img)
            self.std += np.std(img)
        self.mean = self.mean / dataset_size / 255.
        self.std = self.std / dataset_size / 255.
        self.mean = list(self.mean.repeat(3))
        self.std = list(self.std.repeat(3))
            
    def _get_label_list(self):
        self.label_list = [data['label'] for data in self.data]
    
    def _setup_transformation(self, phase):
        self.phase = phase
        transform_list = [
            v2.ToImage(),  
            v2.ToDtype(torch.uint8, scale=True), 
        ]

        if self.phase == 'train':
            transform_list.extend([
                NewPad(),
                v2.Resize(size=(self.args.img_size, self.args.img_size), antialias=True),  
            ])
        else:
            transform_list.extend([
                NewPad(),
                v2.Resize(size=(self.args.img_size, self.args.img_size), antialias=True),  
            ])
        

        transform_list.extend([
            v2.ToDtype(torch.float32, scale=True), 
            v2.Normalize(mean=self.mean, std=self.std),
        ])

        self.transforms = v2.Compose(transform_list)
    

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx]['path'])
        img = self.transforms(img)

        label = self.data[idx]['label']
        label = torch.tensor(label, dtype=torch.long)
        return img, label


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_image, self.next_label = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.next_image
        label = self.next_label 
        self.preload()
        return image, label
