import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import torch

from utils.data_utils import make_dataset


class ImagesDataset(Dataset):

    def __init__(self, source_root, source_transform=None, dezired_size:int=-1, video_ips:int=60, verbose_make:bool=True):
        self.source_paths = sorted(make_dataset(source_root, video_ips, verbose_make))
        self.source_transform = source_transform
        self.dezired_size=dezired_size

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path).convert('RGB')
        w, h = from_im.size
        s = min(w, h)
        from_im = from_im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        if self.dezired_size<=0: self.dezired_size=s
        from_im = from_im.resize((self.dezired_size, self.dezired_size), Image.LANCZOS)
        return fname, transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])(from_im)
