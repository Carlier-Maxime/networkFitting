from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

from utils.data_utils import make_dataset


class ImagesDataset(Dataset):

    def __init__(self, source_root, source_transform=None, dezired_size:int=-1, video_ips:int=60, verbose_make:bool=True, img_mode:str='auto'):
        self.source_paths = sorted(make_dataset(source_root, video_ips, verbose_make))
        self.source_transform = source_transform
        self.dezired_size=dezired_size
        self.img_mode=img_mode

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        fname, from_path = self.source_paths[index]
        from_im = Image.open(from_path)
        if self.img_mode!='auto': from_im =from_im.convert(self.img_mode)
        w, h = from_im.size
        s = min(w, h)
        from_im = from_im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        if self.dezired_size<=0: self.dezired_size=s
        from_im = from_im.resize((self.dezired_size, self.dezired_size), Image.LANCZOS)
        tab_normalize = [0.5, 0.5, 0.5]
        if from_im.mode=="RGBA": tab_normalize += [0.5]
        return fname, transforms.Compose([transforms.ToTensor(), transforms.Normalize(tab_normalize, tab_normalize)])(from_im)
