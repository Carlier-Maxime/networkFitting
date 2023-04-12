import os
import cv2
from PIL import Image
from tqdm import trange

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]

VIDEO_EXTENSIONS = [
    '.mkv', '.mp4'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_video_file(filename):
    return any(filename.endswith(extension) for extension in VIDEO_EXTENSIONS)

def tensor2im(var):
    # var shape: (3, H, W)
    var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
    var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return Image.fromarray(var.astype('uint8'))

def make_dataset_by_dir(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                fname = fname.split('.')[0]
                images.append((fname, path))
    return images

def make_dataset_by_video(video_path:str, ips:int, verbose:bool=True):
    video=cv2.VideoCapture(video_path)
    steps=(video.get(cv2.CAP_PROP_FRAME_COUNT)/video.get(cv2.CAP_PROP_FPS))*ips
    frame_step = int(video.get(cv2.CAP_PROP_FPS)/ips)
    if frame_step==0:
        frame_step=1
    images = []
    read_counter=0
    vname = video_path.split("/")[-1].split(".")[0]
    vdir = f"data/{vname}_imgs"
    os.makedirs(vdir,exist_ok=True)
    for i in trange(1,int(steps+1), desc='Make dataset', unit='image', disable=(not verbose)):
        target_pil = None
        ret,cv2_im = video.read()
        while (read_counter % frame_step != 0):
            ret,cv2_im = video.read()
            if not ret: break
            read_counter+=1
        if not ret: break
        read_counter+=1
        converted = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        img_name = f"frame_{read_counter-1}"
        img_path = f"{vdir}/{img_name}.png"
        Image.fromarray(converted).save(img_path)
        images.append([img_name, img_path])
    return images

def make_dataset(path, ips:int=60, verbose:bool=True):
    assert os.path.isdir(path) or is_video_file(path) or is_image_file(path), '%s is not a valid directory or video or image' % path
    if os.path.isdir(path): return make_dataset_by_dir(path)
    elif is_video_file(path): return make_dataset_by_video(path,ips,verbose)
    elif is_image_file(path): return [[path.split('/')[-1].split('.')[0],path]]
    return []
