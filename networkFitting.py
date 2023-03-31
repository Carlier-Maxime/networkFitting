import click, torch, sys, cv2, PIL, numpy as np, os
from time import perf_counter
from tqdm import trange
sys.path.insert(1, './stylegan-xl')
import dnnlib, legacy
from torch_utils.ops import filtered_lrelu, bias_act

def loadNetwork(network_pkl:str, device:torch.device, verbose:bool=True):
    if verbose: print('Loading networks from "%s"...' % network_pkl)
    device = torch.device(device)
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].to(device) # type: ignore
    return G

def getImagesFromVideo(filename:str, ips:int, device:torch.device, verbose:bool=True, dezired_size:int=-1):
    video=cv2.VideoCapture(filename)
    steps=(video.get(cv2.CAP_PROP_FRAME_COUNT)/video.get(cv2.CAP_PROP_FPS))*ips
    frame_step = int(video.get(cv2.CAP_PROP_FPS)/ips)
    if frame_step==0:
        frame_step=1
    images = []
    read_counter=0
    for i in trange(1,int(steps+1), desc='Loading images', unit='image', disable=(not verbose)):
        target_pil = None
        ret,cv2_im = video.read()
        while (read_counter % frame_step != 0):
            ret,cv2_im = video.read()
            if not ret:
                break
            read_counter+=1
        if not ret:
            break
        converted = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
        target_pil = PIL.Image.fromarray(converted)
        read_counter+=1
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        if dezired_size<=0: dezired_size=s
        target_pil = target_pil.resize((dezired_size, dezired_size), PIL.Image.LANCZOS)
        img_np = np.array(target_pil, dtype=np.uint8)
        images.append(torch.tensor(img_np.transpose([2, 0, 1]), device=device))
    return images

def initPlugins():
    bias_act._init()
    filtered_lrelu._init()

def fitting(**kwargs):
    start_time = perf_counter()
    opts = dnnlib.EasyDict(kwargs)
    device = torch.device(opts.device)
    G = loadNetwork(opts.network_pkl, device, opts.verbose)
    images = getImagesFromVideo(opts.target_fname, opts.ips, device, opts.verbose, G.img_resolution)
    initPlugins()
    

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target video file to project to', required=True, metavar='FILE')
@click.option('--seed', help='Random seed', type=int, default=64, show_default=True)
@click.option('--save-video', help='Save an mp4 video of fitting progress', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--save-latent', help='Save latent in file npz', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--save-video-latent', help='Save video for result latent optimization', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR', default='out')
@click.option('--first-inv-steps', help='Number of inversion steps for first image', type=int, default=1000, show_default=True)
@click.option('--inv-steps', help='Number of inversion steps for image (for first image use --first-inv-steps)', type=int, default=100, show_default=True)
@click.option('--w-init', help='path to inital latent', type=str, default='', show_default=True)
@click.option('--pti-steps', help='Number of pti steps', type=int, default=2500, show_default=True)
@click.option('--ips', help='The number of image used in one second of video', type=int, default=10, show_default=True)
@click.option('--not-verbose', 'verbose', help='this flag disable the verbose mode', default=True, is_flag=True)
@click.option('--device', help='torch device used', default='cuda', metavar='torch.device')
def main(**kwargs):
    fitting(**kwargs)

if __name__ == "__main__":
    main()
