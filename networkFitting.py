import click, torch, os
from time import perf_counter
from torch.utils.data import DataLoader
from stylegan_xl import dnnlib
import sys
sys.path.insert(1,'stylegan_xl')
from torch_utils.ops import filtered_lrelu, bias_act, upfirdn2d

from utils.ImagesDataset import ImagesDataset
from coaches.multi_id_coach import MultiIDCoach
from coaches.single_id_coach import SingleIDCoach
from utils.models_utils import load_network

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def initPlugins(verbose:bool=True):
    verbosity = 'brief' if verbose else 'none'
    bias_act._init(verbosity)
    filtered_lrelu._init(verbosity)
    upfirdn2d._init(verbosity)

def fitting(**kwargs):
    start_time = perf_counter()
    opts = dnnlib.EasyDict(kwargs)
    device = torch.device(opts.device)
    G = load_network(opts.network_path, device)
    dataset = ImagesDataset(
        opts.target_fname,
        None,
        G.img_resolution,
        opts.ips,
        opts.verbose,
        img_mode=opts.img_mode
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    initPlugins(opts.verbose)
    os.makedirs(opts.outdir, exist_ok=True)
    if opts.coache == "multi": coache = MultiIDCoach(device, dataloader, opts.network_path, opts.outdir, opts.save_latent, opts.save_video_latent, opts.save_video, opts.save_img_result, opts.seed, G=G, verbose=opts.verbose, load_w_pivot=opts.load_w_pivot)
    elif opts.coache == "single": coache = SingleIDCoach(device, dataloader, opts.network_path, opts.outdir, opts.save_latent, opts.save_video_latent, opts.save_video, opts.save_img_result, opts.seed, G=G, verbose=opts.verbose, load_w_pivot=opts.load_w_pivot)
    else: raise TypeError("a type of coache is incorrect")
    color = opts.color[1:-1].split(',')
    for i in range(len(color)): color[i]=float(color[i])
    color = torch.tensor(color).to(device)
    try: epsilon=float(opts.epsilon)
    except:
        epsilon=opts.epsilon[1:-1].split(',')
        for i in range(len(epsilon)): epsilon[i]=float(epsilon[i])
        epsilon = torch.tensor(epsilon).to(device)
    coache.train(opts.first_inv_steps, opts.inv_steps, opts.pti_steps, opts.max_images, opts.paste_color, color, epsilon, opts.save_img_step)
    if opts.verbose : print(f'Elapsed time: {(perf_counter()-start_time):.1f} s')
    

@click.command()
@click.option('--network', 'network_path', help='Network file (support pickle (.pkl) and torch (.pt))', required=True)
@click.option('--target', 'target_fname', help='Target video file or directory content target images', required=True, metavar='FILE')
@click.option('--seed', help='Random seed', type=int, default=64, show_default=True)
@click.option('--save-video', help='Save an mp4 video of fitting progress', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--save-latent', help='Save latent in file npz', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--save-video-latent', help='Save video for result latent optimization', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--save-img-result', help='Save image result of fitting target', type=bool, default=False, show_default=True, is_flag=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR', default='out')
@click.option('--first-inv-steps', help='Number of inversion steps for first image', type=int, default=1000, show_default=True)
@click.option('--inv-steps', help='Number of inversion steps for image (for first image use --first-inv-steps)', type=int, default=100, show_default=True)
@click.option('--pti-steps', help='Number of pti steps', type=int, default=2500, show_default=True)
@click.option('--ips', help='The number of image used in one second of input video', type=int, default=12, show_default=True)
@click.option('--not-verbose', 'verbose', help='this flag disable the verbose mode', default=True, is_flag=True)
@click.option('--device', help='torch device used', default='cuda', metavar='torch.device')
@click.option('--max-images', help='max images used for fitting network', default=-1, type=int)
@click.option('--paste-color', help='copies pixels that have the correct color from the generated image to the target image for the loss calculation', default=False, type=bool, is_flag=True)
@click.option('--color', help='color used for paste color', default='[0,255,0]')
@click.option('--epsilon', help='a epsilon used for paste color', default='[150,100,150]')
@click.option('--save-img-step', help='save a image step (Warning: increase step duration)', default=False, type=bool, is_flag=True)
@click.option('--coache', type=click.Choice(["multi", "single"], case_sensitive=False), default="single")
@click.option('--img-mode', type=click.Choice(["auto","RGB","RGBA"]), default='auto', help="choice mode for loading image")
@click.option('--load-w-pivot', type=bool, default=False, help="enable load w_pivot by file in outdir", is_flag=True)
def main(**kwargs):
    fitting(**kwargs)

if __name__ == "__main__":
    main()
